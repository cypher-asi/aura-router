//! LLM proxy handler — receives requests, checks credits, forwards to provider.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

use aura_router_auth::AuthUser;
use aura_router_core::AppError;
use aura_router_proxy::{billing, providers, stats, stream};

use crate::state::AppState;

/// POST /v1/messages — Anthropic-compatible proxy endpoint.
///
/// Flow:
/// 1. Auth (JWT)
/// 2. Extract model from request body
/// 3. Resolve provider
/// 4. Pre-check credits via z-billing
/// 5. [ENRICHMENT HOOK — future: RAG, memory, prompt modification]
/// 6. Forward to provider with platform API key
/// 7. Debit credits + record usage (fire-and-forget)
/// 8. Return response
pub async fn messages(
    auth: AuthUser,
    State(state): State<AppState>,
    body: bytes::Bytes,
) -> Result<Response, AppError> {
    // Parse just the model and stream fields from the request body
    let request_value: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    let model = request_value
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::BadRequest("Missing 'model' field".into()))?;

    let is_streaming = request_value
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Resolve provider from model name
    let provider = providers::resolve_provider(model)
        .ok_or_else(|| AppError::BadRequest(format!("Unsupported model: {model}")))?;

    // Pre-check credits (conservative minimum: 1 credit)
    let balance = billing::check_credits(
        &state.http_client,
        &state.z_billing_url,
        &state.z_billing_api_key,
        &auth.user_id,
        1,
    )
    .await?;

    if !balance.sufficient {
        return Err(AppError::InsufficientCredits {
            balance: balance.balance_cents,
            required: 1,
        });
    }

    // [ENRICHMENT HOOK — v1: pass-through, future: RAG/memory/prompt modification]

    // Forward to provider
    let upstream_url = providers::provider_url(&provider);
    let upstream_headers = providers::provider_headers(&provider, &state.anthropic_api_key);

    let upstream_resp = state
        .http_client
        .post(upstream_url)
        .headers(upstream_headers)
        .body(body.to_vec())
        .send()
        .await
        .map_err(|e| AppError::ProviderError(format!("Provider unreachable: {e}")))?;

    let upstream_status = upstream_resp.status();

    // If provider returned an error, pass it through
    if !upstream_status.is_success() {
        let error_body = upstream_resp.bytes().await.unwrap_or_default();
        return Ok((
            StatusCode::from_u16(upstream_status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            [(header::CONTENT_TYPE, "application/json")],
            Body::from(error_body),
        )
            .into_response());
    }

    if is_streaming {
        return handle_streaming(auth, state, model, upstream_resp).await;
    }

    handle_non_streaming(auth, state, model, upstream_resp).await
}

/// Handle non-streaming response: read full body, extract usage, debit, return.
async fn handle_non_streaming(
    auth: AuthUser,
    state: AppState,
    model: &str,
    upstream_resp: reqwest::Response,
) -> Result<Response, AppError> {
    let response_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| AppError::ProviderError(format!("Failed to read provider response: {e}")))?;

    // Extract token counts from response
    let response_value: serde_json::Value =
        serde_json::from_slice(&response_bytes).unwrap_or_default();

    let input_tokens = response_value
        .pointer("/usage/input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let output_tokens = response_value
        .pointer("/usage/output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    spawn_post_request_tasks(&state, &auth.user_id, model, input_tokens, output_tokens);

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        Body::from(response_bytes),
    )
        .into_response())
}

/// Handle streaming response: tee SSE stream to client while capturing billing data.
async fn handle_streaming(
    auth: AuthUser,
    state: AppState,
    model: &str,
    upstream_resp: reqwest::Response,
) -> Result<Response, AppError> {
    let model_owned = model.to_string();
    let (tee_stream, usage_rx) = stream::proxy_stream(upstream_resp);

    // Spawn task to handle billing after stream completes
    let billing_state = state.clone();
    let user_id = auth.user_id.clone();
    tokio::spawn(async move {
        if let Ok(usage) = usage_rx.await {
            let model = usage.model.as_deref().unwrap_or(&model_owned);
            spawn_post_request_tasks(
                &billing_state,
                &user_id,
                model,
                usage.input_tokens,
                usage.output_tokens,
            );
        }
    });

    let body = Body::from_stream(tee_stream);

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response())
}

/// Fire-and-forget tasks: debit z-billing + record to aura-network.
fn spawn_post_request_tasks(
    state: &AppState,
    user_id: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
) {
    let event_id = uuid::Uuid::new_v4().to_string();
    let model_owned = model.to_string();
    let user_id_owned = user_id.to_string();

    // Debit z-billing
    {
        let client = state.http_client.clone();
        let billing_url = state.z_billing_url.clone();
        let billing_key = state.z_billing_api_key.clone();
        let user_id = user_id_owned.clone();
        let model = model_owned.clone();
        tokio::spawn(async move {
            if let Err(e) = billing::report_usage(
                &client,
                &billing_url,
                &billing_key,
                &event_id,
                &user_id,
                "anthropic",
                &model,
                input_tokens,
                output_tokens,
            )
            .await
            {
                tracing::warn!(error = %e, "Failed to debit credits via z-billing");
            }
        });
    }

    // Record to aura-network
    if let (Some(ref network_url), Some(ref network_token)) =
        (&state.aura_network_url, &state.aura_network_token)
    {
        let client = state.http_client.clone();
        let url = network_url.clone();
        let token = network_token.clone();
        let user_id = user_id_owned;
        let model = model_owned;
        tokio::spawn(async move {
            stats::record_usage(
                &client,
                &url,
                &token,
                &user_id,
                &model,
                input_tokens,
                output_tokens,
                (input_tokens + output_tokens) as f64 * 0.00001,
            )
            .await;
        });
    }
}
