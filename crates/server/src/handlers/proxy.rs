//! LLM proxy handler — receives requests, checks credits, forwards to provider.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

use aura_router_auth::AuthUser;
use aura_router_core::AppError;
use aura_router_proxy::{billing, providers, stats, storage, stream};

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
    headers: axum::http::HeaderMap,
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

    // Extract session context from custom headers (optional, for storage recording)
    let session_ctx = storage::SessionContext::from_headers(&headers);

    // Extract user content from the request for storage (last user message)
    let user_content = request_value
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|msgs| {
            msgs.iter()
                .rfind(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        })
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .unwrap_or("")
        .to_string();

    // [ENRICHMENT HOOK — v1: pass-through, future: RAG/memory/prompt modification]

    // Resolve API key for the provider
    let api_key = match provider {
        providers::Provider::Anthropic => state.anthropic_api_key.clone(),
        providers::Provider::OpenAi => state
            .openai_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("OpenAI provider not configured".into()))?,
    };

    // Forward to provider
    let upstream_url = providers::provider_url(&provider);
    let upstream_headers = providers::provider_headers(&provider, &api_key)
        .ok_or_else(|| AppError::Internal("Invalid API key format".into()))?;

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

    let provider_name = provider.name();

    if is_streaming {
        return handle_streaming(
            auth,
            state,
            model,
            provider_name,
            upstream_resp,
            session_ctx,
            user_content,
        )
        .await;
    }

    handle_non_streaming(
        auth,
        state,
        model,
        provider_name,
        upstream_resp,
        session_ctx,
        user_content,
    )
    .await
}

/// Handle non-streaming response: read full body, extract usage, debit, return.
async fn handle_non_streaming(
    auth: AuthUser,
    state: AppState,
    model: &str,
    provider_name: &str,
    upstream_resp: reqwest::Response,
    session_ctx: Option<storage::SessionContext>,
    user_content: String,
) -> Result<Response, AppError> {
    let response_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| AppError::ProviderError(format!("Failed to read provider response: {e}")))?;

    // Extract token counts and content from response
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

    spawn_post_request_tasks(
        &state,
        &auth.user_id,
        provider_name,
        model,
        input_tokens,
        output_tokens,
    );

    // Store messages to aura-storage if session context is present
    if let Some(ctx) = session_ctx {
        if let (Some(ref storage_url), Some(ref storage_token)) =
            (&state.aura_storage_url, &state.aura_storage_token)
        {
            let assistant_content = response_value
                .get("content")
                .and_then(|v| v.as_array())
                .and_then(|blocks| {
                    blocks
                        .iter()
                        .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                        .and_then(|b| b.get("text").and_then(|t| t.as_str()))
                })
                .unwrap_or("")
                .to_string();

            let client = state.http_client.clone();
            let url = storage_url.clone();
            let token = storage_token.clone();
            let user_id = auth.user_id.clone();
            tokio::spawn(async move {
                storage::store_messages(
                    &client,
                    &url,
                    &token,
                    &ctx,
                    &user_id,
                    &user_content,
                    &assistant_content,
                    None,
                    input_tokens,
                    output_tokens,
                )
                .await;
            });
        }
    }

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
    provider_name: &str,
    upstream_resp: reqwest::Response,
    session_ctx: Option<storage::SessionContext>,
    user_content: String,
) -> Result<Response, AppError> {
    let model_owned = model.to_string();
    let provider_owned = provider_name.to_string();
    let (tee_stream, usage_rx) = stream::proxy_stream(upstream_resp);

    // Spawn task to handle billing + storage after stream completes
    let billing_state = state.clone();
    let user_id = auth.user_id.clone();
    tokio::spawn(async move {
        if let Ok(usage) = usage_rx.await {
            let model = usage.model.as_deref().unwrap_or(&model_owned);
            spawn_post_request_tasks(
                &billing_state,
                &user_id,
                &provider_owned,
                model,
                usage.input_tokens,
                usage.output_tokens,
            );

            // Store messages to aura-storage if session context is present
            if let Some(ctx) = session_ctx {
                if let (Some(ref storage_url), Some(ref storage_token)) = (
                    &billing_state.aura_storage_url,
                    &billing_state.aura_storage_token,
                ) {
                    storage::store_messages(
                        &billing_state.http_client,
                        storage_url,
                        storage_token,
                        &ctx,
                        &user_id,
                        &user_content,
                        "[streamed response]",
                        None,
                        usage.input_tokens,
                        usage.output_tokens,
                    )
                    .await;
                }
            }
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
    provider_name: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
) {
    let event_id = uuid::Uuid::new_v4().to_string();
    let model_owned = model.to_string();
    let user_id_owned = user_id.to_string();
    let provider_owned = provider_name.to_string();

    // Debit z-billing
    {
        let client = state.http_client.clone();
        let billing_url = state.z_billing_url.clone();
        let billing_key = state.z_billing_api_key.clone();
        let user_id = user_id_owned.clone();
        let model = model_owned.clone();
        let provider = provider_owned.clone();
        tokio::spawn(async move {
            if let Err(e) = billing::report_usage(
                &client,
                &billing_url,
                &billing_key,
                &event_id,
                &user_id,
                &provider,
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
