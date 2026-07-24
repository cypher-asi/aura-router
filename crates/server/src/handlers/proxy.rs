//! LLM proxy handler — receives requests, checks credits, forwards to provider.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

use aura_router_auth::{AuthUser, AuthUserOrGuest};
use aura_router_core::AppError;
use aura_router_proxy::{anthropic_compat, billing, providers, stats, storage, stream};

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
    auth: AuthUserOrGuest,
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: bytes::Bytes,
) -> Result<Response, AppError> {
    let request_start = std::time::Instant::now();

    // Rate limit check
    if let Err(retry_after) = state.rate_limiter.check(&auth.user_id) {
        tracing::warn!(user_id = %auth.user_id, retry_after, "Rate limited");
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [
                (header::RETRY_AFTER, retry_after.to_string()),
                (header::CONTENT_TYPE, "application/json".to_string()),
            ],
            Body::from(
                serde_json::json!({
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": format!("Too many requests. Retry after {retry_after} seconds.")
                    }
                })
                .to_string(),
            ),
        )
            .into_response());
    }

    // Parse just the model and stream fields from the request body
    let request_value: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    let requested_model = request_value
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::BadRequest("Missing 'model' field".into()))?;

    let is_streaming = request_value
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Resolve the Aura-facing model id into an upstream provider/model pair.
    let resolved_model = providers::resolve_model(requested_model)
        .ok_or_else(|| AppError::BadRequest(format!("Unsupported model: {requested_model}")))?;
    let provider = resolved_model.provider;
    anthropic_compat::validate_request(provider, &request_value).map_err(AppError::BadRequest)?;

    // OpenAI rejects function tools + reasoning_effort on /v1/chat/completions
    // for the gpt-5 family; route any tool-bearing OpenAI request through the
    // /v1/responses API instead (which supports tools + reasoning together).
    let openai_api = providers::openai_api_for_request(provider, &request_value);

    // Public-guest requests (from the logged-out aura.ai surface) skip
    // billing entirely — cost is capped by the upstream rate limiter in
    // aura-os-server (3 turns/guest, 30/IP/day, global daily ceiling).
    let is_public_guest = auth.user_id == "public-guest";

    if !is_public_guest {
        // Pre-check credits using the z-billing pricing table when possible.
        let balance = billing::check_credits(
            &state.http_client,
            &state.z_billing_url,
            &state.z_billing_api_key,
            &auth.user_id,
            0,
            Some(provider.name()),
            Some(requested_model),
        )
        .await?;

        if !balance.sufficient {
            return Err(AppError::InsufficientCredits {
                balance: balance.balance_cents,
                required: balance.required_cents,
            });
        }
    }

    // Extract session context from custom headers (optional, for storage recording)
    let session_ctx = storage::SessionContext::from_headers(&headers);

    // Extract user content from the request for storage (last user message)
    let user_content = anthropic_compat::extract_last_user_text(&request_value);

    // [ENRICHMENT HOOK — v1: pass-through, future: RAG/memory/prompt modification]

    // Resolve API key for the provider
    let api_key = resolve_provider_api_key(&state, resolved_model.provider)?;

    // Forward to provider. Google encodes the model + streaming mode in the
    // URL path, so it is built separately from the static per-provider URLs.
    let upstream_url: String = match provider {
        providers::Provider::OpenAi => providers::openai_endpoint_url(openai_api).to_string(),
        providers::Provider::Xai => providers::xai_endpoint_url(openai_api).to_string(),
        providers::Provider::Google => {
            providers::google_endpoint_url(resolved_model.upstream_model, is_streaming)
        }
        other => providers::provider_url(&other).to_string(),
    };
    let mut upstream_headers = providers::provider_headers(&provider, &api_key)
        .ok_or_else(|| AppError::Internal("Invalid API key format".into()))?;
    if provider == providers::Provider::Anthropic {
        if let Some(inbound_beta) = headers
            .get("anthropic-beta")
            .and_then(|value| value.to_str().ok())
        {
            providers::merge_anthropic_beta_header(&mut upstream_headers, inbound_beta)
                .ok_or_else(|| AppError::Internal("Invalid Anthropic beta header".into()))?;
        }
    }
    let mut upstream_request_value = match openai_api {
        providers::OpenAiApi::Responses => anthropic_compat::anthropic_request_to_responses(
            provider,
            &request_value,
            resolved_model.upstream_model,
        )
        .map_err(AppError::BadRequest)?,
        providers::OpenAiApi::ChatCompletions => anthropic_compat::request_to_upstream(
            provider,
            resolved_model.upstream_model,
            &request_value,
        )
        .map_err(AppError::BadRequest)?,
    };
    apply_provider_request_controls(
        provider,
        openai_api,
        &mut upstream_headers,
        &mut upstream_request_value,
        session_ctx.as_ref(),
    );
    if provider == providers::Provider::Anthropic && !is_public_guest {
        set_anthropic_user_metadata(&mut upstream_request_value, &auth.user_id)?;
    }
    let upstream_body = serde_json::to_vec(&upstream_request_value)
        .map_err(|e| AppError::Internal(format!("Failed to encode upstream body: {e}")))?;

    let upstream_resp = state
        .http_client
        .post(upstream_url)
        .headers(upstream_headers)
        .body(upstream_body)
        .send()
        .await
        .map_err(|e| AppError::ProviderError(format!("Provider unreachable: {e}")))?;

    let upstream_status = upstream_resp.status();
    let provider_name = provider.name();
    // Convert to AuthUser for internal functions (same fields, stricter type).
    let auth = auth.into_auth_user();

    // Normalize upstream failures into Anthropic-compatible error envelopes.
    if !upstream_status.is_success() {
        let error_body = upstream_resp.bytes().await.unwrap_or_default();
        return Ok(normalize_upstream_error(upstream_status, &error_body));
    }

    let provider_request_id = upstream_request_id(&upstream_resp);

    if is_streaming {
        return handle_streaming(
            auth,
            state,
            requested_model,
            provider,
            openai_api,
            provider_name,
            upstream_resp,
            provider_request_id,
            session_ctx,
            user_content,
            request_start,
        )
        .await;
    }

    handle_non_streaming(
        auth,
        state,
        requested_model,
        provider_name,
        openai_api,
        upstream_resp,
        provider_request_id,
        session_ctx,
        user_content,
        request_start,
    )
    .await
}

fn resolve_provider_api_key(
    state: &AppState,
    provider: providers::Provider,
) -> Result<String, AppError> {
    match provider {
        providers::Provider::Anthropic => Ok(state.anthropic_api_key.clone()),
        providers::Provider::OpenAi => state
            .openai_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("OpenAI provider not configured".into())),
        providers::Provider::Xai => state
            .xai_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("xAI provider not configured".into())),
        providers::Provider::Fireworks => state
            .fireworks_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("Fireworks provider not configured".into())),
        providers::Provider::DeepSeek => state
            .deepseek_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("DeepSeek provider not configured".into())),
        providers::Provider::Google => state
            .google_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("Google provider not configured".into())),
    }
}

/// Handle non-streaming response: read full body, extract usage, debit, return.
async fn handle_non_streaming(
    auth: AuthUser,
    state: AppState,
    model: &str,
    provider_name: &str,
    openai_api: providers::OpenAiApi,
    upstream_resp: reqwest::Response,
    provider_request_id: String,
    session_ctx: Option<storage::SessionContext>,
    user_content: String,
    request_start: std::time::Instant,
) -> Result<Response, AppError> {
    let response_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| AppError::ProviderError(format!("Failed to read provider response: {e}")))?;

    let upstream_value: serde_json::Value = serde_json::from_slice(&response_bytes)
        .map_err(|e| AppError::ProviderError(format!("Provider returned invalid JSON: {e}")))?;
    let provider_reported_cost_ticks = upstream_value
        .pointer("/usage/cost_in_usd_ticks")
        .and_then(|value| value.as_u64());
    let provider_reported_cost_cents =
        provider_reported_cost_ticks.and_then(billing::marked_up_cost_cents_from_usd_ticks);
    let provider_reported_cost_microusd =
        provider_reported_cost_ticks.and_then(billing::provider_cost_microusd_from_usd_ticks);
    let response_value = match openai_api {
        providers::OpenAiApi::Responses => {
            anthropic_compat::openai_responses_to_anthropic(&upstream_value, model)
                .map_err(AppError::ProviderError)?
        }
        providers::OpenAiApi::ChatCompletions => anthropic_compat::response_from_upstream(
            provider_from_name(provider_name),
            model,
            &upstream_value,
        )
        .map_err(AppError::ProviderError)?,
    };
    let normalized_response_bytes = serde_json::to_vec(&response_value)
        .map_err(|e| AppError::Internal(format!("Failed to encode normalized response: {e}")))?;

    let input_tokens = response_value
        .pointer("/usage/input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let output_tokens = response_value
        .pointer("/usage/output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_creation_input_tokens = response_value
        .pointer("/usage/cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_read_input_tokens = response_value
        .pointer("/usage/cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_creation_5m_input_tokens = response_value
        .pointer("/usage/cache_creation/ephemeral_5m_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_creation_1h_input_tokens = response_value
        .pointer("/usage/cache_creation/ephemeral_1h_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let web_search_requests = response_value
        .pointer("/usage/server_tool_use/web_search_requests")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let web_fetch_requests = response_value
        .pointer("/usage/server_tool_use/web_fetch_requests")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let code_execution_requests = response_value
        .pointer("/usage/server_tool_use/code_execution_requests")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let service_tier = response_value
        .pointer("/usage/service_tier")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let inference_geo = response_value
        .pointer("/usage/inference_geo")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let speed = response_value
        .pointer("/usage/speed")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let org_id_ref = session_ctx.as_ref().and_then(|c| c.org_id.as_deref());
    let project_id_ref = session_ctx.as_ref().and_then(|c| c.project_id.as_deref());
    let agent_id_ref = session_ctx
        .as_ref()
        .and_then(|c| c.project_agent_id.as_deref());

    let duration_ms = request_start.elapsed().as_millis() as u64;

    spawn_post_request_tasks(
        &state,
        &auth.user_id,
        org_id_ref,
        project_id_ref,
        agent_id_ref,
        provider_name,
        model,
        billing::UsageCostInput {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_creation_5m_input_tokens,
            cache_creation_1h_input_tokens,
            cache_read_input_tokens,
            web_search_requests,
            web_fetch_requests,
            code_execution_requests,
            service_tier,
            inference_geo,
            speed,
        },
        provider_reported_cost_cents,
        provider_reported_cost_microusd,
        duration_ms,
        billing_event_id(provider_name, &provider_request_id),
    );

    // Store messages to aura-storage if session context is present
    if let Some(ctx) = session_ctx {
        if let (Some(ref storage_url), Some(ref storage_token)) =
            (&state.aura_storage_url, &state.aura_storage_token)
        {
            let assistant_content = anthropic_compat::extract_response_text(&response_value);

            let client = state.http_client.clone();
            let url = storage_url.clone();
            let token = storage_token.clone();
            let user_id = auth.user_id.clone();
            tokio::spawn(async move {
                storage::store_events(
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

    let max_tokens = providers::max_context_tokens(model);
    let context_usage = if max_tokens > 0 {
        input_tokens as f64 / max_tokens as f64
    } else {
        0.0
    };

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/json".to_string()),
            (
                header::HeaderName::from_static("x-context-usage"),
                format!("{context_usage:.4}"),
            ),
            (
                header::HeaderName::from_static("x-model-max-tokens"),
                max_tokens.to_string(),
            ),
            (
                header::HeaderName::from_static("x-request-id"),
                provider_request_id,
            ),
        ],
        Body::from(normalized_response_bytes),
    )
        .into_response())
}

/// Handle streaming response: tee SSE stream to client while capturing billing data.
async fn handle_streaming(
    auth: AuthUser,
    state: AppState,
    model: &str,
    provider: providers::Provider,
    openai_api: providers::OpenAiApi,
    provider_name: &str,
    upstream_resp: reqwest::Response,
    provider_request_id: String,
    session_ctx: Option<storage::SessionContext>,
    user_content: String,
    request_start: std::time::Instant,
) -> Result<Response, AppError> {
    let model_owned = model.to_string();
    let provider_owned = provider_name.to_string();
    let (tee_stream, usage_rx) = stream::proxy_stream(
        provider,
        openai_api,
        model,
        upstream_resp,
        provider_request_id.clone(),
    );

    // Spawn task to handle billing + storage after stream completes
    let billing_state = state.clone();
    let user_id = auth.user_id.clone();
    let stream_org_id = session_ctx.as_ref().and_then(|c| c.org_id.clone());
    let stream_project_id = session_ctx.as_ref().and_then(|c| c.project_id.clone());
    let stream_agent_id = session_ctx
        .as_ref()
        .and_then(|c| c.project_agent_id.clone());
    let billing_provider_request_id = provider_request_id.clone();
    tokio::spawn(async move {
        if let Ok(usage) = usage_rx.await {
            let duration_ms = request_start.elapsed().as_millis() as u64;
            let model = usage.model.as_deref().unwrap_or(&model_owned);
            spawn_post_request_tasks(
                &billing_state,
                &user_id,
                stream_org_id.as_deref(),
                stream_project_id.as_deref(),
                stream_agent_id.as_deref(),
                &provider_owned,
                model,
                billing::UsageCostInput {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cache_creation_input_tokens: usage.cache_creation_input_tokens,
                    cache_creation_5m_input_tokens: usage.cache_creation_5m_input_tokens,
                    cache_creation_1h_input_tokens: usage.cache_creation_1h_input_tokens,
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                    web_search_requests: usage.web_search_requests,
                    web_fetch_requests: usage.web_fetch_requests,
                    code_execution_requests: usage.code_execution_requests,
                    service_tier: usage.service_tier.clone(),
                    inference_geo: usage.inference_geo.clone(),
                    speed: usage.speed.clone(),
                },
                usage.provider_reported_cost_cents,
                usage.provider_reported_cost_microusd,
                duration_ms,
                billing_event_id(&provider_owned, &billing_provider_request_id),
            );

            // Store messages to aura-storage if session context is present
            if usage.completed {
                if let Some(ctx) = session_ctx {
                    if let (Some(ref storage_url), Some(ref storage_token)) = (
                        &billing_state.aura_storage_url,
                        &billing_state.aura_storage_token,
                    ) {
                        storage::store_events(
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
        }
    });

    let max_tokens = providers::max_context_tokens(model);
    let body = Body::from_stream(tee_stream);

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/event-stream".to_string()),
            (header::CACHE_CONTROL, "no-cache".to_string()),
            (
                header::HeaderName::from_static("x-accel-buffering"),
                "no".to_string(),
            ),
            (
                header::HeaderName::from_static("x-model-max-tokens"),
                max_tokens.to_string(),
            ),
            (
                header::HeaderName::from_static("x-request-id"),
                provider_request_id,
            ),
        ],
        body,
    )
        .into_response())
}

fn provider_from_name(provider_name: &str) -> providers::Provider {
    match provider_name {
        "openai" => providers::Provider::OpenAi,
        "xai" => providers::Provider::Xai,
        "fireworks" => providers::Provider::Fireworks,
        "deepseek" => providers::Provider::DeepSeek,
        "google" => providers::Provider::Google,
        _ => providers::Provider::Anthropic,
    }
}

fn apply_provider_request_controls(
    provider: providers::Provider,
    openai_api: providers::OpenAiApi,
    upstream_headers: &mut reqwest::header::HeaderMap,
    upstream_body: &mut serde_json::Value,
    session_ctx: Option<&storage::SessionContext>,
) {
    // The harness owns `prompt_cache_key` derivation and clamps it to
    // OpenAI's 64-char limit before threading it via the
    // `X-Aura-Prompt-Cache-Key` header. The router forwards that value
    // verbatim and never synthesizes its own (a router-built
    // org/project/agent/model key overran the 64-char limit and tripped
    // OpenAI's `invalid_request_error`). For xAI, Chat Completions uses
    // `x-grok-conv-id`, while Responses uses `prompt_cache_key`.
    // When no key is threaded, the request simply goes out without one.
    let Some(cache_key) = session_ctx.and_then(|ctx| ctx.prompt_cache_key.as_deref()) else {
        return;
    };

    match (provider, openai_api) {
        (providers::Provider::OpenAi, _)
        | (providers::Provider::Xai, providers::OpenAiApi::Responses) => {
            let Some(body) = upstream_body.as_object_mut() else {
                return;
            };
            body.entry("prompt_cache_key")
                .or_insert_with(|| serde_json::Value::String(cache_key.to_string()));
            let is_gpt_5_6 = provider == providers::Provider::OpenAi
                && body
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|model| model.starts_with("gpt-5.6"));
            if is_gpt_5_6 {
                body.remove("prompt_cache_retention");
                body.entry("prompt_cache_options")
                    .or_insert_with(|| serde_json::json!({ "ttl": "30m" }));
            }
        }
        (providers::Provider::Xai, providers::OpenAiApi::ChatCompletions) => {
            if let Ok(value) = reqwest::header::HeaderValue::from_str(cache_key) {
                upstream_headers.insert(
                    reqwest::header::HeaderName::from_static("x-grok-conv-id"),
                    value,
                );
            }
        }
        _ => {}
    }
}

fn set_anthropic_user_metadata(
    upstream_body: &mut serde_json::Value,
    user_id: &str,
) -> Result<(), AppError> {
    let body = upstream_body
        .as_object_mut()
        .ok_or_else(|| AppError::Internal("Anthropic request body must be an object".into()))?;
    body.insert(
        "metadata".to_string(),
        serde_json::json!({ "user_id": user_id }),
    );
    Ok(())
}

fn normalize_upstream_error(status: StatusCode, error_body: &[u8]) -> Response {
    let raw_message = extract_upstream_error_message(error_body)
        .unwrap_or_else(|| format!("Upstream provider returned HTTP {}", status.as_u16()));
    let provider_account_error =
        is_provider_account_error(&raw_message) || has_provider_account_error_code(error_body);

    // When the upstream provider ACCOUNT has a billing/account problem — out of
    // credits, suspended, or over its spend quota (a single shared key, never
    // the end user's) — the provider's raw copy (e.g. Anthropic "credit balance
    // too low — Plans & Billing", Fireworks "Account X is suspended", OpenAI
    // "exceeded your current quota") both makes users think it's *their* fault
    // and can leak our account identifiers. Log it loudly so we notice and fix
    // the account, and replace only the user-facing text with a clear our-side
    // message.
    let message = if provider_account_error {
        tracing::error!(
            upstream_status = %status,
            upstream_message = %raw_message,
            "Upstream provider account has a billing/account problem (out of credits, suspended, \
             or over quota) — fix the shared provider account; users are shielded with a generic \
             message"
        );
        "AURA is temporarily unable to reach the model provider. This is an issue on our side, \
         not with your account or credits. Please try again shortly."
            .to_string()
    } else {
        raw_message
    };

    // OpenAI reports an exhausted account quota as 429, even though waiting
    // cannot recover it. Legacy harnesses retry every 429, so normalize this
    // one account-level case to 424 Failed Dependency. Genuine provider rate
    // limits keep their original 429 and Retry-After semantics.
    let response_status = if provider_account_error && status == StatusCode::TOO_MANY_REQUESTS {
        StatusCode::FAILED_DEPENDENCY
    } else {
        status
    };
    let mut error = serde_json::json!({
        "type": anthropic_error_type(response_status),
        "message": message,
    });
    if provider_account_error {
        error["code"] = serde_json::Value::String("provider_account_unavailable".to_string());
    }

    (
        response_status,
        [(header::CONTENT_TYPE, "application/json".to_string())],
        Body::from(
            serde_json::json!({
                "type": "error",
                "error": error,
            })
            .to_string(),
        ),
    )
        .into_response()
}

/// Detect an upstream error that means *the shared provider account* has an
/// account-level problem — out of credits, suspended, or over its spend quota —
/// as opposed to a genuine malformed request. End users never supply their own
/// provider key, so any of these is by definition our account: safe to detect
/// and replace the user-facing text.
///
/// Matches deliberately-specific phrases per provider, NOT bare
/// "billing"/"quota"/"limit", so genuine request errors and transient rate
/// limits (which carry their own actionable messages and should pass through)
/// are never misclassified.
fn is_provider_account_error(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    // Anthropic — out of credits.
    m.contains("credit balance is too low")
        || m.contains("plans & billing")
        || m.contains("plans and billing")
        // Fireworks — account suspended / unpaid / budget cap.
        || m.contains("is suspended")
        || m.contains("spending limit")
        || m.contains("past invoices")
        || m.contains("failure to pay")
        // OpenAI — account quota exhausted (NOT transient rate-limit).
        || m.contains("insufficient_quota")
        || m.contains("exceeded your current quota")
}

fn has_provider_account_error_code(error_body: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(error_body)
        .ok()
        .is_some_and(|value| {
            ["/error/type", "/error/code"].iter().any(|pointer| {
                value
                    .pointer(pointer)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|code| code.eq_ignore_ascii_case("insufficient_quota"))
            })
        })
}

fn anthropic_error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        status if status.as_u16() == 529 => "overloaded_error",
        _ => "api_error",
    }
}

fn extract_upstream_error_message(error_body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(error_body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    value
                        .pointer("/message")
                        .and_then(serde_json::Value::as_str)
                })
                .or_else(|| value.pointer("/error").and_then(serde_json::Value::as_str))
                .map(str::trim)
                .filter(|message| !message.is_empty())
                .map(str::to_string)
        })
        .or_else(|| {
            std::str::from_utf8(error_body)
                .ok()
                .map(str::trim)
                .filter(|message| !message.is_empty())
                .map(str::to_string)
        })
}

/// Fire-and-forget tasks: debit z-billing + record to aura-network.
#[allow(clippy::too_many_arguments)]
fn spawn_post_request_tasks(
    state: &AppState,
    user_id: &str,
    org_id: Option<&str>,
    project_id: Option<&str>,
    agent_id: Option<&str>,
    provider_name: &str,
    model: &str,
    usage: billing::UsageCostInput,
    provider_reported_cost_cents: Option<i64>,
    provider_reported_cost_microusd: Option<i64>,
    duration_ms: u64,
    event_id: String,
) {
    let model_owned = model.to_string();
    let user_id_owned = user_id.to_string();
    let provider_owned = provider_name.to_string();
    let org_id_owned = org_id.map(String::from);
    let project_id_owned = project_id.map(String::from);
    let agent_id_owned = agent_id.map(String::from);
    let static_estimate = billing::estimate_usage_cost(provider_name, model, &usage);
    let has_detailed_cost_override = usage.cache_creation_input_tokens > 0
        || usage.cache_creation_5m_input_tokens > 0
        || usage.cache_creation_1h_input_tokens > 0
        || usage.cache_read_input_tokens > 0
        || usage.web_search_requests > 0
        || usage.inference_geo.as_deref() == Some("us")
        || usage.speed.as_deref() == Some("fast");
    let cost_cents = provider_reported_cost_cents.or_else(|| {
        has_detailed_cost_override
            .then(|| static_estimate.map(|estimate| estimate.billed_cost_cents))
            .flatten()
    });
    let estimated_provider_cost_microusd = provider_reported_cost_microusd
        .or_else(|| static_estimate.map(|estimate| estimate.provider_cost_microusd));
    let cost_estimate_source = if provider_reported_cost_microusd.is_some() {
        "provider_reported"
    } else {
        "static_model_rates"
    };
    let markup_bps = static_estimate.map_or(2_000, |estimate| estimate.markup_bps);
    let metadata = serde_json::json!({
        "org_id": org_id,
        "project_id": project_id,
        "agent_id": agent_id,
        "cost_observability": {
            "schema_version": 1,
            "estimated_provider_cost_microusd": estimated_provider_cost_microusd,
            "estimate_source": cost_estimate_source,
            "pricing_source": billing::PRICING_SOURCE,
            "markup_bps": markup_bps,
            "uncached_input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_creation_5m_input_tokens": usage.cache_creation_5m_input_tokens,
            "cache_creation_1h_input_tokens": usage.cache_creation_1h_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "web_search_requests": usage.web_search_requests,
            "web_fetch_requests": usage.web_fetch_requests,
            "code_execution_requests": usage.code_execution_requests,
            "service_tier": usage.service_tier.clone(),
            "inference_geo": usage.inference_geo.clone(),
            "speed": usage.speed.clone(),
        }
    });
    let input_tokens = usage.input_tokens;
    let output_tokens = usage.output_tokens;

    // Debit z-billing (skip for public-guest — no billing account)
    if user_id != "public-guest" {
        let client = state.http_client.clone();
        let billing_url = state.z_billing_url.clone();
        let billing_key = state.z_billing_api_key.clone();
        let user_id = user_id_owned.clone();
        let model = model_owned.clone();
        let provider = provider_owned.clone();
        let metadata = metadata.clone();
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
                cost_cents,
                Some(metadata),
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
                org_id_owned.as_deref(),
                project_id_owned.as_deref(),
                agent_id_owned.as_deref(),
                &model,
                input_tokens,
                output_tokens,
                // Prefer the cache-aware cost (model-specific rate
                // accounting for cache create/read tokens) computed
                // above for z-billing. Fall back to the legacy flat
                // estimate so the cost field stays populated for
                // providers/models that lack a price entry.
                cost_cents
                    .map(|c| c as f64 / 100.0)
                    .unwrap_or((input_tokens + output_tokens) as f64 * 0.00001),
                duration_ms,
            )
            .await;
        });
    }
}

fn upstream_request_id(response: &reqwest::Response) -> String {
    response
        .headers()
        .get("x-request-id")
        .or_else(|| response.headers().get("request-id"))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

fn billing_event_id(provider: &str, provider_request_id: &str) -> String {
    format!("llm:{provider}:{provider_request_id}")
}

#[cfg(test)]
mod tests {
    use crate::{router, state::AppState};
    use aura_router_auth::{InternalToken, TokenValidator};
    use aura_router_proxy::billing;
    use axum::body::Body;
    use axum::http::{header, Request, StatusCode};
    use axum::routing::post;
    use axum::{Json, Router};
    use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
    use serde::Serialize;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    const SELF_SIGNED_KID: &str = "jFNXMnFjGrSoDafnLQBohoCNalWcFcTjnKEbkRzWFBHyYJFikdLMHP";

    #[derive(Debug, Serialize)]
    struct TestClaims {
        id: String,
    }

    #[test]
    fn anthropic_requests_receive_trusted_opaque_user_metadata() {
        let mut body = json!({
            "model": "claude-sonnet-5",
            "metadata": { "user_id": "client-supplied" }
        });

        super::set_anthropic_user_metadata(&mut body, "aura-user-123").unwrap();

        assert_eq!(body["metadata"]["user_id"], "aura-user-123");
    }

    async fn start_mock_billing() -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route(
                "/v1/usage/check",
                post(|| async {
                    Json(json!({
                        "sufficient": true,
                        "balance_cents": 1_000_000,
                        "required_cents": 1,
                    }))
                }),
            )
            .route(
                "/v1/usage",
                post(|| async {
                    Json(json!({
                        "success": true,
                        "balance_cents": 999_999,
                        "cost_cents": 1,
                        "transaction_id": "txn_test",
                    }))
                }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        (url, handle)
    }

    async fn start_recording_billing(
        check_response: serde_json::Value,
    ) -> (
        String,
        Arc<Mutex<Vec<serde_json::Value>>>,
        tokio::task::JoinHandle<()>,
    ) {
        let recorded_requests = Arc::new(Mutex::new(Vec::new()));
        let recorded_for_check = Arc::clone(&recorded_requests);
        let recorded_for_usage = Arc::clone(&recorded_requests);
        let check_response_for_route = check_response.clone();

        let app = Router::new()
            .route(
                "/v1/usage/check",
                post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                    let recorded_for_check = Arc::clone(&recorded_for_check);
                    let check_response = check_response_for_route.clone();
                    async move {
                        recorded_for_check.lock().unwrap().push(body);
                        Json(check_response)
                    }
                }),
            )
            .route(
                "/v1/usage",
                post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                    let recorded_for_usage = Arc::clone(&recorded_for_usage);
                    async move {
                        let cost_cents = body
                            .get("cost_cents")
                            .and_then(serde_json::Value::as_i64)
                            .unwrap_or(1);
                        recorded_for_usage.lock().unwrap().push(body);
                        Json(json!({
                            "success": true,
                            "balance_cents": 999_999,
                            "cost_cents": cost_cents,
                            "transaction_id": "txn_test",
                        }))
                    }
                }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        (url, recorded_requests, handle)
    }

    fn test_jwt(secret: &str, user_id: &str) -> String {
        let mut header = Header::new(Algorithm::HS256);
        header.kid = Some(SELF_SIGNED_KID.to_string());
        encode(
            &header,
            &TestClaims {
                id: user_id.to_string(),
            },
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .expect("test jwt")
    }

    fn test_state(
        cookie_secret: &str,
        billing_url: String,
        anthropic_api_key: String,
        openai_api_key: Option<String>,
        fireworks_api_key: Option<String>,
    ) -> AppState {
        AppState {
            validator: TokenValidator::new(
                "example.auth0.test".to_string(),
                "aura-router-tests".to_string(),
                cookie_secret.to_string(),
            ),
            internal_token: InternalToken("internal-test-token".to_string()),
            public_rate_limiter: std::sync::Arc::new(aura_router_auth::PublicRateLimiter::new()),
            http_client: reqwest::Client::new(),
            rate_limiter: std::sync::Arc::new(aura_router_proxy::rate_limit::RateLimiter::new(
                120, 60,
            )),
            anthropic_api_key,
            openai_api_key,
            xai_api_key: None,
            fireworks_api_key,
            deepseek_api_key: None,
            google_api_key: None,
            tripo_api_key: None,
            ark_api_key: None,
            z_billing_url: billing_url,
            z_billing_api_key: "billing-test-key".to_string(),
            aura_network_url: None,
            aura_network_token: None,
            aura_storage_url: None,
            aura_storage_token: None,
            s3_config: None,
            watermark_bytes: None,
        }
    }

    #[test]
    fn xai_requires_platform_key() {
        let state = test_state(
            "test-cookie-secret",
            "http://billing.test".to_string(),
            "anthropic-unused".to_string(),
            None,
            None,
        );

        let error =
            super::resolve_provider_api_key(&state, aura_router_proxy::providers::Provider::Xai)
                .expect_err("Grok should require Aura's platform xAI key");

        assert_eq!(
            error.to_string(),
            "Bad request: xAI provider not configured"
        );
    }

    #[test]
    fn xai_uses_platform_key() {
        let mut state = test_state(
            "test-cookie-secret",
            "http://billing.test".to_string(),
            "anthropic-unused".to_string(),
            None,
            None,
        );
        state.xai_api_key = Some("platform-xai-key".to_string());

        let key =
            super::resolve_provider_api_key(&state, aura_router_proxy::providers::Provider::Xai)
                .expect("platform xAI key should configure provider");

        assert_eq!(key, "platform-xai-key");
    }

    #[test]
    fn openai_provider_request_controls_forward_threaded_cache_key() {
        let session = aura_router_proxy::storage::SessionContext {
            session_id: Some("session-ignored".to_string()),
            project_agent_id: Some("agent-a".to_string()),
            project_id: Some("project-a".to_string()),
            org_id: Some("org-a".to_string()),
            prompt_cache_key: Some("instance:abc-123".to_string()),
        };
        let mut upstream = json!({
            "model": "gpt-5.5",
            "messages": []
        });
        let mut upstream_headers = reqwest::header::HeaderMap::new();

        super::apply_provider_request_controls(
            aura_router_proxy::providers::Provider::OpenAi,
            aura_router_proxy::providers::OpenAiApi::ChatCompletions,
            &mut upstream_headers,
            &mut upstream,
            Some(&session),
        );

        // The router forwards the harness-supplied key verbatim and never
        // synthesizes its own org/project/agent/model key.
        assert_eq!(upstream["prompt_cache_key"], "instance:abc-123");
        assert!(upstream.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn gpt_5_6_provider_controls_use_the_new_cache_ttl() {
        let session = aura_router_proxy::storage::SessionContext {
            prompt_cache_key: Some("instance:abc-123".to_string()),
            ..Default::default()
        };
        let mut upstream = json!({
            "model": "gpt-5.6-terra",
            "input": [],
            "prompt_cache_retention": "24h"
        });
        let mut upstream_headers = reqwest::header::HeaderMap::new();

        super::apply_provider_request_controls(
            aura_router_proxy::providers::Provider::OpenAi,
            aura_router_proxy::providers::OpenAiApi::Responses,
            &mut upstream_headers,
            &mut upstream,
            Some(&session),
        );

        assert_eq!(upstream["prompt_cache_key"], "instance:abc-123");
        assert_eq!(upstream["prompt_cache_options"]["ttl"], "30m");
        assert!(upstream.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn openai_provider_request_controls_skip_when_no_threaded_key() {
        let session = aura_router_proxy::storage::SessionContext {
            session_id: Some("sess".to_string()),
            project_agent_id: Some("agent-a".to_string()),
            project_id: Some("project-a".to_string()),
            org_id: Some("org-a".to_string()),
            prompt_cache_key: None,
        };
        let mut upstream = json!({ "model": "gpt-5.5", "messages": [] });
        let mut upstream_headers = reqwest::header::HeaderMap::new();

        super::apply_provider_request_controls(
            aura_router_proxy::providers::Provider::OpenAi,
            aura_router_proxy::providers::OpenAiApi::ChatCompletions,
            &mut upstream_headers,
            &mut upstream,
            Some(&session),
        );

        // No threaded key means no prompt caching (and crucially, no
        // router-built oversized key that OpenAI would reject).
        assert!(upstream.get("prompt_cache_key").is_none());
    }

    #[test]
    fn xai_provider_request_controls_forward_threaded_cache_key() {
        let session = aura_router_proxy::storage::SessionContext {
            prompt_cache_key: Some("instance:abc-123".to_string()),
            ..Default::default()
        };

        let mut chat_headers = reqwest::header::HeaderMap::new();
        let mut chat_upstream = json!({ "model": "grok-4.5", "messages": [] });
        super::apply_provider_request_controls(
            aura_router_proxy::providers::Provider::Xai,
            aura_router_proxy::providers::OpenAiApi::ChatCompletions,
            &mut chat_headers,
            &mut chat_upstream,
            Some(&session),
        );
        assert_eq!(
            chat_headers
                .get("x-grok-conv-id")
                .and_then(|value| value.to_str().ok()),
            Some("instance:abc-123")
        );
        assert!(chat_upstream.get("prompt_cache_key").is_none());

        let mut responses_headers = reqwest::header::HeaderMap::new();
        let mut responses_upstream = json!({ "model": "grok-4.5", "input": [] });
        super::apply_provider_request_controls(
            aura_router_proxy::providers::Provider::Xai,
            aura_router_proxy::providers::OpenAiApi::Responses,
            &mut responses_headers,
            &mut responses_upstream,
            Some(&session),
        );
        assert!(responses_headers.get("x-grok-conv-id").is_none());
        assert_eq!(responses_upstream["prompt_cache_key"], "instance:abc-123");
    }

    #[test]
    fn provider_request_controls_skip_non_openai() {
        let session = aura_router_proxy::storage::SessionContext {
            prompt_cache_key: Some("instance:abc-123".to_string()),
            ..Default::default()
        };
        let mut upstream = json!({ "model": "claude", "messages": [] });
        let mut upstream_headers = reqwest::header::HeaderMap::new();

        super::apply_provider_request_controls(
            aura_router_proxy::providers::Provider::Anthropic,
            aura_router_proxy::providers::OpenAiApi::ChatCompletions,
            &mut upstream_headers,
            &mut upstream,
            Some(&session),
        );

        assert!(upstream.get("prompt_cache_key").is_none());
    }

    #[tokio::test]
    async fn normalizes_upstream_json_errors_to_anthropic_shape() {
        let response = super::normalize_upstream_error(
            StatusCode::BAD_REQUEST,
            br#"{"error":{"message":"tools[0] is invalid"}}"#,
        );

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");

        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "tools[0] is invalid");
    }

    #[tokio::test]
    async fn shields_users_from_provider_account_billing_errors() {
        // Anthropic's verbatim message when the *account's* credit balance is
        // depleted. Forwarding this confuses users into thinking it's their
        // own credits, so the user-facing text must be reworded.
        let response = super::normalize_upstream_error(
            StatusCode::BAD_REQUEST,
            br#"{"error":{"message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.","type":"invalid_request_error"}}"#,
        );

        // Anthropic already returns a terminal 400 for account depletion.
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");

        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "provider_account_unavailable");
        let message = body["error"]["message"].as_str().expect("message string");
        // Exact rendered text — also guards against stray whitespace from the
        // multi-line string continuations in the source.
        assert_eq!(
            message,
            "AURA is temporarily unable to reach the model provider. This is an issue on our \
             side, not with your account or credits. Please try again shortly."
        );
        // The provider's billing copy must not leak to the end user.
        assert!(!message.to_ascii_lowercase().contains("plans & billing"));
        assert!(!message.to_ascii_lowercase().contains("credit balance"));
    }

    #[tokio::test]
    async fn shields_users_from_fireworks_suspension() {
        // Fireworks' verbatim suspension message leaks the account name and a
        // billing URL, and reads like the *user* is suspended. Must be replaced.
        let response = super::normalize_upstream_error(
            StatusCode::PRECONDITION_FAILED,
            br#"{"error":{"message":"Account acme-7 is suspended, possibly due to reaching the monthly spending limit or failure to pay past invoices. Please go to https://fireworks.ai/account/billing for more information.","type":"api_error"}}"#,
        );

        // Fireworks already returns a terminal 412 for account suspension.
        assert_eq!(response.status(), StatusCode::PRECONDITION_FAILED);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        let message = body["error"]["message"].as_str().expect("message string");

        assert_eq!(body["error"]["code"], "provider_account_unavailable");
        assert_eq!(
            message,
            "AURA is temporarily unable to reach the model provider. This is an issue on our \
             side, not with your account or credits. Please try again shortly."
        );
        // No account id, billing URL, or "suspended" framing reaches the user.
        let lower = message.to_ascii_lowercase();
        assert!(!lower.contains("acme"));
        assert!(!lower.contains("suspended"));
        assert!(!lower.contains("fireworks"));
        assert!(!lower.contains("invoices"));
    }

    #[tokio::test]
    async fn shields_users_from_openai_quota_error() {
        let response = super::normalize_upstream_error(
            StatusCode::TOO_MANY_REQUESTS,
            br#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota"}}"#,
        );

        // Unlike a genuine rate limit, account quota exhaustion cannot recover
        // after a backoff. 424 prevents legacy harnesses retrying it as a 429.
        assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        let message = body["error"]["message"].as_str().expect("message string");

        assert_eq!(body["error"]["type"], "api_error");
        assert_eq!(body["error"]["code"], "provider_account_unavailable");
        assert_eq!(
            message,
            "AURA is temporarily unable to reach the model provider. This is an issue on our \
             side, not with your account or credits. Please try again shortly."
        );
        assert!(!message.to_ascii_lowercase().contains("quota"));
    }

    #[tokio::test]
    async fn recognizes_structured_account_quota_when_provider_copy_changes() {
        let response = super::normalize_upstream_error(
            StatusCode::TOO_MANY_REQUESTS,
            br#"{"error":{"message":"Capacity unavailable for this request.","type":"insufficient_quota"}}"#,
        );

        assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");

        assert_eq!(body["error"]["code"], "provider_account_unavailable");
        assert!(!body["error"]["message"]
            .as_str()
            .expect("message string")
            .contains("Capacity unavailable"));
    }

    #[tokio::test]
    async fn preserves_genuine_provider_rate_limit_as_retryable_429() {
        let response = super::normalize_upstream_error(
            StatusCode::TOO_MANY_REQUESTS,
            br#"{"error":{"message":"Rate limit reached for gpt-5.5 in organization org-xxx. Please try again in 1s.","type":"rate_limit_error"}}"#,
        );

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");

        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert!(body["error"].get("code").is_none());
        assert!(body["error"]["message"]
            .as_str()
            .expect("message string")
            .contains("Please try again in 1s"));
    }

    #[tokio::test]
    async fn passes_through_genuine_request_errors() {
        // A real invalid-request error (here an unsupported reasoning effort for a
        // model) is the user's/UI's to act on, so it must NOT be shielded — the
        // provider's actionable message passes through unchanged.
        let response = super::normalize_upstream_error(
            StatusCode::BAD_REQUEST,
            br#"{"error":{"message":"Unsupported value: 'minimal' is not supported with the 'gpt-5.4-nano' model. Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'.","type":"invalid_request_error"}}"#,
        );

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        // Unchanged — the real, actionable provider message reaches the user.
        assert_eq!(
            body["error"]["message"],
            "Unsupported value: 'minimal' is not supported with the 'gpt-5.4-nano' model. Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."
        );
    }

    #[test]
    fn detects_provider_account_errors_across_providers() {
        // Anthropic — out of credits.
        assert!(super::is_provider_account_error(
            "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."
        ));
        // Fireworks — account suspended.
        assert!(super::is_provider_account_error(
            "Account acme-7 is suspended, possibly due to reaching the monthly spending limit or failure to pay past invoices. Please go to https://fireworks.ai/account/billing for more information."
        ));
        // OpenAI — account quota exhausted.
        assert!(super::is_provider_account_error(
            "You exceeded your current quota, please check your plan and billing details."
        ));
        assert!(super::is_provider_account_error("insufficient_quota"));

        // Genuine request errors and transient rate limits carry actionable
        // messages and must NOT be misclassified — they pass through untouched.
        assert!(!super::is_provider_account_error("tools[0] is invalid"));
        assert!(!super::is_provider_account_error(
            "Unsupported value: 'minimal' is not supported with the 'gpt-5.4-nano' model. Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."
        ));
        assert!(!super::is_provider_account_error(
            "Rate limit reached for gpt-5.5 in organization org-xxx. Please try again in 1s."
        ));
    }

    #[tokio::test]
    async fn uses_requested_aura_model_for_model_aware_credit_check() {
        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-credit-check");
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": false,
            "balance_cents": 1,
            "required_cents": 3,
        }))
        .await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            Some("unused".to_string()),
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-gpt-5-5",
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["required_cents"], 0);
        assert_eq!(requests[0]["provider"], "openai");
        assert_eq!(requests[0]["model"], "aura-gpt-5-5");

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value = serde_json::from_slice(&body).expect("json response");
        assert_eq!(response["error"]["code"], "INSUFFICIENT_CREDITS");
        assert!(response["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("required=3"));
    }

    #[tokio::test]
    async fn uses_fireworks_provider_for_deepseek_v4_credit_check() {
        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-deepseek-credit-check");
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": false,
            "balance_cents": 1,
            "required_cents": 3,
        }))
        .await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            None,
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-deepseek-v4-flash",
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["provider"], "fireworks");
        assert_eq!(requests[0]["model"], "aura-deepseek-v4-flash");
    }

    #[tokio::test]
    async fn uses_xai_provider_for_grok_credit_check() {
        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-xai-credit-check");
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": false,
            "balance_cents": 1,
            "required_cents": 3,
        }))
        .await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            None,
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-grok-4-3",
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["provider"], "xai");
        assert_eq!(requests[0]["model"], "aura-grok-4-3");
    }

    #[tokio::test]
    async fn reports_deepseek_cache_aware_usage_cost_to_billing() {
        let cookie_secret = "test-cookie-secret";
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": true,
            "balance_cents": 1_000_000,
            "required_cents": 1,
        }))
        .await;
        let state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);

        super::spawn_post_request_tasks(
            &state,
            "user-deepseek-billing",
            None,
            None,
            None,
            "deepseek",
            "aura-deepseek-v4-flash",
            billing::UsageCostInput {
                input_tokens: 1_000_000,
                output_tokens: 500_000,
                cache_read_input_tokens: 1_000_000,
                ..billing::UsageCostInput::default()
            },
            None,
            None,
            123,
            "test-event-deepseek".to_string(),
        );

        for _ in 0..20 {
            if !recorded_requests.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["user_id"], "user-deepseek-billing");
        assert_eq!(requests[0]["cost_cents"], 20);
        assert_eq!(requests[0]["metric"]["provider"], "deepseek");
        assert_eq!(requests[0]["metric"]["model"], "aura-deepseek-v4-flash");
        assert_eq!(requests[0]["metric"]["input_tokens"], 1_000_000);
        assert_eq!(requests[0]["metric"]["output_tokens"], 500_000);
        assert_eq!(
            requests[0]["metadata"]["cost_observability"]["estimated_provider_cost_microusd"],
            168_000
        );
    }

    #[tokio::test]
    async fn reports_openai_cache_aware_usage_cost_to_billing() {
        let cookie_secret = "test-cookie-secret";
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": true,
            "balance_cents": 1_000_000,
            "required_cents": 1,
        }))
        .await;
        let state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);

        super::spawn_post_request_tasks(
            &state,
            "user-openai-billing",
            Some("org-openai"),
            Some("project-openai"),
            None,
            "openai",
            "aura-gpt-5-5",
            billing::UsageCostInput {
                input_tokens: 1_000_000,
                output_tokens: 500_000,
                cache_read_input_tokens: 1_000_000,
                ..billing::UsageCostInput::default()
            },
            None,
            None,
            123,
            "test-event-openai".to_string(),
        );

        for _ in 0..20 {
            if !recorded_requests.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["user_id"], "user-openai-billing");
        // GPT-5.5's 1M-token prompt uses the published long-context step.
        assert_eq!(requests[0]["cost_cents"], 2820);
        assert_eq!(requests[0]["metric"]["provider"], "openai");
        assert_eq!(requests[0]["metric"]["model"], "aura-gpt-5-5");
        assert_eq!(requests[0]["metric"]["input_tokens"], 1_000_000);
        assert_eq!(requests[0]["metric"]["output_tokens"], 500_000);
    }

    #[tokio::test]
    async fn reports_anthropic_provider_cost_details_to_billing() {
        let cookie_secret = "test-cookie-secret";
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": true,
            "balance_cents": 1_000_000,
            "required_cents": 1,
        }))
        .await;
        let state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);

        super::spawn_post_request_tasks(
            &state,
            "user-anthropic-billing",
            Some("org-anthropic"),
            Some("project-anthropic"),
            None,
            "anthropic",
            "claude-sonnet-5",
            billing::UsageCostInput {
                input_tokens: 100_000,
                output_tokens: 10_000,
                cache_creation_input_tokens: 300_000,
                cache_creation_5m_input_tokens: 200_000,
                cache_creation_1h_input_tokens: 100_000,
                cache_read_input_tokens: 500_000,
                web_search_requests: 2,
                web_fetch_requests: 1,
                code_execution_requests: 4,
                service_tier: Some("standard".to_string()),
                inference_geo: Some("us".to_string()),
                speed: Some("standard".to_string()),
            },
            None,
            None,
            123,
            "test-event-anthropic".to_string(),
        );

        for _ in 0..20 {
            if !recorded_requests.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }

        let expected = billing::estimate_usage_cost(
            "anthropic",
            "claude-sonnet-5",
            &billing::UsageCostInput {
                input_tokens: 100_000,
                output_tokens: 10_000,
                cache_creation_input_tokens: 300_000,
                cache_creation_5m_input_tokens: 200_000,
                cache_creation_1h_input_tokens: 100_000,
                cache_read_input_tokens: 500_000,
                web_search_requests: 2,
                web_fetch_requests: 1,
                code_execution_requests: 4,
                service_tier: Some("standard".to_string()),
                inference_geo: Some("us".to_string()),
                speed: Some("standard".to_string()),
            },
        )
        .unwrap();
        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["cost_cents"], expected.billed_cost_cents);
        assert_eq!(requests[0]["metadata"]["org_id"], "org-anthropic");
        assert_eq!(
            requests[0]["metadata"]["cost_observability"]["estimated_provider_cost_microusd"],
            expected.provider_cost_microusd
        );
        assert_eq!(
            requests[0]["metadata"]["cost_observability"]["cache_creation_1h_input_tokens"],
            100_000
        );
        assert_eq!(
            requests[0]["metadata"]["cost_observability"]["web_search_requests"],
            2
        );
        assert_eq!(
            requests[0]["metadata"]["cost_observability"]["code_execution_requests"],
            4
        );
    }

    #[tokio::test]
    async fn reports_provider_reported_usage_cost_to_billing() {
        let cookie_secret = "test-cookie-secret";
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": true,
            "balance_cents": 1_000_000,
            "required_cents": 1,
        }))
        .await;
        let state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);

        super::spawn_post_request_tasks(
            &state,
            "user-xai-billing",
            None,
            None,
            None,
            "xai",
            "aura-grok-4-3",
            billing::UsageCostInput {
                input_tokens: 10,
                output_tokens: 2,
                ..billing::UsageCostInput::default()
            },
            Some(3),
            Some(25_000),
            123,
            "test-event-xai".to_string(),
        );

        for _ in 0..20 {
            if !recorded_requests.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["user_id"], "user-xai-billing");
        assert_eq!(requests[0]["cost_cents"], 3);
        assert_eq!(requests[0]["metric"]["provider"], "xai");
        assert_eq!(requests[0]["metric"]["model"], "aura-grok-4-3");
        assert_eq!(
            requests[0]["metadata"]["cost_observability"]["estimate_source"],
            "provider_reported"
        );
    }

    #[tokio::test]
    async fn normalizes_unparseable_upstream_errors_without_passthrough() {
        let response = super::normalize_upstream_error(StatusCode::BAD_GATEWAY, b"");

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");

        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "api_error");
        assert_eq!(
            body["error"]["message"],
            "Upstream provider returned HTTP 502"
        );
    }

    #[tokio::test]
    async fn anthropic_live_smoke_for_aura_managed_model() {
        dotenvy::dotenv().ok();
        let Some(anthropic_api_key) = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping Anthropic live smoke test because ANTHROPIC_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-anthropic-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            anthropic_api_key,
            None,
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-claude-sonnet-4-6",
                    "max_tokens": 32,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly ANTHROPIC_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized anthropic response");
        assert_eq!(response["model"], "aura-claude-sonnet-4-6");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("ANTHROPIC_OK"),
            "expected live Anthropic response to contain ANTHROPIC_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Anthropic input token count to be populated: {response}"
        );
        assert!(
            response["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Anthropic output token count to be populated: {response}"
        );
    }

    #[tokio::test]
    async fn openai_live_smoke_for_aura_managed_model() {
        dotenvy::dotenv().ok();
        let Some(openai_api_key) = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping OpenAI live smoke test because OPENAI_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-openai-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            Some(openai_api_key),
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-gpt-5-4-mini",
                    "max_tokens": 32,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly OPENAI_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized anthropic response");
        assert_eq!(response["model"], "aura-gpt-5-4-mini");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("OPENAI_OK"),
            "expected live OpenAI response to contain OPENAI_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected OpenAI input token count to be populated: {response}"
        );
        assert!(
            response["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected OpenAI output token count to be populated: {response}"
        );
    }

    #[tokio::test]
    async fn xai_live_smoke_for_aura_managed_model() {
        dotenvy::dotenv().ok();
        let Some(xai_api_key) = std::env::var("XAI_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping xAI live smoke test because XAI_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-xai-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let mut state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);
        state.xai_api_key = Some(xai_api_key);
        let app = router::create_router().with_state(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-grok-4-3",
                    "max_tokens": 32,
                    "temperature": 0,
                    "reasoning_effort": "minimal",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly XAI_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized xAI response");
        assert_eq!(response["model"], "aura-grok-4-3");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("XAI_OK"),
            "expected live xAI response to contain XAI_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected xAI input token count to be populated: {response}"
        );
        assert!(
            response["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected xAI output token count to be populated: {response}"
        );
    }

    #[tokio::test]
    async fn xai_live_smoke_drops_reasoning_for_grok_build() {
        dotenvy::dotenv().ok();
        let Some(xai_api_key) = std::env::var("XAI_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping xAI Grok Build live smoke test because XAI_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-xai-build-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let mut state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);
        state.xai_api_key = Some(xai_api_key);
        let app = router::create_router().with_state(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-grok-build-0-1",
                    "max_tokens": 32,
                    "temperature": 0,
                    "reasoning_effort": "high",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly XAI_BUILD_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized xAI Grok Build response");
        assert_eq!(response["model"], "aura-grok-build-0-1");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("XAI_BUILD_OK"),
            "expected live xAI Grok Build response to contain XAI_BUILD_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected xAI Grok Build input token count to be populated: {response}"
        );
        assert!(
            response["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected xAI Grok Build output token count to be populated: {response}"
        );
    }

    #[tokio::test]
    async fn fireworks_live_smoke_for_aura_managed_model() {
        dotenvy::dotenv().ok();
        let Some(fireworks_api_key) = std::env::var("FIREWORKS_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping Fireworks live smoke test because FIREWORKS_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-fireworks-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            None,
            Some(fireworks_api_key),
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-kimi-k2-5",
                    "max_tokens": 32,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly FIREWORKS_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized anthropic response");
        assert_eq!(response["model"], "aura-kimi-k2-5");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("FIREWORKS_OK"),
            "expected live Fireworks response to contain FIREWORKS_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Fireworks input token count to be populated: {response}"
        );
        assert!(
            response["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Fireworks output token count to be populated: {response}"
        );
    }

    #[tokio::test]
    async fn google_live_smoke_for_aura_managed_model() {
        dotenvy::dotenv().ok();
        let Some(google_api_key) = std::env::var("GOOGLE_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping Google live smoke test because GOOGLE_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-google-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let mut state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);
        state.google_api_key = Some(google_api_key);
        let app = router::create_router().with_state(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-gemini-2-5-flash",
                    "max_tokens": 32,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly GOOGLE_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized anthropic response");
        assert_eq!(response["model"], "aura-gemini-2-5-flash");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("GOOGLE_OK"),
            "expected live Google response to contain GOOGLE_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Google input token count to be populated: {response}"
        );
    }
}
