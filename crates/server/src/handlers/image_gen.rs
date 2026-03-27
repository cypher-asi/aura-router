//! Image generation handler — generates images via OpenAI or Gemini, uploads to S3.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use aura_router_auth::AuthUser;
use aura_router_core::AppError;
use aura_router_proxy::{billing, image_gen, s3, storage};

use crate::state::AppState;

/// POST /v1/generate-image — Generate an image.
pub async fn generate_image(
    auth: AuthUser,
    State(state): State<AppState>,
    Json(input): Json<image_gen::GenerateImageRequest>,
) -> Result<Response, AppError> {
    // Validate prompt
    if input.prompt.trim().is_empty() {
        return Err(AppError::BadRequest("Prompt must not be empty".into()));
    }

    // Check S3 is configured
    let s3_config = state
        .s3_config
        .as_ref()
        .ok_or_else(|| AppError::Internal("Image generation not configured (S3)".into()))?;

    // Rate limit
    if let Err(retry_after) = state.rate_limiter.check(&auth.user_id) {
        tracing::warn!(user_id = %auth.user_id, retry_after, "Rate limited");
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, retry_after.to_string())],
            axum::body::Body::from(
                serde_json::json!({
                    "error": { "code": "RATE_LIMITED", "message": format!("Retry after {retry_after} seconds.") }
                })
                .to_string(),
            ),
        )
            .into_response());
    }

    // Pre-check credits
    let balance = billing::check_credits(
        &state.http_client,
        &state.z_billing_url,
        &state.z_billing_api_key,
        &auth.user_id,
        26, // image generation minimum: 26 credits ($0.26)
    )
    .await?;

    if !balance.sufficient {
        return Err(AppError::InsufficientCredits {
            balance: balance.balance_cents,
            required: 100,
        });
    }

    // Resolve model + provider
    let (model, provider) =
        image_gen::resolve_image_model(input.model.as_deref(), input.prompt_mode.as_deref());

    // Generate image
    let generated = match provider {
        "google" => {
            let api_key = state
                .google_api_key
                .as_ref()
                .ok_or_else(|| AppError::BadRequest("Google API key not configured".into()))?;
            image_gen::generate_gemini(
                &state.http_client,
                api_key,
                &input.prompt,
                &input.size,
                input.images.as_deref(),
                input.is_iteration,
            )
            .await
            .map_err(|e| AppError::ProviderError(e))?
        }
        _ => {
            let api_key = state
                .openai_api_key
                .as_ref()
                .ok_or_else(|| AppError::BadRequest("OpenAI API key not configured".into()))?;
            image_gen::generate_openai(
                &state.http_client,
                api_key,
                &input.prompt,
                &input.size,
                model,
                input.images.as_deref(),
                input.is_iteration,
            )
            .await
            .map_err(|e| AppError::ProviderError(e))?
        }
    };

    // Decode base64 to bytes
    let image_bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &generated.base64_data,
    )
    .map_err(|e| AppError::Internal(format!("Failed to decode image: {e}")))?;

    // Apply watermark if available
    let final_bytes = if let Some(ref wm_bytes) = state.watermark_bytes {
        match s3::apply_watermark(&image_bytes, wm_bytes) {
            Ok(watermarked) => watermarked,
            Err(e) => {
                tracing::warn!(error = %e, "Watermarking failed, using original");
                image_bytes.clone()
            }
        }
    } else {
        image_bytes.clone()
    };

    // Upload watermarked image to S3
    let image_url = s3_config
        .upload_bytes(final_bytes, &auth.user_id, "image/png", "png")
        .await
        .map_err(|e| AppError::Internal(format!("S3 upload failed: {e}")))?;

    // Upload original (unwatermarked) to S3
    let original_url = s3_config
        .upload_bytes(image_bytes, &auth.user_id, "image/png", "png")
        .await
        .map_err(|e| AppError::Internal(format!("S3 upload failed: {e}")))?;

    // Debit credits (fire-and-forget) — flat cost per generation (+30% markup)
    let cost_cents = match model {
        "gpt-image-1" => 26,           // $0.26
        "dall-e-3" => 20,              // $0.20
        "dall-e-2" => 7,               // $0.07
        "gemini-nano-banana" => 13,    // $0.13
        _ => 26,                       // default
    };
    {
        let client = state.http_client.clone();
        let billing_url = state.z_billing_url.clone();
        let billing_key = state.z_billing_api_key.clone();
        let user_id = auth.user_id.clone();
        let model_owned = model.to_string();
        let provider_owned = provider.to_string();
        tokio::spawn(async move {
            if let Err(e) = billing::report_image_usage(
                &client,
                &billing_url,
                &billing_key,
                &uuid::Uuid::new_v4().to_string(),
                &user_id,
                &provider_owned,
                &model_owned,
                cost_cents,
            )
            .await
            {
                tracing::warn!(error = %e, "Failed to debit credits for image generation");
            }
        });
    }

    // Auto-store artifact in aura-storage (fire-and-forget)
    if let Some(ref project_id) = input.project_id {
        if let (Some(ref storage_url), Some(ref storage_token)) =
            (&state.aura_storage_url, &state.aura_storage_token)
        {
            let client = state.http_client.clone();
            let url = storage_url.clone();
            let token = storage_token.clone();
            let pid = project_id.clone();
            let uid = auth.user_id.clone();
            let asset = image_url.clone();
            let orig = original_url.clone();
            let name = input.name.clone();
            let prompt = input.prompt.clone();
            let pm = input.prompt_mode.clone();
            let m = model.to_string();
            let p = provider.to_string();
            let is_iter = input.is_iteration;
            let parent = input.parent_id.clone();
            tokio::spawn(async move {
                storage::store_artifact(
                    &client,
                    &url,
                    &token,
                    &pid,
                    &uid,
                    "image",
                    &asset,
                    Some(&orig),
                    name.as_deref(),
                    Some(&prompt),
                    pm.as_deref(),
                    &m,
                    &p,
                    is_iter,
                    parent.as_deref(),
                )
                .await;
            });
        }
    }

    let response = image_gen::GenerateImageResponse {
        success: true,
        image_url,
        original_url: Some(original_url),
        meta: image_gen::ImageMeta {
            model: model.to_string(),
            size: input.size.clone(),
            prompt: input.prompt,
            provider: provider.to_string(),
            created: chrono::Utc::now().timestamp(),
        },
    };

    Ok(Json(response).into_response())
}

/// POST /v1/generate-image/stream — Stream image generation with SSE.
pub async fn generate_image_stream(
    auth: AuthUser,
    State(state): State<AppState>,
    Json(input): Json<image_gen::GenerateImageRequest>,
) -> Result<Response, AppError> {
    if input.prompt.trim().is_empty() {
        return Err(AppError::BadRequest("Prompt must not be empty".into()));
    }

    let s3_config = state
        .s3_config
        .clone()
        .ok_or_else(|| AppError::Internal("Image generation not configured (S3)".into()))?;

    if let Err(retry_after) = state.rate_limiter.check(&auth.user_id) {
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, retry_after.to_string())],
            axum::body::Body::from(
                serde_json::json!({
                    "error": { "code": "RATE_LIMITED", "message": format!("Retry after {retry_after} seconds.") }
                })
                .to_string(),
            ),
        )
            .into_response());
    }

    let balance = billing::check_credits(
        &state.http_client,
        &state.z_billing_url,
        &state.z_billing_api_key,
        &auth.user_id,
        100,
    )
    .await?;

    if !balance.sufficient {
        return Err(AppError::InsufficientCredits {
            balance: balance.balance_cents,
            required: 100,
        });
    }

    let (model, provider) = image_gen::resolve_image_model(input.model.as_deref(), input.prompt_mode.as_deref());
    let model_owned = model.to_string();
    let provider_owned = provider.to_string();
    let is_iteration = input.is_iteration;

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<image_gen::ImageStreamEvent>(32);

    // Spawn the generation task
    let gen_state = state.clone();
    let gen_user_id = auth.user_id.clone();
    let gen_prompt = input.prompt.clone();
    let gen_size = input.size.clone();
    let gen_images = input.images.clone();
    let gen_model = model_owned.clone();
    let gen_provider = provider_owned.clone();
    let gen_project_id = input.project_id.clone();
    let gen_parent_id = input.parent_id.clone();
    let gen_name = input.name.clone();
    let gen_prompt_mode = input.prompt_mode.clone();
    let gen_is_iteration = input.is_iteration;
    let event_tx_clone = event_tx.clone();

    tokio::spawn(async move {
        // Generate
        let generated = if gen_provider == "google" {
            let api_key = match gen_state.google_api_key.as_ref() {
                Some(k) => k,
                None => {
                    let _ = event_tx_clone
                        .send(image_gen::ImageStreamEvent::Error {
                            code: "CONFIG_ERROR".to_string(),
                            message: "Google API key not configured".to_string(),
                        })
                        .await;
                    return;
                }
            };
            // Gemini doesn't support streaming — use non-streaming with progress events
            let _ = event_tx_clone
                .send(image_gen::ImageStreamEvent::Start {
                    ts: chrono::Utc::now().to_rfc3339(),
                })
                .await;
            let _ = event_tx_clone
                .send(image_gen::ImageStreamEvent::Progress {
                    percent: 10,
                    message: "Generating with Gemini...".to_string(),
                })
                .await;

            match image_gen::generate_gemini(
                &gen_state.http_client,
                api_key,
                &gen_prompt,
                &gen_size,
                gen_images.as_deref(),
                is_iteration,
            )
            .await
            {
                Ok(img) => {
                    let _ = event_tx_clone
                        .send(image_gen::ImageStreamEvent::Progress {
                            percent: 80,
                            message: "Uploading...".to_string(),
                        })
                        .await;
                    img
                }
                Err(e) => {
                    let _ = event_tx_clone
                        .send(image_gen::ImageStreamEvent::Error {
                            code: "GENERATION_FAILED".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            }
        } else {
            let api_key = match gen_state.openai_api_key.as_ref() {
                Some(k) => k,
                None => {
                    let _ = event_tx_clone
                        .send(image_gen::ImageStreamEvent::Error {
                            code: "CONFIG_ERROR".to_string(),
                            message: "OpenAI API key not configured".to_string(),
                        })
                        .await;
                    return;
                }
            };

            match image_gen::generate_openai_stream(
                &gen_state.http_client,
                api_key,
                &gen_prompt,
                &gen_size,
                &gen_model,
                gen_images.as_deref(),
                is_iteration,
                event_tx_clone.clone(),
            )
            .await
            {
                Ok(img) => img,
                Err(e) => {
                    let _ = event_tx_clone
                        .send(image_gen::ImageStreamEvent::Error {
                            code: "GENERATION_FAILED".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            }
        };

        // Decode, watermark, upload
        let image_bytes = match base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &generated.base64_data,
        ) {
            Ok(b) => b,
            Err(e) => {
                let _ = event_tx_clone
                    .send(image_gen::ImageStreamEvent::Error {
                        code: "DECODE_ERROR".to_string(),
                        message: format!("Failed to decode image: {e}"),
                    })
                    .await;
                return;
            }
        };

        let final_bytes = if let Some(ref wm_bytes) = gen_state.watermark_bytes {
            s3::apply_watermark(&image_bytes, wm_bytes).unwrap_or_else(|_| image_bytes.clone())
        } else {
            image_bytes.clone()
        };

        let image_url = match s3_config
            .upload_bytes(final_bytes, &gen_user_id, "image/png", "png")
            .await
        {
            Ok(url) => url,
            Err(e) => {
                let _ = event_tx_clone
                    .send(image_gen::ImageStreamEvent::Error {
                        code: "UPLOAD_ERROR".to_string(),
                        message: e,
                    })
                    .await;
                return;
            }
        };

        let original_url = s3_config
            .upload_bytes(image_bytes, &gen_user_id, "image/png", "png")
            .await
            .ok();

        // Debit credits (+30% markup)
        let cost_cents = match gen_model.as_str() {
            "gpt-image-1" => 26,           // $0.26
            "dall-e-3" => 20,              // $0.20
            "dall-e-2" => 7,               // $0.07
            "gemini-nano-banana" => 13,    // $0.13
            _ => 26,                       // default
        };
        let _ = billing::report_image_usage(
            &gen_state.http_client,
            &gen_state.z_billing_url,
            &gen_state.z_billing_api_key,
            &uuid::Uuid::new_v4().to_string(),
            &gen_user_id,
            &gen_provider,
            &gen_model,
            cost_cents,
        )
        .await;

        // Auto-store artifact
        if let Some(ref pid) = gen_project_id {
            if let (Some(ref surl), Some(ref stok)) =
                (&gen_state.aura_storage_url, &gen_state.aura_storage_token)
            {
                storage::store_artifact(
                    &gen_state.http_client,
                    surl,
                    stok,
                    pid,
                    &gen_user_id,
                    "image",
                    &image_url,
                    original_url.as_deref(),
                    gen_name.as_deref(),
                    Some(&gen_prompt),
                    gen_prompt_mode.as_deref(),
                    &gen_model,
                    &gen_provider,
                    gen_is_iteration,
                    gen_parent_id.as_deref(),
                )
                .await;
            }
        }

        let _ = event_tx_clone
            .send(image_gen::ImageStreamEvent::Completed {
                image_url,
                original_url,
                meta: image_gen::ImageMeta {
                    model: gen_model,
                    size: gen_size,
                    prompt: gen_prompt,
                    provider: gen_provider,
                    created: chrono::Utc::now().timestamp(),
                },
            })
            .await;
    });

    // Stream SSE events to client
    let stream = async_stream::stream! {
        while let Some(event) = event_rx.recv().await {
            let json = serde_json::to_string(&event).unwrap_or_default();
            let sse = format!("data: {json}\n\n");
            yield Ok::<_, std::convert::Infallible>(sse);
        }
    };

    Ok((
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream".to_string()),
            (axum::http::header::CACHE_CONTROL, "no-cache".to_string()),
            (
                axum::http::header::HeaderName::from_static("x-accel-buffering"),
                "no".to_string(),
            ),
        ],
        axum::body::Body::from_stream(stream),
    )
        .into_response())
}

/// GET /v1/generate-image/config — Available models and ETAs.
pub async fn generate_image_config(
    _auth: AuthUser,
) -> Json<image_gen::ImageGenConfig> {
    Json(image_gen::get_config())
}
