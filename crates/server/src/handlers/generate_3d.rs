//! 3D generation handler — submits image-to-3D tasks via Tripo, polls for results.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use aura_router_auth::AuthUser;
use aura_router_core::AppError;
use aura_router_proxy::{billing, storage, tripo};

use crate::state::AppState;

/// POST /v1/generate-3d — Submit an image-to-3D generation task.
pub async fn generate_3d(
    auth: AuthUser,
    State(state): State<AppState>,
    Json(input): Json<tripo::Generate3dRequest>,
) -> Result<Response, AppError> {
    if input.image_url.trim().is_empty() {
        return Err(AppError::BadRequest("imageUrl must not be empty".into()));
    }

    let tripo_api_key = state
        .tripo_api_key
        .as_ref()
        .ok_or_else(|| AppError::Internal("Tripo not configured".into()))?;

    // Rate limit
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

    // Pre-check credits (3D generation: 50 credits / $0.50)
    let balance = billing::check_credits(
        &state.http_client,
        &state.z_billing_url,
        &state.z_billing_api_key,
        &auth.user_id,
        50,
    )
    .await?;

    if !balance.sufficient {
        return Err(AppError::InsufficientCredits {
            balance: balance.balance_cents,
            required: 50,
        });
    }

    // If image is base64/data URL, upload to S3 first (Tripo requires URL, base64 is unreliable)
    let image_url = if input.image_url.starts_with("data:") {
        let s3_config = state
            .s3_config
            .as_ref()
            .ok_or_else(|| AppError::Internal("S3 not configured for image upload".into()))?;

        s3_config
            .upload_base64(&input.image_url, &auth.user_id)
            .await
            .map_err(|e| AppError::Internal(format!("Failed to upload image to S3: {e}")))?
    } else {
        input.image_url.clone()
    };

    // Submit task to Tripo
    let task_id = tripo::create_task(&state.http_client, tripo_api_key, &image_url)
        .await
        .map_err(|e| AppError::ProviderError(e))?;

    // Debit credits (fire-and-forget)
    {
        let client = state.http_client.clone();
        let billing_url = state.z_billing_url.clone();
        let billing_key = state.z_billing_api_key.clone();
        let user_id = auth.user_id.clone();
        tokio::spawn(async move {
            if let Err(e) = billing::report_image_usage(
                &client,
                &billing_url,
                &billing_key,
                &uuid::Uuid::new_v4().to_string(),
                &user_id,
                "tripo",
                "tripo-v2",
                50, // $0.50 per 3D generation
            )
            .await
            {
                tracing::warn!(error = %e, "Failed to debit credits for 3D generation");
            }
        });
    }

    // Background: poll for completion and auto-store artifact
    if let Some(ref project_id) = input.project_id {
        if let (Some(ref storage_url), Some(ref storage_token)) =
            (&state.aura_storage_url, &state.aura_storage_token)
        {
            let client = state.http_client.clone();
            let api_key = tripo_api_key.clone();
            let tid = task_id.clone();
            let surl = storage_url.clone();
            let stok = storage_token.clone();
            let pid = project_id.clone();
            let uid = auth.user_id.clone();
            let name = input.name.clone();
            let prompt = input.prompt.clone();
            let parent = input.parent_id.clone();
            tokio::spawn(async move {
                match tripo::poll_task(&client, &api_key, &tid).await {
                    Ok(status) if status.status == "success" => {
                        if let Some(ref glb_url) = status.glb_url {
                            storage::store_artifact(
                                &client,
                                &surl,
                                &stok,
                                &pid,
                                &uid,
                                "model",
                                glb_url,
                                None,
                                name.as_deref(),
                                prompt.as_deref(),
                                None,
                                "tripo-v2",
                                "tripo",
                                false,
                                parent.as_deref(),
                            )
                            .await;
                        }
                    }
                    Ok(status) => {
                        tracing::warn!(task_id = %tid, status = %status.status, "3D generation did not succeed");
                    }
                    Err(e) => {
                        tracing::warn!(task_id = %tid, error = %e, "3D generation polling failed");
                    }
                }
            });
        }
    }

    let response = tripo::Generate3dResponse {
        success: true,
        task_id,
        eta_ms: 45000,
    };

    Ok(Json(response).into_response())
}

/// POST /v1/generate-3d/stream — Submit and stream 3D generation progress via SSE.
pub async fn generate_3d_stream(
    auth: AuthUser,
    State(state): State<AppState>,
    Json(input): Json<tripo::Generate3dRequest>,
) -> Result<Response, AppError> {
    if input.image_url.trim().is_empty() {
        return Err(AppError::BadRequest("imageUrl must not be empty".into()));
    }

    let tripo_api_key = state
        .tripo_api_key
        .as_ref()
        .ok_or_else(|| AppError::Internal("Tripo not configured".into()))?
        .clone();

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
        50,
    )
    .await?;

    if !balance.sufficient {
        return Err(AppError::InsufficientCredits {
            balance: balance.balance_cents,
            required: 50,
        });
    }

    // Upload data URL to S3 if needed
    let image_url = if input.image_url.starts_with("data:") {
        let s3_config = state
            .s3_config
            .as_ref()
            .ok_or_else(|| AppError::Internal("S3 not configured".into()))?;
        s3_config
            .upload_base64(&input.image_url, &auth.user_id)
            .await
            .map_err(|e| AppError::Internal(format!("S3 upload failed: {e}")))?
    } else {
        input.image_url.clone()
    };

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<serde_json::Value>(32);

    let gen_state = state.clone();
    let gen_user_id = auth.user_id.clone();
    let gen_project_id = input.project_id.clone();
    let gen_parent_id = input.parent_id.clone();
    let gen_name = input.name.clone();
    let gen_prompt = input.prompt.clone();

    tokio::spawn(async move {
        // Submit task
        let _ = event_tx
            .send(serde_json::json!({"type": "start", "ts": chrono::Utc::now().to_rfc3339()}))
            .await;

        let task_id = match tripo::create_task(&gen_state.http_client, &tripo_api_key, &image_url)
            .await
        {
            Ok(tid) => {
                let _ = event_tx
                    .send(serde_json::json!({"type": "submitted", "taskId": tid}))
                    .await;
                tid
            }
            Err(e) => {
                let _ = event_tx
                    .send(serde_json::json!({"type": "error", "code": "SUBMIT_FAILED", "message": e}))
                    .await;
                return;
            }
        };

        // Debit credits
        let _ = billing::report_image_usage(
            &gen_state.http_client,
            &gen_state.z_billing_url,
            &gen_state.z_billing_api_key,
            &uuid::Uuid::new_v4().to_string(),
            &gen_user_id,
            "tripo",
            "tripo-v2",
            50,
        )
        .await;

        // Poll for completion
        let _ = event_tx
            .send(serde_json::json!({"type": "progress", "percent": 10, "message": "Generating 3D model..."}))
            .await;

        match tripo::poll_task(&gen_state.http_client, &tripo_api_key, &task_id).await {
            Ok(status) if status.status == "success" => {
                // Auto-store artifact
                if let Some(ref pid) = gen_project_id {
                    if let (Some(ref surl), Some(ref stok)) =
                        (&gen_state.aura_storage_url, &gen_state.aura_storage_token)
                    {
                        if let Some(ref glb_url) = status.glb_url {
                            storage::store_artifact(
                                &gen_state.http_client,
                                surl,
                                stok,
                                pid,
                                &gen_user_id,
                                "model",
                                glb_url,
                                None,
                                gen_name.as_deref(),
                                gen_prompt.as_deref(),
                                None,
                                "tripo-v2",
                                "tripo",
                                false,
                                gen_parent_id.as_deref(),
                            )
                            .await;
                        }
                    }
                }

                let _ = event_tx
                    .send(serde_json::json!({
                        "type": "completed",
                        "taskId": task_id,
                        "glbUrl": status.glb_url,
                        "polyCount": status.poly_count,
                    }))
                    .await;
            }
            Ok(status) => {
                let _ = event_tx
                    .send(serde_json::json!({
                        "type": "error",
                        "code": "GENERATION_FAILED",
                        "message": status.error.unwrap_or_else(|| "Generation failed".to_string()),
                    }))
                    .await;
            }
            Err(e) => {
                let _ = event_tx
                    .send(serde_json::json!({"type": "error", "code": "POLL_FAILED", "message": e}))
                    .await;
            }
        }
    });

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

/// GET /v1/generate-3d/:taskId — Check status of a 3D generation task.
pub async fn get_3d_status(
    _auth: AuthUser,
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<Json<tripo::TaskStatusResponse>, AppError> {
    let tripo_api_key = state
        .tripo_api_key
        .as_ref()
        .ok_or_else(|| AppError::Internal("Tripo not configured".into()))?;

    let status = tripo::check_task_status(&state.http_client, tripo_api_key, &task_id)
        .await
        .map_err(|e| AppError::ProviderError(e))?;

    Ok(Json(status))
}
