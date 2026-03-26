//! 3D generation handler — submits image-to-3D tasks via Tripo, polls for results.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use aura_router_auth::AuthUser;
use aura_router_core::AppError;
use aura_router_proxy::{billing, tripo};

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

    let response = tripo::Generate3dResponse {
        success: true,
        task_id,
        eta_ms: 45000,
    };

    Ok(Json(response).into_response())
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
