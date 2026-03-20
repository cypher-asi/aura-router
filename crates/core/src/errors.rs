use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Insufficient credits: balance={balance}, required={required}")]
    InsufficientCredits { balance: i64, required: i64 },

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("Billing service error: {0}")]
    BillingError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            AppError::NotFound(_) => (StatusCode::NOT_FOUND, "NOT_FOUND"),
            AppError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "UNAUTHORIZED"),
            AppError::Forbidden(_) => (StatusCode::FORBIDDEN, "FORBIDDEN"),
            AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "BAD_REQUEST"),
            AppError::InsufficientCredits { .. } => {
                (StatusCode::PAYMENT_REQUIRED, "INSUFFICIENT_CREDITS")
            }
            AppError::ProviderError(_) => (StatusCode::BAD_GATEWAY, "PROVIDER_ERROR"),
            AppError::BillingError(_) => (StatusCode::SERVICE_UNAVAILABLE, "BILLING_UNAVAILABLE"),
            AppError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL"),
        };

        let body = serde_json::json!({
            "error": {
                "code": code,
                "message": self.to_string()
            }
        });

        (status, axum::Json(body)).into_response()
    }
}
