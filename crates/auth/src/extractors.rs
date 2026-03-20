use async_trait::async_trait;
use axum::extract::FromRequestParts;
use axum::http::request::Parts;

use aura_router_core::AppError;

use crate::validate::{TokenClaims, TokenValidator};

/// Authenticated user extracted from JWT in Authorization header.
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub user_id: String,
    pub claims: TokenClaims,
}

#[async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync + AsRef<TokenValidator>,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let token = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or_else(|| AppError::Unauthorized("Missing authorization header".into()))?;

        let validator = state.as_ref();
        let claims = validator
            .validate(token)
            .await
            .map_err(AppError::Unauthorized)?;

        let user_id = claims
            .user_id()
            .ok_or_else(|| AppError::Unauthorized("Token missing user ID".into()))?
            .to_string();

        Ok(AuthUser { user_id, claims })
    }
}

/// Internal service auth extracted from X-Internal-Token header.
#[derive(Debug, Clone)]
pub struct InternalAuth;

/// Wrapper for the internal service token, stored in AppState.
#[derive(Clone)]
pub struct InternalToken(pub String);

#[async_trait]
impl<S> FromRequestParts<S> for InternalAuth
where
    S: Send + Sync + AsRef<InternalToken>,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let token = parts
            .headers
            .get("x-internal-token")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| AppError::Unauthorized("Missing internal token".into()))?;

        let expected = state.as_ref();
        if token != expected.0 {
            return Err(AppError::Unauthorized("Invalid internal token".into()));
        }

        Ok(InternalAuth)
    }
}
