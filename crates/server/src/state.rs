use std::sync::Arc;

use aura_router_auth::{InternalToken, TokenValidator};
use aura_router_proxy::rate_limit::RateLimiter;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub validator: TokenValidator,
    pub internal_token: InternalToken,
    pub http_client: reqwest::Client,
    pub rate_limiter: Arc<RateLimiter>,

    // Provider API keys
    pub anthropic_api_key: String,
    pub openai_api_key: Option<String>,
    pub google_api_key: Option<String>,

    // Service URLs
    pub z_billing_url: String,
    pub z_billing_api_key: String,
    pub aura_network_url: Option<String>,
    pub aura_network_token: Option<String>,
    pub aura_storage_url: Option<String>,
    pub aura_storage_token: Option<String>,

    // Image generation
    pub s3_config: Option<aura_router_proxy::s3::S3Config>,
    pub watermark_bytes: Option<Vec<u8>>,
}

impl AsRef<TokenValidator> for AppState {
    fn as_ref(&self) -> &TokenValidator {
        &self.validator
    }
}

impl AsRef<InternalToken> for AppState {
    fn as_ref(&self) -> &InternalToken {
        &self.internal_token
    }
}
