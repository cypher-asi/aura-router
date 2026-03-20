//! Provider resolution — maps model names to LLM providers.

use reqwest::header::{HeaderMap, HeaderValue};

/// Supported LLM providers.
#[derive(Debug, Clone)]
pub enum Provider {
    Anthropic,
}

/// Resolve a model name to its provider.
pub fn resolve_provider(model: &str) -> Option<Provider> {
    if model.starts_with("claude") {
        Some(Provider::Anthropic)
    } else {
        None
    }
}

/// Get the base URL for a provider's messages endpoint.
pub fn provider_url(provider: &Provider) -> &'static str {
    match provider {
        Provider::Anthropic => "https://api.anthropic.com/v1/messages",
    }
}

/// Build provider-specific headers for the upstream request.
pub fn provider_headers(provider: &Provider, api_key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    match provider {
        Provider::Anthropic => {
            headers.insert("x-api-key", HeaderValue::from_str(api_key).unwrap());
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            headers.insert(
                "anthropic-beta",
                HeaderValue::from_static("prompt-caching-2024-07-31"),
            );
            headers.insert("content-type", HeaderValue::from_static("application/json"));
        }
    }
    headers
}
