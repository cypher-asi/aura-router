//! Provider resolution — maps model names to LLM providers.

use reqwest::header::{HeaderMap, HeaderValue};

/// Supported LLM providers.
#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    Anthropic,
    OpenAi,
}

impl Provider {
    /// Provider name string for billing/stats.
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAi => "openai",
        }
    }
}

/// Resolve a model name to its provider.
pub fn resolve_provider(model: &str) -> Option<Provider> {
    if model.starts_with("claude") {
        Some(Provider::Anthropic)
    } else if model.starts_with("gpt")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("codex")
    {
        Some(Provider::OpenAi)
    } else {
        None
    }
}

/// Get the base URL for a provider's messages endpoint.
pub fn provider_url(provider: &Provider) -> &'static str {
    match provider {
        Provider::Anthropic => "https://api.anthropic.com/v1/messages",
        Provider::OpenAi => "https://api.openai.com/v1/chat/completions",
    }
}

/// Get the maximum context window size for a model (in tokens).
pub fn max_context_tokens(model: &str) -> u64 {
    match model {
        // Anthropic
        m if m.starts_with("claude-opus-4") => 1_000_000,
        m if m.starts_with("claude-sonnet-4") => 1_000_000,
        m if m.starts_with("claude-haiku-4") => 200_000,
        m if m.starts_with("claude-3-5") => 200_000,
        m if m.starts_with("claude-3") => 200_000,
        m if m.starts_with("claude") => 200_000,
        // OpenAI
        m if m.starts_with("gpt-4o") => 128_000,
        m if m.starts_with("gpt-4-turbo") => 128_000,
        m if m.starts_with("gpt-4") => 8_192,
        m if m.starts_with("gpt-3.5") => 16_385,
        m if m.starts_with("o1") => 200_000,
        m if m.starts_with("o3") => 200_000,
        m if m.starts_with("o4") => 200_000,
        m if m.starts_with("codex") => 200_000,
        _ => 200_000, // safe default
    }
}

/// Build provider-specific headers for the upstream request.
///
/// Returns None if the API key contains invalid header characters.
pub fn provider_headers(provider: &Provider, api_key: &str) -> Option<HeaderMap> {
    let mut headers = HeaderMap::new();
    match provider {
        Provider::Anthropic => {
            headers.insert("x-api-key", HeaderValue::from_str(api_key).ok()?);
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            headers.insert(
                "anthropic-beta",
                HeaderValue::from_static("prompt-caching-2024-07-31"),
            );
        }
        Provider::OpenAi => {
            headers.insert(
                "authorization",
                HeaderValue::from_str(&format!("Bearer {api_key}")).ok()?,
            );
        }
    }
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    Some(headers)
}
