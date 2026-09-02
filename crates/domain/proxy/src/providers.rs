//! Provider resolution — maps model names to LLM providers.

use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;

const ANTHROPIC_BETA_HEADER: &str = "anthropic-beta";
const PROMPT_CACHING_BETA: &str = "prompt-caching-2024-07-31";
const COMPUTER_USE_BETA: &str = "computer-use-2025-01-24";

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAi,
    Xai,
    Moonshot,
    Fireworks,
    DeepSeek,
    Google,
}

impl Provider {
    /// Provider name string for billing/stats.
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAi => "openai",
            Provider::Xai => "xai",
            Provider::Moonshot => "moonshot",
            Provider::Fireworks => "fireworks",
            Provider::DeepSeek => "deepseek",
            Provider::Google => "google",
        }
    }
}

/// The company that *makes* a model (its research lab / vendor), as shown
/// to users in the model picker.
///
/// Deliberately distinct from [`Provider`], which identifies the upstream
/// host we actually call: e.g. DeepSeek, Qwen, MiniMax and GLM are all
/// served via `Provider::Fireworks` yet are made by four different
/// companies.
///
/// The display strings returned by [`Maker::display_name`] are the
/// canonical company labels and MUST stay in sync with aura-os
/// `MODEL_VENDOR_LABELS` (`interface/src/constants/models.ts`) and the
/// z-billing `Maker` enum. These three repos share no code, so the set is
/// duplicated and kept aligned by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maker {
    /// Anthropic (Claude).
    Anthropic,
    /// OpenAI (GPT, o-series, GPT-OSS).
    OpenAi,
    /// Google (Gemini, Gemma).
    Google,
    /// xAI (Grok).
    Xai,
    /// DeepSeek AI.
    DeepSeek,
    /// Moonshot AI (Kimi).
    Moonshot,
    /// MiniMax.
    MiniMax,
    /// Z.ai / Zhipu AI (GLM).
    Zai,
    /// Alibaba Cloud (Qwen / Tongyi).
    Alibaba,
}

impl Maker {
    /// Human-facing company name shown in the model picker.
    pub fn display_name(&self) -> &'static str {
        match self {
            Maker::Anthropic => "Anthropic",
            Maker::OpenAi => "OpenAI",
            Maker::Google => "Google",
            Maker::Xai => "xAI",
            Maker::DeepSeek => "DeepSeek AI",
            Maker::Moonshot => "Moonshot AI",
            Maker::MiniMax => "MiniMax",
            Maker::Zai => "Z.ai",
            Maker::Alibaba => "Alibaba Cloud",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel<'a> {
    pub requested_model: &'a str,
    pub upstream_model: &'a str,
    pub provider: Provider,
}

fn aura_model_alias(model: &str) -> Option<ResolvedModel<'_>> {
    match model {
        "aura-claude-fable-5-1" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-fable-5-1",
            provider: Provider::Anthropic,
        }),
        "aura-claude-mythos-5-1" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-mythos-5-1",
            provider: Provider::Anthropic,
        }),
        "aura-claude-fable-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-fable-5",
            provider: Provider::Anthropic,
        }),
        "aura-claude-opus-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-opus-5",
            provider: Provider::Anthropic,
        }),
        "aura-claude-opus-4-8" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-opus-4-8",
            provider: Provider::Anthropic,
        }),
        "aura-claude-opus-4-7" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-opus-4-7",
            provider: Provider::Anthropic,
        }),
        "aura-claude-opus-4-6" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-opus-4-6",
            provider: Provider::Anthropic,
        }),
        "aura-claude-sonnet-4-6" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-sonnet-4-6",
            provider: Provider::Anthropic,
        }),
        "aura-claude-sonnet-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-sonnet-5",
            provider: Provider::Anthropic,
        }),
        "aura-claude-haiku-4-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-haiku-4-5",
            provider: Provider::Anthropic,
        }),
        "aura-gpt-5-6-sol" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.6-sol",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-6-terra" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.6-terra",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-6-luna" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.6-luna",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.5",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-4" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.4",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-4-mini" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.4-mini",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-4-nano" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.4-nano",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-4.1" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-4.1",
            provider: Provider::OpenAi,
        }),
        "aura-o3" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "o3",
            provider: Provider::OpenAi,
        }),
        "aura-o4-mini" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "o4-mini",
            provider: Provider::OpenAi,
        }),
        "aura-grok-4-6" | "xai/grok-4.6" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "grok-4.6",
            provider: Provider::Xai,
        }),
        "aura-grok-4-5" | "xai/grok-4.5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "grok-4.5",
            provider: Provider::Xai,
        }),
        "aura-grok-4-3" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "grok-4.3",
            provider: Provider::Xai,
        }),
        "xai/grok-4.3" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "grok-4.3",
            provider: Provider::Xai,
        }),
        "aura-grok-build-0-1"
        | "xai/grok-build-0.1"
        | "xai/grok-code-fast"
        | "xai/grok-code-fast-1"
        | "xai/grok-code-fast-1-0825" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "grok-build-0.1",
            provider: Provider::Xai,
        }),
        // DeepSeek V4 models are served via Fireworks (which hosts them
        // verbatim) rather than DeepSeek's first-party API, so they reuse the
        // already-provisioned FIREWORKS_API_KEY. Provider::DeepSeek remains for
        // raw first-party passthrough names (see infer_provider).
        "aura-deepseek-v4-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/deepseek-v4-pro",
            provider: Provider::Fireworks,
        }),
        "aura-deepseek-v4-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/deepseek-v4-flash",
            provider: Provider::Fireworks,
        }),
        "deepseek/deepseek-v4-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/deepseek-v4-pro",
            provider: Provider::Fireworks,
        }),
        "deepseek/deepseek-v4-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/deepseek-v4-flash",
            provider: Provider::Fireworks,
        }),
        "aura-kimi-k3" | "moonshot/kimi-k3" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "kimi-k3",
            provider: Provider::Moonshot,
        }),
        "aura-kimi-k2-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/kimi-k2p5",
            provider: Provider::Fireworks,
        }),
        "aura-kimi-k2-6" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/kimi-k2p6",
            provider: Provider::Fireworks,
        }),
        "aura-kimi-k2-7-code" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/kimi-k2p7-code",
            provider: Provider::Fireworks,
        }),
        "aura-oss-120b" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/gpt-oss-120b",
            provider: Provider::Fireworks,
        }),
        "aura-qwen2-5-coder-7b" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/qwen2p5-coder-7b",
            provider: Provider::Fireworks,
        }),
        "aura-minimax-m3" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/minimax-m3",
            provider: Provider::Fireworks,
        }),
        "aura-minimax-m2-7" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/minimax-m2p7",
            provider: Provider::Fireworks,
        }),
        "aura-glm-5-1" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/glm-5p1",
            provider: Provider::Fireworks,
        }),
        "aura-glm-5-2" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/glm-5p2",
            provider: Provider::Fireworks,
        }),
        "aura-qwen3-6-plus" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/qwen3p6-plus",
            provider: Provider::Fireworks,
        }),
        "aura-qwen3-7-plus" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/qwen3p7-plus",
            provider: Provider::Fireworks,
        }),
        // Google Gemini chat models route directly through the Google
        // Generative Language API (`generativelanguage.googleapis.com`)
        // using the platform GOOGLE_API_KEY. Pro models map to their
        // current preview string; stable Flash/Flash-Lite tiers map to the
        // bare stable name.
        "aura-gemini-3-1-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-3.1-pro-preview",
            provider: Provider::Google,
        }),
        "aura-gemini-3-5-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-3.5-flash",
            provider: Provider::Google,
        }),
        "aura-gemini-3-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-3-flash-preview",
            provider: Provider::Google,
        }),
        "aura-gemini-3-1-flash-lite" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-3.1-flash-lite",
            provider: Provider::Google,
        }),
        "aura-gemini-2-5-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-2.5-pro",
            provider: Provider::Google,
        }),
        "aura-gemini-2-5-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-2.5-flash",
            provider: Provider::Google,
        }),
        "aura-gemini-2-5-flash-lite" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-2.5-flash-lite",
            provider: Provider::Google,
        }),
        _ => None,
    }
}

fn infer_provider(model: &str) -> Option<Provider> {
    if model.starts_with("claude") {
        Some(Provider::Anthropic)
    } else if model.starts_with("gpt")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("codex")
    {
        Some(Provider::OpenAi)
    } else if model.starts_with("grok") {
        Some(Provider::Xai)
    } else if model == "kimi-k3" {
        Some(Provider::Moonshot)
    } else if model.starts_with("deepseek-v4")
        || model == "deepseek-chat"
        || model == "deepseek-reasoner"
    {
        Some(Provider::DeepSeek)
    } else if model.starts_with("gemini") {
        Some(Provider::Google)
    } else {
        None
    }
}

/// Resolve an Aura or upstream model name into its upstream provider/model pair.
pub fn resolve_model(model: &str) -> Option<ResolvedModel<'_>> {
    if let Some(alias) = aura_model_alias(model) {
        return Some(alias);
    }

    infer_provider(model).map(|provider| ResolvedModel {
        requested_model: model,
        upstream_model: model,
        provider,
    })
}

/// Resolve a model name to its provider.
pub fn resolve_provider(model: &str) -> Option<Provider> {
    resolve_model(model).map(|resolved| resolved.provider)
}

/// Resolve a model name to the company that makes it.
///
/// Resolves through [`resolve_model`] first so both Aura aliases
/// (`aura-…`) and raw upstream names map correctly — including
/// Fireworks-hosted `accounts/<org>/models/<name>` paths, whose host
/// (`Provider::Fireworks`) says nothing about who built the model.
/// Returns `None` for unrecognized models.
pub fn model_maker(model: &str) -> Option<Maker> {
    let upstream = resolve_model(model)
        .map(|resolved| resolved.upstream_model)
        .unwrap_or(model);
    // Reduce Fireworks-style "accounts/<org>/models/<name>" paths to the
    // bare model name so the checks below match on the model, not the host.
    let name = upstream.rsplit('/').next().unwrap_or(upstream);
    let maker = if name.contains("claude") {
        Maker::Anthropic
    } else if name.starts_with("gpt")
        || name.starts_with("o1")
        || name.starts_with("o3")
        || name.starts_with("o4")
        || name.starts_with("codex")
        || name.contains("oss")
    {
        Maker::OpenAi
    } else if name.contains("gemini") || name.contains("gemma") {
        Maker::Google
    } else if name.contains("grok") {
        Maker::Xai
    } else if name.contains("deepseek") {
        Maker::DeepSeek
    } else if name.contains("kimi") {
        Maker::Moonshot
    } else if name.contains("minimax") {
        Maker::MiniMax
    } else if name.contains("glm") {
        Maker::Zai
    } else if name.contains("qwen") {
        Maker::Alibaba
    } else {
        return None;
    };
    Some(maker)
}

/// Which OpenAI HTTP surface a request should use.
///
/// OpenAI's `/v1/chat/completions` rejects requests that combine function
/// `tools` with `reasoning_effort` for the gpt-5 family ("Please use
/// /v1/responses instead"). The `/v1/responses` API supports tools +
/// reasoning together and is OpenAI's go-forward surface for tool use, so
/// any tool-bearing OpenAI request is routed through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiApi {
    ChatCompletions,
    Responses,
}

/// Decide which OpenAI-compatible surface an inbound `/v1/messages` request
/// maps to.
///
/// OpenAI and xAI both expose a Responses API. For OpenAI we use it when
/// Anthropic function tools are present. For xAI we use it for any tool
/// surface, including native server-side tools and Remote MCP servers, since
/// those are documented on the xAI Responses path.
pub fn openai_api_for_request(provider: Provider, request: &Value) -> OpenAiApi {
    let has_tools = request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    match provider {
        Provider::OpenAi if has_tools => OpenAiApi::Responses,
        Provider::Xai
            if has_tools
                || has_non_empty_array(request, "xai_tools")
                || has_non_empty_array(request, "server_tools")
                || has_non_empty_array(request, "xai_mcp_servers") =>
        {
            OpenAiApi::Responses
        }
        _ => OpenAiApi::ChatCompletions,
    }
}

fn has_non_empty_array(request: &Value, key: &str) -> bool {
    request
        .get(key)
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
}

/// Get the base URL for a provider's messages endpoint.
///
/// For OpenAI this returns the chat/completions surface; callers that may
/// need the Responses API should use [`openai_endpoint_url`] with the
/// [`OpenAiApi`] resolved from the request.
pub fn provider_url(provider: &Provider) -> &'static str {
    match provider {
        Provider::Anthropic => "https://api.anthropic.com/v1/messages",
        Provider::OpenAi => "https://api.openai.com/v1/chat/completions",
        Provider::Xai => "https://api.x.ai/v1/chat/completions",
        Provider::Moonshot => "https://api.moonshot.ai/v1/chat/completions",
        // Intentionally use the stateless chat completions path for Aura's OSS lane.
        // Aura Router centrally avoids Fireworks surfaces that can retain conversation state.
        Provider::Fireworks => "https://api.fireworks.ai/inference/v1/chat/completions",
        Provider::DeepSeek => "https://api.deepseek.com/chat/completions",
        // Google encodes the model + streaming mode in the path, so callers
        // must use `google_endpoint_url` instead. This base is returned only
        // to keep the match exhaustive.
        Provider::Google => "https://generativelanguage.googleapis.com/v1beta",
    }
}

/// Google Generative Language endpoint for a Gemini chat request.
///
/// Unlike the other providers, Gemini puts the upstream model and the
/// streaming mode in the URL path (`:generateContent` vs
/// `:streamGenerateContent?alt=sse`) rather than the request body.
pub fn google_endpoint_url(upstream_model: &str, streaming: bool) -> String {
    let method = if streaming {
        "streamGenerateContent?alt=sse"
    } else {
        "generateContent"
    };
    format!("https://generativelanguage.googleapis.com/v1beta/models/{upstream_model}:{method}")
}

/// Get the OpenAI endpoint URL for the given API surface.
pub fn openai_endpoint_url(api: OpenAiApi) -> &'static str {
    match api {
        OpenAiApi::ChatCompletions => "https://api.openai.com/v1/chat/completions",
        OpenAiApi::Responses => "https://api.openai.com/v1/responses",
    }
}

/// Get the xAI endpoint URL for the given OpenAI-compatible API surface.
pub fn xai_endpoint_url(api: OpenAiApi) -> &'static str {
    match api {
        OpenAiApi::ChatCompletions => "https://api.x.ai/v1/chat/completions",
        OpenAiApi::Responses => "https://api.x.ai/v1/responses",
    }
}

/// Get the maximum context window size for a model (in tokens).
pub fn max_context_tokens(model: &str) -> u64 {
    let resolved_model = resolve_model(model)
        .map(|resolved| resolved.upstream_model)
        .unwrap_or(model);
    match resolved_model {
        // Anthropic
        m if m.starts_with("claude-fable-5") => 1_000_000,
        m if m.starts_with("claude-mythos-5") => 1_000_000,
        m if m.starts_with("claude-opus-5") => 1_000_000,
        m if m.starts_with("claude-opus-4") => 1_000_000,
        m if m.starts_with("claude-sonnet-5") => 1_000_000,
        m if m.starts_with("claude-sonnet-4") => 1_000_000,
        m if m.starts_with("claude-haiku-4") => 200_000,
        m if m.starts_with("claude-3-5") => 200_000,
        m if m.starts_with("claude-3") => 200_000,
        m if m.starts_with("claude") => 200_000,
        // OpenAI
        m if m.starts_with("gpt-5.6") => 1_050_000,
        "gpt-5.5" => 1_050_000,
        "gpt-5.4" => 1_050_000,
        "gpt-5.4-mini" => 400_000,
        "gpt-5.4-nano" => 400_000,
        // xAI
        "grok-4.6" => 500_000,
        "grok-4.5" => 500_000,
        "grok-4.3" => 1_000_000,
        "grok-build-0.1" | "grok-code-fast" | "grok-code-fast-1" | "grok-code-fast-1-0825" => {
            256_000
        }
        // Moonshot
        "kimi-k3" => 1_048_576,
        m if m.starts_with("gpt-4o") => 128_000,
        m if m.starts_with("gpt-4-turbo") => 128_000,
        m if m.starts_with("gpt-4") => 8_192,
        m if m.starts_with("gpt-3.5") => 16_385,
        m if m.starts_with("o1") => 200_000,
        m if m.starts_with("o3") => 200_000,
        m if m.starts_with("o4") => 200_000,
        m if m.starts_with("codex") => 200_000,
        // Fireworks OSS
        "accounts/fireworks/models/kimi-k2p5" => 262_144,
        "accounts/fireworks/models/kimi-k2p6" => 262_144,
        "accounts/fireworks/models/kimi-k2p7-code" => 262_144,
        "accounts/fireworks/models/gpt-oss-120b" => 131_072,
        "accounts/fireworks/models/qwen2p5-coder-7b" => 32_768,
        "accounts/fireworks/models/minimax-m3" => 512_000,
        "accounts/fireworks/models/minimax-m2p7" => 196_608,
        "accounts/fireworks/models/glm-5p1" => 202_752,
        "accounts/fireworks/models/glm-5p2" => 1_048_576,
        "accounts/fireworks/models/qwen3p6-plus" => 262_144,
        "accounts/fireworks/models/qwen3p7-plus" => 262_144,
        // DeepSeek V4 via Fireworks
        "accounts/fireworks/models/deepseek-v4-pro"
        | "accounts/fireworks/models/deepseek-v4-flash" => 1_048_576,
        // DeepSeek V4 direct API
        "deepseek-v4-pro" | "deepseek-v4-flash" | "deepseek-chat" | "deepseek-reasoner" => {
            1_000_000
        }
        // Google Gemini — the 2.5 and 3 families all expose a 1M token window.
        m if m.starts_with("gemini") => 1_000_000,
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
                ANTHROPIC_BETA_HEADER,
                HeaderValue::from_static(PROMPT_CACHING_BETA),
            );
        }
        Provider::OpenAi | Provider::Xai | Provider::Moonshot | Provider::DeepSeek => {
            headers.insert(
                "authorization",
                HeaderValue::from_str(&format!("Bearer {api_key}")).ok()?,
            );
        }
        Provider::Fireworks => {
            headers.insert(
                "authorization",
                HeaderValue::from_str(&format!("Bearer {api_key}")).ok()?,
            );
        }
        Provider::Google => {
            // The Generative Language API authenticates with a bare API key
            // header rather than a Bearer token.
            headers.insert("x-goog-api-key", HeaderValue::from_str(api_key).ok()?);
        }
    }
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    Some(headers)
}

/// Merge Anthropic beta tokens from an inbound proxy request into the
/// provider-bound header. Only beta programs AURA Router explicitly supports
/// are forwarded; arbitrary client-supplied beta values are ignored.
pub fn merge_anthropic_beta_header(headers: &mut HeaderMap, inbound_beta: &str) -> Option<()> {
    let mut tokens = vec![PROMPT_CACHING_BETA.to_string()];
    for token in inbound_beta
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if token == COMPUTER_USE_BETA && !tokens.iter().any(|existing| existing == token) {
            tokens.push(token.to_string());
        }
    }
    headers.insert(
        ANTHROPIC_BETA_HEADER,
        HeaderValue::from_str(&tokens.join(",")).ok()?,
    );
    Some(())
}

#[cfg(test)]
mod tests {
    use super::{
        merge_anthropic_beta_header, model_maker, openai_api_for_request, openai_endpoint_url,
        provider_headers, resolve_model, resolve_provider, xai_endpoint_url, OpenAiApi, Provider,
    };
    use serde_json::json;

    #[test]
    fn openai_tool_requests_route_to_responses_api() {
        let request = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "name": "search", "input_schema": {} }],
        });
        assert_eq!(
            openai_api_for_request(Provider::OpenAi, &request),
            OpenAiApi::Responses
        );
        assert_eq!(
            openai_endpoint_url(openai_api_for_request(Provider::OpenAi, &request)),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn openai_requests_without_tools_use_chat_completions() {
        let request = json!({ "messages": [{ "role": "user", "content": "hi" }] });
        assert_eq!(
            openai_api_for_request(Provider::OpenAi, &request),
            OpenAiApi::ChatCompletions
        );
        let empty_tools = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [],
        });
        assert_eq!(
            openai_api_for_request(Provider::OpenAi, &empty_tools),
            OpenAiApi::ChatCompletions
        );
    }

    #[test]
    fn anthropic_headers_forward_computer_use_beta() {
        let mut headers =
            provider_headers(&Provider::Anthropic, "sk-ant").expect("valid api key header");

        merge_anthropic_beta_header(
            &mut headers,
            "computer-use-2025-01-24,unknown-beta-2099-01-01,prompt-caching-2024-07-31",
        )
        .expect("valid merged beta header");

        assert_eq!(
            headers
                .get("anthropic-beta")
                .and_then(|value| value.to_str().ok()),
            Some("prompt-caching-2024-07-31,computer-use-2025-01-24")
        );
    }

    #[test]
    fn xai_headers_use_bearer_auth() {
        let headers = provider_headers(&Provider::Xai, "xai-test-key").expect("valid headers");
        assert_eq!(
            headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer xai-test-key")
        );
        assert_eq!(
            super::provider_url(&Provider::Xai),
            "https://api.x.ai/v1/chat/completions"
        );
        assert_eq!(
            xai_endpoint_url(OpenAiApi::Responses),
            "https://api.x.ai/v1/responses"
        );
    }

    #[test]
    fn xai_tool_requests_route_to_responses_api() {
        let function_tools = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "name": "search", "input_schema": {} }],
        });
        assert_eq!(
            openai_api_for_request(Provider::Xai, &function_tools),
            OpenAiApi::Responses
        );

        let remote_mcp = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "xai_mcp_servers": [{
                "server_url": "https://mcp.deepwiki.com/mcp",
                "server_label": "deepwiki"
            }],
        });
        assert_eq!(
            openai_api_for_request(Provider::Xai, &remote_mcp),
            OpenAiApi::Responses
        );
    }

    #[test]
    fn other_non_openai_providers_never_select_responses_api() {
        let request = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "name": "search", "input_schema": {} }],
        });
        for provider in [
            Provider::Anthropic,
            Provider::Moonshot,
            Provider::Fireworks,
            Provider::DeepSeek,
        ] {
            assert_eq!(
                openai_api_for_request(provider, &request),
                OpenAiApi::ChatCompletions
            );
        }
    }

    #[test]
    fn resolves_aura_aliases_to_upstream_models() {
        for (alias, upstream) in [
            ("aura-claude-fable-5-1", "claude-fable-5-1"),
            ("aura-claude-mythos-5-1", "claude-mythos-5-1"),
        ] {
            let resolved = resolve_model(alias).expect("Claude 5.1 alias should resolve");
            assert_eq!(resolved.requested_model, alias);
            assert_eq!(resolved.upstream_model, upstream);
            assert_eq!(resolved.provider, Provider::Anthropic);
            assert_eq!(super::max_context_tokens(alias), 1_000_000);
        }

        let resolved = resolve_model("aura-claude-fable-5").expect("model alias should resolve");
        assert_eq!(resolved.requested_model, "aura-claude-fable-5");
        assert_eq!(resolved.upstream_model, "claude-fable-5");
        assert_eq!(resolved.provider, Provider::Anthropic);
        assert_eq!(super::max_context_tokens("aura-claude-fable-5"), 1_000_000);

        let resolved = resolve_model("aura-claude-opus-5").expect("model alias should resolve");
        assert_eq!(resolved.requested_model, "aura-claude-opus-5");
        assert_eq!(resolved.upstream_model, "claude-opus-5");
        assert_eq!(resolved.provider, Provider::Anthropic);
        assert_eq!(super::max_context_tokens("aura-claude-opus-5"), 1_000_000);

        let resolved = resolve_model("aura-claude-sonnet-5").expect("model alias should resolve");
        assert_eq!(resolved.upstream_model, "claude-sonnet-5");
        assert_eq!(resolved.provider, Provider::Anthropic);
        assert_eq!(super::max_context_tokens("aura-claude-sonnet-5"), 1_000_000);

        let resolved = resolve_model("aura-gpt-5-5").expect("model alias should resolve");
        assert_eq!(resolved.requested_model, "aura-gpt-5-5");
        assert_eq!(resolved.upstream_model, "gpt-5.5");
        assert_eq!(resolved.provider, Provider::OpenAi);
        assert_eq!(super::max_context_tokens("aura-gpt-5-5"), 1_050_000);

        for (alias, upstream) in [
            ("aura-gpt-5-6-sol", "gpt-5.6-sol"),
            ("aura-gpt-5-6-terra", "gpt-5.6-terra"),
            ("aura-gpt-5-6-luna", "gpt-5.6-luna"),
        ] {
            let resolved = resolve_model(alias).expect("GPT-5.6 alias should resolve");
            assert_eq!(resolved.upstream_model, upstream);
            assert_eq!(resolved.provider, Provider::OpenAi);
            assert_eq!(super::max_context_tokens(alias), 1_050_000);
        }

        let resolved = resolve_model("aura-gpt-5-4-mini").expect("model alias should resolve");
        assert_eq!(resolved.requested_model, "aura-gpt-5-4-mini");
        assert_eq!(resolved.upstream_model, "gpt-5.4-mini");
        assert_eq!(resolved.provider, Provider::OpenAi);
    }

    #[test]
    fn preserves_upstream_model_names() {
        let resolved = resolve_model("claude-sonnet-4-6").expect("provider should resolve");
        assert_eq!(resolved.requested_model, "claude-sonnet-4-6");
        assert_eq!(resolved.upstream_model, "claude-sonnet-4-6");
        assert_eq!(resolved.provider, Provider::Anthropic);
    }

    #[test]
    fn model_maker_attributes_models_to_their_company_not_their_host() {
        // Fireworks-hosted models still resolve to their real maker, even
        // though their host provider is Fireworks.
        let cases = [
            ("aura-claude-opus-5", "Anthropic"),
            ("aura-gpt-5-5", "OpenAI"),
            ("aura-gpt-5-6-sol", "OpenAI"),
            ("aura-gpt-5-6-terra", "OpenAI"),
            ("aura-gpt-5-6-luna", "OpenAI"),
            ("aura-oss-120b", "OpenAI"),
            ("aura-grok-4-5", "xAI"),
            ("aura-grok-4-3", "xAI"),
            ("aura-grok-build-0-1", "xAI"),
            ("aura-gemini-3-1-pro", "Google"),
            ("aura-deepseek-v4-pro", "DeepSeek AI"),
            ("deepseek/deepseek-v4-flash", "DeepSeek AI"),
            ("aura-kimi-k3", "Moonshot AI"),
            ("aura-kimi-k2-6", "Moonshot AI"),
            ("aura-minimax-m3", "MiniMax"),
            ("aura-minimax-m2-7", "MiniMax"),
            ("aura-glm-5-1", "Z.ai"),
            ("aura-qwen3-6-plus", "Alibaba Cloud"),
        ];
        for (model, expected) in cases {
            assert_eq!(
                model_maker(model).map(|m| m.display_name()),
                Some(expected),
                "maker for {model}"
            );
        }
        assert_eq!(model_maker("totally-unknown-model"), None);
    }

    #[test]
    fn resolve_provider_understands_aura_aliases() {
        assert_eq!(resolve_provider("aura-gpt-5-5"), Some(Provider::OpenAi));
        assert_eq!(resolve_provider("aura-gpt-5-6-sol"), Some(Provider::OpenAi));
        assert_eq!(resolve_provider("aura-gpt-5-4"), Some(Provider::OpenAi));
        assert_eq!(resolve_provider("aura-grok-4-5"), Some(Provider::Xai));
        assert_eq!(resolve_provider("aura-grok-4-6"), Some(Provider::Xai));
        assert_eq!(resolve_provider("aura-grok-4-3"), Some(Provider::Xai));
        assert_eq!(resolve_provider("aura-grok-build-0-1"), Some(Provider::Xai));
        assert_eq!(resolve_provider("aura-kimi-k3"), Some(Provider::Moonshot));
        assert_eq!(
            resolve_provider("aura-kimi-k2-5"),
            Some(Provider::Fireworks)
        );
        assert_eq!(
            resolve_provider("aura-kimi-k2-6"),
            Some(Provider::Fireworks)
        );
        assert_eq!(
            resolve_provider("aura-deepseek-v4-flash"),
            Some(Provider::Fireworks)
        );
    }

    #[test]
    fn resolves_deepseek_v4_models_via_fireworks() {
        // Picker-facing DeepSeek aliases route through Fireworks (which hosts
        // the V4 models verbatim) and reuse FIREWORKS_API_KEY.
        let resolved = resolve_model("aura-deepseek-v4-pro").expect("aura alias");
        assert_eq!(
            resolved.upstream_model,
            "accounts/fireworks/models/deepseek-v4-pro"
        );
        assert_eq!(resolved.provider, Provider::Fireworks);

        let resolved = resolve_model("deepseek/deepseek-v4-flash").expect("provider alias");
        assert_eq!(
            resolved.upstream_model,
            "accounts/fireworks/models/deepseek-v4-flash"
        );
        assert_eq!(resolved.provider, Provider::Fireworks);

        // Raw first-party names still resolve to the direct DeepSeek API.
        let resolved = resolve_model("deepseek-chat").expect("compat alias");
        assert_eq!(resolved.upstream_model, "deepseek-chat");
        assert_eq!(resolved.provider, Provider::DeepSeek);
    }

    #[test]
    fn resolves_kimi_k2_6_to_fireworks() {
        let resolved = resolve_model("aura-kimi-k2-6").expect("aura alias should resolve");
        assert_eq!(
            resolved.upstream_model,
            "accounts/fireworks/models/kimi-k2p6"
        );
        assert_eq!(resolved.provider, Provider::Fireworks);
    }

    #[test]
    fn resolves_kimi_k3_directly_to_moonshot() {
        for model in ["aura-kimi-k3", "moonshot/kimi-k3", "kimi-k3"] {
            let resolved = resolve_model(model).expect("Kimi K3 should resolve");
            assert_eq!(resolved.upstream_model, "kimi-k3");
            assert_eq!(resolved.provider, Provider::Moonshot);
            assert_eq!(super::max_context_tokens(model), 1_048_576);
        }
        assert_eq!(
            super::provider_url(&Provider::Moonshot),
            "https://api.moonshot.ai/v1/chat/completions"
        );
        let headers =
            provider_headers(&Provider::Moonshot, "moonshot-test-key").expect("valid headers");
        assert_eq!(
            headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer moonshot-test-key")
        );
    }

    #[test]
    fn resolves_new_fireworks_models() {
        for (alias, upstream, context) in [
            (
                "aura-minimax-m3",
                "accounts/fireworks/models/minimax-m3",
                512_000,
            ),
            (
                "aura-minimax-m2-7",
                "accounts/fireworks/models/minimax-m2p7",
                196_608,
            ),
            ("aura-glm-5-1", "accounts/fireworks/models/glm-5p1", 202_752),
            (
                "aura-glm-5-2",
                "accounts/fireworks/models/glm-5p2",
                1_048_576,
            ),
            (
                "aura-qwen3-6-plus",
                "accounts/fireworks/models/qwen3p6-plus",
                262_144,
            ),
            (
                "aura-qwen3-7-plus",
                "accounts/fireworks/models/qwen3p7-plus",
                262_144,
            ),
            (
                "aura-kimi-k2-7-code",
                "accounts/fireworks/models/kimi-k2p7-code",
                262_144,
            ),
        ] {
            let resolved = resolve_model(alias).expect("aura alias should resolve");
            assert_eq!(resolved.upstream_model, upstream);
            assert_eq!(resolved.provider, Provider::Fireworks);
            assert_eq!(super::max_context_tokens(alias), context);
        }
        assert_eq!(super::max_context_tokens("aura-kimi-k2-7-code"), 262_144);
    }

    #[test]
    fn resolves_gpt_5_5_api_model() {
        let resolved = resolve_model("gpt-5.5").expect("api model should resolve");
        assert_eq!(resolved.requested_model, "gpt-5.5");
        assert_eq!(resolved.upstream_model, "gpt-5.5");
        assert_eq!(resolved.provider, Provider::OpenAi);
    }

    #[test]
    fn resolves_grok_aliases_to_xai() {
        for (alias, upstream, context) in [
            ("aura-grok-4-6", "grok-4.6", 500_000),
            ("xai/grok-4.6", "grok-4.6", 500_000),
            ("aura-grok-4-5", "grok-4.5", 500_000),
            ("xai/grok-4.5", "grok-4.5", 500_000),
            ("aura-grok-4-3", "grok-4.3", 1_000_000),
            ("xai/grok-4.3", "grok-4.3", 1_000_000),
            ("aura-grok-build-0-1", "grok-build-0.1", 256_000),
            ("xai/grok-build-0.1", "grok-build-0.1", 256_000),
            ("xai/grok-code-fast-1", "grok-build-0.1", 256_000),
        ] {
            let resolved = resolve_model(alias).expect("grok alias should resolve");
            assert_eq!(resolved.upstream_model, upstream);
            assert_eq!(resolved.provider, Provider::Xai);
            assert_eq!(super::max_context_tokens(alias), context);
        }

        let raw = resolve_model("grok-4.3").expect("raw grok model should resolve");
        assert_eq!(raw.upstream_model, "grok-4.3");
        assert_eq!(raw.provider, Provider::Xai);
    }

    #[test]
    fn resolves_gemini_aliases_to_google() {
        for (alias, upstream) in [
            ("aura-gemini-3-1-pro", "gemini-3.1-pro-preview"),
            ("aura-gemini-3-5-flash", "gemini-3.5-flash"),
            ("aura-gemini-3-flash", "gemini-3-flash-preview"),
            ("aura-gemini-3-1-flash-lite", "gemini-3.1-flash-lite"),
            ("aura-gemini-2-5-pro", "gemini-2.5-pro"),
            ("aura-gemini-2-5-flash", "gemini-2.5-flash"),
            ("aura-gemini-2-5-flash-lite", "gemini-2.5-flash-lite"),
        ] {
            let resolved = resolve_model(alias).expect("aura gemini alias should resolve");
            assert_eq!(resolved.upstream_model, upstream);
            assert_eq!(resolved.provider, Provider::Google);
        }
    }

    #[test]
    fn infers_google_provider_for_raw_gemini_names() {
        let resolved = resolve_model("gemini-2.5-flash").expect("api model should resolve");
        assert_eq!(resolved.upstream_model, "gemini-2.5-flash");
        assert_eq!(resolved.provider, Provider::Google);
        assert_eq!(super::max_context_tokens("aura-gemini-2-5-pro"), 1_000_000);
    }

    #[test]
    fn google_endpoint_url_encodes_model_and_streaming_mode() {
        assert_eq!(
            super::google_endpoint_url("gemini-2.5-pro", false),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        );
        assert_eq!(
            super::google_endpoint_url("gemini-2.5-pro", true),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
    }
}
