//! z-billing client for credit checks and usage reporting.

use serde::{Deserialize, Serialize};

use aura_router_core::AppError;

/// Response from POST /v1/usage/check.
#[derive(Debug, Deserialize)]
pub struct CheckBalanceResponse {
    pub sufficient: bool,
    pub balance_cents: i64,
    pub required_cents: i64,
}

/// Response from POST /v1/usage.
#[derive(Debug, Deserialize)]
pub struct UsageResponse {
    pub success: bool,
    pub balance_cents: i64,
    pub cost_cents: i64,
    pub transaction_id: String,
}

/// Usage metric for LLM tokens.
#[derive(Debug, Serialize)]
pub struct LlmMetric {
    #[serde(rename = "type")]
    pub metric_type: String,
    pub provider: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

const DEFAULT_LLM_MARKUP_MULTIPLIER: f64 = 1.20;
const DEFAULT_LLM_MARKUP_BPS: u32 = 2_000;
const USD_TICKS_PER_CENT: f64 = 100_000_000.0;
const MICRO_USD_PER_CENT: f64 = 10_000.0;
const ANTHROPIC_WEB_SEARCH_CENTS_PER_REQUEST: f64 = 1.0;
const OPENAI_LONG_CONTEXT_THRESHOLD: u64 = 272_000;

/// Versioned source identifier emitted with provider-cost estimates.
pub const PRICING_SOURCE: &str = "aura-router-static-2026-07-22";

/// Detailed provider usage used for cost estimation and billing overrides.
#[derive(Debug, Clone, Default)]
pub struct UsageCostInput {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_creation_5m_input_tokens: u64,
    pub cache_creation_1h_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub web_search_requests: u64,
    pub web_fetch_requests: u64,
    pub code_execution_requests: u64,
    pub service_tier: Option<String>,
    pub inference_geo: Option<String>,
    pub speed: Option<String>,
}

/// Estimated provider cost before AURA's markup, plus the matching debit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageCostEstimate {
    /// Estimated provider cost in micro-USD ($0.000001).
    pub provider_cost_microusd: i64,
    /// User debit after applying AURA's standard markup and cent rounding.
    pub billed_cost_cents: i64,
    pub markup_bps: u32,
}

/// Convert provider-reported USD ticks into marked-up billing cents.
///
/// xAI reports exact request cost in `usage.cost_in_usd_ticks` where
/// 10,000,000,000 ticks = $1. Convert that to cents and apply the same LLM
/// markup used by model-table estimates.
pub fn marked_up_cost_cents_from_usd_ticks(ticks: u64) -> Option<i64> {
    if ticks == 0 {
        return None;
    }
    let cents = ticks as f64 / USD_TICKS_PER_CENT * DEFAULT_LLM_MARKUP_MULTIPLIER;
    let rounded = cents.round() as i64;
    Some(rounded.max(1))
}

/// Convert provider-reported USD ticks into micro-USD without AURA markup.
pub fn provider_cost_microusd_from_usd_ticks(ticks: u64) -> Option<i64> {
    if ticks == 0 {
        return None;
    }
    let micro_usd = ticks as f64 / (USD_TICKS_PER_CENT / MICRO_USD_PER_CENT);
    Some(micro_usd.round() as i64)
}

#[derive(Debug, Clone, Copy)]
struct CacheAwareRates {
    /// Rate for uncached input tokens (the prompt portion that is neither
    /// served from cache nor written to cache).
    new_input_cents_per_million: f64,
    /// Rate for tokens being written to cache. For providers that bill
    /// cache writes at the same rate as base input (OpenAI before GPT-5.6,
    /// DeepSeek, Fireworks), this equals `new_input_cents_per_million`.
    /// GPT-5.6 and Anthropic 5m ephemeral writes cost 1.25x base input.
    cache_write_input_cents_per_million: f64,
    /// Rate for tokens served from cache (always cheaper than base input).
    cache_read_input_cents_per_million: f64,
    output_cents_per_million: f64,
    /// Whether the provider's `input_tokens` field reports only the new
    /// (uncached) input portion (Anthropic), or the total request input
    /// including tokens served from cache (OpenAI/DeepSeek/Fireworks).
    input_tokens_is_new_only: bool,
}

/// Calculate a cache-aware cost override for providers whose raw usage has
/// more detail than z-billing's generic input/output metric.
pub fn cache_aware_cost_cents(
    provider: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
) -> Option<i64> {
    if cache_creation_input_tokens == 0 && cache_read_input_tokens == 0 {
        return None;
    }
    estimate_usage_cost(
        provider,
        model,
        &UsageCostInput {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            ..UsageCostInput::default()
        },
    )
    .map(|estimate| estimate.billed_cost_cents)
}

/// Estimate provider list-price cost with micro-USD precision.
///
/// Anthropic's response includes details that the generic z-billing token
/// metric cannot represent: 5-minute vs 1-hour cache writes, cache reads,
/// server-side web searches, service tier, and inference geography. Keeping
/// this estimate separate from the rounded user debit makes margin analysis
/// meaningful even for sub-cent requests.
pub fn estimate_usage_cost(
    provider: &str,
    model: &str,
    usage: &UsageCostInput,
) -> Option<UsageCostEstimate> {
    estimate_usage_cost_on(provider, model, usage, chrono::Utc::now().date_naive())
}

fn estimate_usage_cost_on(
    provider: &str,
    model: &str,
    usage: &UsageCostInput,
    date: chrono::NaiveDate,
) -> Option<UsageCostEstimate> {
    let rates = cache_aware_rates_on(provider, model, date)?;

    let cache_creation_split = usage
        .cache_creation_5m_input_tokens
        .saturating_add(usage.cache_creation_1h_input_tokens);
    let legacy_5m_tokens = usage
        .cache_creation_input_tokens
        .saturating_sub(cache_creation_split);
    let cache_creation_5m_tokens = usage
        .cache_creation_5m_input_tokens
        .saturating_add(legacy_5m_tokens);
    let cache_creation_tokens =
        cache_creation_5m_tokens.saturating_add(usage.cache_creation_1h_input_tokens);

    let new_input_tokens = if rates.input_tokens_is_new_only {
        usage.input_tokens
    } else {
        usage
            .input_tokens
            .saturating_sub(cache_creation_tokens.saturating_add(usage.cache_read_input_tokens))
    };
    let normalized_model = model.strip_prefix("openai/").unwrap_or(model);
    let is_long_context_model = matches!(
        normalized_model,
        "aura-gpt-5-4" | "gpt-5.4" | "aura-gpt-5-5" | "gpt-5.5"
    ) || normalized_model.starts_with("aura-gpt-5-6-")
        || normalized_model.starts_with("gpt-5.6");
    let is_long_context_openai = provider == "openai"
        && usage.input_tokens > OPENAI_LONG_CONTEXT_THRESHOLD
        && is_long_context_model;
    let input_multiplier = if is_long_context_openai { 2.0 } else { 1.0 };
    let output_multiplier = if is_long_context_openai { 1.5 } else { 1.0 };
    let speed_multiplier = anthropic_fast_mode_multiplier(provider, model, usage.speed.as_deref());

    let geo_multiplier = if provider == "anthropic"
        && usage.inference_geo.as_deref() == Some("us")
        && anthropic_supports_us_geo_pricing(model)
    {
        1.10
    } else {
        1.0
    };
    let cache_write_1h_rate = if provider == "anthropic" {
        rates.new_input_cents_per_million * 2.0
    } else {
        rates.cache_write_input_cents_per_million
    };

    let token_cents = ((new_input_tokens as f64
        * rates.new_input_cents_per_million
        * input_multiplier
        * speed_multiplier)
        + (cache_creation_5m_tokens as f64
            * rates.cache_write_input_cents_per_million
            * input_multiplier
            * speed_multiplier)
        + (usage.cache_creation_1h_input_tokens as f64
            * cache_write_1h_rate
            * input_multiplier
            * speed_multiplier)
        + (usage.cache_read_input_tokens as f64
            * rates.cache_read_input_cents_per_million
            * input_multiplier
            * speed_multiplier)
        + (usage.output_tokens as f64
            * rates.output_cents_per_million
            * output_multiplier
            * speed_multiplier))
        * geo_multiplier
        / 1_000_000.0;
    let server_tool_cents = if provider == "anthropic" {
        usage.web_search_requests as f64 * ANTHROPIC_WEB_SEARCH_CENTS_PER_REQUEST
    } else {
        0.0
    };
    let provider_cents = token_cents + server_tool_cents;
    let provider_cost_microusd = (provider_cents * MICRO_USD_PER_CENT).round() as i64;
    let rounded = (provider_cents * DEFAULT_LLM_MARKUP_MULTIPLIER).round() as i64;
    let has_billable_usage = usage.input_tokens > 0
        || usage.output_tokens > 0
        || cache_creation_tokens > 0
        || usage.cache_read_input_tokens > 0
        || usage.web_search_requests > 0;
    let billed_cost_cents = if rounded == 0 && has_billable_usage {
        1
    } else {
        rounded
    };

    Some(UsageCostEstimate {
        provider_cost_microusd,
        billed_cost_cents,
        markup_bps: DEFAULT_LLM_MARKUP_BPS,
    })
}

fn anthropic_fast_mode_multiplier(provider: &str, model: &str, speed: Option<&str>) -> f64 {
    if provider != "anthropic" || speed != Some("fast") {
        return 1.0;
    }
    let model = model.strip_prefix("anthropic/").unwrap_or(model);
    match model {
        "claude-opus-4-8" | "aura-claude-opus-4-8" => 2.0,
        "claude-opus-4-7" | "aura-claude-opus-4-7" => 6.0,
        _ => 1.0,
    }
}

fn anthropic_supports_us_geo_pricing(model: &str) -> bool {
    let model = model.strip_prefix("anthropic/").unwrap_or(model);
    !matches!(
        model,
        "claude-haiku-4-5" | "claude-haiku-4-5-20251001" | "aura-claude-haiku-4-5"
    )
}

fn cache_aware_rates_on(
    provider: &str,
    model: &str,
    date: chrono::NaiveDate,
) -> Option<CacheAwareRates> {
    match provider {
        "anthropic" => anthropic_rates_on(model, date),
        "deepseek" => deepseek_rates(model),
        "openai" => openai_rates(model),
        "xai" => xai_rates(model),
        "fireworks" => fireworks_rates(model),
        "google" => google_rates(model),
        _ => None,
    }
}

/// Per-model rates for xAI Grok chat models (cents per million tokens).
///
/// xAI publishes a discounted cached-input rate for Grok. It does not
/// publish a separate cache-write rate for these models, so cache writes use
/// the base input rate when reported.
fn xai_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model
        .strip_prefix("xai/")
        .or_else(|| model.strip_prefix("grok/"))
        .unwrap_or(model);
    match model {
        "aura-grok-4-5" | "grok-4.5" => Some(CacheAwareRates {
            new_input_cents_per_million: 200.0,
            cache_write_input_cents_per_million: 200.0,
            cache_read_input_cents_per_million: 50.0,
            output_cents_per_million: 600.0,
            input_tokens_is_new_only: false,
        }),
        "aura-grok-4-3" | "grok-4.3" => Some(CacheAwareRates {
            new_input_cents_per_million: 125.0,
            cache_write_input_cents_per_million: 125.0,
            cache_read_input_cents_per_million: 20.0,
            output_cents_per_million: 250.0,
            input_tokens_is_new_only: false,
        }),
        "aura-grok-build-0-1"
        | "grok-build-0.1"
        | "grok-code-fast"
        | "grok-code-fast-1"
        | "grok-code-fast-1-0825" => Some(CacheAwareRates {
            new_input_cents_per_million: 100.0,
            cache_write_input_cents_per_million: 100.0,
            cache_read_input_cents_per_million: 20.0,
            output_cents_per_million: 200.0,
            input_tokens_is_new_only: false,
        }),
        _ => None,
    }
}

/// Per-model rates for Google Gemini chat models (cents per million tokens),
/// matching both the `aura-gemini-*` aliases and the raw upstream model
/// names the streaming path may report. Gemini bills `promptTokenCount` as
/// the full prompt (cached tokens included), so `input_tokens_is_new_only`
/// is false; cached input is ~10% of the base input rate. Pro tiers use the
/// flat (<=200k prompt) rate.
fn google_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("google/").unwrap_or(model);
    match model {
        "aura-gemini-3-1-pro" | "gemini-3.1-pro-preview" => Some(CacheAwareRates {
            new_input_cents_per_million: 200.0,
            cache_write_input_cents_per_million: 200.0,
            cache_read_input_cents_per_million: 20.0,
            output_cents_per_million: 1200.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gemini-3-5-flash" | "gemini-3.5-flash" => Some(CacheAwareRates {
            new_input_cents_per_million: 150.0,
            cache_write_input_cents_per_million: 150.0,
            cache_read_input_cents_per_million: 15.0,
            output_cents_per_million: 900.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gemini-3-flash" | "gemini-3-flash-preview" => Some(CacheAwareRates {
            new_input_cents_per_million: 50.0,
            cache_write_input_cents_per_million: 50.0,
            cache_read_input_cents_per_million: 5.0,
            output_cents_per_million: 300.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gemini-3-1-flash-lite" | "gemini-3.1-flash-lite" => Some(CacheAwareRates {
            new_input_cents_per_million: 25.0,
            cache_write_input_cents_per_million: 25.0,
            cache_read_input_cents_per_million: 2.5,
            output_cents_per_million: 150.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gemini-2-5-pro" | "gemini-2.5-pro" => Some(CacheAwareRates {
            new_input_cents_per_million: 125.0,
            cache_write_input_cents_per_million: 125.0,
            cache_read_input_cents_per_million: 12.5,
            output_cents_per_million: 1000.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gemini-2-5-flash" | "gemini-2.5-flash" => Some(CacheAwareRates {
            new_input_cents_per_million: 30.0,
            cache_write_input_cents_per_million: 30.0,
            cache_read_input_cents_per_million: 3.0,
            output_cents_per_million: 250.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gemini-2-5-flash-lite" | "gemini-2.5-flash-lite" => Some(CacheAwareRates {
            new_input_cents_per_million: 10.0,
            cache_write_input_cents_per_million: 10.0,
            cache_read_input_cents_per_million: 1.0,
            output_cents_per_million: 40.0,
            input_tokens_is_new_only: false,
        }),
        _ => None,
    }
}

fn anthropic_rates_on(model: &str, date: chrono::NaiveDate) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("anthropic/").unwrap_or(model);

    // Anthropic returns `input_tokens` as the new (uncached) portion only;
    // cache creation and cache read tokens are reported separately.
    // 5-minute ephemeral cache: write = 1.25× base input, read = 0.10× base input.
    match model {
        "claude-fable-5" | "aura-claude-fable-5" => Some(CacheAwareRates {
            new_input_cents_per_million: 1000.0,
            cache_write_input_cents_per_million: 1250.0,
            cache_read_input_cents_per_million: 100.0,
            output_cents_per_million: 5000.0,
            input_tokens_is_new_only: true,
        }),
        "claude-opus-4-6"
        | "claude-opus-4-7"
        | "claude-opus-4-8"
        | "aura-claude-opus-4-6"
        | "aura-claude-opus-4-7"
        | "aura-claude-opus-4-8" => Some(CacheAwareRates {
            new_input_cents_per_million: 500.0,
            cache_write_input_cents_per_million: 625.0,
            cache_read_input_cents_per_million: 50.0,
            output_cents_per_million: 2500.0,
            input_tokens_is_new_only: true,
        }),
        "claude-sonnet-4-6" | "aura-claude-sonnet-4-6" => Some(CacheAwareRates {
            new_input_cents_per_million: 300.0,
            cache_write_input_cents_per_million: 375.0,
            cache_read_input_cents_per_million: 30.0,
            output_cents_per_million: 1500.0,
            input_tokens_is_new_only: true,
        }),
        "claude-sonnet-5" | "aura-claude-sonnet-5" => {
            let promotional =
                date <= chrono::NaiveDate::from_ymd_opt(2026, 8, 31).expect("valid pricing date");
            let base_input = if promotional { 200.0 } else { 300.0 };
            let output = if promotional { 1000.0 } else { 1500.0 };
            Some(CacheAwareRates {
                new_input_cents_per_million: base_input,
                cache_write_input_cents_per_million: base_input * 1.25,
                cache_read_input_cents_per_million: base_input * 0.10,
                output_cents_per_million: output,
                input_tokens_is_new_only: true,
            })
        }
        "claude-haiku-4-5" | "claude-haiku-4-5-20251001" | "aura-claude-haiku-4-5" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 100.0,
                cache_write_input_cents_per_million: 125.0,
                cache_read_input_cents_per_million: 10.0,
                output_cents_per_million: 500.0,
                input_tokens_is_new_only: true,
            })
        }
        _ => None,
    }
}

fn deepseek_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("deepseek/").unwrap_or(model);

    match model {
        "aura-deepseek-v4-pro" | "deepseek-v4-pro" => Some(CacheAwareRates {
            new_input_cents_per_million: 174.0,
            cache_write_input_cents_per_million: 174.0,
            cache_read_input_cents_per_million: 14.5,
            output_cents_per_million: 348.0,
            input_tokens_is_new_only: false,
        }),
        "aura-deepseek-v4-flash" | "deepseek-v4-flash" | "deepseek-chat" | "deepseek-reasoner" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 14.0,
                cache_write_input_cents_per_million: 14.0,
                cache_read_input_cents_per_million: 2.8,
                output_cents_per_million: 28.0,
                input_tokens_is_new_only: false,
            })
        }
        _ => None,
    }
}

fn openai_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("openai/").unwrap_or(model);

    match model {
        "aura-gpt-5-6-sol" | "gpt-5.6" | "gpt-5.6-sol" => Some(CacheAwareRates {
            new_input_cents_per_million: 500.0,
            cache_write_input_cents_per_million: 625.0,
            cache_read_input_cents_per_million: 50.0,
            output_cents_per_million: 3000.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-6-terra" | "gpt-5.6-terra" => Some(CacheAwareRates {
            new_input_cents_per_million: 250.0,
            cache_write_input_cents_per_million: 312.5,
            cache_read_input_cents_per_million: 25.0,
            output_cents_per_million: 1500.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-6-luna" | "gpt-5.6-luna" => Some(CacheAwareRates {
            new_input_cents_per_million: 100.0,
            cache_write_input_cents_per_million: 125.0,
            cache_read_input_cents_per_million: 10.0,
            output_cents_per_million: 600.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-5" | "gpt-5.5" => Some(CacheAwareRates {
            new_input_cents_per_million: 500.0,
            cache_write_input_cents_per_million: 500.0,
            cache_read_input_cents_per_million: 50.0,
            output_cents_per_million: 3000.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-4" | "gpt-5.4" => Some(CacheAwareRates {
            new_input_cents_per_million: 250.0,
            cache_write_input_cents_per_million: 250.0,
            cache_read_input_cents_per_million: 25.0,
            output_cents_per_million: 1500.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-4-mini" | "gpt-5.4-mini" => Some(CacheAwareRates {
            new_input_cents_per_million: 75.0,
            cache_write_input_cents_per_million: 75.0,
            cache_read_input_cents_per_million: 7.5,
            output_cents_per_million: 450.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-4-nano" | "gpt-5.4-nano" => Some(CacheAwareRates {
            new_input_cents_per_million: 20.0,
            cache_write_input_cents_per_million: 20.0,
            cache_read_input_cents_per_million: 2.0,
            output_cents_per_million: 125.0,
            input_tokens_is_new_only: false,
        }),
        _ => None,
    }
}

fn fireworks_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("fireworks/").unwrap_or(model);

    // Fireworks normalizes through the OpenAI-shaped path, so `input_tokens`
    // is the total prompt size. Fireworks discounts cache reads at 50% off
    // base input and does not charge a premium for cache writes.
    match model {
        "aura-kimi-k2-5" | "accounts/fireworks/models/kimi-k2p5" => Some(CacheAwareRates {
            new_input_cents_per_million: 60.0,
            cache_write_input_cents_per_million: 60.0,
            cache_read_input_cents_per_million: 30.0,
            output_cents_per_million: 300.0,
            input_tokens_is_new_only: false,
        }),
        "accounts/fireworks/models/kimi-k2p5-turbo"
        | "accounts/fireworks/routers/kimi-k2p5-turbo" => Some(CacheAwareRates {
            new_input_cents_per_million: 99.0,
            cache_write_input_cents_per_million: 99.0,
            cache_read_input_cents_per_million: 49.5,
            output_cents_per_million: 494.0,
            input_tokens_is_new_only: false,
        }),
        "aura-kimi-k2-6" | "accounts/fireworks/models/kimi-k2p6" => Some(CacheAwareRates {
            new_input_cents_per_million: 95.0,
            cache_write_input_cents_per_million: 95.0,
            cache_read_input_cents_per_million: 47.5,
            output_cents_per_million: 400.0,
            input_tokens_is_new_only: false,
        }),
        // Kimi K2.7 Code — Moonshot's coding variant. Input/output match
        // K2.6's published rates; the cache-hit rate is its own $0.19/1M.
        "aura-kimi-k2-7-code" | "accounts/fireworks/models/kimi-k2p7-code" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 95.0,
                cache_write_input_cents_per_million: 95.0,
                cache_read_input_cents_per_million: 19.0,
                output_cents_per_million: 400.0,
                input_tokens_is_new_only: false,
            })
        }
        "accounts/fireworks/models/kimi-k2p6-turbo"
        | "accounts/fireworks/routers/kimi-k2p6-turbo" => Some(CacheAwareRates {
            new_input_cents_per_million: 200.0,
            cache_write_input_cents_per_million: 200.0,
            cache_read_input_cents_per_million: 100.0,
            output_cents_per_million: 800.0,
            input_tokens_is_new_only: false,
        }),
        "accounts/fireworks/models/kimi-k2-thinking"
        | "accounts/fireworks/models/kimi-k2-instruct-0905" => Some(CacheAwareRates {
            new_input_cents_per_million: 60.0,
            cache_write_input_cents_per_million: 60.0,
            cache_read_input_cents_per_million: 30.0,
            output_cents_per_million: 250.0,
            input_tokens_is_new_only: false,
        }),
        "aura-oss-120b" | "accounts/fireworks/models/gpt-oss-120b" => Some(CacheAwareRates {
            new_input_cents_per_million: 15.0,
            cache_write_input_cents_per_million: 15.0,
            cache_read_input_cents_per_million: 7.5,
            output_cents_per_million: 60.0,
            input_tokens_is_new_only: false,
        }),
        // DeepSeek V4 models are served via Fireworks, so they bill under the
        // "fireworks" provider. Base rates match DeepSeek's published pricing;
        // the standard markup is applied by `cache_aware_cost_cents`.
        "aura-deepseek-v4-pro" | "accounts/fireworks/models/deepseek-v4-pro" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 174.0,
                cache_write_input_cents_per_million: 174.0,
                cache_read_input_cents_per_million: 14.5,
                output_cents_per_million: 348.0,
                input_tokens_is_new_only: false,
            })
        }
        "aura-deepseek-v4-flash" | "accounts/fireworks/models/deepseek-v4-flash" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 14.0,
                cache_write_input_cents_per_million: 14.0,
                cache_read_input_cents_per_million: 2.8,
                output_cents_per_million: 28.0,
                input_tokens_is_new_only: false,
            })
        }
        "aura-minimax-m3" | "accounts/fireworks/models/minimax-m3" => Some(CacheAwareRates {
            new_input_cents_per_million: 40.0,
            cache_write_input_cents_per_million: 40.0,
            cache_read_input_cents_per_million: 8.0,
            output_cents_per_million: 160.0,
            input_tokens_is_new_only: false,
        }),
        "aura-minimax-m2-7" | "accounts/fireworks/models/minimax-m2p7" => Some(CacheAwareRates {
            new_input_cents_per_million: 30.0,
            cache_write_input_cents_per_million: 30.0,
            cache_read_input_cents_per_million: 6.0,
            output_cents_per_million: 120.0,
            input_tokens_is_new_only: false,
        }),
        "aura-glm-5-1" | "accounts/fireworks/models/glm-5p1" => Some(CacheAwareRates {
            new_input_cents_per_million: 140.0,
            cache_write_input_cents_per_million: 140.0,
            cache_read_input_cents_per_million: 26.0,
            output_cents_per_million: 440.0,
            input_tokens_is_new_only: false,
        }),
        "aura-glm-5-2" | "accounts/fireworks/models/glm-5p2" => Some(CacheAwareRates {
            new_input_cents_per_million: 140.0,
            cache_write_input_cents_per_million: 140.0,
            cache_read_input_cents_per_million: 26.0,
            output_cents_per_million: 440.0,
            input_tokens_is_new_only: false,
        }),
        "aura-qwen3-6-plus" | "accounts/fireworks/models/qwen3p6-plus" => Some(CacheAwareRates {
            new_input_cents_per_million: 50.0,
            cache_write_input_cents_per_million: 50.0,
            cache_read_input_cents_per_million: 10.0,
            output_cents_per_million: 300.0,
            input_tokens_is_new_only: false,
        }),
        "aura-qwen3-7-plus" | "accounts/fireworks/models/qwen3p7-plus" => Some(CacheAwareRates {
            new_input_cents_per_million: 40.0,
            cache_write_input_cents_per_million: 40.0,
            cache_read_input_cents_per_million: 8.0,
            output_cents_per_million: 160.0,
            input_tokens_is_new_only: false,
        }),
        _ => None,
    }
}

/// Pre-check whether a user has sufficient credits.
pub async fn check_credits(
    client: &reqwest::Client,
    billing_url: &str,
    api_key: &str,
    user_id: &str,
    required_cents: i64,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<CheckBalanceResponse, AppError> {
    let url = format!("{billing_url}/v1/usage/check");
    let mut request_body = serde_json::json!({
        "user_id": user_id,
        "required_cents": required_cents
    });
    if let Some(provider) = provider {
        request_body["provider"] = serde_json::Value::String(provider.to_string());
    }
    if let Some(model) = model {
        request_body["model"] = serde_json::Value::String(model.to_string());
    }

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("x-service-name", "aura-router")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing unreachable: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError::BillingError(format!(
            "z-billing check failed ({status}): {body}"
        )));
    }

    resp.json::<CheckBalanceResponse>()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing response parse error: {e}")))
}

/// Report image generation usage and debit a flat cost.
pub async fn report_image_usage(
    client: &reqwest::Client,
    billing_url: &str,
    api_key: &str,
    event_id: &str,
    user_id: &str,
    provider: &str,
    model: &str,
    cost_cents: i64,
) -> Result<UsageResponse, AppError> {
    let url = format!("{billing_url}/v1/usage");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("x-service-name", "aura-router")
        .json(&serde_json::json!({
            "event_id": event_id,
            "user_id": user_id,
            "cost_cents": cost_cents,
            "metric": {
                "type": "llm_tokens",
                "provider": provider,
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0
            }
        }))
        .send()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing unreachable: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        tracing::warn!(status = %status, body = %body, "z-billing image usage report failed");
        return Err(AppError::BillingError(format!(
            "z-billing usage report failed ({status})"
        )));
    }

    resp.json::<UsageResponse>()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing response parse error: {e}")))
}

/// Report LLM usage and debit credits.
pub async fn report_usage(
    client: &reqwest::Client,
    billing_url: &str,
    api_key: &str,
    event_id: &str,
    user_id: &str,
    provider: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cost_cents: Option<i64>,
    metadata: Option<serde_json::Value>,
) -> Result<UsageResponse, AppError> {
    let url = format!("{billing_url}/v1/usage");
    let mut body = serde_json::json!({
        "event_id": event_id,
        "user_id": user_id,
        "metric": {
            "type": "llm_tokens",
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    });
    if let Some(cost_cents) = cost_cents {
        body["cost_cents"] = serde_json::Value::from(cost_cents);
    }
    if let Some(metadata) = metadata {
        body["metadata"] = metadata;
    }

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("x-service-name", "aura-router")
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing unreachable: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        tracing::warn!(status = %status, body = %body, "z-billing usage report failed");
        return Err(AppError::BillingError(format!(
            "z-billing usage report failed ({status})"
        )));
    }

    resp.json::<UsageResponse>()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing response parse error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::{
        cache_aware_cost_cents, estimate_usage_cost_on, marked_up_cost_cents_from_usd_ticks,
        provider_cost_microusd_from_usd_ticks, UsageCostInput,
    };

    #[test]
    fn deepseek_cache_aware_cost_uses_cache_buckets_and_markup() {
        assert_eq!(
            cache_aware_cost_cents(
                "deepseek",
                "aura-deepseek-v4-pro",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
            Some(226)
        );
        assert_eq!(
            cache_aware_cost_cents(
                "deepseek",
                "deepseek/deepseek-v4-flash",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
            Some(20)
        );
    }

    #[test]
    fn deepseek_cost_override_is_absent_without_cache_buckets() {
        assert_eq!(
            cache_aware_cost_cents(
                "deepseek",
                "aura-deepseek-v4-flash",
                1_000_000,
                500_000,
                0,
                0,
            ),
            None
        );
        // Unknown provider with cache buckets still returns None.
        assert_eq!(
            cache_aware_cost_cents("cohere", "command-r-plus", 1_000_000, 500_000, 0, 1_000_000,),
            None
        );
    }

    #[test]
    fn openai_cache_aware_cost_discounts_cached_tokens() {
        assert_eq!(
            cache_aware_cost_cents("openai", "aura-gpt-5-5", 1_000_000, 500_000, 0, 1_000_000,),
            Some(2820)
        );
        assert_eq!(
            cache_aware_cost_cents("openai", "gpt-5.4-mini", 1_000_000, 500_000, 0, 1_000_000,),
            Some(279)
        );
    }

    #[test]
    fn gpt_5_6_cache_writes_use_the_published_premium() {
        // Long-context Sol: 500k new @ $10/M + 200k writes @ $12.50/M +
        // 300k reads @ $1/M + 500k output @ $45/M, then 20% markup.
        assert_eq!(
            cache_aware_cost_cents(
                "openai",
                "aura-gpt-5-6-sol",
                1_000_000,
                500_000,
                200_000,
                300_000,
            ),
            Some(3636)
        );

        // Below the threshold, Luna cache writes remain 1.25x input.
        assert_eq!(
            cache_aware_cost_cents("openai", "gpt-5.6-luna", 200_000, 100_000, 100_000, 50_000,),
            Some(94)
        );
    }

    #[test]
    fn openai_cache_cost_steps_up_only_above_the_long_context_threshold() {
        assert_eq!(
            cache_aware_cost_cents("openai", "gpt-5.5", 272_000, 100_000, 0, 272_000),
            Some(376)
        );
        assert_eq!(
            cache_aware_cost_cents("openai", "gpt-5.5", 272_001, 100_000, 0, 272_001),
            Some(573)
        );
    }

    #[test]
    fn openai_cost_override_is_absent_without_cached_tokens_or_rates() {
        assert_eq!(
            cache_aware_cost_cents("openai", "aura-gpt-5-5", 1_000_000, 500_000, 0, 0),
            None
        );
        assert_eq!(
            cache_aware_cost_cents("openai", "gpt-4.1", 1_000_000, 500_000, 0, 1_000_000),
            None
        );
    }

    #[test]
    fn xai_grok_cache_aware_cost_discounts_cached_tokens() {
        // grok-4.5: new 0, cache_read 1M x 50, output 500k x 600.
        // (50_000_000 + 300_000_000) x 1.2 / 1M = 420
        assert_eq!(
            cache_aware_cost_cents("xai", "aura-grok-4-5", 1_000_000, 500_000, 0, 1_000_000),
            Some(420)
        );
        // grok-4.3: new 0, cache_read 1M x 20, output 500k x 250.
        // (20_000_000 + 125_000_000) x 1.2 / 1M = 174
        assert_eq!(
            cache_aware_cost_cents("xai", "aura-grok-4-3", 1_000_000, 500_000, 0, 1_000_000),
            Some(174)
        );
        // grok-build-0.1: new 0, cache_read 1M x 20, output 500k x 200.
        // (20_000_000 + 100_000_000) x 1.2 / 1M = 144
        assert_eq!(
            cache_aware_cost_cents(
                "xai",
                "aura-grok-build-0-1",
                1_000_000,
                500_000,
                0,
                1_000_000
            ),
            Some(144)
        );
    }

    #[test]
    fn provider_reported_usd_ticks_convert_to_marked_up_cents() {
        assert_eq!(marked_up_cost_cents_from_usd_ticks(250_000_000), Some(3));
        assert_eq!(
            provider_cost_microusd_from_usd_ticks(250_000_000),
            Some(25_000)
        );
        assert_eq!(marked_up_cost_cents_from_usd_ticks(1), Some(1));
        assert_eq!(marked_up_cost_cents_from_usd_ticks(0), None);
    }

    #[test]
    fn google_gemini_cache_aware_cost_discounts_cached_tokens() {
        // gemini-2.5-pro: new 0, cache_read 1M × 12.5, output 500k × 1000.
        // (12_500_000 + 500_000_000) × 1.2 / 1M = 615
        assert_eq!(
            cache_aware_cost_cents(
                "google",
                "aura-gemini-2-5-pro",
                1_000_000,
                500_000,
                0,
                1_000_000
            ),
            Some(615)
        );
        // Raw upstream name resolves to the same rate table.
        assert_eq!(
            cache_aware_cost_cents(
                "google",
                "gemini-2.5-flash",
                1_000_000,
                500_000,
                0,
                1_000_000
            ),
            cache_aware_cost_cents(
                "google",
                "aura-gemini-2-5-flash",
                1_000_000,
                500_000,
                0,
                1_000_000
            ),
        );
    }

    #[test]
    fn google_cost_override_is_absent_without_cached_tokens() {
        assert_eq!(
            cache_aware_cost_cents("google", "aura-gemini-2-5-pro", 1_000_000, 500_000, 0, 0),
            None
        );
    }

    #[test]
    fn anthropic_opus_charges_cache_read_tokens() {
        // Mirrors a realistic Claude Code follow-up turn: tiny new input,
        // most of the prompt served from cache, modest output.
        // new: 1 × 500 = 500
        // read: 50_000 × 50 = 2_500_000
        // output: 595 × 2500 = 1_487_500
        // total: 3_988_000 / 1M = 3.988 × 1.20 = 4.7856 → 5
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-opus-4-7", 1, 595, 0, 50_000),
            Some(5)
        );
    }

    #[test]
    fn anthropic_sonnet_charges_cache_creation_at_premium() {
        // Cache-write turn with no new input outside the cache block.
        // write: 50_000 × 375 = 18_750_000
        // output: 600 × 1500 = 900_000
        // total: 19_650_000 / 1M = 19.65 × 1.20 = 23.58 → 24
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-sonnet-4-6", 0, 600, 50_000, 0),
            Some(24)
        );
    }

    #[test]
    fn anthropic_estimate_includes_cache_ttl_tools_and_geography() {
        let estimate = estimate_usage_cost_on(
            "anthropic",
            "claude-sonnet-5",
            &UsageCostInput {
                input_tokens: 100_000,
                output_tokens: 10_000,
                cache_creation_input_tokens: 300_000,
                cache_creation_5m_input_tokens: 200_000,
                cache_creation_1h_input_tokens: 100_000,
                cache_read_input_tokens: 500_000,
                web_search_requests: 2,
                web_fetch_requests: 3,
                service_tier: Some("standard".to_string()),
                inference_geo: Some("us".to_string()),
                ..UsageCostInput::default()
            },
            chrono::NaiveDate::from_ymd_opt(2026, 7, 22).unwrap(),
        )
        .expect("Sonnet 5 pricing");

        // Promotional token cost is 130 cents, US-only inference adds 10%, and two web
        // searches add 2 cents. Web fetch has no separate request charge.
        assert_eq!(estimate.provider_cost_microusd, 1_450_000);
        assert_eq!(estimate.billed_cost_cents, 174);
        assert_eq!(estimate.markup_bps, 2_000);
    }

    #[test]
    fn sonnet_5_introductory_pricing_switches_after_august() {
        let usage = UsageCostInput {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..UsageCostInput::default()
        };
        let promotional = estimate_usage_cost_on(
            "anthropic",
            "claude-sonnet-5",
            &usage,
            chrono::NaiveDate::from_ymd_opt(2026, 8, 31).unwrap(),
        )
        .unwrap();
        let standard = estimate_usage_cost_on(
            "anthropic",
            "claude-sonnet-5",
            &usage,
            chrono::NaiveDate::from_ymd_opt(2026, 9, 1).unwrap(),
        )
        .unwrap();

        assert_eq!(promotional.provider_cost_microusd, 12_000_000);
        assert_eq!(standard.provider_cost_microusd, 18_000_000);
    }

    #[test]
    fn anthropic_fast_mode_uses_premium_opus_rates() {
        let usage = UsageCostInput {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            speed: Some("fast".to_string()),
            ..UsageCostInput::default()
        };
        let date = chrono::NaiveDate::from_ymd_opt(2026, 7, 22).unwrap();
        let opus_48 = estimate_usage_cost_on("anthropic", "claude-opus-4-8", &usage, date).unwrap();
        let opus_47 = estimate_usage_cost_on("anthropic", "claude-opus-4-7", &usage, date).unwrap();

        assert_eq!(opus_48.provider_cost_microusd, 60_000_000);
        assert_eq!(opus_47.provider_cost_microusd, 180_000_000);
    }

    #[test]
    fn anthropic_fable_charges_all_cache_buckets() {
        for model in ["claude-fable-5", "aura-claude-fable-5"] {
            assert_eq!(
                cache_aware_cost_cents(
                    "anthropic",
                    model,
                    1_000_000,
                    1_000_000,
                    1_000_000,
                    1_000_000,
                ),
                Some(8_820),
                "cache-aware cost mismatch for {model}"
            );
        }
    }

    #[test]
    fn anthropic_haiku_applies_minimum_credit_floor() {
        // Tiny call: rounded cost is < 1 cent but cache tokens are present,
        // so the 1-credit floor applies.
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-haiku-4-5-20251001", 10, 100, 0, 1_000),
            Some(1)
        );
    }

    #[test]
    fn anthropic_treats_input_tokens_as_new_only() {
        // For Anthropic, `input_tokens` already excludes cache_read tokens.
        // The function must NOT subtract cache_read from input_tokens.
        // new: 10 × 500 = 5_000
        // read: 100_000 × 50 = 5_000_000
        // total: 5_005_000 / 1M = 5.005 × 1.20 = 6.006 → 6
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-opus-4-7", 10, 0, 0, 100_000),
            Some(6)
        );
    }

    #[test]
    fn anthropic_cost_override_is_absent_without_cache_tokens() {
        // No cache fields → fall back to z-billing's token-only math.
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-opus-4-7", 100, 100, 0, 0),
            None
        );
    }

    #[test]
    fn anthropic_aura_aliases_resolve_to_same_rates() {
        // aura-* aliases must produce identical costs to canonical model names.
        let canonical = cache_aware_cost_cents("anthropic", "claude-opus-4-7", 1, 595, 0, 50_000);
        let alias = cache_aware_cost_cents("anthropic", "aura-claude-opus-4-7", 1, 595, 0, 50_000);
        assert_eq!(canonical, alias);
        assert!(canonical.is_some());
    }

    #[test]
    fn fireworks_kimi_discounts_cache_reads() {
        // new: (100_000 − 80_000) × 60 = 1_200_000
        // read: 80_000 × 30 = 2_400_000
        // output: 500 × 300 = 150_000
        // total: 3_750_000 / 1M = 3.75 × 1.20 = 4.5 → 5
        assert_eq!(
            cache_aware_cost_cents("fireworks", "aura-kimi-k2-5", 100_000, 500, 0, 80_000),
            Some(5)
        );
    }

    #[test]
    fn anthropic_mixed_cache_creation_and_read() {
        // Realistic: an agentic turn that both writes new content to cache
        // and reads existing cached content.
        // new: 5 × 500 = 2_500
        // write: 20_000 × 625 = 12_500_000
        // read: 80_000 × 50 = 4_000_000
        // output: 400 × 2500 = 1_000_000
        // total: 17_502_500 / 1M = 17.5025 × 1.20 = 21.003 → 21
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-opus-4-7", 5, 400, 20_000, 80_000),
            Some(21)
        );
    }

    #[test]
    fn anthropic_unknown_model_returns_none() {
        // Models not in the rates table fall through so z-billing's
        // per-model pricing applies (or default fallback).
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-3-5-sonnet", 100, 100, 0, 1_000),
            None
        );
    }

    #[test]
    fn fireworks_unknown_model_returns_none() {
        assert_eq!(
            cache_aware_cost_cents(
                "fireworks",
                "accounts/fireworks/models/qwen2p5-coder-7b",
                100,
                100,
                0,
                1_000,
            ),
            None
        );
    }

    #[test]
    fn openai_cache_creation_billed_at_base_input_rate() {
        // Locks in pre-existing behavior: for OpenAI, cache_creation
        // tokens are billed at the same rate as new input (no premium).
        // new: 0 (all input was creation)
        // write: 10_000 × 500 = 5_000_000
        // output: 500 × 3000 = 1_500_000
        // total: 6_500_000 / 1M = 6.5 × 1.20 = 7.8 → 8
        assert_eq!(
            cache_aware_cost_cents("openai", "aura-gpt-5-5", 10_000, 500, 10_000, 0),
            Some(8)
        );
    }

    #[test]
    fn deepseek_via_fireworks_matches_direct_deepseek_cost() {
        // DeepSeek now bills under the "fireworks" provider; the cost must be
        // identical to the prior "deepseek"-provider result so the routing
        // change does not alter what users pay.
        for model in ["aura-deepseek-v4-pro", "aura-deepseek-v4-flash"] {
            let via_fireworks =
                cache_aware_cost_cents("fireworks", model, 1_000_000, 500_000, 0, 1_000_000);
            let via_deepseek =
                cache_aware_cost_cents("deepseek", model, 1_000_000, 500_000, 0, 1_000_000);
            assert_eq!(via_fireworks, via_deepseek, "cost parity for {model}");
            assert!(via_fireworks.is_some(), "rate present for {model}");
        }

        // The Fireworks upstream id resolves to the same rate as the aura id.
        assert_eq!(
            cache_aware_cost_cents(
                "fireworks",
                "accounts/fireworks/models/deepseek-v4-pro",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
            cache_aware_cost_cents(
                "fireworks",
                "aura-deepseek-v4-pro",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
        );
    }

    #[test]
    fn fireworks_aura_aliases_resolve_to_same_rates() {
        let canonical = cache_aware_cost_cents(
            "fireworks",
            "accounts/fireworks/models/kimi-k2p5",
            100_000,
            500,
            0,
            80_000,
        );
        let alias = cache_aware_cost_cents("fireworks", "aura-kimi-k2-5", 100_000, 500, 0, 80_000);
        assert_eq!(canonical, alias);
        assert!(canonical.is_some());
    }

    #[test]
    fn new_fireworks_models_aura_aliases_resolve_to_same_rates() {
        for (alias, upstream) in [
            ("aura-minimax-m3", "accounts/fireworks/models/minimax-m3"),
            (
                "aura-minimax-m2-7",
                "accounts/fireworks/models/minimax-m2p7",
            ),
            ("aura-glm-5-1", "accounts/fireworks/models/glm-5p1"),
            ("aura-glm-5-2", "accounts/fireworks/models/glm-5p2"),
            (
                "aura-qwen3-6-plus",
                "accounts/fireworks/models/qwen3p6-plus",
            ),
            (
                "aura-qwen3-7-plus",
                "accounts/fireworks/models/qwen3p7-plus",
            ),
            (
                "aura-kimi-k2-7-code",
                "accounts/fireworks/models/kimi-k2p7-code",
            ),
        ] {
            let via_alias =
                cache_aware_cost_cents("fireworks", alias, 1_000_000, 500_000, 0, 200_000);
            let via_upstream =
                cache_aware_cost_cents("fireworks", upstream, 1_000_000, 500_000, 0, 200_000);
            assert_eq!(via_alias, via_upstream, "rate parity for {alias}");
            assert!(via_alias.is_some(), "rate present for {alias}");
        }
    }
}
