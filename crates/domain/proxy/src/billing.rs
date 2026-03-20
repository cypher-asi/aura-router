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

/// Pre-check whether a user has sufficient credits.
pub async fn check_credits(
    client: &reqwest::Client,
    billing_url: &str,
    api_key: &str,
    user_id: &str,
    required_cents: i64,
) -> Result<CheckBalanceResponse, AppError> {
    let url = format!("{billing_url}/v1/usage/check");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("x-service-name", "aura-router")
        .json(&serde_json::json!({
            "user_id": user_id,
            "required_cents": required_cents
        }))
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
) -> Result<UsageResponse, AppError> {
    let url = format!("{billing_url}/v1/usage");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("x-service-name", "aura-router")
        .json(&serde_json::json!({
            "event_id": event_id,
            "user_id": user_id,
            "metric": {
                "type": "llm_tokens",
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }))
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
