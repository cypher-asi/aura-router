//! aura-network usage recording client.

/// Record token usage to aura-network (fire-and-forget).
///
/// Calls POST /internal/usage with X-Internal-Token.
/// Errors are logged but do not block the response.
pub async fn record_usage(
    client: &reqwest::Client,
    network_url: &str,
    token: &str,
    user_id: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
) {
    let url = format!("{network_url}/internal/usage");

    let result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "orgId": null,
            "userId": user_id,
            "agentId": null,
            "model": model,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "estimatedCostUsd": cost_usd
        }))
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!(user_id = %user_id, model = %model, "Usage recorded to aura-network");
        }
        Ok(resp) => {
            tracing::warn!(
                status = %resp.status(),
                "Failed to record usage to aura-network"
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-network for usage recording");
        }
    }
}
