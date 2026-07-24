//! aura-network usage recording client.

fn optional_uuid(value: Option<&str>) -> Option<String> {
    value
        .and_then(|candidate| uuid::Uuid::parse_str(candidate).ok())
        .map(|id| id.to_string())
}

fn usage_payload(
    user_id: &str,
    org_id: Option<&str>,
    project_id: Option<&str>,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
    duration_ms: u64,
) -> serde_json::Value {
    // aura-network deserializes userId/orgId/projectId as UUIDs before its
    // handler can resolve zeroUserId. Auth0 subjects and Home/local project
    // labels are valid router context but are not UUIDs, so forwarding them
    // verbatim produces a 422 even though the provider request succeeded.
    //
    // A nil userId is only a schema-safe placeholder. aura-network replaces
    // it from zeroUserId before inserting usage. Optional attribution fields
    // are omitted when they are not network UUIDs.
    let network_user_id = uuid::Uuid::parse_str(user_id)
        .unwrap_or_else(|_| uuid::Uuid::nil())
        .to_string();
    let input_tokens = input_tokens.min(i64::MAX as u64) as i64;
    let output_tokens = output_tokens.min(i64::MAX as u64) as i64;
    let duration_ms = duration_ms.min(i64::MAX as u64) as i64;
    let cost_usd = if cost_usd.is_finite() { cost_usd } else { 0.0 };

    serde_json::json!({
        "orgId": optional_uuid(org_id),
        "userId": network_user_id,
        "zeroUserId": user_id,
        "agentId": null,
        "projectId": optional_uuid(project_id),
        "model": model,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "estimatedCostUsd": cost_usd,
        "durationMs": duration_ms
    })
}

/// Record token usage to aura-network (fire-and-forget).
///
/// Calls POST /internal/usage with X-Internal-Token. Any of `org_id`,
/// `project_id`, `agent_id` may be `None`; the receiver stores `null`
/// and aggregations scoped by those columns simply exclude the row.
///
/// IMPORTANT: `agent_id` is currently swallowed and NOT sent to
/// aura-network — passing it would trigger
/// `token_usage_daily_agent_id_fkey` FK violations. The header
/// `x-aura-agent-id` carries aura-code's `project_agents.id`, but
/// aura-network's FK references its own `agents` table — different
/// tables in different services. Until proper id translation lands,
/// we keep the legacy behaviour of sending `agentId: null` so the
/// row inserts cleanly. Per-agent attribution is a follow-up.
/// Errors are logged but do not block the response.
#[allow(clippy::too_many_arguments)]
pub async fn record_usage(
    client: &reqwest::Client,
    network_url: &str,
    token: &str,
    user_id: &str,
    org_id: Option<&str>,
    project_id: Option<&str>,
    _agent_id: Option<&str>,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
    duration_ms: u64,
) {
    let url = format!("{network_url}/internal/usage");

    let result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&usage_payload(
            user_id,
            org_id,
            project_id,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
            duration_ms,
        ))
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!(user_id = %user_id, model = %model, "Usage recorded to aura-network");
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|error| format!("<failed to read response body: {error}>"));
            let body_preview: String = body.chars().take(1_000).collect();
            tracing::warn!(
                status = %status,
                user_id = %user_id,
                model = %model,
                body = %body_preview,
                "Failed to record usage to aura-network"
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-network for usage recording");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::usage_payload;

    #[test]
    fn usage_payload_preserves_network_uuids() {
        let user_id = uuid::Uuid::new_v4();
        let org_id = uuid::Uuid::new_v4();
        let project_id = uuid::Uuid::new_v4();

        let payload = usage_payload(
            &user_id.to_string(),
            Some(&org_id.to_string()),
            Some(&project_id.to_string()),
            "aura-claude-haiku-4-5",
            12,
            7,
            0.01,
            250,
        );

        assert_eq!(payload["userId"], user_id.to_string());
        assert_eq!(payload["zeroUserId"], user_id.to_string());
        assert_eq!(payload["orgId"], org_id.to_string());
        assert_eq!(payload["projectId"], project_id.to_string());
        assert_eq!(payload["inputTokens"], 12);
        assert_eq!(payload["outputTokens"], 7);
        assert_eq!(payload["durationMs"], 250);
    }

    #[test]
    fn usage_payload_sanitizes_non_uuid_router_context() {
        let payload = usage_payload(
            "auth0|customer-725",
            Some("my-team"),
            Some("home"),
            "aura-grok-4-5",
            u64::MAX,
            u64::MAX,
            f64::NAN,
            u64::MAX,
        );

        assert_eq!(payload["userId"], uuid::Uuid::nil().to_string());
        assert_eq!(payload["zeroUserId"], "auth0|customer-725");
        assert!(payload["orgId"].is_null());
        assert!(payload["projectId"].is_null());
        assert_eq!(payload["inputTokens"], i64::MAX);
        assert_eq!(payload["outputTokens"], i64::MAX);
        assert_eq!(payload["estimatedCostUsd"], 0.0);
        assert_eq!(payload["durationMs"], i64::MAX);
    }
}
