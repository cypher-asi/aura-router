//! aura-storage message recording client.
//!
//! Stores LLM prompts and responses to aura-storage for conversation history.
//! Requires session context headers from the client (X-Aura-Session-Id, etc.).

/// Context headers from the client request, used for storage recording.
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub session_id: String,
    pub project_agent_id: String,
    pub project_id: String,
    pub org_id: Option<String>,
}

impl SessionContext {
    /// Extract session context from request headers.
    /// Returns None if required headers are missing.
    pub fn from_headers(headers: &axum::http::HeaderMap) -> Option<Self> {
        let session_id = headers
            .get("x-aura-session-id")
            .and_then(|v| v.to_str().ok())?
            .to_string();
        let project_agent_id = headers
            .get("x-aura-agent-id")
            .and_then(|v| v.to_str().ok())?
            .to_string();
        let project_id = headers
            .get("x-aura-project-id")
            .and_then(|v| v.to_str().ok())?
            .to_string();
        let org_id = headers
            .get("x-aura-org-id")
            .and_then(|v| v.to_str().ok())
            .map(String::from);

        Some(Self {
            session_id,
            project_agent_id,
            project_id,
            org_id,
        })
    }
}

/// Store a user prompt and assistant response to aura-storage (fire-and-forget).
///
/// Calls POST /internal/messages for each message.
/// Errors are logged but do not block the response.
pub async fn store_messages(
    client: &reqwest::Client,
    storage_url: &str,
    token: &str,
    ctx: &SessionContext,
    user_id: &str,
    user_content: &str,
    assistant_content: &str,
    thinking: Option<&str>,
    input_tokens: u64,
    output_tokens: u64,
) {
    let url = format!("{storage_url}/internal/messages");

    // Store user prompt
    let user_result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "sessionId": ctx.session_id,
            "projectAgentId": ctx.project_agent_id,
            "projectId": ctx.project_id,
            "orgId": ctx.org_id,
            "createdBy": user_id,
            "role": "user",
            "content": user_content,
            "inputTokens": null,
            "outputTokens": null
        }))
        .send()
        .await;

    match user_result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("User message stored to aura-storage");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "Failed to store user message");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-storage for user message");
        }
    }

    // Store assistant response
    let assistant_result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "sessionId": ctx.session_id,
            "projectAgentId": ctx.project_agent_id,
            "projectId": ctx.project_id,
            "orgId": ctx.org_id,
            "createdBy": null,
            "role": "assistant",
            "content": assistant_content,
            "thinking": thinking,
            "inputTokens": input_tokens as i32,
            "outputTokens": output_tokens as i32
        }))
        .send()
        .await;

    match assistant_result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("Assistant message stored to aura-storage");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "Failed to store assistant message");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-storage for assistant message");
        }
    }
}
