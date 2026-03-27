//! aura-storage event recording client.
//!
//! Stores LLM prompts and responses as session events to aura-storage.
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

/// Store a user prompt and assistant response as session events (fire-and-forget).
///
/// Calls POST /internal/events for each event.
/// Errors are logged but do not block the response.
pub async fn store_events(
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
    let url = format!("{storage_url}/internal/events");

    // Store user prompt as event
    let user_result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "sessionId": ctx.session_id,
            "userId": user_id,
            "agentId": ctx.project_agent_id,
            "sender": "user",
            "projectId": ctx.project_id,
            "orgId": ctx.org_id,
            "type": "message_saved",
            "content": {
                "role": "user",
                "text": user_content
            }
        }))
        .send()
        .await;

    match user_result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("User event stored to aura-storage");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "Failed to store user event");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-storage for user event");
        }
    }

    // Store assistant response as event
    let mut content = serde_json::json!({
        "role": "assistant",
        "text": assistant_content,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens
    });

    if let Some(thinking_text) = thinking {
        content["thinking"] = serde_json::Value::String(thinking_text.to_string());
    }

    let assistant_result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "sessionId": ctx.session_id,
            "agentId": ctx.project_agent_id,
            "sender": "agent",
            "projectId": ctx.project_id,
            "orgId": ctx.org_id,
            "type": "message_saved",
            "content": content
        }))
        .send()
        .await;

    match assistant_result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("Assistant event stored to aura-storage");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "Failed to store assistant event");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-storage for assistant event");
        }
    }
}

/// Store a generated artifact in aura-storage (fire-and-forget).
///
/// Calls POST /internal/artifacts.
/// Errors are logged but do not block the response.
pub async fn store_artifact(
    client: &reqwest::Client,
    storage_url: &str,
    token: &str,
    project_id: &str,
    created_by: &str,
    artifact_type: &str,
    asset_url: &str,
    original_url: Option<&str>,
    name: Option<&str>,
    prompt: Option<&str>,
    prompt_mode: Option<&str>,
    model: &str,
    provider: &str,
    is_iteration: bool,
    parent_id: Option<&str>,
) {
    let url = format!("{storage_url}/internal/artifacts");

    let result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "projectId": project_id,
            "createdBy": created_by,
            "type": artifact_type,
            "assetUrl": asset_url,
            "originalUrl": original_url,
            "name": name,
            "prompt": prompt,
            "promptMode": prompt_mode,
            "model": model,
            "provider": provider,
            "isIteration": is_iteration,
            "parentId": parent_id,
        }))
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("Artifact stored to aura-storage");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "Failed to store artifact");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-storage for artifact");
        }
    }
}
