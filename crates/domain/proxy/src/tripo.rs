//! Tripo 3D generation client.
//!
//! Submits image-to-3D tasks and polls for completion.
//! Always uses S3 URLs for input (base64 is unreliable with Tripo).

use serde::{Deserialize, Serialize};

const TRIPO_API_BASE: &str = "https://api.tripo3d.ai/v2/openapi";
const MODEL_VERSION: &str = "v2.0-20240919";
const POLL_INTERVAL_MS: u64 = 2000;
const MAX_POLL_ATTEMPTS: u32 = 60;
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Request to create a 3D generation task.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Generate3dRequest {
    /// Image URL (must be a publicly accessible URL, ideally S3).
    pub image_url: String,
    /// Optional text prompt for guidance.
    pub prompt: Option<String>,
}

/// Response from creating a 3D generation task.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Generate3dResponse {
    pub success: bool,
    pub task_id: String,
    pub eta_ms: u64,
}

/// Response from polling a 3D generation task.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatusResponse {
    pub status: String,
    pub task_id: String,
    pub glb_url: Option<String>,
    pub poly_count: Option<u64>,
    pub error: Option<String>,
}

/// Submit an image-to-3D task to Tripo.
pub async fn create_task(
    client: &reqwest::Client,
    api_key: &str,
    image_url: &str,
) -> Result<String, String> {
    let body = serde_json::json!({
        "type": "image_to_model",
        "file": {
            "type": "url",
            "url": image_url
        },
        "model_version": MODEL_VERSION
    });

    let resp = client
        .post(format!("{TRIPO_API_BASE}/task"))
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Tripo request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("Tripo error: {error}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    let task_id = data
        .pointer("/data/task_id")
        .and_then(|v| v.as_str())
        .ok_or("No task_id in Tripo response")?
        .to_string();

    Ok(task_id)
}

/// Poll a Tripo task until completion or failure.
/// Returns the GLB URL and metadata on success.
pub async fn poll_task(
    client: &reqwest::Client,
    api_key: &str,
    task_id: &str,
) -> Result<TaskStatusResponse, String> {
    for attempt in 0..MAX_POLL_ATTEMPTS {
        tokio::time::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS)).await;

        let resp = client
            .get(format!("{TRIPO_API_BASE}/task/{task_id}"))
            .header("authorization", format!("Bearer {api_key}"))
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .send()
            .await
            .map_err(|e| format!("Tripo poll failed: {e}"))?;

        if !resp.status().is_success() {
            let error = resp.text().await.unwrap_or_default();
            tracing::warn!(attempt, task_id, error = %error, "Tripo poll error, retrying");
            continue;
        }

        let data: serde_json::Value =
            resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

        let status = data
            .pointer("/data/status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match status {
            "success" => {
                let output = data.pointer("/data/output");
                let glb_url = output.and_then(|o| extract_glb_url(o));
                let poly_count = data
                    .pointer("/data/output/pbr_model/poly_count")
                    .and_then(|v| v.as_u64());

                return Ok(TaskStatusResponse {
                    status: "success".to_string(),
                    task_id: task_id.to_string(),
                    glb_url,
                    poly_count,
                    error: None,
                });
            }
            "failed" => {
                let error_code = data
                    .pointer("/data/error_code")
                    .and_then(|v| v.as_u64());
                let error_message = data
                    .pointer("/data/error_message")
                    .or_else(|| data.pointer("/data/message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");

                let user_message = match error_code {
                    Some(1001) => "Tripo validation failed. Requires real photographs with: 1) Clear single object, 2) Good lighting/contrast, 3) Simple background, 4) Min 200x200px. AI-generated images often fail.".to_string(),
                    Some(1002) => "Image processing failed — corrupted or unsupported format.".to_string(),
                    Some(1003) => "Content moderation failed — inappropriate content detected.".to_string(),
                    _ => error_message.to_string(),
                };

                return Ok(TaskStatusResponse {
                    status: "failed".to_string(),
                    task_id: task_id.to_string(),
                    glb_url: None,
                    poly_count: None,
                    error: Some(user_message),
                });
            }
            "processing" | "queued" => {
                tracing::debug!(attempt, task_id, status, "Tripo task still processing");
                continue;
            }
            _ => {
                tracing::warn!(attempt, task_id, status, "Unknown Tripo status");
                continue;
            }
        }
    }

    Err(format!(
        "Tripo task {task_id} timed out after {} seconds",
        MAX_POLL_ATTEMPTS as u64 * POLL_INTERVAL_MS / 1000
    ))
}

/// Check the status of a task without polling (single check).
pub async fn check_task_status(
    client: &reqwest::Client,
    api_key: &str,
    task_id: &str,
) -> Result<TaskStatusResponse, String> {
    let resp = client
        .get(format!("{TRIPO_API_BASE}/task/{task_id}"))
        .header("authorization", format!("Bearer {api_key}"))
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .send()
        .await
        .map_err(|e| format!("Tripo status check failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("Tripo error: {error}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    let status = data
        .pointer("/data/status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let glb_url = if status == "success" {
        data.pointer("/data/output").and_then(|o| extract_glb_url(o))
    } else {
        None
    };

    let poly_count = data
        .pointer("/data/output/pbr_model/poly_count")
        .and_then(|v| v.as_u64());

    let error = if status == "failed" {
        let error_code = data.pointer("/data/error_code").and_then(|v| v.as_u64());
        let error_message = data
            .pointer("/data/error_message")
            .or_else(|| data.pointer("/data/message"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");

        Some(match error_code {
            Some(1001) => "Tripo validation failed. Requires real photographs with: 1) Clear single object, 2) Good lighting/contrast, 3) Simple background, 4) Min 200x200px. AI-generated images often fail.".to_string(),
            Some(1002) => "Image processing failed — corrupted or unsupported format.".to_string(),
            Some(1003) => "Content moderation failed — inappropriate content detected.".to_string(),
            _ => error_message.to_string(),
        })
    } else {
        None
    };

    Ok(TaskStatusResponse {
        status,
        task_id: task_id.to_string(),
        glb_url,
        poly_count,
        error,
    })
}

/// Extract the best GLB URL from Tripo's nested output structure.
/// Uses a scoring algorithm to find the most likely GLB model URL.
fn extract_glb_url(output: &serde_json::Value) -> Option<String> {
    let mut candidates: Vec<(String, u32)> = Vec::new();

    fn walk(value: &serde_json::Value, key: Option<&str>, candidates: &mut Vec<(String, u32)>) {
        match value {
            serde_json::Value::String(s) if s.starts_with("http") => {
                let mut score: u32 = 1;
                if s.to_lowercase().contains(".glb") {
                    score += 5;
                }
                if s.to_lowercase().contains("glb") {
                    score += 2;
                }
                if s.to_lowercase().contains("model") {
                    score += 1;
                }
                if let Some(k) = key {
                    let k_lower = k.to_lowercase();
                    if k_lower.contains("glb") {
                        score += 2;
                    }
                    if k_lower.contains("model") {
                        score += 1;
                    }
                    if k_lower.contains("url") {
                        score += 1;
                    }
                }
                candidates.push((s.clone(), score));
            }
            serde_json::Value::Array(arr) => {
                for v in arr {
                    walk(v, key, candidates);
                }
            }
            serde_json::Value::Object(obj) => {
                for (k, v) in obj {
                    walk(v, Some(k), candidates);
                }
            }
            _ => {}
        }
    }

    walk(output, None, &mut candidates);

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    Some(candidates[0].0.clone())
}
