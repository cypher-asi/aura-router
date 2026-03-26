//! Image generation clients for OpenAI and Gemini.

use base64::Engine;
use serde::{Deserialize, Serialize};

/// Style lock prompt appended to all image generation requests.
pub const STYLE_LOCK: &str = ", standalone product only, 3/4 angle view, single object centered, fully in frame with no cropping, no other objects or elements in frame, jet black background with subtle vignette, photorealistic, high-poly, textured 3D sculpture, subject pops from background, cinematic depth, isolated product presentation";

/// Image generation request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateImageRequest {
    pub prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    pub model: Option<String>,
    pub images: Option<Vec<ImageInput>>,
}

fn default_size() -> String {
    "1024x1024".to_string()
}

/// Image input — either a URL or base64 data URL.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ImageInput {
    Url(String),
    Object { url: String, name: Option<String> },
}

impl ImageInput {
    pub fn url(&self) -> &str {
        match self {
            ImageInput::Url(u) => u,
            ImageInput::Object { url, .. } => url,
        }
    }
}

/// Image generation response.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateImageResponse {
    pub success: bool,
    pub image_url: String,
    pub original_url: Option<String>,
    pub meta: ImageMeta,
}

/// Image generation metadata.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageMeta {
    pub model: String,
    pub size: String,
    pub prompt: String,
    pub provider: String,
    pub created: i64,
}

/// Config for available models.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenConfig {
    pub models: Vec<ImageModelConfig>,
    pub default_model: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageModelConfig {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub eta_ms: u64,
    pub supports_references: bool,
}

pub fn get_config() -> ImageGenConfig {
    ImageGenConfig {
        default_model: "gpt-image-1".to_string(),
        models: vec![
            ImageModelConfig {
                id: "gpt-image-1".to_string(),
                name: "GPT Image 1".to_string(),
                provider: "openai".to_string(),
                eta_ms: 20000,
                supports_references: true,
            },
            ImageModelConfig {
                id: "dall-e-3".to_string(),
                name: "DALL-E 3".to_string(),
                provider: "openai".to_string(),
                eta_ms: 15000,
                supports_references: false,
            },
            ImageModelConfig {
                id: "gemini-nano-banana".to_string(),
                name: "Gemini Flash Image".to_string(),
                provider: "google".to_string(),
                eta_ms: 25000,
                supports_references: true,
            },
        ],
    }
}

/// Result from image generation — base64 image data.
pub struct GeneratedImage {
    pub base64_data: String,
    pub mime_type: String,
    pub model: String,
    pub provider: String,
}

/// Generate an image using OpenAI's API.
pub async fn generate_openai(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    model: &str,
    images: Option<&[ImageInput]>,
) -> Result<GeneratedImage, String> {
    let full_prompt = format!("{prompt}{STYLE_LOCK}");
    let has_images = images.map_or(false, |imgs| !imgs.is_empty());

    if has_images {
        generate_openai_edit(client, api_key, &full_prompt, size, model, images.unwrap()).await
    } else {
        generate_openai_create(client, api_key, &full_prompt, size, model).await
    }
}

/// OpenAI image generation (no reference images).
async fn generate_openai_create(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    model: &str,
) -> Result<GeneratedImage, String> {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1
    });

    let resp = client
        .post("https://api.openai.com/v1/images/generations")
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("OpenAI request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI error: {error}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    let b64 = data["data"][0]["b64_json"]
        .as_str()
        .ok_or("No image data in response")?
        .to_string();

    Ok(GeneratedImage {
        base64_data: b64,
        mime_type: "image/png".to_string(),
        model: model.to_string(),
        provider: "openai".to_string(),
    })
}

/// OpenAI image edit (with reference images).
async fn generate_openai_edit(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    model: &str,
    images: &[ImageInput],
) -> Result<GeneratedImage, String> {
    // Fetch first reference image as bytes
    let image_url = images[0].url();
    let image_bytes = fetch_image_bytes(client, image_url).await?;

    // Build multipart form
    let form = reqwest::multipart::Form::new()
        .text("model", model.to_string())
        .text("prompt", prompt.to_string())
        .text("size", size.to_string())
        .part(
            "image",
            reqwest::multipart::Part::bytes(image_bytes)
                .file_name("image.png")
                .mime_str("image/png")
                .map_err(|e| format!("MIME error: {e}"))?,
        );

    let resp = client
        .post("https://api.openai.com/v1/images/edits")
        .header("authorization", format!("Bearer {api_key}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("OpenAI edit request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        // Fallback to generation without references
        tracing::warn!("OpenAI edit failed, falling back to generation: {error}");
        return generate_openai_create(client, api_key, prompt, size, model).await;
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    let b64 = data["data"][0]["b64_json"]
        .as_str()
        .ok_or("No image data in response")?
        .to_string();

    Ok(GeneratedImage {
        base64_data: b64,
        mime_type: "image/png".to_string(),
        model: model.to_string(),
        provider: "openai".to_string(),
    })
}

/// Generate an image using Google Gemini.
pub async fn generate_gemini(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    images: Option<&[ImageInput]>,
) -> Result<GeneratedImage, String> {
    let full_prompt = format!("{prompt}{STYLE_LOCK}");

    // Add size instruction
    let size_instruction = match size {
        "1024x1024" | "256x256" | "512x512" => " Generate a square image (1:1 aspect ratio).",
        "1536x1024" => " Generate a landscape image (3:2 aspect ratio).",
        "1024x1536" => " Generate a portrait image (2:3 aspect ratio).",
        _ => "",
    };

    let prompt_with_size = format!("{full_prompt}{size_instruction}");

    // Build content parts
    let mut parts = Vec::new();

    // Add reference images if provided
    if let Some(imgs) = images {
        for img in imgs {
            if let Ok((data, mime)) = fetch_image_as_base64(client, img.url()).await {
                parts.push(serde_json::json!({
                    "inlineData": {
                        "mimeType": mime,
                        "data": data
                    }
                }));
            }
        }
    }

    // Add text prompt
    parts.push(serde_json::json!({ "text": prompt_with_size }));

    let body = serde_json::json!({
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        }
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={api_key}"
    );

    let resp = client
        .post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Gemini request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("Gemini error: {error}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    // Extract image from response
    let parts = data
        .pointer("/candidates/0/content/parts")
        .and_then(|p| p.as_array())
        .ok_or("No parts in Gemini response")?;

    for part in parts {
        if let Some(inline_data) = part.get("inlineData") {
            let b64 = inline_data["data"]
                .as_str()
                .ok_or("No data in inlineData")?
                .to_string();
            let mime = inline_data["mimeType"]
                .as_str()
                .unwrap_or("image/png")
                .to_string();

            return Ok(GeneratedImage {
                base64_data: b64,
                mime_type: mime,
                model: "gemini-2.5-flash-image".to_string(),
                provider: "google".to_string(),
            });
        }
    }

    Err("No image found in Gemini response".to_string())
}

/// Fetch an image URL as raw bytes.
async fn fetch_image_bytes(client: &reqwest::Client, url: &str) -> Result<Vec<u8>, String> {
    if url.starts_with("data:") {
        // Base64 data URL
        let b64 = url
            .find(',')
            .map(|i| &url[i + 1..])
            .unwrap_or(url);
        return base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("Invalid base64: {e}"));
    }

    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch image: {e}"))?;

    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("Failed to read image: {e}"))
}

/// Fetch an image URL as base64 data + MIME type.
async fn fetch_image_as_base64(
    client: &reqwest::Client,
    url: &str,
) -> Result<(String, String), String> {
    if url.starts_with("data:") {
        // Extract MIME and base64 from data URL
        let mime = url
            .strip_prefix("data:")
            .and_then(|s| s.split(';').next())
            .unwrap_or("image/png")
            .to_string();
        let b64 = url.find(',').map(|i| &url[i + 1..]).unwrap_or(url);
        return Ok((b64.to_string(), mime));
    }

    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch image: {e}"))?;

    let mime = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("image/png")
        .to_string();

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read image: {e}"))?;

    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok((b64, mime))
}

/// SSE event types for streaming image generation.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ImageStreamEvent {
    #[serde(rename = "start")]
    Start { ts: String },
    #[serde(rename = "progress")]
    Progress { percent: u32, message: String },
    #[serde(rename = "partial-image")]
    PartialImage { data: String },
    #[serde(rename = "completed")]
    Completed {
        #[serde(rename = "imageUrl")]
        image_url: String,
        #[serde(rename = "originalUrl")]
        original_url: Option<String>,
        meta: ImageMeta,
    },
    #[serde(rename = "error")]
    Error { code: String, message: String },
}

/// Generate an image using OpenAI with streaming (partial images).
pub async fn generate_openai_stream(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    model: &str,
    images: Option<&[ImageInput]>,
    event_tx: tokio::sync::mpsc::Sender<ImageStreamEvent>,
) -> Result<GeneratedImage, String> {
    let full_prompt = format!("{prompt}{STYLE_LOCK}");
    let has_images = images.map_or(false, |imgs| !imgs.is_empty());

    // Send start event
    let _ = event_tx
        .send(ImageStreamEvent::Start {
            ts: chrono::Utc::now().to_rfc3339(),
        })
        .await;

    let _ = event_tx
        .send(ImageStreamEvent::Progress {
            percent: 10,
            message: "Generating image...".to_string(),
        })
        .await;

    // For streaming, OpenAI supports stream=true on images.generate
    // but the edit endpoint doesn't support streaming well.
    // Use non-streaming for edits, streaming for generation.
    if has_images {
        let _ = event_tx
            .send(ImageStreamEvent::Progress {
                percent: 30,
                message: "Processing reference images...".to_string(),
            })
            .await;

        let result =
            generate_openai_edit(client, api_key, &full_prompt, size, model, images.unwrap())
                .await?;

        let _ = event_tx
            .send(ImageStreamEvent::Progress {
                percent: 90,
                message: "Uploading...".to_string(),
            })
            .await;

        return Ok(result);
    }

    // Try streaming generation
    let body = serde_json::json!({
        "model": model,
        "prompt": full_prompt,
        "size": size,
        "n": 1,
        "stream": true,
        "partial_images": 2
    });

    let resp = client
        .post("https://api.openai.com/v1/images/generations")
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("OpenAI streaming request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI error: {error}"));
    }

    // Parse SSE stream from OpenAI
    let mut final_b64 = String::new();
    let mut stream = resp.bytes_stream();

    use futures_util::StreamExt;
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream error: {e}"))?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        // Process complete SSE lines
        while let Some(newline_pos) = buffer.find("\n\n") {
            let event_block = buffer[..newline_pos].to_string();
            buffer = buffer[newline_pos + 2..].to_string();

            // Parse event
            let mut event_type = String::new();
            let mut data = String::new();

            for line in event_block.lines() {
                if let Some(et) = line.strip_prefix("event: ") {
                    event_type = et.to_string();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = d.to_string();
                }
            }

            if data == "[DONE]" {
                break;
            }

            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&data) {
                match event_type.as_str() {
                    "image_generation.partial_image" | "image_edit.partial_image" => {
                        if let Some(b64) = value.get("b64_json").and_then(|v| v.as_str()) {
                            let _ = event_tx
                                .send(ImageStreamEvent::PartialImage {
                                    data: format!("data:image/png;base64,{b64}"),
                                })
                                .await;
                        }

                        let _ = event_tx
                            .send(ImageStreamEvent::Progress {
                                percent: 50,
                                message: "Refining...".to_string(),
                            })
                            .await;
                    }
                    "image_generation.completed" | "image_edit.completed" => {
                        if let Some(b64) = value.get("b64_json").and_then(|v| v.as_str()) {
                            final_b64 = b64.to_string();
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if final_b64.is_empty() {
        return Err("No final image in stream".to_string());
    }

    let _ = event_tx
        .send(ImageStreamEvent::Progress {
            percent: 90,
            message: "Uploading...".to_string(),
        })
        .await;

    Ok(GeneratedImage {
        base64_data: final_b64,
        mime_type: "image/png".to_string(),
        model: model.to_string(),
        provider: "openai".to_string(),
    })
}

/// Resolve which provider/model to use.
pub fn resolve_image_model(model: Option<&str>) -> (&str, &str) {
    match model {
        Some("gemini-nano-banana") | Some("gemini") => ("gemini-nano-banana", "google"),
        Some("dall-e-3") => ("dall-e-3", "openai"),
        Some("dall-e-2") => ("dall-e-2", "openai"),
        Some(m) if m.starts_with("gpt") => (m, "openai"),
        _ => ("gpt-image-1", "openai"),
    }
}
