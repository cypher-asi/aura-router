use serde_json::{json, Value};

use crate::google_compat;
use crate::providers::Provider;

pub fn validate_request(provider: Provider, request: &Value) -> Result<(), String> {
    match provider {
        Provider::Anthropic => Ok(()),
        // Gemini accepts the same content-block subset the OpenAI lane
        // validates (text/image/tool_result/tool_use) and uses
        // `reasoning_effort` rather than Anthropic `thinking`, so the shared
        // validator applies.
        Provider::OpenAi | Provider::Xai | Provider::DeepSeek | Provider::Google => {
            validate_openai_request(request)
        }
        Provider::Moonshot => validate_moonshot_request(request),
        Provider::Fireworks => {
            validate_openai_request(request)?;
            validate_fireworks_privacy_policy(request)
        }
    }
}

pub fn request_to_upstream(
    provider: Provider,
    upstream_model: &str,
    request: &Value,
) -> Result<Value, String> {
    match provider {
        Provider::Anthropic => {
            let mut next = request.clone();
            next["model"] = Value::String(upstream_model.to_string());
            // Provider-neutral reasoning/cache hints are translated only
            // for OpenAI-family providers. Anthropic uses native
            // `output_config`/`thinking` and content-block `cache_control`,
            // so forwarding these unknown top-level keys would 400.
            if let Some(obj) = next.as_object_mut() {
                for key in [
                    "reasoning_effort",
                    "prompt_cache_key",
                    "prompt_cache_retention",
                ] {
                    obj.remove(key);
                }
            }
            Ok(next)
        }
        Provider::OpenAi | Provider::Xai | Provider::Fireworks | Provider::DeepSeek => {
            validate_openai_request(request)?;
            let mut upstream = anthropic_request_to_openai(request, upstream_model)?;
            apply_reasoning_effort(provider, upstream_model, request, &mut upstream);
            apply_openai_prompt_cache_controls(provider, upstream_model, request, &mut upstream);
            Ok(upstream)
        }
        Provider::Moonshot => {
            validate_moonshot_request(request)?;
            let mut upstream = anthropic_request_to_openai(request, upstream_model)?;
            // K3 fixes these sampling values and rejects client overrides.
            // They must be omitted even when the Anthropic-compatible caller
            // supplied temperature/top_p.
            if let Some(object) = upstream.as_object_mut() {
                object.remove("temperature");
                object.remove("top_p");
            }
            apply_reasoning_effort(provider, upstream_model, request, &mut upstream);
            apply_openai_prompt_cache_controls(provider, upstream_model, request, &mut upstream);
            Ok(upstream)
        }
        Provider::Google => {
            validate_openai_request(request)?;
            google_compat::request_to_gemini(upstream_model, request)
        }
    }
}

/// Translate the provider-neutral `reasoning_effort` tier carried on the
/// inbound `/v1/messages` body into each provider's native control.
///
/// `anthropic_request_to_openai` builds a fresh upstream object, so the
/// neutral hint never leaks through on its own — it is only attached
/// here, clamped to the values the target provider accepts:
/// - OpenAI GPT-5.4/5.5 models accept
///   `none`/`low`/`medium`/`high`/`xhigh`; GPT-5.6 adds `max`.
///   Aura's neutral `minimal` endpoint maps to `none`.
/// - xAI Grok reasoning models accept `none`/`low`/`medium`/`high`; Aura's
///   `minimal` maps to `none`, while larger unsupported tiers fold to `high`.
/// - Fireworks open-weight models (e.g. GPT-OSS) accept
///   `low`/`medium`/`high` (`minimal` folds to `low`).
/// - Moonshot Kimi K3 accepts `low`/`high`/`max`; neutral intermediate
///   tiers fold to the nearest supported native tier.
/// - DeepSeek has no discrete effort knob, so the hint is dropped.
fn apply_reasoning_effort(
    provider: Provider,
    upstream_model: &str,
    request: &Value,
    upstream: &mut Value,
) {
    let Some(tier) = request.get("reasoning_effort").and_then(Value::as_str) else {
        return;
    };
    let mapped = match provider {
        Provider::OpenAi => openai_reasoning_effort(upstream_model, tier),
        Provider::Xai if xai_model_supports_reasoning_effort(upstream_model) => {
            xai_reasoning_effort(tier)
        }
        Provider::Xai => None,
        Provider::Moonshot => match tier.trim().to_ascii_lowercase().as_str() {
            "minimal" | "low" => Some("low"),
            "medium" | "high" | "xhigh" => Some("high"),
            "max" => Some("max"),
            _ => None,
        },
        Provider::Fireworks => match tier.trim().to_ascii_lowercase().as_str() {
            "minimal" | "low" => Some("low"),
            "medium" => Some("medium"),
            "high" | "xhigh" | "max" => Some("high"),
            _ => None,
        },
        // Gemini effort maps to a `thinkingConfig` budget inside
        // `google_compat`, never an OpenAI-style top-level field.
        Provider::Anthropic | Provider::DeepSeek | Provider::Google => None,
    };
    if let (Some(effort), Some(obj)) = (mapped, upstream.as_object_mut()) {
        obj.insert(
            "reasoning_effort".to_string(),
            Value::String(effort.to_string()),
        );
    }
}

fn xai_model_supports_reasoning_effort(upstream_model: &str) -> bool {
    let model = upstream_model
        .strip_prefix("xai/")
        .or_else(|| upstream_model.strip_prefix("grok/"))
        .unwrap_or(upstream_model);
    model == "grok-4.5" || model == "grok-4.3" || model.starts_with("grok-4.20-multi-agent")
}

fn xai_reasoning_effort(tier: &str) -> Option<&'static str> {
    match tier.trim().to_ascii_lowercase().as_str() {
        "minimal" | "none" => Some("none"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" | "xhigh" | "max" => Some("high"),
        _ => None,
    }
}

/// Clamp Aura's provider-neutral effort tier to the values the selected
/// OpenAI model accepts. GPT-5.4/5.5 replaced `minimal` with `none` and
/// added `xhigh`; GPT-5.6 adds a distinct `max` tier above `xhigh`.
/// Older models retain the legacy ladder.
fn openai_reasoning_effort(upstream_model: &str, tier: &str) -> Option<&'static str> {
    let is_gpt_5_6 = upstream_model.starts_with("gpt-5.6");
    let current_ladder = is_gpt_5_6
        || upstream_model.starts_with("gpt-5.4")
        || upstream_model.starts_with("gpt-5.5");
    match (
        current_ladder,
        is_gpt_5_6,
        tier.trim().to_ascii_lowercase().as_str(),
    ) {
        (true, _, "minimal" | "none") => Some("none"),
        (true, _, "low") => Some("low"),
        (true, _, "medium") => Some("medium"),
        (true, _, "high") => Some("high"),
        (true, _, "xhigh") => Some("xhigh"),
        (true, true, "max") => Some("max"),
        (true, false, "max") => Some("xhigh"),
        (false, _, "minimal") => Some("minimal"),
        (false, _, "low") => Some("low"),
        (false, _, "medium") => Some("medium"),
        (false, _, "high" | "xhigh" | "max") => Some("high"),
        _ => None,
    }
}

/// Preserve the cache controls the harness carries on its
/// Anthropic-compatible body. GPT-5.6 requires a stable key for reliable
/// matching and replaces the legacy retention field with a 30-minute TTL.
/// GPT-5.5 accepts only 24-hour retention; GPT-5.4 also accepts in-memory
/// retention. Smaller 5.4 variants must not receive an unsupported value.
fn apply_openai_prompt_cache_controls(
    provider: Provider,
    upstream_model: &str,
    request: &Value,
    upstream: &mut Value,
) {
    if !matches!(provider, Provider::OpenAi | Provider::Moonshot) {
        return;
    }
    let Some(object) = upstream.as_object_mut() else {
        return;
    };
    let cache_key = request
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(key) = cache_key {
        object.insert(
            "prompt_cache_key".to_string(),
            Value::String(key.to_string()),
        );
    }

    // Kimi K3 uses automatic prompt caching and accepts the stable cache
    // key, but not OpenAI's retention/TTL controls.
    if provider == Provider::Moonshot {
        return;
    }

    if cache_key.is_some() && upstream_model.starts_with("gpt-5.6") {
        object.insert("prompt_cache_options".to_string(), json!({ "ttl": "30m" }));
        return;
    }

    let retention = request
        .get("prompt_cache_retention")
        .and_then(Value::as_str)
        .map(str::trim);
    let supported = match (upstream_model, retention) {
        ("gpt-5.5", Some("24h")) => Some("24h"),
        ("gpt-5.4", Some("24h")) => Some("24h"),
        ("gpt-5.4", Some("in_memory")) => Some("in_memory"),
        _ => None,
    };
    if let Some(retention) = supported {
        object.insert(
            "prompt_cache_retention".to_string(),
            Value::String(retention.to_string()),
        );
    }
}

/// Translate an inbound Anthropic `/v1/messages` request into an OpenAI
/// **Responses API** (`/v1/responses`) request body.
///
/// Used for OpenAI requests that carry function tools: chat/completions
/// rejects `tools` + `reasoning_effort` for the gpt-5 family, and the
/// Responses API is OpenAI's go-forward tool-use surface. The conversation
/// is replayed as `input` items (`message` + `function_call` +
/// `function_call_output`), tools use the flat Responses function shape,
/// and effort is carried as `reasoning.effort` (only when the request
/// actually requested an effort tier, so non-reasoning models such as
/// gpt-4.1 are not handed an invalid `reasoning` param).
pub fn anthropic_request_to_openai_responses(
    request: &Value,
    upstream_model: &str,
) -> Result<Value, String> {
    anthropic_request_to_responses(Provider::OpenAi, request, upstream_model)
}

pub fn anthropic_request_to_responses(
    provider: Provider,
    request: &Value,
    upstream_model: &str,
) -> Result<Value, String> {
    let request_messages = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "Anthropic request is missing messages array".to_string())?;

    let mut input: Vec<Value> = Vec::new();
    for message in request_messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| "Anthropic message is missing role".to_string())?;
        let blocks = message_content_blocks(message.get("content"))?;
        match role {
            "user" => append_responses_user_items(&blocks, &mut input),
            "assistant" => append_responses_assistant_items(&blocks, &mut input),
            other => return Err(format!("Unsupported Anthropic role `{other}`")),
        }
    }

    // The Responses API retains conversation state server-side by default
    // (`store: true`). Force the stateless path to match the privacy
    // posture of the chat/completions lane.
    let mut upstream = json!({
        "model": upstream_model,
        "input": input,
        "store": false,
    });

    if let Some(system) = request.get("system") {
        let system_text = flatten_text_blocks(system);
        if !system_text.is_empty() {
            upstream["instructions"] = Value::String(system_text);
        }
    }

    if let Some(max_tokens) = request.get("max_tokens").and_then(Value::as_u64) {
        upstream["max_output_tokens"] = Value::from(max_tokens);
    }

    // Deliberately omit temperature/top_p: gpt-5 and o-series reasoning
    // models reject them on the Responses API.

    let is_streaming = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if is_streaming {
        upstream["stream"] = Value::Bool(true);
    }

    let response_tools = responses_tools(provider, request)?;
    if !response_tools.is_empty() {
        upstream["tools"] = Value::Array(response_tools);
    }

    if let Some(tool_choice) = request.get("tool_choice").and_then(Value::as_object) {
        let mapped = match tool_choice.get("type").and_then(Value::as_str) {
            Some("auto") => Some(Value::String("auto".to_string())),
            Some("any") => Some(Value::String("required".to_string())),
            Some("none") => Some(Value::String("none".to_string())),
            Some("tool") => tool_choice.get("name").and_then(Value::as_str).map(|name| {
                json!({
                    "type": "function",
                    "name": name,
                })
            }),
            _ => None,
        };
        if let Some(value) = mapped {
            upstream["tool_choice"] = value;
        }
    }

    if let Some(tier) = request.get("reasoning_effort").and_then(Value::as_str) {
        if let Some(effort) = responses_reasoning_effort(provider, upstream_model, tier) {
            upstream["reasoning"] = json!({ "effort": effort });
        }
    }
    apply_openai_prompt_cache_controls(provider, upstream_model, request, &mut upstream);

    Ok(upstream)
}

fn responses_reasoning_effort(
    provider: Provider,
    upstream_model: &str,
    tier: &str,
) -> Option<&'static str> {
    match provider {
        Provider::Xai if xai_model_supports_reasoning_effort(upstream_model) => {
            xai_reasoning_effort(tier)
        }
        Provider::Xai => None,
        _ => openai_reasoning_effort(upstream_model, tier),
    }
}

fn responses_tools(provider: Provider, request: &Value) -> Result<Vec<Value>, String> {
    let mut tools = Vec::new();

    if let Some(request_tools) = request.get("tools").and_then(Value::as_array) {
        for tool in request_tools {
            if provider == Provider::Xai && is_xai_server_tool(tool) {
                tools.push(tool.clone());
            } else {
                tools.push(json!({
                    "type": "function",
                    "name": tool.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "description": tool.get("description").and_then(Value::as_str).unwrap_or_default(),
                    "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                }));
            }
        }
    }

    if provider == Provider::Xai {
        for key in ["xai_tools", "server_tools"] {
            if let Some(server_tools) = request.get(key).and_then(Value::as_array) {
                for tool in server_tools {
                    if !is_xai_server_tool(tool) {
                        return Err(format!(
                            "xAI server-side tool entries in `{key}` require a supported `type`."
                        ));
                    }
                    tools.push(tool.clone());
                }
            }
        }
        if let Some(mcp_servers) = request.get("xai_mcp_servers").and_then(Value::as_array) {
            for server in mcp_servers {
                tools.push(xai_mcp_server_tool(server)?);
            }
        }
    }

    Ok(tools)
}

fn is_xai_server_tool(tool: &Value) -> bool {
    matches!(
        tool.get("type").and_then(Value::as_str),
        Some(
            "mcp"
                | "web_search"
                | "x_search"
                | "code_execution"
                | "code_interpreter"
                | "file_search"
                | "attachment_search"
                | "collections_search"
        )
    )
}

fn xai_mcp_server_tool(server: &Value) -> Result<Value, String> {
    let object = server
        .as_object()
        .ok_or_else(|| "`xai_mcp_servers` entries must be objects.".to_string())?;
    let server_url = object
        .get("server_url")
        .or_else(|| object.get("url"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "`xai_mcp_servers` entries require `server_url`.".to_string())?;
    let server_label = object
        .get("server_label")
        .or_else(|| object.get("label"))
        .or_else(|| object.get("name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "`xai_mcp_servers` entries require `server_label`.".to_string())?;

    let mut tool = serde_json::Map::new();
    tool.insert("type".to_string(), Value::String("mcp".to_string()));
    tool.insert(
        "server_url".to_string(),
        Value::String(server_url.to_string()),
    );
    tool.insert(
        "server_label".to_string(),
        Value::String(server_label.to_string()),
    );
    for key in ["server_description", "authorization", "headers"] {
        if let Some(value) = object.get(key) {
            tool.insert(key.to_string(), value.clone());
        }
    }
    if let Some(allowed_tools) = object
        .get("allowed_tools")
        .or_else(|| object.get("allowedTools"))
        .cloned()
    {
        tool.insert("allowed_tools".to_string(), allowed_tools);
    }

    Ok(Value::Object(tool))
}

fn append_responses_user_items(blocks: &[Value], input: &mut Vec<Value>) {
    let mut content_parts: Vec<Value> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                content_parts.push(json!({ "type": "input_text", "text": text }));
            }
            Some("image") => {
                let source = block.get("source").unwrap_or(&Value::Null);
                let media_type = source
                    .get("media_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                let data = source
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                content_parts.push(json!({
                    "type": "input_image",
                    "image_url": format!("data:{media_type};base64,{data}"),
                }));
            }
            Some("tool_result") => {
                // Flush any accumulated user content first so item ordering
                // matches the original message (tool outputs interleaved with
                // text stay in sequence).
                if !content_parts.is_empty() {
                    input.push(json!({
                        "role": "user",
                        "content": std::mem::take(&mut content_parts),
                    }));
                }
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": block.get("tool_use_id").and_then(Value::as_str).unwrap_or_default(),
                    "output": stringify_tool_result_content(block.get("content")),
                }));
            }
            _ => {}
        }
    }

    if !content_parts.is_empty() {
        input.push(json!({
            "role": "user",
            "content": content_parts,
        }));
    }
}

fn append_responses_assistant_items(blocks: &[Value], input: &mut Vec<Value>) {
    let mut text_parts: Vec<String> = Vec::new();

    let flush_text = |text_parts: &mut Vec<String>, input: &mut Vec<Value>| {
        if !text_parts.is_empty() {
            input.push(json!({
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": std::mem::take(text_parts).join("\n\n"),
                }],
            }));
        }
    };

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(text.to_string());
                }
            }
            Some("tool_use") => {
                flush_text(&mut text_parts, input);
                let arguments = block.get("input").cloned().unwrap_or_else(|| json!({}));
                input.push(json!({
                    "type": "function_call",
                    "call_id": block.get("id").and_then(Value::as_str).unwrap_or_default(),
                    "name": block.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "arguments": arguments.to_string(),
                }));
            }
            _ => {}
        }
    }

    flush_text(&mut text_parts, input);
}

/// Translate a non-streaming OpenAI Responses API response into the
/// Anthropic `/v1/messages` response shape.
pub fn openai_responses_to_anthropic(
    response: &Value,
    requested_model: &str,
) -> Result<Value, String> {
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| "OpenAI Responses response is missing output".to_string())?;

    let mut content = Vec::new();
    let mut saw_function_call = false;

    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let text = item
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .filter(|part| {
                                part.get("type").and_then(Value::as_str) == Some("output_text")
                            })
                            .filter_map(|part| part.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join("")
                    })
                    .unwrap_or_default();
                if !text.trim().is_empty() {
                    content.push(json!({ "type": "text", "text": text }));
                }
            }
            Some("function_call") => {
                saw_function_call = true;
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let parsed_input = serde_json::from_str::<Value>(arguments)
                    .unwrap_or_else(|_| Value::String(arguments.to_string()));
                content.push(json!({
                    "type": "tool_use",
                    "id": item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .or_else(|| item.get("id").and_then(Value::as_str))
                        .unwrap_or_default(),
                    "name": item.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "input": parsed_input,
                }));
            }
            _ => {}
        }
    }

    let stop_reason = if saw_function_call {
        "tool_use"
    } else if response
        .pointer("/incomplete_details/reason")
        .and_then(Value::as_str)
        == Some("max_output_tokens")
    {
        "max_tokens"
    } else {
        "end_turn"
    };

    let usage = response.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read_input_tokens = usage
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64);
    let cache_creation_input_tokens = usage
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(Value::as_u64);

    Ok(json!({
        "id": response.get("id").and_then(Value::as_str).unwrap_or_default(),
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
    }))
}

pub fn response_from_upstream(
    provider: Provider,
    requested_model: &str,
    response: &Value,
) -> Result<Value, String> {
    match provider {
        Provider::Anthropic => {
            let mut next = response.clone();
            next["model"] = Value::String(requested_model.to_string());
            Ok(next)
        }
        Provider::Moonshot => openai_response_to_anthropic(response, requested_model, true),
        Provider::OpenAi | Provider::Xai | Provider::Fireworks | Provider::DeepSeek => {
            openai_response_to_anthropic(response, requested_model, false)
        }
        Provider::Google => google_compat::response_from_gemini(requested_model, response),
    }
}

pub fn extract_last_user_text(request: &Value) -> String {
    request
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .rfind(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        })
        .map(|message| flatten_anthropic_content_text(message.get("content")))
        .unwrap_or_default()
}

pub fn extract_response_text(response: &Value) -> String {
    flatten_anthropic_content_text(response.get("content"))
}

fn anthropic_request_to_openai(request: &Value, upstream_model: &str) -> Result<Value, String> {
    let mut messages = Vec::new();

    if let Some(system) = request.get("system") {
        let system_text = flatten_text_blocks(system);
        if !system_text.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": system_text,
            }));
        }
    }

    let request_messages = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "Anthropic request is missing messages array".to_string())?;

    for message in request_messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| "Anthropic message is missing role".to_string())?;
        let blocks = message_content_blocks(message.get("content"))?;

        match role {
            "user" => append_user_messages(&blocks, &mut messages),
            "assistant" => messages.push(build_assistant_message(&blocks)),
            other => return Err(format!("Unsupported Anthropic role `{other}`")),
        }
    }

    let mut upstream = json!({
        "model": upstream_model,
        "messages": messages,
    });

    if let Some(max_tokens) = request.get("max_tokens").and_then(Value::as_u64) {
        upstream["max_completion_tokens"] = Value::from(max_tokens);
    }

    if let Some(temperature) = request.get("temperature").and_then(Value::as_f64) {
        upstream["temperature"] = Value::from(temperature);
    }

    if let Some(top_p) = request.get("top_p").and_then(Value::as_f64) {
        upstream["top_p"] = Value::from(top_p);
    }

    if let Some(stop_sequences) = request.get("stop_sequences").and_then(Value::as_array) {
        let stops = stop_sequences
            .iter()
            .filter_map(Value::as_str)
            .map(|value| Value::String(value.to_string()))
            .collect::<Vec<_>>();
        if !stops.is_empty() {
            upstream["stop"] = Value::Array(stops);
        }
    }

    let is_streaming = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if is_streaming {
        upstream["stream"] = Value::Bool(true);
        let mut stream_options = request
            .get("stream_options")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        stream_options.insert("include_usage".to_string(), Value::Bool(true));
        upstream["stream_options"] = Value::Object(stream_options);
    }

    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        if !tools.is_empty() {
            upstream["tools"] = Value::Array(
                tools.iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.get("name").and_then(Value::as_str).unwrap_or_default(),
                                "description": tool.get("description").and_then(Value::as_str).unwrap_or_default(),
                                "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                            }
                        })
                    })
                    .collect(),
            );
        }
    }

    if let Some(tool_choice) = request.get("tool_choice").and_then(Value::as_object) {
        let mapped = match tool_choice.get("type").and_then(Value::as_str) {
            Some("auto") => Some(Value::String("auto".to_string())),
            Some("any") => Some(Value::String("required".to_string())),
            Some("tool") => tool_choice.get("name").and_then(Value::as_str).map(|name| {
                json!({
                    "type": "function",
                    "function": { "name": name }
                })
            }),
            Some("none") => Some(Value::String("none".to_string())),
            _ => None,
        };
        if let Some(value) = mapped {
            upstream["tool_choice"] = value;
        }
    }

    Ok(upstream)
}

fn validate_openai_request(request: &Value) -> Result<(), String> {
    validate_openai_compatible_request(request, false)
}

fn validate_moonshot_request(request: &Value) -> Result<(), String> {
    validate_openai_compatible_request(request, true)
}

fn validate_openai_compatible_request(
    request: &Value,
    allow_assistant_thinking_blocks: bool,
) -> Result<(), String> {
    if request.get("thinking").is_some() {
        return Err(
            "OpenAI-backed Aura models do not support Anthropic thinking on /v1/messages yet"
                .to_string(),
        );
    }

    validate_system_blocks(request.get("system"))?;
    validate_message_blocks(request.get("messages"), allow_assistant_thinking_blocks)?;

    Ok(())
}

fn validate_fireworks_privacy_policy(request: &Value) -> Result<(), String> {
    if request.get("store").is_some() {
        return Err(
            "Fireworks-backed Aura models do not allow request-level persistence controls on /v1/messages; Aura Router enforces the non-stored inference path centrally"
                .to_string(),
        );
    }

    if request.get("response_id").is_some() {
        return Err(
            "Fireworks-backed Aura models do not support response-state replay on /v1/messages; Aura Router uses stateless inference for privacy"
                .to_string(),
        );
    }

    Ok(())
}

fn validate_system_blocks(system: Option<&Value>) -> Result<(), String> {
    let Some(system) = system else {
        return Ok(());
    };

    match system {
        Value::String(_) => Ok(()),
        Value::Array(blocks) => {
            for block in blocks {
                let block_type = block
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if block_type != "text" {
                    return Err(format!(
                        "OpenAI-backed Aura models only support text system blocks on /v1/messages (got `{block_type}`)"
                    ));
                }
            }
            Ok(())
        }
        _ => Err("System prompt must be a string or content block array".to_string()),
    }
}

fn validate_message_blocks(
    messages: Option<&Value>,
    allow_assistant_thinking_blocks: bool,
) -> Result<(), String> {
    let messages = messages
        .and_then(Value::as_array)
        .ok_or_else(|| "Anthropic request is missing messages array".to_string())?;

    for (message_index, message) in messages.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Anthropic message {message_index} is missing role"))?;
        if !matches!(role, "user" | "assistant") {
            return Err(format!(
                "OpenAI-backed Aura models do not support Anthropic role `{role}`"
            ));
        }
        let blocks = match message.get("content") {
            Some(Value::String(_)) => continue,
            Some(Value::Array(blocks)) => blocks,
            Some(_) => {
                return Err(format!(
                    "Anthropic message {message_index} content must be a string or content array"
                ));
            }
            None => {
                return Err(format!(
                    "Anthropic message {message_index} is missing content"
                ));
            }
        };

        for block in blocks {
            let block_type = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let supported = match role {
                "user" => matches!(block_type, "text" | "image" | "tool_result"),
                "assistant" => {
                    matches!(block_type, "text" | "tool_use")
                        || (allow_assistant_thinking_blocks && block_type == "thinking")
                }
                _ => false,
            };

            if !supported {
                return Err(format!(
                    "OpenAI-backed Aura models do not support `{block_type}` blocks for `{role}` messages on /v1/messages"
                ));
            }
        }
    }

    Ok(())
}

fn append_user_messages(blocks: &[Value], messages: &mut Vec<Value>) {
    if blocks
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
    {
        append_mixed_user_and_tool_result_messages(blocks, messages);
        return;
    }

    append_non_tool_user_message(blocks, messages);
}

fn append_mixed_user_and_tool_result_messages(blocks: &[Value], messages: &mut Vec<Value>) {
    let mut user_blocks = Vec::new();

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("tool_result") => {
                let content = stringify_tool_result_content(block.get("content"));
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id").and_then(Value::as_str).unwrap_or_default(),
                    "content": content,
                }));
            }
            Some("text") | Some("image") => user_blocks.push(block.clone()),
            _ => {}
        }
    }

    append_non_tool_user_message(&user_blocks, messages);
}

fn append_non_tool_user_message(blocks: &[Value], messages: &mut Vec<Value>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut rich_parts: Vec<Value> = Vec::new();
    let mut has_images = false;

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if has_images {
                    rich_parts.push(json!({ "type": "text", "text": text }));
                } else {
                    text_parts.push(text);
                }
            }
            Some("image") => {
                if !has_images {
                    for text in text_parts.drain(..) {
                        rich_parts.push(json!({ "type": "text", "text": text }));
                    }
                }
                has_images = true;
                let source = block.get("source").unwrap_or(&Value::Null);
                let media_type = source
                    .get("media_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                let data = source
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                rich_parts.push(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{media_type};base64,{data}")
                    }
                }));
            }
            _ => {}
        }
    }

    push_pending_user_message(&mut text_parts, &mut rich_parts, has_images, messages);
}

fn push_pending_user_message(
    text_parts: &mut Vec<String>,
    rich_parts: &mut Vec<Value>,
    has_images: bool,
    messages: &mut Vec<Value>,
) {
    if has_images {
        if !rich_parts.is_empty() {
            messages.push(json!({
                "role": "user",
                "content": Value::Array(std::mem::take(rich_parts)),
            }));
        }
        text_parts.clear();
        return;
    }

    let text = text_parts
        .iter()
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !text.is_empty() {
        messages.push(json!({
            "role": "user",
            "content": text,
        }));
    }
    text_parts.clear();
    rich_parts.clear();
}

fn build_assistant_message(blocks: &[Value]) -> Value {
    let mut reasoning_parts = Vec::new();
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("thinking") => {
                if let Some(thinking) = block.get("thinking").and_then(Value::as_str) {
                    reasoning_parts.push(thinking.to_string());
                }
            }
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(text.to_string());
                }
            }
            Some("tool_use") => {
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": block.get("id").and_then(Value::as_str).unwrap_or_default(),
                    "type": "function",
                    "function": {
                        "name": block.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "arguments": input.to_string(),
                    }
                }));
            }
            _ => {}
        }
    }

    let mut message = json!({
        "role": "assistant",
        "content": if text_parts.is_empty() {
            Value::Null
        } else {
            Value::String(text_parts.join("\n\n"))
        },
    });

    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    if !reasoning_parts.is_empty() {
        message["reasoning_content"] = Value::String(reasoning_parts.join(""));
    }

    message
}

fn openai_response_to_anthropic(
    response: &Value,
    requested_model: &str,
    preserve_reasoning_content: bool,
) -> Result<Value, String> {
    let choice = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| "OpenAI response is missing choices".to_string())?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| "OpenAI response is missing message".to_string())?;

    let mut content = Vec::new();
    if preserve_reasoning_content {
        if let Some(reasoning) = message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|reasoning| !reasoning.is_empty())
        {
            content.push(json!({
                "type": "thinking",
                "thinking": reasoning,
                // Moonshot has no Anthropic-style cryptographic thinking
                // signature. A stable marker lets Aura retain and replay this
                // block through its provider-neutral schema.
                "signature": "moonshot"
            }));
        }
    }
    if let Some(text) = extract_openai_text(message.get("content")) {
        if !text.trim().is_empty() {
            content.push(json!({
                "type": "text",
                "text": text,
            }));
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            let arguments = tool_call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let parsed_input = serde_json::from_str::<Value>(arguments)
                .unwrap_or_else(|_| Value::String(arguments.to_string()));
            content.push(json!({
                "type": "tool_use",
                "id": tool_call.get("id").and_then(Value::as_str).unwrap_or_default(),
                "name": tool_call.pointer("/function/name").and_then(Value::as_str).unwrap_or_default(),
                "input": parsed_input,
            }));
        }
    }

    let stop_reason = match choice.get("finish_reason").and_then(Value::as_str) {
        Some("tool_calls") => "tool_use",
        Some("length") => "max_tokens",
        Some("stop") => "end_turn",
        Some("content_filter") => "end_turn",
        _ => "end_turn",
    };

    let usage = response.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read_input_tokens = usage
        .get("prompt_cache_hit_tokens")
        .and_then(Value::as_u64)
        .or_else(|| usage.get("cached_tokens").and_then(Value::as_u64))
        .or_else(|| {
            usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
        });
    let cache_creation_input_tokens = usage
        .get("prompt_cache_miss_tokens")
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .pointer("/prompt_tokens_details/cache_write_tokens")
                .and_then(Value::as_u64)
        });

    let cache_read_input_tokens = cache_read_input_tokens
        .or_else(|| usage.get("cache_read_input_tokens").and_then(Value::as_u64));
    let cache_creation_input_tokens = cache_creation_input_tokens.or_else(|| {
        usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64)
    });

    Ok(json!({
        "id": response.get("id").and_then(Value::as_str).unwrap_or_default(),
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
    }))
}

fn extract_openai_text(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or_else(|| {
                            part.pointer("/text/value")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn message_content_blocks(content: Option<&Value>) -> Result<Vec<Value>, String> {
    match content {
        Some(Value::Array(blocks)) => Ok(blocks.clone()),
        Some(Value::String(text)) => Ok(vec![json!({
            "type": "text",
            "text": text,
        })]),
        Some(_) => Err("Anthropic message content must be a string or content array".to_string()),
        None => Err("Anthropic message is missing content".to_string()),
    }
}

fn flatten_anthropic_content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.trim().to_string(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(extract_text_from_block)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn extract_text_from_block(block: &Value) -> Option<String> {
    block
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            block
                .pointer("/text/value")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn flatten_text_blocks(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_string(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(extract_text_from_block)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn stringify_tool_result_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(_)) => {
            let text = flatten_anthropic_content_text(content);
            if text.is_empty() {
                serde_json::to_string(content.unwrap_or(&Value::Null)).unwrap_or_default()
            } else {
                text
            }
        }
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        anthropic_request_to_openai_responses, anthropic_request_to_responses,
        extract_last_user_text, extract_response_text, openai_responses_to_anthropic,
        request_to_upstream, response_from_upstream, validate_request,
    };
    use crate::providers::Provider;
    use serde_json::{json, Value};

    #[test]
    fn translates_reasoning_effort_to_openai_native() {
        let request = json!({
            "model": "aura-gpt-5-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "reasoning_effort": "minimal"
        });
        let upstream =
            request_to_upstream(Provider::OpenAi, "gpt-5.5", &request).expect("translation");
        assert_eq!(
            upstream["reasoning_effort"],
            Value::String("none".to_string())
        );

        // Aura's neutral `max` maps to OpenAI's native `xhigh`.
        let request = json!({
            "model": "aura-gpt-5-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "reasoning_effort": "max"
        });
        let upstream =
            request_to_upstream(Provider::OpenAi, "gpt-5.5", &request).expect("translation");
        assert_eq!(
            upstream["reasoning_effort"],
            Value::String("xhigh".to_string())
        );
    }

    #[test]
    fn forwards_supported_openai_prompt_cache_controls() {
        let request = json!({
            "model": "aura-gpt-5-5",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "instance:abc-123",
            "prompt_cache_retention": "24h"
        });
        let chat = request_to_upstream(Provider::OpenAi, "gpt-5.5", &request).expect("translation");
        assert_eq!(chat["prompt_cache_key"], "instance:abc-123");
        assert_eq!(chat["prompt_cache_retention"], "24h");

        let responses = anthropic_request_to_openai_responses(&request, "gpt-5.5")
            .expect("responses translation");
        assert_eq!(responses["prompt_cache_key"], "instance:abc-123");
        assert_eq!(responses["prompt_cache_retention"], "24h");
    }

    #[test]
    fn translates_gpt_5_6_effort_and_cache_controls() {
        let request = json!({
            "model": "aura-gpt-5-6-sol",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "max",
            "prompt_cache_key": "instance:abc-123",
            "prompt_cache_retention": "24h"
        });
        let chat =
            request_to_upstream(Provider::OpenAi, "gpt-5.6-sol", &request).expect("translation");
        assert_eq!(chat["reasoning_effort"], "max");
        assert_eq!(chat["prompt_cache_key"], "instance:abc-123");
        assert_eq!(chat["prompt_cache_options"]["ttl"], "30m");
        assert!(chat.get("prompt_cache_retention").is_none());

        let mut xhigh = request.clone();
        xhigh["reasoning_effort"] = json!("xhigh");
        let responses = anthropic_request_to_openai_responses(&xhigh, "gpt-5.6-sol")
            .expect("responses translation");
        assert_eq!(responses["reasoning"]["effort"], "xhigh");
        assert_eq!(responses["prompt_cache_options"]["ttl"], "30m");
    }

    #[test]
    fn drops_unsupported_extended_retention_for_gpt_5_4_mini() {
        let request = json!({
            "model": "aura-gpt-5-4-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "instance:abc-123",
            "prompt_cache_retention": "24h"
        });
        let upstream =
            request_to_upstream(Provider::OpenAi, "gpt-5.4-mini", &request).expect("translation");
        assert_eq!(upstream["prompt_cache_key"], "instance:abc-123");
        assert!(upstream.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn translates_reasoning_effort_to_xai_native() {
        let request = json!({
            "model": "aura-grok-4-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "reasoning_effort": "high"
        });
        let upstream =
            request_to_upstream(Provider::Xai, "grok-4.5", &request).expect("translation");
        assert_eq!(
            upstream["reasoning_effort"],
            Value::String("high".to_string())
        );

        let request = json!({
            "model": "aura-grok-4-3",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "reasoning_effort": "minimal"
        });
        let upstream =
            request_to_upstream(Provider::Xai, "grok-4.3", &request).expect("translation");
        assert_eq!(
            upstream["reasoning_effort"],
            Value::String("none".to_string())
        );

        let request = json!({
            "model": "aura-grok-4-3",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "reasoning_effort": "max"
        });
        let upstream =
            request_to_upstream(Provider::Xai, "grok-4.3", &request).expect("translation");
        assert_eq!(
            upstream["reasoning_effort"],
            Value::String("high".to_string())
        );
    }

    #[test]
    fn drops_reasoning_effort_for_xai_grok_build_chat_completions() {
        let request = json!({
            "model": "aura-grok-build-0-1",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "reasoning_effort": "high"
        });
        let upstream =
            request_to_upstream(Provider::Xai, "grok-build-0.1", &request).expect("translation");
        assert_eq!(upstream["model"], "grok-build-0.1");
        assert!(
            upstream.get("reasoning_effort").is_none(),
            "Grok Build must not receive unsupported reasoning_effort: {upstream}"
        );
    }

    #[test]
    fn folds_minimal_to_low_for_fireworks_open_weight() {
        let request = json!({
            "model": "aura-oss-120b",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "reasoning_effort": "minimal"
        });
        let upstream = request_to_upstream(Provider::Fireworks, "gpt-oss-120b", &request)
            .expect("translation");
        assert_eq!(
            upstream["reasoning_effort"],
            Value::String("low".to_string())
        );
    }

    #[test]
    fn maps_kimi_k3_reasoning_effort_and_prompt_cache_key() {
        for (tier, expected) in [
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "max"),
        ] {
            let request = json!({
                "model": "aura-kimi-k3",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "max_tokens": 1024,
                "reasoning_effort": tier,
                "prompt_cache_key": "workspace-123",
                "temperature": 0,
                "top_p": 1
            });
            let upstream =
                request_to_upstream(Provider::Moonshot, "kimi-k3", &request).expect("translation");
            assert_eq!(upstream["reasoning_effort"], expected);
            assert_eq!(upstream["prompt_cache_key"], "workspace-123");
            assert!(upstream.get("temperature").is_none());
            assert!(upstream.get("top_p").is_none());
            assert!(upstream.get("prompt_cache_options").is_none());
            assert!(upstream.get("prompt_cache_retention").is_none());
        }
    }

    #[test]
    fn preserves_kimi_k3_reasoning_content_for_tool_turn_replay() {
        let request = json!({
            "model": "aura-kimi-k3",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Check the weather."}]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should call the weather tool.",
                            "signature": "moonshot"
                        },
                        {
                            "type": "tool_use",
                            "id": "call_weather",
                            "name": "get_weather",
                            "input": {"city": "Karachi"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_weather",
                        "content": "Sunny"
                    }]
                }
            ],
            "max_tokens": 1024
        });

        let upstream =
            request_to_upstream(Provider::Moonshot, "kimi-k3", &request).expect("translation");
        let assistant = upstream["messages"]
            .as_array()
            .and_then(|messages| {
                messages
                    .iter()
                    .find(|message| message["role"] == "assistant")
            })
            .expect("assistant message");
        assert_eq!(
            assistant["reasoning_content"],
            "I should call the weather tool."
        );
        assert_eq!(assistant["tool_calls"][0]["id"], "call_weather");
    }

    #[test]
    fn strips_provider_neutral_controls_for_anthropic_upstream() {
        let request = json!({
            "model": "aura-claude-opus-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
            "reasoning_effort": "max",
            "prompt_cache_key": "agent:abc-123",
            "prompt_cache_retention": "24h"
        });
        let upstream = request_to_upstream(Provider::Anthropic, "claude-opus-5", &request)
            .expect("translation");
        for key in [
            "reasoning_effort",
            "prompt_cache_key",
            "prompt_cache_retention",
        ] {
            assert!(
                upstream.get(key).is_none(),
                "Anthropic must not receive neutral key {key}: {upstream}"
            );
        }
        // Anthropic-native reasoning controls are left untouched.
        assert_eq!(
            upstream["output_config"]["effort"],
            Value::String("high".to_string())
        );
        assert_eq!(upstream["thinking"]["type"], "adaptive");
    }

    #[test]
    fn responses_request_replays_conversation_as_input_items() {
        let request = json!({
            "model": "aura-gpt-5-5",
            "system": [{"type": "text", "text": "Be helpful"}],
            "max_tokens": 2048,
            "reasoning_effort": "high",
            "tool_choice": {"type": "auto"},
            "tools": [{
                "name": "search_repo",
                "description": "Search repos",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Find a repo"}]},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_repo",
                        "input": {"query": "aura"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "found it"
                    }]
                }
            ]
        });

        let upstream = anthropic_request_to_openai_responses(&request, "gpt-5.5")
            .expect("responses translation");

        assert_eq!(upstream["model"], "gpt-5.5");
        assert_eq!(upstream["store"], Value::Bool(false));
        assert_eq!(upstream["instructions"], "Be helpful");
        assert_eq!(upstream["max_output_tokens"], 2048);
        assert_eq!(upstream["reasoning"]["effort"], "high");
        assert_eq!(upstream["tool_choice"], "auto");

        // Tools use the flat Responses function shape.
        assert_eq!(upstream["tools"][0]["type"], "function");
        assert_eq!(upstream["tools"][0]["name"], "search_repo");
        assert!(upstream["tools"][0]["parameters"].is_object());

        let input = upstream["input"].as_array().expect("input array");
        assert_eq!(input.len(), 3, "input items: {input:?}");

        // user text message
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "Find a repo");

        // assistant function_call (arguments stringified)
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "toolu_1");
        assert_eq!(input[1]["name"], "search_repo");
        assert_eq!(input[1]["arguments"], "{\"query\":\"aura\"}");

        // function_call_output referencing the same call_id
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "toolu_1");
        assert_eq!(input[2]["output"], "found it");
    }

    #[test]
    fn responses_request_omits_reasoning_without_effort() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "max_tokens": 512,
            "tools": [{"name": "noop", "input_schema": {}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        });
        let upstream =
            anthropic_request_to_openai_responses(&request, "gpt-4.1").expect("translation");
        assert!(
            upstream.get("reasoning").is_none(),
            "non-reasoning model with tools must not receive a reasoning param: {upstream}"
        );
    }

    #[test]
    fn xai_responses_request_maps_reasoning_and_remote_mcp_tools() {
        let request = json!({
            "model": "aura-grok-4-3",
            "max_tokens": 512,
            "reasoning_effort": "minimal",
            "xai_mcp_servers": [{
                "server_url": "https://mcp.deepwiki.com/mcp",
                "server_label": "deepwiki",
                "server_description": "Docs search",
                "allowed_tools": ["ask_question"]
            }],
            "xai_tools": [{
                "type": "web_search"
            }],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        });
        let upstream = anthropic_request_to_responses(Provider::Xai, &request, "grok-4.3")
            .expect("xai responses translation");

        assert_eq!(upstream["model"], "grok-4.3");
        assert_eq!(upstream["reasoning"]["effort"], "none");
        assert_eq!(upstream["tools"][0]["type"], "web_search");
        assert_eq!(upstream["tools"][1]["type"], "mcp");
        assert_eq!(
            upstream["tools"][1]["server_url"],
            "https://mcp.deepwiki.com/mcp"
        );
        assert_eq!(upstream["tools"][1]["allowed_tools"][0], "ask_question");
    }

    #[test]
    fn xai_responses_request_omits_reasoning_for_grok_build() {
        let request = json!({
            "model": "aura-grok-build-0-1",
            "max_tokens": 512,
            "reasoning_effort": "high",
            "xai_tools": [{
                "type": "web_search"
            }],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        });
        let upstream = anthropic_request_to_responses(Provider::Xai, &request, "grok-build-0.1")
            .expect("xai responses translation");

        assert_eq!(upstream["model"], "grok-build-0.1");
        assert!(
            upstream.get("reasoning").is_none(),
            "Grok Build must not receive unsupported reasoning: {upstream}"
        );
        assert_eq!(upstream["tools"][0]["type"], "web_search");
    }

    #[test]
    fn responses_response_maps_text_and_tool_calls() {
        let response = json!({
            "id": "resp_123",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Here you go"}]
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "search_repo",
                    "arguments": "{\"query\":\"aura\"}"
                }
            ],
            "usage": {
                "input_tokens": 42,
                "output_tokens": 7,
                "input_tokens_details": {
                    "cached_tokens": 12,
                    "cache_write_tokens": 5
                }
            }
        });

        let normalized =
            openai_responses_to_anthropic(&response, "aura-gpt-5-5").expect("normalize");

        assert_eq!(normalized["id"], "resp_123");
        assert_eq!(normalized["model"], "aura-gpt-5-5");
        assert_eq!(normalized["stop_reason"], "tool_use");
        assert_eq!(normalized["content"][0]["type"], "text");
        assert_eq!(normalized["content"][0]["text"], "Here you go");
        assert_eq!(normalized["content"][1]["type"], "tool_use");
        assert_eq!(normalized["content"][1]["id"], "call_1");
        assert_eq!(normalized["content"][1]["name"], "search_repo");
        assert_eq!(normalized["content"][1]["input"]["query"], "aura");
        assert_eq!(normalized["usage"]["input_tokens"], 42);
        assert_eq!(normalized["usage"]["output_tokens"], 7);
        assert_eq!(normalized["usage"]["cache_creation_input_tokens"], 5);
        assert_eq!(normalized["usage"]["cache_read_input_tokens"], 12);
    }

    #[test]
    fn responses_response_maps_max_tokens_stop_reason() {
        let response = json!({
            "id": "resp_456",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "partial"}]
            }],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let normalized = openai_responses_to_anthropic(&response, "gpt-5.5").expect("normalize");
        assert_eq!(normalized["stop_reason"], "max_tokens");
    }

    #[test]
    fn drops_reasoning_effort_for_deepseek() {
        let request = json!({
            "model": "aura-deepseek-v4-pro",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 1024,
            "reasoning_effort": "high"
        });
        let upstream = request_to_upstream(Provider::DeepSeek, "deepseek-chat", &request)
            .expect("translation");
        assert!(
            upstream.get("reasoning_effort").is_none(),
            "DeepSeek has no effort knob: {upstream}"
        );
    }

    #[test]
    fn translates_anthropic_request_to_openai_tools_format() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "system": [{"type": "text", "text": "Be helpful"}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Find a repo"}]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_repo",
                        "input": {"query": "aura"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "done"
                    }]
                }
            ],
            "tools": [{
                "name": "search_repo",
                "description": "Search repositories",
                "input_schema": {"type": "object"}
            }],
            "tool_choice": {"type": "tool", "name": "search_repo"},
            "max_tokens": 2048
        });

        let translated =
            request_to_upstream(Provider::OpenAi, "gpt-4.1", &request).expect("translation");

        assert_eq!(translated["model"], "gpt-4.1");
        assert_eq!(translated["messages"][0]["role"], "system");
        assert_eq!(translated["messages"][1]["role"], "user");
        assert_eq!(
            translated["messages"][2]["tool_calls"][0]["function"]["name"],
            "search_repo"
        );
        assert_eq!(translated["messages"][3]["role"], "tool");
        assert_eq!(translated["tool_choice"]["function"]["name"], "search_repo");
        assert_eq!(translated["max_completion_tokens"], 2048);
    }

    #[test]
    fn translates_mixed_tool_results_before_follow_up_user_content() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_repo",
                        "input": {"query": "aura"}
                    }]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the screenshot."},
                        {
                            "type": "image",
                            "source": {
                                "media_type": "image/png",
                                "data": "ZmFrZQ=="
                            }
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "repo found"}]
                        },
                        {"type": "text", "text": "Please summarize it."}
                    ]
                }
            ]
        });

        let translated =
            request_to_upstream(Provider::OpenAi, "gpt-4.1", &request).expect("translation");
        let messages = translated["messages"].as_array().expect("messages array");

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "toolu_1");
        assert_eq!(messages[1]["content"], "repo found");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "text");
        assert_eq!(messages[2]["content"][0]["text"], "Here is the screenshot.");
        assert_eq!(messages[2]["content"][1]["type"], "image_url");
        assert_eq!(messages[2]["content"][2]["type"], "text");
        assert_eq!(messages[2]["content"][2]["text"], "Please summarize it.");
    }

    #[test]
    fn translates_openai_response_to_anthropic_blocks() {
        let response = json!({
            "id": "chatcmpl_123",
            "model": "gpt-4.1",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Let me check that.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_repo",
                            "arguments": "{\"query\":\"aura\"}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "prompt_tokens_details": {
                    "cached_tokens": 8,
                    "cache_write_tokens": 3
                }
            }
        });

        let translated = response_from_upstream(Provider::OpenAi, "aura-gpt-4.1", &response)
            .expect("translation");

        assert_eq!(translated["model"], "aura-gpt-4.1");
        assert_eq!(translated["stop_reason"], "tool_use");
        assert_eq!(translated["content"][0]["type"], "text");
        assert_eq!(translated["content"][1]["type"], "tool_use");
        assert_eq!(translated["usage"]["input_tokens"], 12);
        assert_eq!(translated["usage"]["output_tokens"], 7);
        assert_eq!(translated["usage"]["cache_creation_input_tokens"], 3);
        assert_eq!(translated["usage"]["cache_read_input_tokens"], 8);
    }

    #[test]
    fn translates_moonshot_top_level_cached_tokens() {
        let response = json!({
            "id": "chatcmpl_kimi",
            "model": "kimi-k3",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "reasoning_content": "I can answer directly.",
                    "content": "Done"
                }
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 40
            }
        });

        let translated = response_from_upstream(Provider::Moonshot, "aura-kimi-k3", &response)
            .expect("translation");
        assert_eq!(translated["model"], "aura-kimi-k3");
        assert_eq!(translated["usage"]["input_tokens"], 100);
        assert_eq!(translated["usage"]["output_tokens"], 20);
        assert_eq!(translated["usage"]["cache_read_input_tokens"], 40);
        assert_eq!(translated["content"][0]["type"], "thinking");
        assert_eq!(
            translated["content"][0]["thinking"],
            "I can answer directly."
        );
        assert_eq!(translated["content"][1]["type"], "text");
    }

    #[test]
    fn translates_deepseek_cache_usage_aliases() {
        let response = json!({
            "id": "deepseek_123",
            "model": "deepseek-v4-flash",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Done"
                }
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_miss_tokens": 30,
                "prompt_cache_hit_tokens": 70
            }
        });

        let translated =
            response_from_upstream(Provider::DeepSeek, "aura-deepseek-v4-flash", &response)
                .expect("translation");

        assert_eq!(translated["model"], "aura-deepseek-v4-flash");
        assert_eq!(translated["usage"]["input_tokens"], 100);
        assert_eq!(translated["usage"]["output_tokens"], 20);
        assert_eq!(translated["usage"]["cache_creation_input_tokens"], 30);
        assert_eq!(translated["usage"]["cache_read_input_tokens"], 70);
    }

    #[test]
    fn rejects_openai_requests_with_anthropic_thinking() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "system": [{"type": "text", "text": "Be helpful"}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }],
            "thinking": {"type": "enabled", "budget_tokens": 2048}
        });

        let error = validate_request(Provider::OpenAi, &request).expect_err("request should fail");
        assert!(error.contains("do not support Anthropic thinking"));
    }

    #[test]
    fn rejects_openai_requests_with_unsupported_message_blocks() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "messages": [{
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "secret"}]
            }]
        });

        let error = validate_request(Provider::OpenAi, &request).expect_err("request should fail");
        assert!(error.contains("do not support `thinking` blocks"));
    }

    #[test]
    fn rejects_fireworks_requests_with_store_flag() {
        let request = json!({
            "model": "aura-kimi-k2-6",
            "store": true,
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        });

        let error =
            validate_request(Provider::Fireworks, &request).expect_err("request should fail");
        assert!(error.contains("do not allow request-level persistence controls"));
    }

    #[test]
    fn rejects_fireworks_requests_with_response_state_replay() {
        let request = json!({
            "model": "aura-kimi-k2-6",
            "response_id": "resp_123",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        });

        let error =
            validate_request(Provider::Fireworks, &request).expect_err("request should fail");
        assert!(error.contains("stateless inference for privacy"));
    }

    #[test]
    fn translates_string_shorthand_messages_for_openai_requests() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello from shorthand"
                },
                {
                    "role": "assistant",
                    "content": "Partial answer"
                }
            ]
        });

        let translated =
            request_to_upstream(Provider::OpenAi, "gpt-4.1", &request).expect("translation");

        assert_eq!(translated["messages"][0]["role"], "user");
        assert_eq!(translated["messages"][0]["content"], "Hello from shorthand");
        assert_eq!(translated["messages"][1]["role"], "assistant");
        assert_eq!(translated["messages"][1]["content"], "Partial answer");
    }

    #[test]
    fn forwards_streaming_and_high_value_compat_fields_for_openai_requests() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "stream": true,
            "top_p": 0.9,
            "stop_sequences": ["DONE", "STOP"],
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        });

        let translated =
            request_to_upstream(Provider::OpenAi, "gpt-4.1", &request).expect("translation");

        assert_eq!(translated["stream"], Value::Bool(true));
        assert_eq!(
            translated["stream_options"]["include_usage"],
            Value::Bool(true)
        );
        assert_eq!(translated["top_p"], json!(0.9));
        assert_eq!(translated["stop"], json!(["DONE", "STOP"]));
    }

    #[test]
    fn extracts_last_user_text_from_string_and_block_content() {
        let request = json!({
            "messages": [
                {
                    "role": "user",
                    "content": "Earlier"
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Reply"}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Latest"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                        {"type": "text", "text": "Question"}
                    ]
                }
            ]
        });

        assert_eq!(extract_last_user_text(&request), "Latest\n\nQuestion");
    }

    #[test]
    fn extracts_response_text_from_text_blocks() {
        let response = json!({
            "content": [
                {"type": "text", "text": "First"},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "aura"}},
                {"type": "text", "text": "Second"}
            ]
        });

        assert_eq!(extract_response_text(&response), "First\n\nSecond");
    }
}
