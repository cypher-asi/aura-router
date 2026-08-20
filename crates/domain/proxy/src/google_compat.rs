//! Anthropic `/v1/messages` <-> Google Gemini `generateContent` translation.
//!
//! The router speaks one hosted protocol (Anthropic Messages) to clients.
//! Gemini uses a different request/response shape (`contents`/`parts`,
//! `systemInstruction`, `functionDeclarations`, `generationConfig` with a
//! `thinkingConfig` budget), so this module bridges the two for the chat
//! proxy. Streaming is handled separately in `stream.rs`.

use std::collections::HashMap;

use serde_json::{json, Map, Value};
use uuid::Uuid;

/// Translate an inbound Anthropic request body into a Gemini
/// `generateContent` request body. The upstream model and streaming mode
/// are carried in the URL (see `providers::google_endpoint_url`), so they
/// are not part of the body.
pub fn request_to_gemini(_upstream_model: &str, request: &Value) -> Result<Value, String> {
    let request_messages = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "Anthropic request is missing messages array".to_string())?;

    // Gemini `functionResponse` parts need the function *name*, but Anthropic
    // `tool_result` blocks only reference the originating `tool_use_id`. Build
    // an id->name map from the assistant `tool_use` blocks in the same
    // conversation so tool results can be resolved.
    let tool_names = collect_tool_names(request_messages);

    let mut contents: Vec<Value> = Vec::new();
    for message in request_messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| "Anthropic message is missing role".to_string())?;
        let blocks = message_content_blocks(message.get("content"))?;
        match role {
            "user" => append_user_contents(&blocks, &tool_names, &mut contents),
            "assistant" => append_assistant_content(&blocks, &mut contents),
            other => return Err(format!("Unsupported Anthropic role `{other}`")),
        }
    }

    let mut body = json!({ "contents": contents });

    if let Some(system) = request.get("system") {
        let system_text = flatten_text_blocks(system);
        if !system_text.is_empty() {
            body["systemInstruction"] = json!({
                "parts": [{ "text": system_text }],
            });
        }
    }

    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        let declarations: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "description": tool
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    "parameters": sanitize_schema(
                        tool.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                    ),
                })
            })
            .collect();
        if !declarations.is_empty() {
            body["tools"] = json!([{ "functionDeclarations": declarations }]);
        }
    }

    if let Some(tool_choice) = request.get("tool_choice").and_then(Value::as_object) {
        if let Some(config) = tool_config_from_choice(tool_choice) {
            body["toolConfig"] = config;
        }
    }

    let mut generation_config = Map::new();
    if let Some(max_tokens) = request.get("max_tokens").and_then(Value::as_u64) {
        generation_config.insert("maxOutputTokens".to_string(), Value::from(max_tokens));
    }
    if let Some(temperature) = request.get("temperature").and_then(Value::as_f64) {
        generation_config.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = request.get("top_p").and_then(Value::as_f64) {
        generation_config.insert("topP".to_string(), Value::from(top_p));
    }
    if let Some(stop_sequences) = request.get("stop_sequences").and_then(Value::as_array) {
        let stops: Vec<Value> = stop_sequences
            .iter()
            .filter_map(Value::as_str)
            .map(|value| Value::String(value.to_string()))
            .collect();
        if !stops.is_empty() {
            generation_config.insert("stopSequences".to_string(), Value::Array(stops));
        }
    }
    if let Some(tier) = request.get("reasoning_effort").and_then(Value::as_str) {
        if let Some(budget) = thinking_budget_for_effort(tier) {
            generation_config.insert(
                "thinkingConfig".to_string(),
                json!({
                    "thinkingBudget": budget,
                    "includeThoughts": false,
                }),
            );
        }
    }
    if !generation_config.is_empty() {
        body["generationConfig"] = Value::Object(generation_config);
    }

    Ok(body)
}

/// Translate a non-streaming Gemini `generateContent` response into the
/// Anthropic `/v1/messages` response shape.
pub fn response_from_gemini(requested_model: &str, response: &Value) -> Result<Value, String> {
    let candidate = response
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first());

    let mut content = Vec::new();
    let mut saw_tool_call = false;

    if let Some(parts) = candidate
        .and_then(|candidate| candidate.pointer("/content/parts"))
        .and_then(Value::as_array)
    {
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    content.push(json!({ "type": "text", "text": text }));
                }
            } else if let Some(function_call) = part.get("functionCall") {
                saw_tool_call = true;
                content.push(json!({
                    "type": "tool_use",
                    "id": format!("toolu_{}", Uuid::new_v4().simple()),
                    "name": function_call.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "input": function_call.get("args").cloned().unwrap_or_else(|| json!({})),
                }));
            }
        }
    }

    let finish_reason = candidate
        .and_then(|candidate| candidate.get("finishReason"))
        .and_then(Value::as_str);
    let stop_reason = map_finish_reason(finish_reason, saw_tool_call);

    let usage = response
        .get("usageMetadata")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let (input_tokens, output_tokens, cache_read_input_tokens) = extract_usage(&usage);

    Ok(json!({
        "id": format!("msg_{}", Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": Value::Null,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
    }))
}

/// Extract `(input_tokens, output_tokens, cache_read_input_tokens)` from a
/// Gemini `usageMetadata` object. Gemini reports `promptTokenCount` as the
/// full prompt (including cached tokens) and surfaces thinking tokens
/// separately, so the output total folds in `thoughtsTokenCount`.
pub fn extract_usage(usage: &Value) -> (u64, u64, Option<u64>) {
    let input_tokens = usage
        .get("promptTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let candidates_tokens = usage
        .get("candidatesTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let thoughts_tokens = usage
        .get("thoughtsTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = usage.get("cachedContentTokenCount").and_then(Value::as_u64);
    (
        input_tokens,
        candidates_tokens.saturating_add(thoughts_tokens),
        cache_read,
    )
}

/// Map a Gemini `finishReason` (plus whether a function call was emitted)
/// onto an Anthropic `stop_reason`.
pub fn map_finish_reason(finish_reason: Option<&str>, saw_tool_call: bool) -> &'static str {
    if saw_tool_call {
        return "tool_use";
    }
    match finish_reason {
        Some("MAX_TOKENS") => "max_tokens",
        Some("STOP") => "end_turn",
        _ => "end_turn",
    }
}

/// Translate the provider-neutral `reasoning_effort` tier into a Gemini
/// `thinkingBudget` (in tokens). `max` maps to `-1` (dynamic budget, the
/// model decides). Returns `None` for unrecognized tiers so the caller
/// omits `thinkingConfig` entirely.
fn thinking_budget_for_effort(tier: &str) -> Option<i64> {
    match tier.trim().to_ascii_lowercase().as_str() {
        "minimal" => Some(0),
        "low" => Some(4_096),
        "medium" => Some(8_192),
        "high" => Some(16_384),
        "xhigh" | "max" => Some(-1),
        _ => None,
    }
}

fn collect_tool_names(messages: &[Value]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                if let (Some(id), Some(name)) = (
                    block.get("id").and_then(Value::as_str),
                    block.get("name").and_then(Value::as_str),
                ) {
                    names.insert(id.to_string(), name.to_string());
                }
            }
        }
    }
    names
}

fn append_user_contents(
    blocks: &[Value],
    tool_names: &HashMap<String, String>,
    contents: &mut Vec<Value>,
) {
    let mut user_parts: Vec<Value> = Vec::new();
    let mut function_responses: Vec<Value> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block.get("text").and_then(Value::as_str).unwrap_or_default();
                user_parts.push(json!({ "text": text }));
            }
            Some("image") => {
                let source = block.get("source").unwrap_or(&Value::Null);
                let media_type = source
                    .get("media_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                let data = source.get("data").and_then(Value::as_str).unwrap_or_default();
                user_parts.push(json!({
                    "inlineData": { "mimeType": media_type, "data": data },
                }));
            }
            Some("tool_result") => {
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let name = tool_names
                    .get(tool_use_id)
                    .cloned()
                    .unwrap_or_else(|| tool_use_id.to_string());
                function_responses.push(json!({
                    "functionResponse": {
                        "name": name,
                        "response": {
                            "result": stringify_tool_result_content(block.get("content")),
                        }
                    }
                }));
            }
            _ => {}
        }
    }

    // Gemini expects `functionResponse` parts in a `user`-role turn. Emit any
    // plain user content first so ordering matches the original message.
    if !user_parts.is_empty() {
        contents.push(json!({ "role": "user", "parts": user_parts }));
    }
    if !function_responses.is_empty() {
        contents.push(json!({ "role": "user", "parts": function_responses }));
    }
}

fn append_assistant_content(blocks: &[Value], contents: &mut Vec<Value>) {
    let mut parts: Vec<Value> = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(json!({ "text": text }));
                }
            }
            Some("tool_use") => {
                parts.push(json!({
                    "functionCall": {
                        "name": block.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "args": block.get("input").cloned().unwrap_or_else(|| json!({})),
                    }
                }));
            }
            _ => {}
        }
    }
    if !parts.is_empty() {
        contents.push(json!({ "role": "model", "parts": parts }));
    }
}

fn tool_config_from_choice(tool_choice: &Map<String, Value>) -> Option<Value> {
    let mode = match tool_choice.get("type").and_then(Value::as_str) {
        Some("auto") => "AUTO",
        Some("any") | Some("tool") => "ANY",
        Some("none") => "NONE",
        _ => return None,
    };
    let mut function_calling_config = json!({ "mode": mode });
    if mode == "ANY" {
        if let Some(name) = tool_choice.get("name").and_then(Value::as_str) {
            function_calling_config["allowedFunctionNames"] = json!([name]);
        }
    }
    Some(json!({ "functionCallingConfig": function_calling_config }))
}

/// Gemini rejects several JSON Schema keywords that Anthropic/OpenAI tool
/// schemas commonly include (`$schema`, `additionalProperties`, `$ref`,
/// etc.). Strip the most common offenders recursively so tool calls do not
/// 400. Unknown-but-harmless keys are left in place.
fn sanitize_schema(schema: Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut cleaned = Map::new();
            for (key, value) in map {
                if matches!(
                    key.as_str(),
                    "$schema" | "additionalProperties" | "$ref" | "$defs" | "definitions"
                ) {
                    continue;
                }
                // Gemini's Schema message represents every `enum` entry as
                // a string, even when the declared OpenAPI type is integer,
                // number, or boolean. Sending JSON numbers here makes Google
                // reject the entire tool manifest before inference (for
                // example, `enum: [4, 6, 8]`). Preserve the schema type while
                // encoding primitive enum values in the wire format Gemini
                // expects. Composite JSON values are outside Gemini's enum
                // contract, so omit them rather than forwarding a request
                // that is guaranteed to fail.
                if key == "enum" {
                    if let Value::Array(items) = value {
                        let normalized: Vec<Value> =
                            items.into_iter().filter_map(gemini_enum_value).collect();
                        if !normalized.is_empty() {
                            cleaned.insert(key, Value::Array(normalized));
                        }
                    }
                    continue;
                }
                cleaned.insert(key, sanitize_schema(value));
            }
            Value::Object(cleaned)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sanitize_schema).collect()),
        other => other,
    }
}

fn gemini_enum_value(value: Value) -> Option<Value> {
    match value {
        Value::String(_) => Some(value),
        Value::Number(number) => Some(Value::String(number.to_string())),
        Value::Bool(boolean) => Some(Value::String(boolean.to_string())),
        Value::Null => Some(Value::String("null".to_string())),
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn message_content_blocks(content: Option<&Value>) -> Result<Vec<Value>, String> {
    match content {
        Some(Value::Array(blocks)) => Ok(blocks.clone()),
        Some(Value::String(text)) => Ok(vec![json!({ "type": "text", "text": text })]),
        Some(_) => Err("Anthropic message content must be a string or content array".to_string()),
        None => Err("Anthropic message is missing content".to_string()),
    }
}

fn flatten_text_blocks(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_string(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
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
        Some(Value::Array(blocks)) => {
            let text = blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n");
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
    use super::*;

    #[test]
    fn translates_system_messages_and_generation_config() {
        let request = json!({
            "model": "aura-gemini-2-5-pro",
            "system": [{"type": "text", "text": "Be helpful"}],
            "max_tokens": 1024,
            "temperature": 0.5,
            "reasoning_effort": "high",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]}
            ]
        });

        let body = request_to_gemini("gemini-2.5-pro", &request).expect("translation");

        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be helpful");
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "Hello");
        assert_eq!(body["contents"][1]["role"], "model");
        assert_eq!(body["contents"][1]["parts"][0]["text"], "Hi there");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 1024);
        assert_eq!(body["generationConfig"]["temperature"], 0.5);
        assert_eq!(body["generationConfig"]["thinkingConfig"]["thinkingBudget"], 16_384);
        // Streaming is encoded in the URL, never the body.
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn translates_tools_and_tool_results_with_resolved_names() {
        let request = json!({
            "model": "aura-gemini-2-5-flash",
            "tools": [{
                "name": "search_repo",
                "description": "Search repos",
                "input_schema": {
                    "type": "object",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": false,
                    "properties": {"query": {"type": "string"}}
                }
            }],
            "tool_choice": {"type": "auto"},
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "find aura"}]},
                {"role": "assistant", "content": [{
                    "type": "tool_use", "id": "toolu_1", "name": "search_repo",
                    "input": {"query": "aura"}
                }]},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "toolu_1", "content": "found it"
                }]}
            ]
        });

        let body = request_to_gemini("gemini-2.5-flash", &request).expect("translation");

        // Tool declaration uses Gemini's functionDeclarations shape and the
        // schema is stripped of unsupported keywords.
        let decl = &body["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "search_repo");
        assert!(decl["parameters"].get("$schema").is_none());
        assert!(decl["parameters"].get("additionalProperties").is_none());
        assert_eq!(decl["parameters"]["properties"]["query"]["type"], "string");

        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["mode"],
            "AUTO"
        );

        // Assistant tool_use -> functionCall; tool_result -> functionResponse
        // with the name resolved from the prior tool_use.
        assert_eq!(
            body["contents"][1]["parts"][0]["functionCall"]["name"],
            "search_repo"
        );
        let fr = &body["contents"][2]["parts"][0]["functionResponse"];
        assert_eq!(fr["name"], "search_repo");
        assert_eq!(fr["response"]["result"], "found it");
    }

    #[test]
    fn normalizes_numeric_tool_schema_enums_for_gemini() {
        let request = json!({
            "model": "aura-gemini-3-5-flash",
            "tools": [{
                "name": "render_preview",
                "description": "Render a preview at the requested scale",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scale": {
                            "type": "integer",
                            "enum": [4, 6, 8]
                        },
                        "enabled": {
                            "type": "boolean",
                            "enum": [true, false]
                        }
                    }
                }
            }],
            "messages": [{"role": "user", "content": "Render it"}]
        });

        let body = request_to_gemini("gemini-3.5-flash", &request).expect("translation");
        let properties = &body["tools"][0]["functionDeclarations"][0]["parameters"]["properties"];

        assert_eq!(properties["scale"]["type"], "integer");
        assert_eq!(properties["scale"]["enum"], json!(["4", "6", "8"]));
        assert_eq!(properties["enabled"]["type"], "boolean");
        assert_eq!(properties["enabled"]["enum"], json!(["true", "false"]));
    }

    #[test]
    fn maps_gemini_response_text_and_function_calls() {
        let response = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Let me search"},
                        {"functionCall": {"name": "search_repo", "args": {"query": "aura"}}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 42,
                "candidatesTokenCount": 7,
                "thoughtsTokenCount": 5,
                "cachedContentTokenCount": 10
            }
        });

        let normalized =
            response_from_gemini("aura-gemini-2-5-pro", &response).expect("normalize");

        assert_eq!(normalized["model"], "aura-gemini-2-5-pro");
        assert_eq!(normalized["stop_reason"], "tool_use");
        assert_eq!(normalized["content"][0]["type"], "text");
        assert_eq!(normalized["content"][0]["text"], "Let me search");
        assert_eq!(normalized["content"][1]["type"], "tool_use");
        assert_eq!(normalized["content"][1]["name"], "search_repo");
        assert_eq!(normalized["content"][1]["input"]["query"], "aura");
        assert_eq!(normalized["usage"]["input_tokens"], 42);
        // candidates + thoughts tokens fold into output.
        assert_eq!(normalized["usage"]["output_tokens"], 12);
        assert_eq!(normalized["usage"]["cache_read_input_tokens"], 10);
    }

    #[test]
    fn maps_max_tokens_finish_reason() {
        let response = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "partial"}]},
                "finishReason": "MAX_TOKENS"
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2}
        });
        let normalized = response_from_gemini("gemini-2.5-flash", &response).expect("normalize");
        assert_eq!(normalized["stop_reason"], "max_tokens");
        assert_eq!(normalized["usage"]["output_tokens"], 2);
    }
}
