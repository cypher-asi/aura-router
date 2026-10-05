//! Managed Kimi must accept the reasoning history emitted by its own adapter.
use super::{request_to_upstream, response_from_upstream, validate_request};
use crate::providers::{resolve_model, Provider};
use serde_json::{json, Value};

fn replay_request(with_tool: bool) -> Value {
    let mut message = json!({
        "role": "assistant",
        "reasoning_content": "I should check the file first.",
        "content": "I will check the file."
    });
    if with_tool {
        message["tool_calls"] = json!([{
            "id": "read_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{\"path\":\"example.txt\"}"}
        }]);
    }
    let response = json!({
        "id": "kimi_response",
        "choices": [{
            "finish_reason": if with_tool { "tool_calls" } else { "stop" },
            "message": message
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 40}
    });
    let resolved = resolve_model("aura-kimi-k3").expect("managed Kimi route");
    assert_eq!(resolved.provider, Provider::Fireworks);
    let reply = response_from_upstream(resolved.provider, "aura-kimi-k3", &response)
        .expect("translate Kimi response");
    assert_eq!(reply["content"][0]["type"], "thinking");
    assert_eq!(reply["content"][0]["signature"], "moonshot");

    json!({
        "model": "aura-kimi-k3",
        "max_tokens": 16384,
        "reasoning_effort": "max",
        "messages": [
            {"role": "user", "content": "Check the file."},
            {"role": "assistant", "content": reply["content"]},
            {"role": "user", "content": if with_tool {
                json!([{"type": "tool_result", "tool_use_id": "read_1", "content": "File contents"}])
            } else {
                json!("Please continue.")
            }}
        ]
    })
}

#[test]
fn managed_kimi_replays_its_own_reasoning_on_followup() {
    let request = replay_request(false);
    let resolved = resolve_model("aura-kimi-k3").unwrap();
    validate_request(resolved.provider, &request).expect("accept Kimi's own history");
    let upstream = request_to_upstream(resolved.provider, resolved.upstream_model, &request)
        .expect("translate follow-up");
    assert_eq!(upstream["reasoning_effort"], "max");
    assert_eq!(
        upstream["messages"][1]["reasoning_content"],
        "I should check the file first."
    );
    assert_eq!(upstream["messages"][1]["content"], "I will check the file.");
    assert_eq!(upstream["messages"][2]["content"], "Please continue.");
    assert!(upstream.get("thinking").is_none());
}

#[test]
fn managed_kimi_replays_reasoning_with_completed_tool_results() {
    let request = replay_request(true);
    let resolved = resolve_model("aura-kimi-k3").unwrap();
    validate_request(resolved.provider, &request).expect("accept Kimi tool history");
    let upstream = request_to_upstream(resolved.provider, resolved.upstream_model, &request)
        .expect("translate tool continuation");
    assert_eq!(upstream["messages"].as_array().unwrap().len(), 3);
    assert_eq!(
        upstream["messages"][1]["reasoning_content"],
        "I should check the file first."
    );
    let tool = &upstream["messages"][1]["tool_calls"][0];
    assert_eq!(tool["id"], "read_1");
    assert_eq!(tool["function"]["name"], "read_file");
    assert_eq!(
        serde_json::from_str::<Value>(tool["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"path": "example.txt"})
    );
    assert_eq!(upstream["messages"][2]["role"], "tool");
    assert_eq!(upstream["messages"][2]["tool_call_id"], "read_1");
    assert_eq!(upstream["messages"][2]["content"], "File contents");
}

#[test]
fn kimi_exception_does_not_enable_thinking_for_other_routes() {
    for (provider, model, upstream_model) in [
        (
            Provider::Fireworks,
            "aura-kimi-k2-6",
            "accounts/fireworks/models/kimi-k2p6",
        ),
        (
            Provider::Fireworks,
            "aura-gpt-oss-120b",
            "accounts/fireworks/models/gpt-oss-120b",
        ),
        (Provider::OpenAi, "aura-gpt-5-5", "gpt-5.5"),
        (Provider::Xai, "aura-grok-4-7", "grok-4.7"),
        (Provider::DeepSeek, "deepseek-chat", "deepseek-chat"),
        (Provider::Google, "aura-gemini-3-pro", "gemini-3-pro"),
    ] {
        let mut request = replay_request(false);
        request["model"] = json!(model);
        for error in [
            validate_request(provider, &request).expect_err("other routes stay strict"),
            request_to_upstream(provider, upstream_model, &request)
                .expect_err("translation stays strict"),
        ] {
            assert!(
                error.contains("do not support `thinking` blocks"),
                "{model}: {error}"
            );
        }
    }
}

#[test]
fn managed_kimi_keeps_fireworks_privacy_guards() {
    let resolved = resolve_model("aura-kimi-k3").unwrap();
    for (key, value, expected) in [
        ("store", json!(true), "persistence controls"),
        ("response_id", json!("resp_123"), "stateless inference"),
    ] {
        let mut request = replay_request(true);
        request[key] = value;
        for error in [
            validate_request(resolved.provider, &request).expect_err("privacy guard"),
            request_to_upstream(resolved.provider, resolved.upstream_model, &request)
                .expect_err("translation privacy guard"),
        ] {
            assert!(error.contains(expected), "{key}: {error}");
        }
    }
}

#[test]
fn managed_kimi_still_rejects_anthropic_thinking_configuration() {
    let resolved = resolve_model("aura-kimi-k3").unwrap();
    let mut request = replay_request(false);
    request["thinking"] = json!({"type": "adaptive"});
    for error in [
        validate_request(resolved.provider, &request)
            .expect_err("native Anthropic config is unsupported"),
        request_to_upstream(resolved.provider, resolved.upstream_model, &request)
            .expect_err("translation rejects config"),
    ] {
        assert!(error.contains("do not support Anthropic thinking"));
    }
}

#[test]
fn managed_kimi_rejects_foreign_or_unsigned_reasoning_history() {
    let resolved = resolve_model("aura-kimi-k3").unwrap();
    for signature in [json!("anthropic-signed-thinking"), Value::Null] {
        let mut request = replay_request(false);
        request["messages"][1]["content"][0]["signature"] = signature;
        for error in [
            validate_request(resolved.provider, &request).expect_err("foreign reasoning"),
            request_to_upstream(resolved.provider, resolved.upstream_model, &request)
                .expect_err("translation rejects foreign reasoning"),
        ] {
            assert!(error.contains("only support replaying Kimi reasoning blocks"));
        }
    }
}

#[test]
fn managed_kimi_rejects_thinking_in_user_messages() {
    let resolved = resolve_model("aura-kimi-k3").unwrap();
    let mut request = replay_request(false);
    request["messages"][0]["content"] = json!([{
        "type": "thinking", "thinking": "Not assistant reasoning", "signature": "moonshot"
    }]);
    for error in [
        validate_request(resolved.provider, &request).expect_err("user thinking is unsupported"),
        request_to_upstream(resolved.provider, resolved.upstream_model, &request)
            .expect_err("translation rejects user thinking"),
    ] {
        assert!(error.contains("do not support `thinking` blocks for `user`"));
    }
}
