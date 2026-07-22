//! Provider-aware SSE proxy with billing capture.
//!
//! Anthropic streams are passed through as-is while we parse usage metadata.
//! OpenAI chat-completion streams are translated into Anthropic-style `/v1/messages`
//! events so Aura clients can keep speaking one hosted protocol.

use std::collections::{BTreeMap, VecDeque};

use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::{
    billing,
    providers::{self, OpenAiApi, Provider},
};

/// Token usage extracted from an SSE stream.
#[derive(Debug, Clone, Default)]
pub struct StreamUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_creation_5m_input_tokens: u64,
    pub cache_creation_1h_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub web_search_requests: u64,
    pub web_fetch_requests: u64,
    pub code_execution_requests: u64,
    pub service_tier: Option<String>,
    pub inference_geo: Option<String>,
    pub speed: Option<String>,
    pub provider_reported_cost_cents: Option<i64>,
    pub provider_reported_cost_microusd: Option<i64>,
    pub model: Option<String>,
    /// True only after the upstream protocol emitted a terminal success
    /// marker. Observed usage from interrupted streams is still billable,
    /// but those streams must not be persisted as completed assistant turns.
    pub completed: bool,
}

/// Proxy an SSE stream from the upstream provider to the client,
/// capturing billing data along the way.
pub fn proxy_stream(
    provider: Provider,
    api: OpenAiApi,
    requested_model: &str,
    upstream: reqwest::Response,
    provider_request_id: String,
) -> (
    impl futures_util::Stream<Item = Result<Bytes, std::convert::Infallible>>,
    oneshot::Receiver<StreamUsage>,
) {
    let (usage_tx, usage_rx) = oneshot::channel();
    let byte_stream = upstream.bytes_stream();

    let tee_stream = TeeStream {
        inner: Box::pin(byte_stream),
        adapter: StreamAdapter::new(provider, api, requested_model),
        usage_tx: Some(usage_tx),
        finished: false,
        pending_output: VecDeque::new(),
        provider_request_id,
    };

    (tee_stream, usage_rx)
}

/// A stream that either forwards raw Anthropic SSE bytes or emits translated
/// Anthropic-compatible SSE for OpenAI-backed models.
struct TeeStream<S> {
    inner: std::pin::Pin<Box<S>>,
    adapter: StreamAdapter,
    usage_tx: Option<oneshot::Sender<StreamUsage>>,
    finished: bool,
    pending_output: VecDeque<Bytes>,
    provider_request_id: String,
}

impl<S, E> futures_util::Stream for TeeStream<S>
where
    S: futures_util::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    type Item = Result<Bytes, std::convert::Infallible>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if let Some(bytes) = self.pending_output.pop_front() {
            return std::task::Poll::Ready(Some(Ok(bytes)));
        }

        if self.finished {
            return std::task::Poll::Ready(None);
        }

        loop {
            match self.inner.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    let mut emitted = VecDeque::new();
                    self.adapter.feed(bytes, &mut emitted);
                    self.pending_output.extend(emitted);
                    if let Some(next) = self.pending_output.pop_front() {
                        return std::task::Poll::Ready(Some(Ok(next)));
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    let terminal_error_emitted = self.adapter.terminal_error_emitted();
                    let usage = self.adapter.abort();
                    if let Some(tx) = self.usage_tx.take() {
                        let _ = tx.send(usage.clone());
                    }
                    // Some providers close the transport noisily after a
                    // protocol-level completion marker. Do not turn that
                    // into a retry that would duplicate a completed turn.
                    if usage.completed || terminal_error_emitted {
                        return std::task::Poll::Ready(None);
                    }
                    tracing::warn!(
                        error = %e,
                        provider_request_id = %self.provider_request_id,
                        "provider SSE transport closed before message completion"
                    );
                    return std::task::Poll::Ready(Some(Ok(interrupted_stream_error(
                        &self.provider_request_id,
                    ))));
                }
                std::task::Poll::Ready(None) => {
                    self.finished = true;
                    let mut emitted = VecDeque::new();
                    let terminal_error_emitted = self.adapter.terminal_error_emitted();
                    let usage = self.adapter.finish(&mut emitted);
                    self.pending_output.extend(emitted);
                    if usage.completed {
                        let model = usage.model.as_deref().unwrap_or("unknown");
                        let max_tokens = providers::max_context_tokens(model);
                        let context_usage = if max_tokens > 0 {
                            usage.input_tokens as f64 / max_tokens as f64
                        } else {
                            0.0
                        };

                        self.pending_output.push_back(Bytes::from(format!(
                            "event: x_context_usage\ndata: {{\"contextUsage\":{context_usage:.4},\"inputTokens\":{},\"outputTokens\":{},\"maxTokens\":{max_tokens}}}\n\n",
                            usage.input_tokens, usage.output_tokens
                        )));
                    } else if !terminal_error_emitted {
                        let provider_request_id = self.provider_request_id.clone();
                        self.pending_output
                            .push_back(interrupted_stream_error(&provider_request_id));
                    }

                    if let Some(tx) = self.usage_tx.take() {
                        let _ = tx.send(usage);
                    }

                    if let Some(next) = self.pending_output.pop_front() {
                        return std::task::Poll::Ready(Some(Ok(next)));
                    }

                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => {
                    if let Some(next) = self.pending_output.pop_front() {
                        return std::task::Poll::Ready(Some(Ok(next)));
                    }
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

enum StreamAdapter {
    Anthropic(AnthropicPassthrough),
    OpenAi(OpenAiCompatStream),
    OpenAiResponses(OpenAiResponsesStream),
    Google(GoogleStream),
}

impl StreamAdapter {
    fn new(provider: Provider, api: OpenAiApi, requested_model: &str) -> Self {
        match provider {
            Provider::Anthropic => Self::Anthropic(AnthropicPassthrough {
                parser: SseParser::new(),
            }),
            Provider::OpenAi | Provider::Xai if api == OpenAiApi::Responses => {
                Self::OpenAiResponses(OpenAiResponsesStream::new(requested_model))
            }
            Provider::OpenAi | Provider::Xai | Provider::Fireworks | Provider::DeepSeek => {
                Self::OpenAi(OpenAiCompatStream::new(requested_model))
            }
            Provider::Google => Self::Google(GoogleStream::new(requested_model)),
        }
    }

    fn feed(&mut self, bytes: Bytes, output: &mut VecDeque<Bytes>) {
        match self {
            Self::Anthropic(adapter) => adapter.feed(bytes, output),
            Self::OpenAi(adapter) => adapter.feed(bytes, output),
            Self::OpenAiResponses(adapter) => adapter.feed(bytes, output),
            Self::Google(adapter) => adapter.feed(bytes, output),
        }
    }

    fn finish(&mut self, output: &mut VecDeque<Bytes>) -> StreamUsage {
        match self {
            Self::Anthropic(adapter) => adapter.finish(output),
            Self::OpenAi(adapter) => adapter.finish(output),
            Self::OpenAiResponses(adapter) => adapter.finish(output),
            Self::Google(adapter) => adapter.finish(output),
        }
    }

    /// Return usage observed before an interrupted stream without emitting
    /// synthetic success frames such as `message_stop`.
    fn abort(&mut self) -> StreamUsage {
        match self {
            Self::Anthropic(adapter) => adapter.parser.finalize(),
            Self::OpenAi(adapter) => {
                let mut usage = adapter.usage.clone();
                usage.completed = adapter.finalized && !adapter.terminal_error;
                usage
            }
            Self::OpenAiResponses(adapter) => {
                let mut usage = adapter.usage.clone();
                usage.completed = adapter.finalized && !adapter.terminal_error;
                usage
            }
            Self::Google(adapter) => {
                let mut usage = adapter.usage.clone();
                usage.completed = adapter.finalized && !adapter.terminal_error;
                usage
            }
        }
    }

    fn terminal_error_emitted(&self) -> bool {
        match self {
            Self::Anthropic(adapter) => adapter.parser.terminal_error,
            Self::OpenAi(adapter) => adapter.terminal_error,
            Self::OpenAiResponses(adapter) => adapter.terminal_error,
            Self::Google(adapter) => adapter.terminal_error,
        }
    }
}

struct AnthropicPassthrough {
    parser: SseParser,
}

impl AnthropicPassthrough {
    fn feed(&mut self, bytes: Bytes, output: &mut VecDeque<Bytes>) {
        self.parser.feed(&bytes);
        output.push_back(bytes);
    }

    fn finish(&mut self, _output: &mut VecDeque<Bytes>) -> StreamUsage {
        self.parser.finalize()
    }
}

struct OpenAiCompatStream {
    requested_model: String,
    buffer: String,
    usage: StreamUsage,
    message_id: Option<String>,
    message_started: bool,
    text_block_index: Option<usize>,
    text_block_open: bool,
    tool_blocks: BTreeMap<usize, ToolBlockState>,
    pending_stop_reason: Option<String>,
    finalized: bool,
    terminal_error: bool,
}

#[derive(Default)]
struct ToolBlockState {
    content_index: usize,
    id: String,
    name: String,
    started: bool,
    pending_json: String,
    stopped: bool,
}

impl OpenAiCompatStream {
    fn new(requested_model: &str) -> Self {
        Self {
            requested_model: requested_model.to_string(),
            buffer: String::new(),
            usage: StreamUsage {
                model: Some(requested_model.to_string()),
                ..StreamUsage::default()
            },
            message_id: None,
            message_started: false,
            text_block_index: None,
            text_block_open: false,
            tool_blocks: BTreeMap::new(),
            pending_stop_reason: None,
            finalized: false,
            terminal_error: false,
        }
    }

    fn feed(&mut self, bytes: Bytes, output: &mut VecDeque<Bytes>) {
        let text = String::from_utf8_lossy(&bytes);
        self.buffer.push_str(&text);

        while let Some((event, delimiter_len)) = next_sse_event(&self.buffer) {
            let event_str = self.buffer[..event].to_string();
            self.buffer = self.buffer[event + delimiter_len..].to_string();
            self.process_event(&event_str, output);
        }
    }

    fn finish(&mut self, output: &mut VecDeque<Bytes>) -> StreamUsage {
        if self.pending_stop_reason.is_some() {
            self.finish_message(output);
        }
        let mut usage = self.usage.clone();
        usage.completed = self.finalized && !self.terminal_error;
        usage
    }

    fn process_event(&mut self, event_str: &str, output: &mut VecDeque<Bytes>) {
        let mut data_lines = Vec::new();

        for line in event_str.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data: ") {
                data_lines.push(data.to_string());
            } else if let Some(data) = line.strip_prefix("data:") {
                data_lines.push(data.trim().to_string());
            }
        }

        if data_lines.is_empty() {
            return;
        }

        let data = data_lines.join("\n");
        if data == "[DONE]" {
            self.finish_message(output);
            return;
        }

        let Ok(chunk) = serde_json::from_str::<Value>(&data) else {
            output.push_back(anthropic_sse(
                "error",
                json!({
                    "type": "error",
                    "error": {
                        "message": "malformed OpenAI SSE JSON"
                    }
                }),
            ));
            self.terminal_error = true;
            self.finalized = true;
            return;
        };

        if let Some(id) = chunk.get("id").and_then(Value::as_str) {
            self.message_id.get_or_insert_with(|| id.to_string());
        }

        if let Some(model) = chunk.get("model").and_then(Value::as_str) {
            self.usage.model = Some(model.to_string());
        }

        if let Some(usage) = chunk.get("usage").and_then(Value::as_object) {
            self.usage.input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.input_tokens);
            self.usage.output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.output_tokens);
            self.usage.cache_creation_input_tokens = usage
                .get("prompt_cache_miss_tokens")
                .and_then(Value::as_u64)
                .or_else(|| {
                    usage
                        .get("prompt_tokens_details")
                        .and_then(|details| details.get("cache_write_tokens"))
                        .and_then(Value::as_u64)
                })
                .or_else(|| {
                    usage
                        .get("cache_creation_input_tokens")
                        .and_then(Value::as_u64)
                })
                .unwrap_or(self.usage.cache_creation_input_tokens);
            self.usage.cache_read_input_tokens = usage
                .get("prompt_cache_hit_tokens")
                .and_then(Value::as_u64)
                .or_else(|| {
                    usage
                        .get("prompt_tokens_details")
                        .and_then(|details| details.get("cached_tokens"))
                        .and_then(Value::as_u64)
                })
                .or_else(|| usage.get("cache_read_input_tokens").and_then(Value::as_u64))
                .unwrap_or(self.usage.cache_read_input_tokens);
            if let Some(ticks) = usage.get("cost_in_usd_ticks").and_then(Value::as_u64) {
                self.usage.provider_reported_cost_cents =
                    billing::marked_up_cost_cents_from_usd_ticks(ticks);
                self.usage.provider_reported_cost_microusd =
                    billing::provider_cost_microusd_from_usd_ticks(ticks);
            }
            if self.pending_stop_reason.is_some() {
                self.finish_message(output);
            }
        }

        let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
            return;
        };

        if !choices.is_empty() {
            self.ensure_message_started(output);
        }

        for choice in choices {
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                if let Some(content) = delta.get("content").and_then(Value::as_str) {
                    if !content.is_empty() {
                        self.ensure_text_block_started(output);
                        if let Some(index) = self.text_block_index {
                            output.push_back(anthropic_sse(
                                "content_block_delta",
                                json!({
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": content,
                                    }
                                }),
                            ));
                        }
                    }
                }

                if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for tool_call in tool_calls {
                        self.process_tool_call_delta(tool_call, output);
                    }
                }
            }

            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.pending_stop_reason = Some(finish_reason.to_string());
                if self.usage.output_tokens > 0 || finish_reason == "tool_calls" {
                    self.finish_message(output);
                }
            }
        }
    }

    fn process_tool_call_delta(&mut self, tool_call: &Value, output: &mut VecDeque<Bytes>) {
        let tool_index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        let content_index = self.tool_content_index(tool_index);
        let state = self
            .tool_blocks
            .entry(tool_index)
            .or_insert_with(|| ToolBlockState {
                content_index,
                ..ToolBlockState::default()
            });

        if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
            state.id = id.to_string();
        }

        if let Some(name) = tool_call.pointer("/function/name").and_then(Value::as_str) {
            if state.name.is_empty() {
                state.name = name.to_string();
            } else if !name.is_empty() && !state.name.ends_with(name) {
                state.name.push_str(name);
            }
        }

        if !state.started && !state.id.is_empty() && !state.name.is_empty() {
            output.push_back(anthropic_sse(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": state.content_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": state.id,
                        "name": state.name,
                        "input": {},
                    }
                }),
            ));
            state.started = true;

            if !state.pending_json.is_empty() {
                output.push_back(anthropic_sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": state.content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": state.pending_json,
                        }
                    }),
                ));
                state.pending_json.clear();
            }
        }

        if let Some(arguments) = tool_call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
        {
            if state.started {
                output.push_back(anthropic_sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": state.content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        }
                    }),
                ));
            } else {
                state.pending_json.push_str(arguments);
            }
        }
    }

    fn ensure_message_started(&mut self, output: &mut VecDeque<Bytes>) {
        if self.message_started {
            return;
        }
        self.message_started = true;

        let message_id = self
            .message_id
            .clone()
            .unwrap_or_else(|| format!("msg_{}", Uuid::new_v4().simple()));

        output.push_back(anthropic_sse(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.requested_model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {
                        "input_tokens": self.usage.input_tokens,
                    }
                }
            }),
        ));
    }

    fn ensure_text_block_started(&mut self, output: &mut VecDeque<Bytes>) {
        if self.text_block_open {
            return;
        }

        let index = if let Some(index) = self.text_block_index {
            index
        } else if self.tool_blocks.is_empty() {
            0
        } else {
            self.tool_blocks
                .values()
                .map(|state| state.content_index)
                .max()
                .unwrap_or(0)
                + 1
        };

        self.text_block_index = Some(index);
        self.text_block_open = true;
        output.push_back(anthropic_sse(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "text",
                    "text": "",
                }
            }),
        ));
    }

    fn tool_content_index(&self, tool_index: usize) -> usize {
        if self.text_block_index == Some(0) {
            tool_index + 1
        } else {
            tool_index
        }
    }

    fn finish_message(&mut self, output: &mut VecDeque<Bytes>) {
        if self.finalized || !self.message_started {
            return;
        }

        if self.text_block_open {
            if let Some(index) = self.text_block_index {
                output.push_back(anthropic_sse(
                    "content_block_stop",
                    json!({
                        "type": "content_block_stop",
                        "index": index,
                    }),
                ));
            }
            self.text_block_open = false;
        }

        for state in self.tool_blocks.values_mut() {
            if state.started && !state.stopped {
                output.push_back(anthropic_sse(
                    "content_block_stop",
                    json!({
                        "type": "content_block_stop",
                        "index": state.content_index,
                    }),
                ));
                state.stopped = true;
            }
        }

        output.push_back(anthropic_sse(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": map_openai_finish_reason(self.pending_stop_reason.as_deref()),
                    "stop_sequence": Value::Null,
                },
                "usage": {
                    "output_tokens": self.usage.output_tokens,
                }
            }),
        ));
        output.push_back(anthropic_sse(
            "message_stop",
            json!({
                "type": "message_stop",
            }),
        ));
        self.finalized = true;
    }
}

/// Translates an OpenAI **Responses API** (`/v1/responses`) SSE stream into
/// Anthropic-style `/v1/messages` events.
///
/// The Responses stream is a sequence of typed events
/// (`response.output_item.added`, `response.output_text.delta`,
/// `response.function_call_arguments.delta`, `response.output_item.done`,
/// `response.completed`, ...). We map them onto the same Anthropic content
/// block lifecycle the chat-completions adapter emits, assigning a
/// contiguous Anthropic `index` per emitted block (reasoning items are
/// skipped, so they never occupy a client-visible index).
struct OpenAiResponsesStream {
    requested_model: String,
    buffer: String,
    usage: StreamUsage,
    message_id: Option<String>,
    message_started: bool,
    blocks: BTreeMap<usize, ResponsesBlock>,
    next_index: usize,
    saw_tool_call: bool,
    incomplete_max_tokens: bool,
    finalized: bool,
    terminal_error: bool,
}

#[derive(Default)]
struct ResponsesBlock {
    anthropic_index: usize,
    is_tool: bool,
    started: bool,
    stopped: bool,
    tool_id: String,
    tool_name: String,
    pending_args: String,
}

impl OpenAiResponsesStream {
    fn new(requested_model: &str) -> Self {
        Self {
            requested_model: requested_model.to_string(),
            buffer: String::new(),
            usage: StreamUsage {
                model: Some(requested_model.to_string()),
                ..StreamUsage::default()
            },
            message_id: None,
            message_started: false,
            blocks: BTreeMap::new(),
            next_index: 0,
            saw_tool_call: false,
            incomplete_max_tokens: false,
            finalized: false,
            terminal_error: false,
        }
    }

    fn feed(&mut self, bytes: Bytes, output: &mut VecDeque<Bytes>) {
        let text = String::from_utf8_lossy(&bytes);
        self.buffer.push_str(&text);

        while let Some((event, delimiter_len)) = next_sse_event(&self.buffer) {
            let event_str = self.buffer[..event].to_string();
            self.buffer = self.buffer[event + delimiter_len..].to_string();
            self.process_event(&event_str, output);
        }
    }

    fn finish(&mut self, _output: &mut VecDeque<Bytes>) -> StreamUsage {
        let mut usage = self.usage.clone();
        usage.completed = self.finalized && !self.terminal_error;
        usage
    }

    fn process_event(&mut self, event_str: &str, output: &mut VecDeque<Bytes>) {
        // Responses streams carry both `event:` and `data:` lines; the data
        // JSON repeats the event name in its `type` field, so we parse the
        // JSON and branch on `type` (ignoring the `event:` line).
        let mut data_lines = Vec::new();
        for line in event_str.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data: ") {
                data_lines.push(data.to_string());
            } else if let Some(data) = line.strip_prefix("data:") {
                data_lines.push(data.trim().to_string());
            }
        }
        if data_lines.is_empty() {
            return;
        }
        let data = data_lines.join("\n");
        if data == "[DONE]" {
            self.finish_message(output);
            return;
        }
        let Ok(event) = serde_json::from_str::<Value>(&data) else {
            output.push_back(anthropic_sse(
                "error",
                json!({
                    "type": "error",
                    "error": { "message": "malformed OpenAI Responses SSE JSON" }
                }),
            ));
            self.terminal_error = true;
            self.finalized = true;
            return;
        };

        match event.get("type").and_then(Value::as_str) {
            Some("response.created") | Some("response.in_progress") => {
                if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                    self.message_id.get_or_insert_with(|| id.to_string());
                }
                if let Some(model) = event.pointer("/response/model").and_then(Value::as_str) {
                    self.usage.model = Some(model.to_string());
                }
            }
            Some("response.output_item.added") => {
                self.on_output_item_added(&event, output);
            }
            Some("response.output_text.delta") => {
                self.on_output_text_delta(&event, output);
            }
            Some("response.function_call_arguments.delta") => {
                self.on_function_args_delta(&event, output);
            }
            Some("response.output_item.done") => {
                self.on_output_item_done(&event, output);
            }
            Some("response.completed") | Some("response.incomplete") => {
                self.capture_response_usage(&event);
                self.finish_message(output);
            }
            Some("response.failed") | Some("error") => {
                let message = event
                    .pointer("/response/error/message")
                    .and_then(Value::as_str)
                    .or_else(|| event.pointer("/error/message").and_then(Value::as_str))
                    .or_else(|| event.get("message").and_then(Value::as_str))
                    .unwrap_or("OpenAI Responses stream error");
                output.push_back(anthropic_sse(
                    "error",
                    json!({
                        "type": "error",
                        "error": { "message": message }
                    }),
                ));
                self.terminal_error = true;
                self.finalized = true;
            }
            _ => {}
        }
    }

    fn on_output_item_added(&mut self, event: &Value, output: &mut VecDeque<Bytes>) {
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return;
        };
        let output_index = output_index as usize;
        let item = event.get("item").unwrap_or(&Value::Null);
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                self.ensure_message_started(output);
                self.saw_tool_call = true;
                let anthropic_index = self.next_index;
                self.next_index += 1;
                let tool_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("id").and_then(Value::as_str))
                    .unwrap_or_default()
                    .to_string();
                let tool_name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let mut block = ResponsesBlock {
                    anthropic_index,
                    is_tool: true,
                    tool_id,
                    tool_name,
                    ..ResponsesBlock::default()
                };
                if !block.tool_name.is_empty() {
                    Self::emit_tool_start(&block, output);
                    block.started = true;
                }
                self.blocks.insert(output_index, block);
            }
            // Text (`message`) and `reasoning` items defer: text blocks start
            // on the first text delta; reasoning items are dropped.
            _ => {}
        }
    }

    fn on_output_text_delta(&mut self, event: &Value, output: &mut VecDeque<Bytes>) {
        let Some(delta) = event.get("delta").and_then(Value::as_str) else {
            return;
        };
        if delta.is_empty() {
            return;
        }
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return;
        };
        let output_index = output_index as usize;
        self.ensure_message_started(output);

        if !self.blocks.contains_key(&output_index) {
            let anthropic_index = self.next_index;
            self.next_index += 1;
            self.blocks.insert(
                output_index,
                ResponsesBlock {
                    anthropic_index,
                    is_tool: false,
                    started: true,
                    ..ResponsesBlock::default()
                },
            );
            output.push_back(anthropic_sse(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": anthropic_index,
                    "content_block": { "type": "text", "text": "" }
                }),
            ));
        }

        let index = self.blocks.get(&output_index).map(|b| b.anthropic_index);
        if let Some(index) = index {
            output.push_back(anthropic_sse(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "text_delta", "text": delta }
                }),
            ));
        }
    }

    fn on_function_args_delta(&mut self, event: &Value, output: &mut VecDeque<Bytes>) {
        let Some(delta) = event.get("delta").and_then(Value::as_str) else {
            return;
        };
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return;
        };
        let output_index = output_index as usize;
        if let Some(block) = self.blocks.get_mut(&output_index) {
            if block.started {
                let index = block.anthropic_index;
                output.push_back(anthropic_sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "input_json_delta", "partial_json": delta }
                    }),
                ));
            } else {
                block.pending_args.push_str(delta);
            }
        }
    }

    fn on_output_item_done(&mut self, event: &Value, output: &mut VecDeque<Bytes>) {
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return;
        };
        let output_index = output_index as usize;
        let item_name = event
            .pointer("/item/name")
            .and_then(Value::as_str)
            .map(str::to_string);
        let Some(block) = self.blocks.get_mut(&output_index) else {
            return;
        };

        // A tool block whose name only arrived on the `done` item: open it now
        // and flush any buffered argument fragments before closing.
        if block.is_tool && !block.started {
            if let Some(name) = item_name {
                if block.tool_name.is_empty() {
                    block.tool_name = name;
                }
            }
            Self::emit_tool_start(block, output);
            block.started = true;
            if !block.pending_args.is_empty() {
                let index = block.anthropic_index;
                let pending = std::mem::take(&mut block.pending_args);
                output.push_back(anthropic_sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "input_json_delta", "partial_json": pending }
                    }),
                ));
            }
        }

        if block.started && !block.stopped {
            let index = block.anthropic_index;
            output.push_back(anthropic_sse(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": index }),
            ));
            block.stopped = true;
        }
    }

    fn capture_response_usage(&mut self, event: &Value) {
        if let Some(model) = event.pointer("/response/model").and_then(Value::as_str) {
            self.usage.model = Some(model.to_string());
        }
        if let Some(usage) = event.pointer("/response/usage") {
            self.usage.input_tokens = usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.input_tokens);
            self.usage.output_tokens = usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.output_tokens);
            self.usage.cache_creation_input_tokens = usage
                .pointer("/input_tokens_details/cache_write_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.cache_creation_input_tokens);
            self.usage.cache_read_input_tokens = usage
                .pointer("/input_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.cache_read_input_tokens);
            if let Some(ticks) = usage.get("cost_in_usd_ticks").and_then(Value::as_u64) {
                self.usage.provider_reported_cost_cents =
                    billing::marked_up_cost_cents_from_usd_ticks(ticks);
                self.usage.provider_reported_cost_microusd =
                    billing::provider_cost_microusd_from_usd_ticks(ticks);
            }
        }
        if event
            .pointer("/response/incomplete_details/reason")
            .and_then(Value::as_str)
            == Some("max_output_tokens")
        {
            self.incomplete_max_tokens = true;
        }
    }

    fn emit_tool_start(block: &ResponsesBlock, output: &mut VecDeque<Bytes>) {
        output.push_back(anthropic_sse(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": block.anthropic_index,
                "content_block": {
                    "type": "tool_use",
                    "id": block.tool_id,
                    "name": block.tool_name,
                    "input": {},
                }
            }),
        ));
    }

    fn ensure_message_started(&mut self, output: &mut VecDeque<Bytes>) {
        if self.message_started {
            return;
        }
        self.message_started = true;
        let message_id = self
            .message_id
            .clone()
            .unwrap_or_else(|| format!("msg_{}", Uuid::new_v4().simple()));
        output.push_back(anthropic_sse(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.requested_model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": { "input_tokens": self.usage.input_tokens }
                }
            }),
        ));
    }

    fn finish_message(&mut self, output: &mut VecDeque<Bytes>) {
        if self.finalized || !self.message_started {
            return;
        }

        for block in self.blocks.values_mut() {
            if block.started && !block.stopped {
                output.push_back(anthropic_sse(
                    "content_block_stop",
                    json!({ "type": "content_block_stop", "index": block.anthropic_index }),
                ));
                block.stopped = true;
            }
        }

        let stop_reason = if self.saw_tool_call {
            "tool_use"
        } else if self.incomplete_max_tokens {
            "max_tokens"
        } else {
            "end_turn"
        };

        output.push_back(anthropic_sse(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": Value::Null,
                },
                "usage": { "output_tokens": self.usage.output_tokens }
            }),
        ));
        output.push_back(anthropic_sse(
            "message_stop",
            json!({ "type": "message_stop" }),
        ));
        self.finalized = true;
    }
}

/// Translates a Google Gemini `streamGenerateContent?alt=sse` stream into
/// Anthropic-style `/v1/messages` events.
///
/// Each Gemini SSE `data:` line carries a full `GenerateContentResponse`
/// chunk: `candidates[0].content.parts` holds incremental `text` (and, for
/// tool use, a complete `functionCall`), `usageMetadata` is cumulative, and
/// `finishReason` lands on the final chunk. We map text onto a single
/// Anthropic text block and each function call onto its own `tool_use`
/// block.
struct GoogleStream {
    requested_model: String,
    buffer: String,
    usage: StreamUsage,
    message_started: bool,
    text_block_open: bool,
    text_block_index: Option<usize>,
    next_index: usize,
    saw_tool_call: bool,
    pending_finish_reason: Option<String>,
    finalized: bool,
    terminal_error: bool,
}

impl GoogleStream {
    fn new(requested_model: &str) -> Self {
        Self {
            requested_model: requested_model.to_string(),
            buffer: String::new(),
            usage: StreamUsage {
                model: Some(requested_model.to_string()),
                ..StreamUsage::default()
            },
            message_started: false,
            text_block_open: false,
            text_block_index: None,
            next_index: 0,
            saw_tool_call: false,
            pending_finish_reason: None,
            finalized: false,
            terminal_error: false,
        }
    }

    fn feed(&mut self, bytes: Bytes, output: &mut VecDeque<Bytes>) {
        let text = String::from_utf8_lossy(&bytes);
        self.buffer.push_str(&text);

        while let Some((event, delimiter_len)) = next_sse_event(&self.buffer) {
            let event_str = self.buffer[..event].to_string();
            self.buffer = self.buffer[event + delimiter_len..].to_string();
            self.process_event(&event_str, output);
        }
    }

    fn finish(&mut self, output: &mut VecDeque<Bytes>) -> StreamUsage {
        if self.pending_finish_reason.is_some() {
            self.finish_message(output);
        }
        let mut usage = self.usage.clone();
        usage.completed = self.finalized && !self.terminal_error;
        usage
    }

    fn process_event(&mut self, event_str: &str, output: &mut VecDeque<Bytes>) {
        let mut data_lines = Vec::new();
        for line in event_str.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data: ") {
                data_lines.push(data.to_string());
            } else if let Some(data) = line.strip_prefix("data:") {
                data_lines.push(data.trim().to_string());
            }
        }
        if data_lines.is_empty() {
            return;
        }
        let data = data_lines.join("\n");
        if data == "[DONE]" {
            self.finish_message(output);
            return;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(&data) else {
            output.push_back(anthropic_sse(
                "error",
                json!({
                    "type": "error",
                    "error": { "message": "malformed Google SSE JSON" }
                }),
            ));
            self.terminal_error = true;
            self.finalized = true;
            return;
        };

        if let Some(model) = chunk.get("modelVersion").and_then(Value::as_str) {
            self.usage.model = Some(model.to_string());
        }
        if let Some(usage) = chunk.get("usageMetadata") {
            let (input_tokens, output_tokens, cache_read) =
                crate::google_compat::extract_usage(usage);
            if input_tokens > 0 {
                self.usage.input_tokens = input_tokens;
            }
            if output_tokens > 0 {
                self.usage.output_tokens = output_tokens;
            }
            if let Some(cache_read) = cache_read {
                self.usage.cache_read_input_tokens = cache_read;
            }
        }

        let parts = chunk
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !parts.is_empty() {
            self.ensure_message_started(output);
        }
        for part in &parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    self.emit_text_delta(text, output);
                }
            } else if let Some(function_call) = part.get("functionCall") {
                self.emit_function_call(function_call, output);
            }
        }

        if let Some(reason) = chunk
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        {
            self.pending_finish_reason = Some(reason.to_string());
        }
    }

    fn ensure_message_started(&mut self, output: &mut VecDeque<Bytes>) {
        if self.message_started {
            return;
        }
        self.message_started = true;
        output.push_back(anthropic_sse(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": format!("msg_{}", Uuid::new_v4().simple()),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.requested_model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": { "input_tokens": self.usage.input_tokens }
                }
            }),
        ));
    }

    fn emit_text_delta(&mut self, text: &str, output: &mut VecDeque<Bytes>) {
        if !self.text_block_open {
            let index = self.next_index;
            self.next_index += 1;
            self.text_block_index = Some(index);
            self.text_block_open = true;
            output.push_back(anthropic_sse(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": { "type": "text", "text": "" }
                }),
            ));
        }
        if let Some(index) = self.text_block_index {
            output.push_back(anthropic_sse(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "text_delta", "text": text }
                }),
            ));
        }
    }

    fn emit_function_call(&mut self, function_call: &Value, output: &mut VecDeque<Bytes>) {
        // Close any open text block so tool blocks get a fresh index.
        if self.text_block_open {
            if let Some(index) = self.text_block_index.take() {
                output.push_back(anthropic_sse(
                    "content_block_stop",
                    json!({ "type": "content_block_stop", "index": index }),
                ));
            }
            self.text_block_open = false;
        }

        self.saw_tool_call = true;
        let index = self.next_index;
        self.next_index += 1;
        let name = function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let args = function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| json!({}));

        output.push_back(anthropic_sse(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": format!("toolu_{}", Uuid::new_v4().simple()),
                    "name": name,
                    "input": {},
                }
            }),
        ));
        // Gemini delivers the complete arguments object in a single chunk, so
        // emit it as one input_json_delta and immediately close the block.
        output.push_back(anthropic_sse(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": args.to_string(),
                }
            }),
        ));
        output.push_back(anthropic_sse(
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": index }),
        ));
    }

    fn finish_message(&mut self, output: &mut VecDeque<Bytes>) {
        if self.finalized || !self.message_started {
            return;
        }
        if self.text_block_open {
            if let Some(index) = self.text_block_index.take() {
                output.push_back(anthropic_sse(
                    "content_block_stop",
                    json!({ "type": "content_block_stop", "index": index }),
                ));
            }
            self.text_block_open = false;
        }

        let stop_reason = crate::google_compat::map_finish_reason(
            self.pending_finish_reason.as_deref(),
            self.saw_tool_call,
        );
        output.push_back(anthropic_sse(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": Value::Null,
                },
                "usage": { "output_tokens": self.usage.output_tokens }
            }),
        ));
        output.push_back(anthropic_sse(
            "message_stop",
            json!({ "type": "message_stop" }),
        ));
        self.finalized = true;
    }
}

/// Parses Anthropic SSE lines to extract billing-relevant data.
struct SseParser {
    buffer: String,
    current_event: Option<String>,
    usage: StreamUsage,
    completed: bool,
    terminal_error: bool,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            current_event: None,
            usage: StreamUsage::default(),
            completed: false,
            terminal_error: false,
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();
            self.process_line(&line);
        }
    }

    fn process_line(&mut self, line: &str) {
        if let Some(event_type) = line.strip_prefix("event: ") {
            self.current_event = Some(event_type.to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            if let Some(ref event_type) = self.current_event.clone() {
                self.process_event(event_type, data);
            }
        } else if line.is_empty() {
            self.current_event = None;
        }
    }

    fn process_event(&mut self, event_type: &str, data: &str) {
        match event_type {
            "message_start" => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(usage) = value.pointer("/message/usage") {
                        capture_anthropic_usage(&mut self.usage, usage);
                    }
                    if let Some(model) = value.pointer("/message/model").and_then(|v| v.as_str()) {
                        self.usage.model = Some(model.to_string());
                    }
                }
            }
            "message_delta" => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(usage) = value.get("usage") {
                        capture_anthropic_usage(&mut self.usage, usage);
                    }
                }
            }
            "message_stop" => {
                self.completed = true;
            }
            "error" => {
                self.terminal_error = true;
            }
            _ => {}
        }
    }

    fn finalize(&self) -> StreamUsage {
        let mut usage = self.usage.clone();
        usage.completed = self.completed;
        usage
    }
}

fn interrupted_stream_error(provider_request_id: &str) -> Bytes {
    anthropic_sse(
        "error",
        json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Provider stream interrupted before completion",
            },
            "request_id": provider_request_id,
        }),
    )
}

fn capture_anthropic_usage(target: &mut StreamUsage, usage: &Value) {
    if let Some(n) = usage.get("input_tokens").and_then(Value::as_u64) {
        target.input_tokens = n;
    }
    if let Some(n) = usage.get("output_tokens").and_then(Value::as_u64) {
        target.output_tokens = n;
    }
    if let Some(n) = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
    {
        target.cache_creation_input_tokens = n;
    }
    if let Some(n) = usage
        .pointer("/cache_creation/ephemeral_5m_input_tokens")
        .and_then(Value::as_u64)
    {
        target.cache_creation_5m_input_tokens = n;
    }
    if let Some(n) = usage
        .pointer("/cache_creation/ephemeral_1h_input_tokens")
        .and_then(Value::as_u64)
    {
        target.cache_creation_1h_input_tokens = n;
    }
    if let Some(n) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
        target.cache_read_input_tokens = n;
    }
    if let Some(n) = usage
        .pointer("/server_tool_use/web_search_requests")
        .and_then(Value::as_u64)
    {
        target.web_search_requests = n;
    }
    if let Some(n) = usage
        .pointer("/server_tool_use/web_fetch_requests")
        .and_then(Value::as_u64)
    {
        target.web_fetch_requests = n;
    }
    if let Some(n) = usage
        .pointer("/server_tool_use/code_execution_requests")
        .and_then(Value::as_u64)
    {
        target.code_execution_requests = n;
    }
    if let Some(value) = usage.get("service_tier").and_then(Value::as_str) {
        target.service_tier = Some(value.to_string());
    }
    if let Some(value) = usage.get("inference_geo").and_then(Value::as_str) {
        target.inference_geo = Some(value.to_string());
    }
    if let Some(value) = usage.get("speed").and_then(Value::as_str) {
        target.speed = Some(value.to_string());
    }
}

fn next_sse_event(buffer: &str) -> Option<(usize, usize)> {
    let event_end = buffer.find("\n\n").or_else(|| buffer.find("\r\n\r\n"))?;
    let delimiter_len = if buffer[event_end..].starts_with("\r\n\r\n") {
        4
    } else {
        2
    };
    Some((event_end, delimiter_len))
}

fn anthropic_sse(event_type: &str, payload: Value) -> Bytes {
    Bytes::from(format!("event: {event_type}\ndata: {payload}\n\n"))
}

fn map_openai_finish_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("tool_calls") => "tool_use",
        Some("length") => "max_tokens",
        Some("stop") => "end_turn",
        Some("content_filter") => "end_turn",
        _ => "end_turn",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    fn bytes_stream(
        chunks: Vec<&'static str>,
    ) -> impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Unpin {
        futures_util::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok(bytes::Bytes::from(c.to_string()))),
        )
    }

    #[tokio::test]
    async fn anthropic_passthrough_preserves_stream_and_usage() {
        let stream = bytes_stream(vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12,\"cache_creation_input_tokens\":8,\"cache_creation\":{\"ephemeral_5m_input_tokens\":3,\"ephemeral_1h_input_tokens\":5},\"cache_read_input_tokens\":20,\"service_tier\":\"standard\",\"inference_geo\":\"us\",\"speed\":\"fast\"},\"model\":\"aura-claude-sonnet-4-6\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":7,\"server_tool_use\":{\"web_search_requests\":2,\"web_fetch_requests\":1,\"code_execution_requests\":4}}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::Anthropic,
                OpenAiApi::ChatCompletions,
                "aura-claude-sonnet-4-6",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_anthropic".to_string(),
        };

        let mut seen = Vec::new();
        while let Some(chunk) = tee.next().await {
            seen.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }

        let usage = rx.await.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cache_creation_5m_input_tokens, 3);
        assert_eq!(usage.cache_creation_1h_input_tokens, 5);
        assert_eq!(usage.cache_read_input_tokens, 20);
        assert_eq!(usage.web_search_requests, 2);
        assert_eq!(usage.web_fetch_requests, 1);
        assert_eq!(usage.code_execution_requests, 4);
        assert_eq!(usage.service_tier.as_deref(), Some("standard"));
        assert_eq!(usage.inference_geo.as_deref(), Some("us"));
        assert_eq!(usage.speed.as_deref(), Some("fast"));
        assert!(seen.iter().any(|chunk| chunk.contains("message_start")));
        assert!(seen.iter().any(|chunk| chunk.contains("x_context_usage")));
        assert!(usage.completed);
    }

    #[tokio::test]
    async fn transport_failure_emits_structured_error_and_reports_incomplete_usage() {
        let stream = futures_util::stream::iter(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5},\"model\":\"claude-sonnet-5\"}}\n\n",
            )),
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "connection closed before message completed",
            )),
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::Anthropic,
                OpenAiApi::ChatCompletions,
                "aura-claude-sonnet-5",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_interrupted".to_string(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }
        let joined = emitted.join("");
        assert!(joined.contains("Provider stream interrupted before completion"));
        assert!(joined.contains("req_interrupted"));
        assert!(!joined.contains("message_stop"));

        let usage = rx.await.unwrap();
        assert_eq!(usage.input_tokens, 5);
        assert!(!usage.completed);
    }

    #[tokio::test]
    async fn clean_eof_without_completion_is_not_synthesized_as_success() {
        let stream = bytes_stream(vec![
            "data: {\"id\":\"chatcmpl-cut-off\",\"model\":\"gpt-4.1\",\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::OpenAi,
                OpenAiApi::ChatCompletions,
                "aura-gpt-4.1",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_clean_eof".to_string(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }
        let joined = emitted.join("");
        assert!(joined.contains("Provider stream interrupted before completion"));
        assert!(!joined.contains("message_stop"));
        assert!(!rx.await.unwrap().completed);
    }

    #[tokio::test]
    async fn upstream_error_event_is_not_followed_by_duplicate_interruption_error() {
        let stream = bytes_stream(vec![
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"upstream failed\"}}\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::Anthropic,
                OpenAiApi::ChatCompletions,
                "aura-claude-sonnet-5",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_upstream_error".to_string(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }
        let joined = emitted.join("");
        assert_eq!(joined.matches("event: error").count(), 1);
        assert!(!joined.contains("Provider stream interrupted before completion"));
        assert!(!rx.await.unwrap().completed);
    }

    #[tokio::test]
    async fn noisy_transport_close_after_completion_does_not_trigger_retry() {
        let stream = futures_util::stream::iter(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            )),
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "late transport close",
            )),
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::Anthropic,
                OpenAiApi::ChatCompletions,
                "aura-claude-sonnet-5",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_completed".to_string(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }
        assert!(!emitted
            .join("")
            .contains("Provider stream interrupted before completion"));
        assert!(rx.await.unwrap().completed);
    }

    #[tokio::test]
    async fn google_stream_translates_text_tool_calls_and_usage() {
        let stream = bytes_stream(vec![
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Let me \"}]}}],\"usageMetadata\":{\"promptTokenCount\":11}}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"search\"}]}}],\"modelVersion\":\"gemini-2.5-pro\"}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"search_repo\",\"args\":{\"query\":\"aura\"}}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":11,\"candidatesTokenCount\":6,\"thoughtsTokenCount\":3}}\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::Google,
                OpenAiApi::ChatCompletions,
                "aura-gemini-2-5-pro",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_google".to_string(),
        };

        let mut seen = Vec::new();
        while let Some(chunk) = tee.next().await {
            seen.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }
        let joined = seen.join("");

        let usage = rx.await.unwrap();
        assert_eq!(usage.input_tokens, 11);
        // candidates (6) + thoughts (3) fold into output tokens.
        assert_eq!(usage.output_tokens, 9);
        assert!(joined.contains("message_start"));
        assert!(joined.contains("\"text_delta\""));
        assert!(joined.contains("Let me "));
        assert!(joined.contains("\"type\":\"tool_use\""));
        assert!(joined.contains("search_repo"));
        assert!(joined.contains("\"input_json_delta\""));
        assert!(joined.contains("\"stop_reason\":\"tool_use\""));
        assert!(joined.contains("x_context_usage"));
    }

    #[tokio::test]
    async fn openai_responses_stream_translates_text_and_tool_calls() {
        let stream = bytes_stream(vec![
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5.6-sol\"}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"msg_1\",\"delta\":\"Hello\"}\n\n",
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n",
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"search_repo\"}}\n\n",
            "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"item_id\":\"fc_1\",\"delta\":\"{\\\"query\\\":\"}\n\n",
            "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"item_id\":\"fc_1\",\"delta\":\"\\\"aura\\\"}\"}\n\n",
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"search_repo\"}}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5.6-sol\",\"usage\":{\"input_tokens\":21,\"output_tokens\":9,\"input_tokens_details\":{\"cached_tokens\":4,\"cache_write_tokens\":3},\"cost_in_usd_ticks\":250000000}}}\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(Provider::OpenAi, OpenAiApi::Responses, "aura-gpt-5-6-sol"),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_responses".to_string(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }
        let joined = emitted.join("");

        assert!(
            joined.contains("message_start"),
            "missing message_start: {joined}"
        );
        assert!(
            joined.contains("\"type\":\"text\""),
            "missing text block: {joined}"
        );
        assert!(joined.contains("Hello"), "missing text delta: {joined}");
        assert!(
            joined.contains("\"type\":\"tool_use\""),
            "missing tool_use: {joined}"
        );
        assert!(
            joined.contains("search_repo"),
            "missing tool name: {joined}"
        );
        assert!(
            joined.contains("input_json_delta"),
            "missing args delta: {joined}"
        );
        assert!(
            joined.contains("\"stop_reason\":\"tool_use\""),
            "expected tool_use stop reason: {joined}"
        );
        assert!(
            joined.contains("message_stop"),
            "missing message_stop: {joined}"
        );

        let usage = rx.await.unwrap();
        assert_eq!(usage.input_tokens, 21);
        assert_eq!(usage.output_tokens, 9);
        assert_eq!(usage.cache_creation_input_tokens, 3);
        assert_eq!(usage.cache_read_input_tokens, 4);
        assert_eq!(usage.provider_reported_cost_cents, Some(3));
    }

    #[tokio::test]
    async fn xai_responses_stream_uses_responses_adapter() {
        let stream = bytes_stream(vec![
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_xai\",\"model\":\"grok-4.3\"}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"msg_1\",\"delta\":\"Grok\"}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_xai\",\"model\":\"grok-4.3\",\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"cost_in_usd_ticks\":100000000}}}\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(Provider::Xai, OpenAiApi::Responses, "aura-grok-4-3"),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_xai".to_string(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }
        let joined = emitted.join("");
        assert!(
            joined.contains("message_start"),
            "missing message_start: {joined}"
        );
        assert!(joined.contains("Grok"), "missing text delta: {joined}");

        let usage = rx.await.unwrap();
        assert_eq!(usage.model.as_deref(), Some("grok-4.3"));
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.provider_reported_cost_cents, Some(1));
    }

    #[tokio::test]
    async fn openai_text_stream_translates_to_anthropic_events() {
        let stream = bytes_stream(vec![
            "data: {\"id\":\"chatcmpl-123\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-123\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-123\",\"model\":\"gpt-5.6-sol\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":4,\"cache_write_tokens\":3}}}\n\n",
            "data: [DONE]\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::OpenAi,
                OpenAiApi::ChatCompletions,
                "aura-gpt-4.1",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_openai_text".to_string(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }

        let usage = rx.await.unwrap();
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.cache_creation_input_tokens, 3);
        assert_eq!(usage.cache_read_input_tokens, 4);
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("event: message_start")));
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("\"text\":\"Hello\"")));
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("\"text\":\" world\"")));
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("\"stop_reason\":\"end_turn\"")));
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("event: message_stop")));
    }

    #[tokio::test]
    async fn openai_tool_call_stream_translates_to_tool_use_blocks() {
        let stream = bytes_stream(vec![
            "data: {\"id\":\"chatcmpl-456\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-456\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"aura\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-456\",\"model\":\"gpt-4.1\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}\n\n",
            "data: [DONE]\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::OpenAi,
                OpenAiApi::ChatCompletions,
                "aura-gpt-4.1",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_openai_tool".to_string(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }

        let usage = rx.await.unwrap();
        assert_eq!(usage.output_tokens, 4);
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("\"type\":\"tool_use\"")));
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("\"partial_json\":\"{\\\"q\\\":\"")));
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("\"partial_json\":\"\\\"aura\\\"}\"")));
        assert!(emitted
            .iter()
            .any(|chunk| chunk.contains("\"stop_reason\":\"tool_use\"")));
    }

    #[tokio::test]
    async fn deepseek_stream_captures_cache_usage_aliases() {
        let stream = bytes_stream(vec![
            "data: {\"id\":\"deepseek-123\",\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
            "data: {\"id\":\"deepseek-123\",\"model\":\"deepseek-v4-flash\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":8,\"prompt_cache_miss_tokens\":30,\"prompt_cache_hit_tokens\":70}}\n\n",
            "data: [DONE]\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(
                Provider::DeepSeek,
                OpenAiApi::ChatCompletions,
                "aura-deepseek-v4-flash",
            ),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
            provider_request_id: "req_deepseek".to_string(),
        };

        while let Some(chunk) = tee.next().await {
            let _ = chunk.unwrap();
        }

        let usage = rx.await.unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 8);
        assert_eq!(usage.cache_creation_input_tokens, 30);
        assert_eq!(usage.cache_read_input_tokens, 70);
    }
}
