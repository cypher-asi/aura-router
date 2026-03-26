//! SSE stream proxy with billing capture.
//!
//! Tees the upstream SSE byte stream: one copy goes to the client unmodified,
//! while a billing parser extracts token counts from `message_start` and
//! `message_delta` events. This preserves exact SSE framing and minimizes latency.

use bytes::Bytes;
use tokio::sync::oneshot;

use crate::providers;

/// Token usage extracted from an SSE stream.
#[derive(Debug, Default)]
pub struct StreamUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub model: Option<String>,
}

/// Proxy an SSE stream from the upstream provider to the client,
/// capturing billing data along the way.
///
/// Returns an async stream of `Result<Bytes>` for the client body,
/// and a oneshot receiver that delivers final usage when the stream ends.
pub fn proxy_stream(
    upstream: reqwest::Response,
) -> (
    impl futures_util::Stream<Item = Result<Bytes, reqwest::Error>>,
    oneshot::Receiver<StreamUsage>,
) {
    let (usage_tx, usage_rx) = oneshot::channel();
    let byte_stream = upstream.bytes_stream();

    let tee_stream = TeeStream {
        inner: Box::pin(byte_stream),
        parser: SseParser::new(),
        usage_tx: Some(usage_tx),
        finished: false,
    };

    (tee_stream, usage_rx)
}

/// A stream that passes bytes through while parsing SSE events for billing.
/// Injects an `x_context_usage` SSE event at the end of the stream with context usage data.
struct TeeStream<S> {
    inner: std::pin::Pin<Box<S>>,
    parser: SseParser,
    usage_tx: Option<oneshot::Sender<StreamUsage>>,
    finished: bool,
}

impl<S> futures_util::Stream for TeeStream<S>
where
    S: futures_util::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.finished {
            return std::task::Poll::Ready(None);
        }

        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                // Feed bytes to the billing parser
                self.parser.feed(&bytes);
                std::task::Poll::Ready(Some(Ok(bytes)))
            }
            std::task::Poll::Ready(Some(Err(e))) => std::task::Poll::Ready(Some(Err(e))),
            std::task::Poll::Ready(None) => {
                // Stream ended — build context usage event and send final usage
                self.finished = true;

                let usage = self.parser.finalize();
                let model = usage.model.as_deref().unwrap_or("unknown");
                let max_tokens = providers::max_context_tokens(model);
                let context_usage = if max_tokens > 0 {
                    usage.input_tokens as f64 / max_tokens as f64
                } else {
                    0.0
                };

                let event = format!(
                    "event: x_context_usage\ndata: {{\"contextUsage\":{context_usage:.4},\"inputTokens\":{},\"outputTokens\":{},\"maxTokens\":{max_tokens}}}\n\n",
                    usage.input_tokens, usage.output_tokens
                );

                if let Some(tx) = self.usage_tx.take() {
                    let _ = tx.send(usage);
                }

                std::task::Poll::Ready(Some(Ok(Bytes::from(event))))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

/// Parses SSE lines to extract billing-relevant data.
///
/// Only cares about two event types:
/// - `message_start`: contains `message.usage.input_tokens` and cache metrics
/// - `message_delta`: contains `usage.output_tokens`
///
/// All other events are ignored — the raw bytes pass through unmodified.
struct SseParser {
    buffer: String,
    current_event: Option<String>,
    usage: StreamUsage,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            current_event: None,
            usage: StreamUsage::default(),
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);

        // Process complete lines
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
            // Empty line = end of SSE event
            self.current_event = None;
        }
    }

    fn process_event(&mut self, event_type: &str, data: &str) {
        match event_type {
            "message_start" => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
                    // Extract input tokens from message.usage
                    if let Some(usage) = value.pointer("/message/usage") {
                        if let Some(n) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                            self.usage.input_tokens = n;
                        }
                        if let Some(n) = usage
                            .get("cache_creation_input_tokens")
                            .and_then(|v| v.as_u64())
                        {
                            self.usage.cache_creation_input_tokens = n;
                        }
                        if let Some(n) = usage
                            .get("cache_read_input_tokens")
                            .and_then(|v| v.as_u64())
                        {
                            self.usage.cache_read_input_tokens = n;
                        }
                    }
                    // Extract model
                    if let Some(model) = value.pointer("/message/model").and_then(|v| v.as_str()) {
                        self.usage.model = Some(model.to_string());
                    }
                }
            }
            "message_delta" => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(n) = value
                        .pointer("/usage/output_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        self.usage.output_tokens = n;
                    }
                }
            }
            _ => {} // Ignore all other events
        }
    }

    fn finalize(&self) -> StreamUsage {
        StreamUsage {
            input_tokens: self.usage.input_tokens,
            output_tokens: self.usage.output_tokens,
            cache_creation_input_tokens: self.usage.cache_creation_input_tokens,
            cache_read_input_tokens: self.usage.cache_read_input_tokens,
            model: self.usage.model.clone(),
        }
    }
}
