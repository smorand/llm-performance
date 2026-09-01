//! Single-request probe execution for OpenAI and Anthropic endpoints.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::payloads::{PROBE_PROMPT, TEST_PNG_BASE64, TEST_WAV_BASE64};
use crate::stats::TrialMetric;
use crate::thinking::ThinkingLevel;

/// Number of retry attempts on 5xx server errors.
const MAX_5XX_RETRIES: usize = 3;

/// Delay between 5xx retry attempts.
const RETRY_DELAY: Duration = Duration::from_millis(500);

/// Modality payload kind used during capability probing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbePayload {
    /// Standard minimal text prompt.
    #[default]
    Text,
    /// Image payload with embedded 8x8 PNG base64.
    Image,
    /// Audio payload with embedded 20ms WAV base64.
    Audio,
}

impl ProbePayload {
    /// Return the string identifier for this payload kind.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }
}

/// Execute a single OpenAI chat completions probe request with retry logic on 5xx errors.
///
/// # Errors
///
/// Returns [`CoreError::Api`] if the upstream returns an unrecoverable client error or exhausts retries.
/// Returns [`CoreError::Http`] if network transport fails after retries.
/// Returns [`CoreError::Json`] if JSON serialization or deserialization fails.
#[tracing::instrument(skip(http, key), fields(model = %model, stream = %stream, payload = ?payload_kind, thinking = ?thinking))]
pub async fn probe_openai_chat(
    http: &reqwest::Client,
    base_url: &str,
    key: &str,
    model: &str,
    payload_kind: ProbePayload,
    thinking: ThinkingLevel,
    stream: bool,
) -> Result<TrialMetric, CoreError> {
    let key = key.trim();
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let messages = build_openai_messages(payload_kind);

    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });

    if stream {
        body["stream_options"] = serde_json::json!({ "include_usage": true });
    }

    if let Some(effort) = thinking.to_openai_reasoning_effort() {
        body["reasoning_effort"] = serde_json::json!(effort);
    } else {
        body["max_tokens"] = serde_json::json!(16);
    }

    for attempt in 0..=MAX_5XX_RETRIES {
        let start_time = Instant::now();

        let response_res = http.post(&url).bearer_auth(key).json(&body).send().await;

        let response = match response_res {
            Ok(resp) => resp,
            Err(e) => {
                if attempt < MAX_5XX_RETRIES {
                    tracing::warn!(attempt = attempt + 1, error = %e, "retrying OpenAI probe on transport error");
                    tokio::time::sleep(RETRY_DELAY).await;
                    continue;
                }
                return Err(CoreError::Http(Box::new(e)));
            }
        };

        let status = response.status();
        if status.is_server_error() {
            if attempt < MAX_5XX_RETRIES {
                tracing::warn!(attempt = attempt + 1, status = %status, "retrying OpenAI probe on 5xx response");
                tokio::time::sleep(RETRY_DELAY).await;
                continue;
            }
            let message = response.text().await.unwrap_or_default();
            return Err(CoreError::Api { status, message });
        }

        if !status.is_success() {
            let message = response.text().await.unwrap_or_default();
            return Err(CoreError::Api { status, message });
        }

        return if stream {
            parse_openai_sse_stream(response, start_time).await
        } else {
            parse_openai_json_response(response, start_time).await
        };
    }

    Err(CoreError::Api {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: "exhausted retry attempts".to_string(),
    })
}

/// Execute a single Anthropic messages probe request with retry logic on 5xx errors.
///
/// # Errors
///
/// Returns [`CoreError::Api`] if the upstream returns an unrecoverable client error or exhausts retries.
/// Returns [`CoreError::Http`] if network transport fails after retries.
/// Returns [`CoreError::Json`] if JSON serialization or deserialization fails.
#[tracing::instrument(skip(http, key), fields(model = %model, stream = %stream, thinking = ?thinking))]
pub async fn probe_anthropic_messages(
    http: &reqwest::Client,
    base_url: &str,
    key: &str,
    model: &str,
    thinking: ThinkingLevel,
    stream: bool,
) -> Result<TrialMetric, CoreError> {
    let key = key.trim();
    let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));

    let mut body = serde_json::json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROBE_PROMPT,
            }
        ],
        "max_tokens": thinking.anthropic_max_tokens(),
        "stream": stream,
    });

    if let Some(budget) = thinking.to_anthropic_budget_tokens() {
        body["thinking"] = serde_json::json!({
            "type": "enabled",
            "budget_tokens": budget,
        });
    }

    for attempt in 0..=MAX_5XX_RETRIES {
        let start_time = Instant::now();

        let response_res = http
            .post(&url)
            .header("x-api-key", key)
            .header("authorization", format!("Bearer {key}"))
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await;

        let response = match response_res {
            Ok(resp) => resp,
            Err(e) => {
                if attempt < MAX_5XX_RETRIES {
                    tracing::warn!(attempt = attempt + 1, error = %e, "retrying Anthropic probe on transport error");
                    tokio::time::sleep(RETRY_DELAY).await;
                    continue;
                }
                return Err(CoreError::Http(Box::new(e)));
            }
        };

        let status = response.status();
        if status.is_server_error() {
            if attempt < MAX_5XX_RETRIES {
                tracing::warn!(attempt = attempt + 1, status = %status, "retrying Anthropic probe on 5xx response");
                tokio::time::sleep(RETRY_DELAY).await;
                continue;
            }
            let message = response.text().await.unwrap_or_default();
            return Err(CoreError::Api { status, message });
        }

        if !status.is_success() {
            let message = response.text().await.unwrap_or_default();
            return Err(CoreError::Api { status, message });
        }

        return if stream {
            parse_anthropic_sse_stream(response, start_time).await
        } else {
            parse_anthropic_json_response(response, start_time).await
        };
    }

    Err(CoreError::Api {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: "exhausted retry attempts".to_string(),
    })
}

/// Construct OpenAI messages array for a given probe modality payload.
fn build_openai_messages(payload: ProbePayload) -> serde_json::Value {
    match payload {
        ProbePayload::Text => serde_json::json!([
            {
                "role": "user",
                "content": PROBE_PROMPT,
            }
        ]),
        ProbePayload::Image => serde_json::json!([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROBE_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/png;base64,{TEST_PNG_BASE64}")
                        }
                    }
                ]
            }
        ]),
        ProbePayload::Audio => serde_json::json!([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROBE_PROMPT,
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": TEST_WAV_BASE64,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]),
    }
}

/// Consume an OpenAI SSE byte stream and calculate latency, TTFT, TPOT, and TPS.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]
async fn parse_openai_sse_stream(
    response: reqwest::Response,
    start_time: Instant,
) -> Result<TrialMetric, CoreError> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut ttft_ms: Option<u64> = None;
    let mut chunk_token_count: u64 = 0;
    let mut reported_output_tokens: Option<u64> = None;
    let mut reported_reasoning_tokens: Option<u64> = None;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| CoreError::Http(Box::new(e)))?;
        let chunk_str = String::from_utf8_lossy(&chunk);
        buffer.push_str(&chunk_str);

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
            buffer.drain(..=newline_pos);

            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with(':') {
                continue;
            }

            if let Some(data) = trimmed.strip_prefix("data:") {
                let data = data.trim();
                if data == "[DONE]" {
                    break;
                }

                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(choices) = json_val.get("choices").and_then(|c| c.as_array()) {
                        for choice in choices {
                            if let Some(delta) = choice.get("delta") {
                                let has_content = delta
                                    .get("content")
                                    .and_then(|c| c.as_str())
                                    .is_some_and(|s| !s.is_empty());
                                let has_reasoning = delta
                                    .get("reasoning_content")
                                    .and_then(|c| c.as_str())
                                    .is_some_and(|s| !s.is_empty())
                                    || delta
                                        .get("reasoning")
                                        .and_then(|c| c.as_str())
                                        .is_some_and(|s| !s.is_empty());

                                if has_content || has_reasoning {
                                    if ttft_ms.is_none() {
                                        ttft_ms = Some(
                                            u64::try_from(start_time.elapsed().as_millis())
                                                .unwrap_or(u64::MAX),
                                        );
                                    }
                                    chunk_token_count += 1;
                                }
                            }
                        }
                    }

                    if let Some(usage) = json_val.get("usage") {
                        if let Some(ct) = usage
                            .get("completion_tokens")
                            .and_then(serde_json::Value::as_u64)
                        {
                            reported_output_tokens = Some(ct);
                        }
                        if let Some(rt) = usage
                            .get("completion_tokens_details")
                            .and_then(|d| d.get("reasoning_tokens"))
                            .and_then(serde_json::Value::as_u64)
                        {
                            reported_reasoning_tokens = Some(rt);
                        }
                    }
                }
            }
        }
    }

    let latency_ms = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);
    let output_tokens = reported_output_tokens.unwrap_or(chunk_token_count.max(1));

    let tpot_ms = match ttft_ms {
        Some(ttft) if output_tokens > 1 => {
            Some((latency_ms.saturating_sub(ttft)) as f64 / (output_tokens - 1) as f64)
        }
        Some(ttft) => Some(latency_ms.saturating_sub(ttft) as f64),
        None if output_tokens > 0 => Some(latency_ms as f64 / output_tokens as f64),
        None => None,
    };

    let tps = if output_tokens > 0 {
        Some((output_tokens as f64) / (latency_ms.max(1) as f64 / 1000.0))
    } else {
        None
    };

    Ok(TrialMetric {
        latency_ms,
        ttft_ms,
        tpot_ms,
        tps,
        output_tokens,
        reasoning_tokens: reported_reasoning_tokens,
    })
}

/// Parse non-streaming OpenAI chat completions JSON response.
#[allow(clippy::cast_precision_loss)]
async fn parse_openai_json_response(
    response: reqwest::Response,
    start_time: Instant,
) -> Result<TrialMetric, CoreError> {
    let body_text = response.text().await?;
    let latency_ms = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);

    let json_val: serde_json::Value = serde_json::from_str(&body_text)?;

    let output_tokens = json_val
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);

    let reasoning_tokens = json_val
        .get("usage")
        .and_then(|u| u.get("completion_tokens_details"))
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(serde_json::Value::as_u64);

    let tpot_ms = if output_tokens > 0 {
        Some(latency_ms as f64 / output_tokens as f64)
    } else {
        None
    };

    let tps = if output_tokens > 0 {
        Some((output_tokens as f64) / (latency_ms.max(1) as f64 / 1000.0))
    } else {
        None
    };

    Ok(TrialMetric {
        latency_ms,
        ttft_ms: None,
        tpot_ms,
        tps,
        output_tokens,
        reasoning_tokens,
    })
}

/// Consume an Anthropic SSE byte stream and calculate latency, TTFT, TPOT, and TPS.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]
async fn parse_anthropic_sse_stream(
    response: reqwest::Response,
    start_time: Instant,
) -> Result<TrialMetric, CoreError> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut ttft_ms: Option<u64> = None;
    let mut chunk_token_count: u64 = 0;
    let mut reported_output_tokens: Option<u64> = None;
    let mut reported_reasoning_tokens: Option<u64> = None;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| CoreError::Http(Box::new(e)))?;
        let chunk_str = String::from_utf8_lossy(&chunk);
        buffer.push_str(&chunk_str);

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
            buffer.drain(..=newline_pos);

            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with(':') {
                continue;
            }

            if let Some(data) = trimmed.strip_prefix("data:") {
                let data = data.trim();
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(data) {
                    let event_type = json_val
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default();

                    match event_type {
                        "content_block_delta" => {
                            if let Some(delta) = json_val.get("delta") {
                                let has_text = delta
                                    .get("text")
                                    .and_then(|t| t.as_str())
                                    .is_some_and(|s| !s.is_empty());
                                let has_thinking = delta
                                    .get("thinking")
                                    .and_then(|t| t.as_str())
                                    .is_some_and(|s| !s.is_empty());

                                if has_text || has_thinking {
                                    if ttft_ms.is_none() {
                                        ttft_ms = Some(
                                            u64::try_from(start_time.elapsed().as_millis())
                                                .unwrap_or(u64::MAX),
                                        );
                                    }
                                    chunk_token_count += 1;
                                }
                            }
                        }
                        "content_block_start" => {
                            if let Some(cb) = json_val.get("content_block")
                                && let Some(text) = cb.get("text").and_then(|t| t.as_str())
                                && !text.is_empty()
                            {
                                if ttft_ms.is_none() {
                                    ttft_ms = Some(
                                        u64::try_from(start_time.elapsed().as_millis())
                                            .unwrap_or(u64::MAX),
                                    );
                                }
                                chunk_token_count += 1;
                            }
                        }
                        "message_delta" => {
                            if let Some(usage) = json_val.get("usage") {
                                if let Some(ot) = usage
                                    .get("output_tokens")
                                    .and_then(serde_json::Value::as_u64)
                                {
                                    reported_output_tokens = Some(ot);
                                }
                                if let Some(tt) = usage
                                    .get("thinking_tokens")
                                    .and_then(serde_json::Value::as_u64)
                                {
                                    reported_reasoning_tokens = Some(tt);
                                }
                            }
                        }
                        "message_start" => {
                            if let Some(msg) = json_val.get("message")
                                && let Some(usage) = msg.get("usage")
                                && let Some(ot) = usage
                                    .get("output_tokens")
                                    .and_then(serde_json::Value::as_u64)
                            {
                                reported_output_tokens = Some(ot);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let latency_ms = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);
    let output_tokens = reported_output_tokens.unwrap_or(chunk_token_count.max(1));

    let tpot_ms = match ttft_ms {
        Some(ttft) if output_tokens > 1 => {
            Some((latency_ms.saturating_sub(ttft)) as f64 / (output_tokens - 1) as f64)
        }
        Some(ttft) => Some(latency_ms.saturating_sub(ttft) as f64),
        None if output_tokens > 0 => Some(latency_ms as f64 / output_tokens as f64),
        None => None,
    };

    let tps = if output_tokens > 0 {
        Some((output_tokens as f64) / (latency_ms.max(1) as f64 / 1000.0))
    } else {
        None
    };

    Ok(TrialMetric {
        latency_ms,
        ttft_ms,
        tpot_ms,
        tps,
        output_tokens,
        reasoning_tokens: reported_reasoning_tokens,
    })
}

/// Parse non-streaming Anthropic messages JSON response.
#[allow(clippy::cast_precision_loss)]
async fn parse_anthropic_json_response(
    response: reqwest::Response,
    start_time: Instant,
) -> Result<TrialMetric, CoreError> {
    let body_text = response.text().await?;
    let latency_ms = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);

    let json_val: serde_json::Value = serde_json::from_str(&body_text)?;

    let output_tokens = json_val
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);

    let reasoning_tokens = json_val
        .get("usage")
        .and_then(|u| u.get("thinking_tokens"))
        .and_then(serde_json::Value::as_u64);

    let tpot_ms = if output_tokens > 0 {
        Some(latency_ms as f64 / output_tokens as f64)
    } else {
        None
    };

    let tps = if output_tokens > 0 {
        Some((output_tokens as f64) / (latency_ms.max(1) as f64 / 1000.0))
    } else {
        None
    };

    Ok(TrialMetric {
        latency_ms,
        ttft_ms: None,
        tpot_ms,
        tps,
        output_tokens,
        reasoning_tokens,
    })
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use axum::http::StatusCode as AxumStatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use axum::{Json, Router};

    use super::*;

    #[test]
    fn probe_payload_as_str() {
        assert_eq!(ProbePayload::Text.as_str(), "text");
        assert_eq!(ProbePayload::Image.as_str(), "image");
        assert_eq!(ProbePayload::Audio.as_str(), "audio");
    }

    #[test]
    fn build_openai_messages_structures() {
        let text_msg = build_openai_messages(ProbePayload::Text);
        assert_eq!(text_msg[0]["role"], "user");
        assert_eq!(text_msg[0]["content"], PROBE_PROMPT);

        let image_msg = build_openai_messages(ProbePayload::Image);
        let contents = image_msg[0]["content"].as_array().unwrap();
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["type"], "text");
        assert_eq!(contents[1]["type"], "image_url");

        let audio_msg = build_openai_messages(ProbePayload::Audio);
        let contents = audio_msg[0]["content"].as_array().unwrap();
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["type"], "text");
        assert_eq!(contents[1]["type"], "input_audio");
    }

    async fn mock_openai_handler(Json(payload): Json<serde_json::Value>) -> Response {
        let stream = payload
            .get("stream")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if stream {
            let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":10,\"completion_tokens_details\":{\"reasoning_tokens\":4}}}\n\ndata: [DONE]\n\n";
            ([("content-type", "text/event-stream")], sse.to_string()).into_response()
        } else {
            Json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "pong"}}],
                "usage": {
                    "completion_tokens": 12,
                    "completion_tokens_details": {
                        "reasoning_tokens": 3
                    }
                }
            }))
            .into_response()
        }
    }

    async fn mock_anthropic_handler(Json(payload): Json<serde_json::Value>) -> Response {
        let stream = payload
            .get("stream")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if stream {
            let sse = "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":8,\"thinking_tokens\":6}}\n\n";
            ([("content-type", "text/event-stream")], sse.to_string()).into_response()
        } else {
            Json(serde_json::json!({
                "content": [{"type": "text", "text": "hello"}],
                "usage": {
                    "output_tokens": 7,
                    "thinking_tokens": 5
                }
            }))
            .into_response()
        }
    }

    async fn setup_probe_mock_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_openai_handler))
            .route("/v1/messages", post(mock_anthropic_handler));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (addr, handle)
    }

    #[tokio::test]
    async fn test_probe_openai_streaming_and_non_streaming() {
        let (addr, handle) = setup_probe_mock_server().await;
        let http = reqwest::Client::new();
        let base_url = format!("http://{addr}");

        // Streaming
        let metric_stream = probe_openai_chat(
            &http,
            &base_url,
            "test-key",
            "gpt-4o",
            ProbePayload::Text,
            ThinkingLevel::Off,
            true,
        )
        .await
        .expect("streaming probe");

        assert!(metric_stream.ttft_ms.is_some());
        assert_eq!(metric_stream.output_tokens, 10);
        assert_eq!(metric_stream.reasoning_tokens, Some(4));
        assert!(metric_stream.tpot_ms.is_some());
        assert!(metric_stream.tps.is_some());

        // Non-streaming
        let metric_non_stream = probe_openai_chat(
            &http,
            &base_url,
            "test-key",
            "gpt-4o",
            ProbePayload::Image,
            ThinkingLevel::Medium,
            false,
        )
        .await
        .expect("non-streaming probe");

        assert_eq!(metric_non_stream.ttft_ms, None);
        assert_eq!(metric_non_stream.output_tokens, 12);
        assert_eq!(metric_non_stream.reasoning_tokens, Some(3));
        assert!(metric_non_stream.tpot_ms.is_some());
        assert!(metric_non_stream.tps.is_some());

        handle.abort();
    }

    #[tokio::test]
    async fn test_probe_anthropic_streaming_and_non_streaming() {
        let (addr, handle) = setup_probe_mock_server().await;
        let http = reqwest::Client::new();
        let base_url = format!("http://{addr}");

        // Streaming
        let metric_stream = probe_anthropic_messages(
            &http,
            &base_url,
            "test-key",
            "claude-3-5-sonnet",
            ThinkingLevel::Low,
            true,
        )
        .await
        .expect("anthropic streaming probe");

        assert!(metric_stream.ttft_ms.is_some());
        assert_eq!(metric_stream.output_tokens, 8);
        assert_eq!(metric_stream.reasoning_tokens, Some(6));

        // Non-streaming
        let metric_non_stream = probe_anthropic_messages(
            &http,
            &base_url,
            "test-key",
            "claude-3-5-sonnet",
            ThinkingLevel::Off,
            false,
        )
        .await
        .expect("anthropic non-streaming probe");

        assert_eq!(metric_non_stream.ttft_ms, None);
        assert_eq!(metric_non_stream.output_tokens, 7);
        assert_eq!(metric_non_stream.reasoning_tokens, Some(5));

        handle.abort();
    }

    #[tokio::test]
    async fn test_probe_client_error_fails_without_retry() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async { (AxumStatusCode::UNAUTHORIZED, "invalid key") }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let http = reqwest::Client::new();
        let err = probe_openai_chat(
            &http,
            &format!("http://{addr}"),
            "bad-key",
            "gpt-4o",
            ProbePayload::Text,
            ThinkingLevel::Off,
            false,
        )
        .await
        .unwrap_err();

        assert!(matches!(err, CoreError::Api { status, .. } if status == StatusCode::UNAUTHORIZED));
        handle.abort();
    }
}
