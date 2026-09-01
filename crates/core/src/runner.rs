//! Benchmark orchestrator coordinating capability discovery, trial execution, and aggregation.

use std::sync::Arc;

use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::error::CoreError;
use crate::models::{
    compile_patterns, fetch_models, filter_models, is_anthropic_model, is_audio_model,
    is_vision_model,
};
use crate::probe::{ProbePayload, probe_anthropic_messages, probe_openai_chat};
use crate::stats::{AggregateStats, TrialMetric, aggregate_trials};
use crate::thinking::ThinkingLevel;

/// Options controlling which model capabilities and modalities to benchmark.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ProbeRunOptions {
    /// Whether to test base text generation.
    pub test_text: bool,

    /// Whether to test vision / image inputs on vision-capable models.
    pub test_image: bool,

    /// Whether to test audio inputs on audio-capable models.
    pub test_audio: bool,

    /// Whether to test reasoning / thinking capability.
    pub test_thinking: bool,

    /// Discrete thinking levels to evaluate if `test_thinking` is enabled.
    pub thinking_levels: Vec<ThinkingLevel>,

    /// Number of repeat trial requests per probe.
    pub test_tries: usize,

    /// Optional regular expression filter restricting tested models.
    pub only: Option<Regex>,
}

impl Default for ProbeRunOptions {
    fn default() -> Self {
        Self {
            test_text: true,
            test_image: true,
            test_audio: true,
            test_thinking: true,
            thinking_levels: vec![
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ],
            test_tries: crate::config::DEFAULT_TEST_TRIES,
            only: None,
        }
    }
}

/// Aggregated benchmark outcomes for a specific model ID across all executed probes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelBenchResult {
    /// Identifier of the tested model.
    pub model_id: String,

    /// Results of all capability probes executed for this model.
    pub probes: Vec<SingleProbeResult>,
}

/// Aggregated statistical outcomes for a single capability probe configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingleProbeResult {
    /// Descriptive name of the probe (e.g. `openai-text`, `anthropic-messages`, `thinking-low`).
    pub name: String,

    /// Upstream protocol channel used (`openai` or `anthropic`).
    pub channel: String,

    /// Modality payload tested (`text`, `image`, or `audio`).
    pub payload: String,

    /// Thinking / reasoning level applied.
    pub thinking_level: ThinkingLevel,

    /// Statistical aggregation over all trial attempts.
    pub stats: AggregateStats,

    /// Sample error messages collected from failed trials, if any.
    pub error_samples: Vec<String>,
}

/// Internal descriptor defining an individual probe to run against a model.
#[derive(Debug, Clone)]
struct ProbeTask {
    name: String,
    channel: String,
    payload: ProbePayload,
    thinking_level: ThinkingLevel,
}

/// Execute full capability benchmarks across discovered models according to the given options.
///
/// # Errors
///
/// Returns [`CoreError::NoKeys`] if no API authentication key is configured.
/// Returns [`CoreError::NoModels`] if model discovery returns empty or all models are filtered out.
/// Returns [`CoreError::Api`] or [`CoreError::Http`] if communication with the upstream API fails.
#[tracing::instrument(skip(config, http, options), fields(parallelism = config.test_parallelism, tries = options.test_tries))]
#[allow(clippy::too_many_lines)]
pub async fn run_bench(
    config: &Config,
    http: &reqwest::Client,
    options: &ProbeRunOptions,
) -> Result<Vec<ModelBenchResult>, CoreError> {
    let api_key = config.require_api_key()?;
    let all_models = fetch_models(config, http).await?;

    let filtered_models = filter_models(&all_models, options.only.as_ref());
    if filtered_models.is_empty() {
        tracing::warn!("no models matched filter criteria");
        return Err(CoreError::NoModels);
    }

    let anthropic_regexes = compile_patterns(&config.anthropic_patterns);
    let vision_regexes = compile_patterns(&config.vision_patterns);
    let audio_regexes = compile_patterns(&config.audio_patterns);

    let semaphore = Arc::new(Semaphore::new(config.test_parallelism.max(1)));
    let tries = options.test_tries.max(1);

    let mut model_bench_results = Vec::with_capacity(filtered_models.len());

    for model_id in &filtered_models {
        let is_anthropic = is_anthropic_model(model_id, &anthropic_regexes);
        let is_vision = is_vision_model(model_id, &vision_regexes);
        let is_audio = is_audio_model(model_id, &audio_regexes);

        let mut probe_tasks = Vec::new();

        if options.test_text {
            if is_anthropic {
                probe_tasks.push(ProbeTask {
                    name: "anthropic-messages".to_string(),
                    channel: "anthropic".to_string(),
                    payload: ProbePayload::Text,
                    thinking_level: ThinkingLevel::Off,
                });
            } else {
                probe_tasks.push(ProbeTask {
                    name: "openai-text".to_string(),
                    channel: "openai".to_string(),
                    payload: ProbePayload::Text,
                    thinking_level: ThinkingLevel::Off,
                });
            }
        }

        if options.test_image && is_vision && !is_anthropic {
            probe_tasks.push(ProbeTask {
                name: "openai-image".to_string(),
                channel: "openai".to_string(),
                payload: ProbePayload::Image,
                thinking_level: ThinkingLevel::Off,
            });
        }

        if options.test_audio && is_audio && !is_anthropic {
            probe_tasks.push(ProbeTask {
                name: "openai-audio".to_string(),
                channel: "openai".to_string(),
                payload: ProbePayload::Audio,
                thinking_level: ThinkingLevel::Off,
            });
        }

        if options.test_thinking {
            for &level in &options.thinking_levels {
                if level == ThinkingLevel::Off {
                    continue;
                }
                let channel = if is_anthropic { "anthropic" } else { "openai" };
                probe_tasks.push(ProbeTask {
                    name: format!("thinking-{level}"),
                    channel: channel.to_string(),
                    payload: ProbePayload::Text,
                    thinking_level: level,
                });
            }
        }

        let mut single_probe_results = Vec::with_capacity(probe_tasks.len());

        for task in probe_tasks {
            let mut join_handles = Vec::with_capacity(tries);

            for _ in 0..tries {
                let sem = Arc::clone(&semaphore);
                let client = http.clone();
                let base_url = config.upstream_url.clone();
                let key = api_key.to_string();
                let model = model_id.clone();
                let task_clone = task.clone();

                let handle = tokio::spawn(async move {
                    let Ok(_permit) = sem.acquire().await else {
                        return Err("semaphore acquired failed".to_string());
                    };

                    let result: Result<TrialMetric, CoreError> =
                        if task_clone.channel == "anthropic" {
                            probe_anthropic_messages(
                                &client,
                                &base_url,
                                &key,
                                &model,
                                task_clone.thinking_level,
                                true,
                            )
                            .await
                        } else {
                            probe_openai_chat(
                                &client,
                                &base_url,
                                &key,
                                &model,
                                task_clone.payload,
                                task_clone.thinking_level,
                                true,
                            )
                            .await
                        };

                    result.map_err(|e| e.to_string())
                });

                join_handles.push(handle);
            }

            let mut trial_results = Vec::with_capacity(tries);
            let mut error_samples = Vec::new();

            for handle in join_handles {
                match handle.await {
                    Ok(Ok(metric)) => trial_results.push(Ok(metric)),
                    Ok(Err(err_msg)) => {
                        if error_samples.len() < 3 && !error_samples.contains(&err_msg) {
                            error_samples.push(err_msg.clone());
                        }
                        trial_results.push(Err(err_msg));
                    }
                    Err(join_err) => {
                        let msg = format!("join error: {join_err}");
                        if error_samples.len() < 3 && !error_samples.contains(&msg) {
                            error_samples.push(msg.clone());
                        }
                        trial_results.push(Err(msg));
                    }
                }
            }

            let stats = aggregate_trials(&trial_results);

            single_probe_results.push(SingleProbeResult {
                name: task.name,
                channel: task.channel,
                payload: task.payload.as_str().to_string(),
                thinking_level: task.thinking_level,
                stats,
                error_samples,
            });
        }

        model_bench_results.push(ModelBenchResult {
            model_id: model_id.clone(),
            probes: single_probe_results,
        });
    }

    Ok(model_bench_results)
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::extract::State;
    use axum::http::StatusCode as AxumStatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};

    use super::*;

    #[derive(Clone)]
    struct MockState {
        chat_attempts: Arc<AtomicUsize>,
    }

    async fn mock_models_handler() -> impl IntoResponse {
        Json(serde_json::json!({
            "object": "list",
            "data": [
                { "id": "gpt-4o" },
                { "id": "claude-3-5-sonnet" }
            ]
        }))
    }

    async fn mock_openai_chat_handler(
        State(state): State<MockState>,
        Json(_payload): Json<serde_json::Value>,
    ) -> Response {
        let attempt = state.chat_attempts.fetch_add(1, Ordering::SeqCst);

        // Inject 500 error on the very first attempt to verify retry logic
        if attempt == 0 {
            return (
                AxumStatusCode::INTERNAL_SERVER_ERROR,
                "temporary server error",
            )
                .into_response();
        }

        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"pong\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":5,\"completion_tokens_details\":{\"reasoning_tokens\":2}}}\n\ndata: [DONE]\n\n";

        (
            [("content-type", "text/event-stream")],
            sse_body.to_string(),
        )
            .into_response()
    }

    async fn mock_anthropic_messages_handler(Json(_payload): Json<serde_json::Value>) -> Response {
        let sse_body = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"output_tokens\":1}}}\n\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":4,\"thinking_tokens\":10}}\n\n";

        (
            [("content-type", "text/event-stream")],
            sse_body.to_string(),
        )
            .into_response()
    }

    async fn setup_mock_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let state = MockState {
            chat_attempts: Arc::new(AtomicUsize::new(0)),
        };

        let app = Router::new()
            .route("/v1/models", get(mock_models_handler))
            .route("/v1/chat/completions", post(mock_openai_chat_handler))
            .route("/v1/messages", post(mock_anthropic_messages_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        (addr, handle)
    }

    #[tokio::test]
    async fn test_run_bench_against_mock_server() {
        let (addr, server_handle) = setup_mock_server().await;
        let http = reqwest::Client::builder().build().unwrap();

        let config = Config {
            upstream_url: format!("http://{addr}"),
            api_key: Some("test-api-key".to_string()),
            test_tries: 2,
            test_parallelism: 2,
            request_timeout_secs: 5,
            anthropic_patterns: vec!["^claude".to_string()],
            vision_patterns: vec!["^gpt-".to_string()],
            audio_patterns: vec![],
            otel_destination: None,
            otel_api_key: None,
        };

        let options = ProbeRunOptions {
            test_text: true,
            test_image: true,
            test_audio: false,
            test_thinking: true,
            thinking_levels: vec![ThinkingLevel::Minimal],
            test_tries: 2,
            only: None,
        };

        let results = run_bench(&config, &http, &options)
            .await
            .expect("benchmarks should succeed");

        assert_eq!(results.len(), 2);

        // Verify gpt-4o results
        let gpt4 = results
            .iter()
            .find(|r| r.model_id == "gpt-4o")
            .expect("gpt-4o found");
        let gpt4_probe_names: Vec<&str> = gpt4.probes.iter().map(|p| p.name.as_str()).collect();
        assert!(gpt4_probe_names.contains(&"openai-text"));
        assert!(gpt4_probe_names.contains(&"openai-image"));
        assert!(gpt4_probe_names.contains(&"thinking-minimal"));

        // Verify claude results
        let claude = results
            .iter()
            .find(|r| r.model_id == "claude-3-5-sonnet")
            .expect("claude found");
        let claude_probe_names: Vec<&str> = claude.probes.iter().map(|p| p.name.as_str()).collect();
        assert!(claude_probe_names.contains(&"anthropic-messages"));
        assert!(claude_probe_names.contains(&"thinking-minimal"));

        server_handle.abort();
    }

    #[tokio::test]
    async fn test_run_bench_no_models_matching_filter() {
        let (addr, server_handle) = setup_mock_server().await;
        let http = reqwest::Client::builder().build().unwrap();

        let config = Config {
            upstream_url: format!("http://{addr}"),
            api_key: Some("test-api-key".to_string()),
            ..Default::default()
        };

        let options = ProbeRunOptions {
            only: Some(Regex::new("^nonexistent-model").unwrap()),
            ..Default::default()
        };

        let err = run_bench(&config, &http, &options).await.unwrap_err();
        assert!(matches!(err, CoreError::NoModels));

        server_handle.abort();
    }
}
