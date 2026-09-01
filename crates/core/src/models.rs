//! Model discovery, pattern matching, and capability detection.

use regex::Regex;
use serde::Deserialize;

use crate::config::Config;
use crate::error::CoreError;

/// Raw model entry in upstream OpenAI-compatible models list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ModelEntry {
    Object { id: String },
    NamedObject { name: String },
    String(String),
}

impl ModelEntry {
    fn into_id(self) -> String {
        match self {
            Self::Object { id } => id,
            Self::NamedObject { name } => name,
            Self::String(s) => s,
        }
    }
}

/// Flexible container for OpenAI or alternative `/v1/models` JSON payloads.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ModelsResponseWrapper {
    Envelope {
        #[serde(default)]
        data: Vec<ModelEntry>,
        #[serde(default)]
        models: Vec<ModelEntry>,
    },
    List(Vec<ModelEntry>),
}

/// Fetch available models from `{upstream_url}/v1/models` using Bearer authentication.
///
/// # Errors
///
/// Returns [`CoreError::NoKeys`] if no API key is configured.
/// Returns [`CoreError::Api`] if the upstream server returns a non-2xx status code.
/// Returns [`CoreError::Http`] if network transport fails.
/// Returns [`CoreError::Json`] if the response body cannot be parsed.
/// Returns [`CoreError::NoModels`] if the response contains zero models.
#[tracing::instrument(skip(config, http), fields(http.url = %config.upstream_url))]
pub async fn fetch_models(
    config: &Config,
    http: &reqwest::Client,
) -> Result<Vec<String>, CoreError> {
    let api_key = config.require_api_key()?;
    let url = format!("{}/v1/models", config.upstream_url.trim_end_matches('/'));

    let response = http
        .get(&url)
        .bearer_auth(api_key)
        .timeout(config.request_timeout())
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let error_body = response.text().await.unwrap_or_default();
        return Err(CoreError::Api {
            status,
            message: error_body,
        });
    }

    let text = response.text().await?;
    let parsed: ModelsResponseWrapper = serde_json::from_str(&text)?;

    let entries = match parsed {
        ModelsResponseWrapper::Envelope { data, models } => {
            if data.is_empty() {
                models
            } else {
                data
            }
        }
        ModelsResponseWrapper::List(list) => list,
    };

    let mut model_ids: Vec<String> = entries
        .into_iter()
        .map(ModelEntry::into_id)
        .filter(|id| !id.trim().is_empty())
        .collect();

    if model_ids.is_empty() {
        return Err(CoreError::NoModels);
    }

    model_ids.sort();
    model_ids.dedup();

    tracing::info!(count = model_ids.len(), "discovered available models");
    Ok(model_ids)
}

/// Compile a slice of regex pattern strings into [`Regex`] instances.
///
/// Any invalid patterns are logged with `tracing::warn` and omitted from the returned list.
#[must_use]
pub fn compile_patterns(patterns: &[String]) -> Vec<Regex> {
    patterns
        .iter()
        .filter_map(|p| match Regex::new(p) {
            Ok(re) => Some(re),
            Err(e) => {
                tracing::warn!(pattern = %p, error = %e, "ignoring invalid regex pattern");
                None
            }
        })
        .collect()
}

/// Strictly compile a slice of regex pattern strings into [`Regex`] instances or fail on the first error.
///
/// # Errors
///
/// Returns [`CoreError::InvalidRegex`] if any pattern fails compilation.
pub fn try_compile_patterns(patterns: &[String]) -> Result<Vec<Regex>, CoreError> {
    patterns
        .iter()
        .map(|p| Regex::new(p).map_err(CoreError::from))
        .collect()
}

/// Test whether a model ID matches Anthropic native messaging endpoints.
#[must_use]
pub fn is_anthropic_model(id: &str, patterns: &[Regex]) -> bool {
    patterns.iter().any(|re| re.is_match(id))
}

/// Test whether a model ID supports vision / image payloads.
#[must_use]
pub fn is_vision_model(id: &str, patterns: &[Regex]) -> bool {
    patterns.iter().any(|re| re.is_match(id))
}

/// Test whether a model ID supports audio payloads.
#[must_use]
pub fn is_audio_model(id: &str, patterns: &[Regex]) -> bool {
    patterns.iter().any(|re| re.is_match(id))
}

/// Filter a list of model IDs by an optional regular expression filter.
///
/// If `pattern` is `None`, all models are returned untouched.
#[must_use]
pub fn filter_models(models: &[String], pattern: Option<&Regex>) -> Vec<String> {
    match pattern {
        Some(re) => models.iter().filter(|m| re.is_match(m)).cloned().collect(),
        None => models.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_compilation() {
        let patterns = vec![
            "^claude-3-5".to_string(),
            "[invalid-regex".to_string(),
            "^gpt-4o".to_string(),
        ];
        let compiled = compile_patterns(&patterns);
        assert_eq!(compiled.len(), 2);
        assert!(compiled[0].is_match("claude-3-5-sonnet"));
        assert!(compiled[1].is_match("gpt-4o-mini"));

        assert!(try_compile_patterns(&patterns).is_err());
        let valid_patterns = vec!["^claude-".to_string(), "^gpt-".to_string()];
        assert!(try_compile_patterns(&valid_patterns).is_ok());
    }

    #[test]
    fn capability_detection() {
        let anthropic_regexes = compile_patterns(&["^claude".to_string()]);
        let vision_regexes = compile_patterns(&[
            "^claude-".to_string(),
            "^gpt-".to_string(),
            "^gemini-".to_string(),
        ]);
        let audio_regexes = compile_patterns(&["^gemini-".to_string()]);

        assert!(is_anthropic_model("claude-3-5-sonnet", &anthropic_regexes));
        assert!(!is_anthropic_model("gpt-4o", &anthropic_regexes));

        assert!(is_vision_model("claude-3-5-haiku", &vision_regexes));
        assert!(is_vision_model("gpt-4o", &vision_regexes));
        assert!(is_vision_model("gemini-2.0-flash", &vision_regexes));
        assert!(!is_vision_model("llama-3-70b", &vision_regexes));

        assert!(is_audio_model("gemini-1.5-pro", &audio_regexes));
        assert!(!is_audio_model("claude-3-5-sonnet", &audio_regexes));
    }

    #[test]
    fn model_filtering() {
        let models = vec![
            "claude-3-5-sonnet".to_string(),
            "gpt-4o".to_string(),
            "llama-3-8b".to_string(),
        ];

        let unfiltered = filter_models(&models, None);
        assert_eq!(unfiltered, models);

        let re = Regex::new("^claude").unwrap();
        let filtered = filter_models(&models, Some(&re));
        assert_eq!(filtered, vec!["claude-3-5-sonnet".to_string()]);
    }

    #[tokio::test]
    async fn fetch_models_success() {
        use axum::routing::get;
        use axum::{Json, Router};
        use serde_json::json;

        let app = Router::new().route(
            "/v1/models",
            get(|| async {
                Json(json!({
                    "object": "list",
                    "data": [
                        {"id": "model-b"},
                        {"id": "model-a"}
                    ]
                }))
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let config = Config {
            upstream_url: format!("http://127.0.0.1:{port}"),
            api_key: Some("test-key".to_string()),
            ..Default::default()
        };

        let http = reqwest::Client::new();
        let models = fetch_models(&config, &http).await.unwrap();
        assert_eq!(models, vec!["model-a", "model-b"]);
    }

    #[tokio::test]
    async fn fetch_models_missing_api_key() {
        let config = Config::default();
        let http = reqwest::Client::new();
        let result = fetch_models(&config, &http).await;
        assert!(matches!(result, Err(CoreError::NoKeys)));
    }

    #[tokio::test]
    async fn fetch_models_upstream_error() {
        use axum::Router;
        use axum::http::StatusCode;
        use axum::routing::get;

        let app = Router::new().route(
            "/v1/models",
            get(|| async { (StatusCode::UNAUTHORIZED, "invalid token") }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let config = Config {
            upstream_url: format!("http://127.0.0.1:{port}"),
            api_key: Some("bad-key".to_string()),
            ..Default::default()
        };

        let http = reqwest::Client::new();
        let result = fetch_models(&config, &http).await;
        match result {
            Err(CoreError::Api { status, message }) => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert_eq!(message, "invalid token");
            }
            other => panic!("expected CoreError::Api, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fetch_models_empty_data() {
        use axum::routing::get;
        use axum::{Json, Router};
        use serde_json::json;

        let app = Router::new().route(
            "/v1/models",
            get(|| async {
                Json(json!({
                    "object": "list",
                    "data": []
                }))
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let config = Config {
            upstream_url: format!("http://127.0.0.1:{port}"),
            api_key: Some("test-key".to_string()),
            ..Default::default()
        };

        let http = reqwest::Client::new();
        let result = fetch_models(&config, &http).await;
        assert!(matches!(result, Err(CoreError::NoModels)));
    }
}
