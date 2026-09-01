//! Configuration loading, layering, and defaults.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use etcetera::{BaseStrategy, choose_base_strategy};
use figment::Figment;
use figment::providers::{Env, Format, Serialized, Toml};
use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// Application identifier used for configuration and cache directories.
pub const APP_NAME: &str = "llm-performance";

/// Default upstream base URL for IBM ICA API.
pub const DEFAULT_UPSTREAM_URL: &str = "https://api.nextgen-beta.ica.ibm.com/ica";

/// Default number of trials per model capability benchmark.
pub const DEFAULT_TEST_TRIES: usize = 10;

/// Default concurrent worker parallelism.
pub const DEFAULT_TEST_PARALLELISM: usize = 4;

/// Default HTTP request timeout in seconds.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;

/// Fully resolved application configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Upstream gateway or API base URL.
    pub upstream_url: String,

    /// API authentication key (Bearer token).
    pub api_key: Option<String>,

    /// Number of repeat trials for benchmark tests.
    pub test_tries: usize,

    /// Number of concurrent requests during benchmarking.
    pub test_parallelism: usize,

    /// HTTP request timeout in seconds.
    pub request_timeout_secs: u64,

    /// Regex patterns identifying native Anthropic models.
    pub anthropic_patterns: Vec<String>,

    /// Regex patterns identifying models supporting vision / image input.
    pub vision_patterns: Vec<String>,

    /// Regex patterns identifying models supporting audio input.
    pub audio_patterns: Vec<String>,

    /// Optional OpenTelemetry OTLP destination endpoint (e.g. `http://localhost:4317`).
    pub otel_destination: Option<String>,

    /// Optional Bearer authorization key for OpenTelemetry OTLP export.
    pub otel_api_key: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            upstream_url: DEFAULT_UPSTREAM_URL.to_string(),
            api_key: None,
            test_tries: DEFAULT_TEST_TRIES,
            test_parallelism: DEFAULT_TEST_PARALLELISM,
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            anthropic_patterns: vec!["^claude".to_string()],
            vision_patterns: vec![
                "^claude-".to_string(),
                "^gpt-".to_string(),
                "^gemini-".to_string(),
            ],
            audio_patterns: vec!["^gemini-".to_string()],
            otel_destination: None,
            otel_api_key: None,
        }
    }
}

/// Sparse configuration overrides provided by CLI flags or programmatic invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigOverrides {
    /// Override upstream base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_url: Option<String>,

    /// Override API authentication key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Override number of benchmark trials.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_tries: Option<usize>,

    /// Override benchmark parallelism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_parallelism: Option<usize>,

    /// Override HTTP request timeout in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_timeout_secs: Option<u64>,

    /// Override Anthropic model regex patterns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_patterns: Option<Vec<String>>,

    /// Override vision-capable model regex patterns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_patterns: Option<Vec<String>>,

    /// Override audio-capable model regex patterns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_patterns: Option<Vec<String>>,

    /// Override OpenTelemetry destination endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub otel_destination: Option<String>,

    /// Override OpenTelemetry API key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub otel_api_key: Option<String>,
}

impl Config {
    /// Return the XDG configuration directory (`$XDG_CONFIG_HOME/llm-performance` or `~/.config/llm-performance`).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Dirs`] if standard base directories cannot be resolved.
    pub fn config_dir() -> Result<PathBuf, CoreError> {
        let strategy = choose_base_strategy().map_err(|e| CoreError::Dirs(e.to_string()))?;
        Ok(strategy.config_dir().join(APP_NAME))
    }

    /// Return the XDG cache root directory (`$XDG_CACHE_HOME/llm-performance` or `~/.cache/llm-performance`).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Dirs`] if standard base directories cannot be resolved.
    pub fn cache_dir() -> Result<PathBuf, CoreError> {
        let strategy = choose_base_strategy().map_err(|e| CoreError::Dirs(e.to_string()))?;
        Ok(strategy.cache_dir().join(APP_NAME))
    }

    /// Return the directory where rolling logs are stored (`<cache_dir>/logs`).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Dirs`] if standard base directories cannot be resolved.
    pub fn log_dir() -> Result<PathBuf, CoreError> {
        Ok(Self::cache_dir()?.join("logs"))
    }

    /// Return the HTTP request timeout as a [`Duration`].
    #[must_use]
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }

    /// Resolve effective API key or return [`CoreError::NoKeys`] if absent or empty.
    ///
    /// Automatically extracts the first valid key token when given a multi-line,
    /// colon-delimited, or commented string (e.g. `sk-key # comment` or `sk-1:sk-2`).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::NoKeys`] if `api_key` is `None` or blank.
    pub fn require_api_key(&self) -> Result<&str, CoreError> {
        match &self.api_key {
            Some(raw) => {
                for line in raw.lines() {
                    let uncommented = line.split('#').next().unwrap_or("").trim();
                    for token in uncommented.split(':') {
                        let trimmed = token.trim();
                        if !trimmed.is_empty() {
                            return Ok(trimmed);
                        }
                    }
                }
                Err(CoreError::NoKeys)
            }
            None => Err(CoreError::NoKeys),
        }
    }

    /// Load layered configuration with precedence:
    /// Defaults < TOML file < Fallback Env (`IBM_ICA_*`) < Env (`LLM_PERFM_*`) < Overrides (CLI).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Config`] or [`CoreError::Dirs`] if loading or extraction fails.
    pub fn load(
        config_path: Option<&Path>,
        overrides: &ConfigOverrides,
    ) -> Result<Self, CoreError> {
        let path = match config_path {
            Some(p) => p.to_path_buf(),
            None => Self::config_dir()?.join("config.toml"),
        };

        let mut fallback_map = BTreeMap::new();
        if let Ok(url) = std::env::var("IBM_ICA_BASE_URL")
            && !url.trim().is_empty()
        {
            fallback_map.insert("upstream_url".to_string(), url.trim().to_string());
        }
        if let Ok(key) = std::env::var("IBM_ICA_MODEL_KEY")
            && !key.trim().is_empty()
        {
            fallback_map.insert("api_key".to_string(), key.trim().to_string());
        }

        let mut figment = Figment::from(Serialized::defaults(Config::default()));

        if !fallback_map.is_empty() {
            figment = figment.merge(Serialized::defaults(fallback_map));
        }

        figment
            .merge(Toml::file(path))
            .merge(Env::prefixed("LLM_PERFM_"))
            .merge(Serialized::defaults(overrides))
            .extract()
            .map_err(CoreError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = Config::default();
        assert_eq!(cfg.upstream_url, DEFAULT_UPSTREAM_URL);
        assert_eq!(cfg.api_key, None);
        assert_eq!(cfg.test_tries, 10);
        assert_eq!(cfg.test_parallelism, 4);
        assert_eq!(cfg.request_timeout_secs, 60);
        assert_eq!(cfg.request_timeout(), Duration::from_secs(60));
        assert_eq!(cfg.anthropic_patterns, vec!["^claude".to_string()]);
        assert_eq!(
            cfg.vision_patterns,
            vec![
                "^claude-".to_string(),
                "^gpt-".to_string(),
                "^gemini-".to_string()
            ]
        );
        assert_eq!(cfg.audio_patterns, vec!["^gemini-".to_string()]);
        assert_eq!(cfg.otel_destination, None);
        assert_eq!(cfg.otel_api_key, None);
        assert!(matches!(cfg.require_api_key(), Err(CoreError::NoKeys)));
    }

    #[test]
    fn require_api_key_valid() {
        let cfg = Config {
            api_key: Some("test-key-123".to_string()),
            ..Default::default()
        };
        assert_eq!(cfg.require_api_key().unwrap(), "test-key-123");

        let blank_cfg = Config {
            api_key: Some("   ".to_string()),
            ..Default::default()
        };
        assert!(matches!(
            blank_cfg.require_api_key(),
            Err(CoreError::NoKeys)
        ));
    }

    #[test]
    fn xdg_directories_resolution() {
        let config_dir = Config::config_dir().expect("config_dir should resolve");
        assert!(config_dir.ends_with(APP_NAME));

        let cache_dir = Config::cache_dir().expect("cache_dir should resolve");
        assert!(cache_dir.ends_with(APP_NAME));

        let log_dir = Config::log_dir().expect("log_dir should resolve");
        assert!(log_dir.ends_with(format!("{APP_NAME}/logs")));
    }

    #[test]
    fn load_with_overrides() {
        let overrides = ConfigOverrides {
            upstream_url: Some("https://custom.gateway.com".to_string()),
            api_key: Some("secret-override".to_string()),
            test_tries: Some(25),
            test_parallelism: Some(8),
            request_timeout_secs: Some(120),
            anthropic_patterns: Some(vec!["^anthropic".to_string()]),
            vision_patterns: Some(vec!["^vision".to_string()]),
            audio_patterns: Some(vec!["^audio".to_string()]),
            otel_destination: Some("http://localhost:4317".to_string()),
            otel_api_key: Some("otel-key".to_string()),
        };

        // Load with non-existent custom path so it only uses defaults + overrides
        let dummy_path = Path::new("/tmp/non_existent_config_for_test.toml");
        let cfg = Config::load(Some(dummy_path), &overrides).expect("should load config");

        assert_eq!(cfg.upstream_url, "https://custom.gateway.com");
        assert_eq!(cfg.api_key.as_deref(), Some("secret-override"));
        assert_eq!(cfg.test_tries, 25);
        assert_eq!(cfg.test_parallelism, 8);
        assert_eq!(cfg.request_timeout_secs, 120);
        assert_eq!(cfg.anthropic_patterns, vec!["^anthropic".to_string()]);
        assert_eq!(cfg.vision_patterns, vec!["^vision".to_string()]);
        assert_eq!(cfg.audio_patterns, vec!["^audio".to_string()]);
        assert_eq!(
            cfg.otel_destination.as_deref(),
            Some("http://localhost:4317")
        );
        assert_eq!(cfg.otel_api_key.as_deref(), Some("otel-key"));
    }

    #[test]
    fn load_with_toml_string() {
        let toml_content = r#"
            upstream_url = "https://toml.upstream.com"
            api_key = "toml-key"
            test_tries = 5
            test_parallelism = 2
            request_timeout_secs = 30
            anthropic_patterns = ["^claude-3-5"]
            vision_patterns = ["^vision-only"]
            audio_patterns = ["^audio-only"]
            otel_destination = "http://otel.internal:4317"
            otel_api_key = "otel-toml-key"
        "#;

        let config_file = std::env::temp_dir().join("llm_perf_test_config.toml");
        std::fs::write(&config_file, toml_content).unwrap();

        let overrides = ConfigOverrides::default();
        let cfg = Config::load(Some(&config_file), &overrides).expect("load from toml");
        let _ = std::fs::remove_file(&config_file);

        assert_eq!(cfg.upstream_url, "https://toml.upstream.com");
        assert_eq!(cfg.api_key.as_deref(), Some("toml-key"));
        assert_eq!(cfg.test_tries, 5);
        assert_eq!(cfg.test_parallelism, 2);
        assert_eq!(cfg.request_timeout_secs, 30);
        assert_eq!(cfg.anthropic_patterns, vec!["^claude-3-5".to_string()]);
        assert_eq!(cfg.vision_patterns, vec!["^vision-only".to_string()]);
        assert_eq!(cfg.audio_patterns, vec!["^audio-only".to_string()]);
        assert_eq!(
            cfg.otel_destination.as_deref(),
            Some("http://otel.internal:4317")
        );
        assert_eq!(cfg.otel_api_key.as_deref(), Some("otel-toml-key"));
    }

    #[test]
    fn load_partial_toml_keeps_defaults() {
        let toml_content = r#"
            upstream_url = "https://custom.upstream.org/v1"
        "#;

        let config_file = std::env::temp_dir().join("llm_perf_test_partial.toml");
        std::fs::write(&config_file, toml_content).unwrap();

        let overrides = ConfigOverrides::default();
        let cfg = Config::load(Some(&config_file), &overrides).expect("load partial config");
        let _ = std::fs::remove_file(&config_file);

        assert_eq!(cfg.upstream_url, "https://custom.upstream.org/v1");
        assert_eq!(cfg.test_tries, DEFAULT_TEST_TRIES);
        assert_eq!(cfg.test_parallelism, DEFAULT_TEST_PARALLELISM);
        assert_eq!(cfg.request_timeout_secs, DEFAULT_REQUEST_TIMEOUT_SECS);
        assert_eq!(cfg.anthropic_patterns, vec!["^claude".to_string()]);
    }

    #[test]
    fn config_overrides_equality_and_debug() {
        let o1 = ConfigOverrides {
            upstream_url: Some("url1".to_string()),
            ..Default::default()
        };
        let o2 = ConfigOverrides {
            upstream_url: Some("url1".to_string()),
            ..Default::default()
        };
        let o3 = ConfigOverrides {
            upstream_url: Some("url2".to_string()),
            ..Default::default()
        };
        assert_eq!(o1, o2);
        assert_ne!(o1, o3);
        assert!(format!("{o1:?}").contains("url1"));
    }
}
