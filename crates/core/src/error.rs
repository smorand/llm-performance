//! Error types for llm-performance-core.

use reqwest::StatusCode;

/// Core library error type.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Configuration loading or parsing error.
    #[error("configuration error: {0}")]
    Config(#[source] Box<figment::Error>),

    /// Directory resolution failure (XDG).
    #[error("directory resolution failed: {0}")]
    Dirs(String),

    /// HTTP client or transport error.
    #[error("http request failed: {0}")]
    Http(#[source] Box<reqwest::Error>),

    /// Upstream API error response with status code.
    #[error("upstream API error (status {status}): {message}")]
    Api {
        /// HTTP status code returned by upstream.
        status: StatusCode,
        /// Error message or body from upstream.
        message: String,
    },

    /// Missing API authentication key.
    #[error("no API key configured: set LLM_PERFM_API_KEY, IBM_ICA_MODEL_KEY, or pass --api-key")]
    NoKeys,

    /// No models discovered or matched.
    #[error("no models available or matched")]
    NoModels,

    /// Invalid regular expression pattern.
    #[error("invalid regular expression: {0}")]
    InvalidRegex(#[source] regex::Error),

    /// JSON serialization or deserialization error.
    #[error("json error: {0}")]
    Json(#[source] serde_json::Error),

    /// Invalid thinking / reasoning level string.
    #[error("invalid thinking level '{0}': expected off, minimal, low, medium, or high")]
    InvalidThinkingLevel(String),

    /// Telemetry or OpenTelemetry initialization error.
    #[error("telemetry error: {0}")]
    Telemetry(String),

    /// Standard I/O error.
    #[error("io error: {0}")]
    Io(#[source] std::io::Error),
}

impl From<figment::Error> for CoreError {
    fn from(e: figment::Error) -> Self {
        Self::Config(Box::new(e))
    }
}

impl From<reqwest::Error> for CoreError {
    fn from(e: reqwest::Error) -> Self {
        Self::Http(Box::new(e))
    }
}

impl From<regex::Error> for CoreError {
    fn from(e: regex::Error) -> Self {
        Self::InvalidRegex(e)
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_formatting() {
        let err = CoreError::Dirs("home not found".to_string());
        assert_eq!(
            err.to_string(),
            "directory resolution failed: home not found"
        );

        let err = CoreError::NoKeys;
        assert!(err.to_string().contains("no API key configured"));

        let err = CoreError::NoModels;
        assert_eq!(err.to_string(), "no models available or matched");

        let err = CoreError::InvalidThinkingLevel("extreme".to_string());
        assert!(err.to_string().contains("invalid thinking level 'extreme'"));

        let err = CoreError::Telemetry("otel init failed".to_string());
        assert_eq!(err.to_string(), "telemetry error: otel init failed");

        let err = CoreError::Api {
            status: StatusCode::UNAUTHORIZED,
            message: "invalid bearer token".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "upstream API error (status 401 Unauthorized): invalid bearer token"
        );
    }

    #[test]
    #[allow(clippy::invalid_regex)]
    fn from_conversions() {
        let json_err = serde_json::from_str::<i32>("not an int").unwrap_err();
        let core_err: CoreError = json_err.into();
        assert!(matches!(core_err, CoreError::Json(_)));

        let regex_err = regex::Regex::new("[").unwrap_err();
        let core_err: CoreError = regex_err.into();
        assert!(matches!(core_err, CoreError::InvalidRegex(_)));

        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let core_err: CoreError = io_err.into();
        assert!(matches!(core_err, CoreError::Io(_)));
    }
}
