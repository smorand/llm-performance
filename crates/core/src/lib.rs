//! Core library for `llm-performance`: model capability testing, benchmarking, configuration, and telemetry.

pub mod config;
pub mod error;
pub mod models;
pub mod payloads;
pub mod probe;
pub mod runner;
pub mod stats;
pub mod telemetry;
pub mod thinking;

pub use config::{
    APP_NAME, Config, ConfigOverrides, DEFAULT_REQUEST_TIMEOUT_SECS, DEFAULT_TEST_PARALLELISM,
    DEFAULT_TEST_TRIES, DEFAULT_UPSTREAM_URL,
};
pub use error::CoreError;
pub use models::{
    compile_patterns, fetch_models, filter_models, is_anthropic_model, is_audio_model,
    is_vision_model, try_compile_patterns,
};
pub use payloads::{
    PROBE_PROMPT, TEST_PNG_BASE64, TEST_WAV_BASE64, decode_test_png, decode_test_wav,
};
pub use probe::{ProbePayload, probe_anthropic_messages, probe_openai_chat};
pub use runner::{ModelBenchResult, ProbeRunOptions, SingleProbeResult, run_bench};
pub use stats::{
    AggregateStats, PercentileStats, TrialMetric, aggregate_trials, compute_percentiles,
};
pub use telemetry::{TelemetryGuard, init as init_telemetry};
pub use thinking::ThinkingLevel;

/// Crate package version from Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Helper future listening for `SIGINT` (Ctrl+C) or `SIGTERM` signals for graceful shutdown.
pub async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c().await.expect("install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            tracing::info!("received Ctrl+C shutdown signal");
        },
        () = terminate => {
            tracing::info!("received SIGTERM shutdown signal");
        },
    }
}
