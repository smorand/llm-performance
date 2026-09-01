//! Observability setup combining tracing, daily rolling JSON file logs, and optional OpenTelemetry OTLP export.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{WithExportConfig, WithTonicConfig};
use opentelemetry_sdk::trace::SdkTracerProvider;
use tonic::metadata::MetadataMap;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, fmt};

use crate::config::Config;
use crate::error::CoreError;

/// Guard maintaining active background logging workers and flushing telemetry on drop.
pub struct TelemetryGuard {
    _file_guard: Option<WorkerGuard>,
    tracer_provider: Option<SdkTracerProvider>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.tracer_provider.take()
            && let Err(e) = provider.shutdown()
        {
            eprintln!("failed to cleanly shut down OpenTelemetry tracer provider: {e}");
        }
    }
}

/// Initialize tracing with:
/// - Stderr layer (human-readable, filtered by `RUST_LOG` or default)
/// - Daily rolling JSON file layer in `<cache_dir>/logs`
/// - Optional OpenTelemetry OTLP export if `otel_destination` is configured
///
/// # Errors
///
/// Returns [`CoreError::Telemetry`] or [`CoreError::Dirs`] if log directories cannot be created
/// or OpenTelemetry fails to initialize.
pub fn init(
    config: &Config,
    default_level: Option<tracing::Level>,
) -> Result<TelemetryGuard, CoreError> {
    let default_directive = default_level.map_or("info", |l| l.as_str());

    // 1. Stderr human-readable layer
    let stderr_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directive));
    let stderr_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(stderr_filter);

    // 2. Rolling JSON file layer
    let (file_layer, file_guard) = match Config::log_dir() {
        Ok(log_dir) => {
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                eprintln!(
                    "warning: failed to create log directory {}: {e}",
                    log_dir.display()
                );
                (None, None)
            } else {
                let file_appender =
                    tracing_appender::rolling::daily(&log_dir, "llm-performance.log");
                let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
                let file_filter =
                    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));
                let layer = fmt::layer()
                    .json()
                    .with_writer(non_blocking)
                    .with_filter(file_filter);
                (Some(layer), Some(guard))
            }
        }
        Err(e) => {
            eprintln!("warning: failed to determine cache log dir: {e}");
            (None, None)
        }
    };

    // 3. Optional OpenTelemetry OTLP layer
    let (otel_layer, tracer_provider) = if let Some(endpoint) = &config.otel_destination {
        let mut builder = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint);

        if let Some(key) = &config.otel_api_key {
            let mut metadata = MetadataMap::new();
            let auth_header = format!("Bearer {key}");
            let header_val = auth_header
                .parse()
                .map_err(|e| CoreError::Telemetry(format!("invalid otel auth header: {e}")))?;
            metadata.insert("authorization", header_val);
            builder = builder.with_metadata(metadata);
        }

        let exporter = builder.build().map_err(|e| {
            CoreError::Telemetry(format!("failed to build OTLP span exporter: {e}"))
        })?;

        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build();

        let tracer = provider.tracer("llm-performance");
        let layer = tracing_opentelemetry::layer().with_tracer(tracer);

        (Some(layer), Some(provider))
    } else {
        (None, None)
    };

    // Compose layers into global subscriber registry
    let registry = tracing_subscriber::registry()
        .with(stderr_layer)
        .with(file_layer)
        .with(otel_layer);

    registry.try_init().map_err(|e| {
        CoreError::Telemetry(format!("failed to initialize tracing subscriber: {e}"))
    })?;

    Ok(TelemetryGuard {
        _file_guard: file_guard,
        tracer_provider,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_guard_drop_safety() {
        let guard = TelemetryGuard {
            _file_guard: None,
            tracer_provider: None,
        };
        drop(guard);
    }

    #[test]
    fn test_telemetry_init_default() {
        let config = Config::default();
        // init may succeed or fail if global subscriber is already initialized by another test
        let _ = init(&config, Some(tracing::Level::DEBUG));
    }
}
