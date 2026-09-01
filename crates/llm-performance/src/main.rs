//! CLI binary for `llm-performance`: model capability testing, benchmarking, and thinking evaluation.

use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use clap_verbosity_flag::{InfoLevel, Verbosity};
use regex::Regex;

use llm_performance_core::config::{Config, ConfigOverrides};
use llm_performance_core::models::{
    compile_patterns, fetch_models, filter_models, is_anthropic_model, is_audio_model,
    is_vision_model,
};
use llm_performance_core::runner::{ModelBenchResult, ProbeRunOptions, run_bench};
use llm_performance_core::telemetry::init as init_telemetry;
use llm_performance_core::thinking::ThinkingLevel;

#[derive(Debug, Parser)]
#[command(
    name = "llm-performance",
    about = "CLI tool to probe LLM model capabilities, evaluate generation performance, and measure the impact of reasoning/thinking levels.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    verbosity: Verbosity<InfoLevel>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// List models exposed by the upstream API and show detected capabilities.
    Models(ModelsArgs),

    /// Run capability and performance benchmarks against models.
    Test(TestArgs),
}

#[derive(Debug, Args)]
struct ModelsArgs {
    /// Filter model IDs matching this regular expression.
    #[arg(long, short = 'm')]
    model: Option<String>,

    /// Path to a custom TOML configuration file.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Output results as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
#[allow(clippy::struct_excessive_bools)]
struct TestArgs {
    /// Filter model IDs matching this regular expression.
    #[arg(long, short = 'm')]
    model: Option<String>,

    /// Number of repeat trial requests per probe (overrides config).
    #[arg(long, short = 't')]
    test_tries: Option<usize>,

    /// Test standard text generation (OpenAI and Anthropic native where eligible).
    #[arg(long)]
    test_text: bool,

    /// Test vision / image input capabilities on compatible models.
    #[arg(long)]
    image: bool,

    /// Test audio input capabilities on compatible models.
    #[arg(long)]
    audio: bool,

    /// Test reasoning / thinking capabilities across thinking levels.
    #[arg(long)]
    thinking: bool,

    /// Comma-separated list of thinking levels to evaluate (e.g. "off,minimal,low,medium,high").
    #[arg(long, value_delimiter = ',')]
    thinking_levels: Option<Vec<String>>,

    /// Show only probes or models that experienced failures.
    #[arg(long)]
    failures_only: bool,

    /// Output results as formatted JSON.
    #[arg(long)]
    json: bool,

    /// Path to a custom TOML configuration file.
    #[arg(long)]
    config: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(cli) {
        Ok(has_failures) => {
            if has_failures {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(1)
        }
    }
}

#[tokio::main]
async fn run(cli: Cli) -> Result<bool> {
    match cli.command {
        Commands::Models(args) => {
            let overrides = ConfigOverrides::default();
            let config = Config::load(args.config.as_deref(), &overrides)
                .context("failed to load configuration")?;
            let _guard = init_telemetry(&config, cli.verbosity.tracing_level_filter().into())
                .context("failed to initialize telemetry")?;

            let http = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
                .build()
                .context("failed to construct HTTP client")?;

            run_models(&config, &http, args).await?;
            Ok(false)
        }
        Commands::Test(args) => {
            let overrides = ConfigOverrides {
                test_tries: args.test_tries,
                ..ConfigOverrides::default()
            };
            let config = Config::load(args.config.as_deref(), &overrides)
                .context("failed to load configuration")?;
            let _guard = init_telemetry(&config, cli.verbosity.tracing_level_filter().into())
                .context("failed to initialize telemetry")?;

            let http = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
                .build()
                .context("failed to construct HTTP client")?;

            let has_failures = run_test(&config, &http, args).await?;
            Ok(has_failures)
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct ModelInfo {
    id: String,
    anthropic_native: bool,
    vision: bool,
    audio: bool,
}

async fn run_models(config: &Config, http: &reqwest::Client, args: ModelsArgs) -> Result<()> {
    let all_models = fetch_models(config, http)
        .await
        .context("failed to fetch models list from upstream API")?;

    let filter_re = if let Some(pattern) = &args.model {
        Some(Regex::new(pattern).with_context(|| format!("invalid regex filter: {pattern}"))?)
    } else {
        None
    };

    let filtered = filter_models(&all_models, filter_re.as_ref());
    if filtered.is_empty() {
        if args.json {
            println!("[]");
        } else {
            println!("No models matched filter criteria.");
        }
        return Ok(());
    }

    let anthropic_res = compile_patterns(&config.anthropic_patterns);
    let vision_res = compile_patterns(&config.vision_patterns);
    let audio_res = compile_patterns(&config.audio_patterns);

    let models_info: Vec<ModelInfo> = filtered
        .iter()
        .map(|id| ModelInfo {
            id: id.clone(),
            anthropic_native: is_anthropic_model(id, &anthropic_res),
            vision: is_vision_model(id, &vision_res),
            audio: is_audio_model(id, &audio_res),
        })
        .collect();

    if args.json {
        let json_str = serde_json::to_string_pretty(&models_info)
            .context("failed to serialize models JSON")?;
        println!("{json_str}");
        return Ok(());
    }

    print_models_table(&models_info);
    Ok(())
}

fn format_centered_icon(icon: &'static str, col_width: usize) -> String {
    let display_width = 2;
    let pad = col_width.saturating_sub(display_width);
    let left_pad = pad / 2;
    let right_pad = pad - left_pad;
    format!("{}{}{}", " ".repeat(left_pad), icon, " ".repeat(right_pad))
}

fn print_models_table(models: &[ModelInfo]) {
    let max_id_len = models
        .iter()
        .map(|m| m.id.len())
        .max()
        .unwrap_or(8)
        .max("MODEL ID".len());

    println!(
        "{:<width$}  {:^8}  {:^18}  {:^8}  {:^8}",
        "MODEL ID",
        "TEXT",
        "ANTHROPIC NATIVE",
        "VISION",
        "AUDIO",
        width = max_id_len
    );
    println!("{:-<width$}", "", width = max_id_len + 50);

    for m in models {
        let text_cell = format_centered_icon("✅", 8);
        let anthropic_cell = format_centered_icon(if m.anthropic_native { "✅" } else { "⏭️" }, 18);
        let vision_cell = format_centered_icon(if m.vision { "✅" } else { "❌" }, 8);
        let audio_cell = format_centered_icon(if m.audio { "✅" } else { "❌" }, 8);

        println!(
            "{:<width$}  {}  {}  {}  {}",
            m.id,
            text_cell,
            anthropic_cell,
            vision_cell,
            audio_cell,
            width = max_id_len
        );
    }
    println!("\nTotal: {} models", models.len());
}

async fn run_test(config: &Config, http: &reqwest::Client, args: TestArgs) -> Result<bool> {
    let filter_re = if let Some(pattern) = &args.model {
        Some(Regex::new(pattern).with_context(|| format!("invalid regex filter: {pattern}"))?)
    } else {
        None
    };

    let thinking_levels = if let Some(raw_levels) = &args.thinking_levels {
        let mut parsed = Vec::new();
        for s in raw_levels {
            let level = ThinkingLevel::from_str(s.trim())
                .map_err(|e| anyhow::anyhow!("unknown thinking level '{s}': {e}"))?;
            parsed.push(level);
        }
        parsed
    } else {
        vec![
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
        ]
    };

    // If no specific capability flag was passed, default to text test
    let any_specific_flag = args.test_text || args.image || args.audio || args.thinking;
    let test_text = if any_specific_flag {
        args.test_text
    } else {
        true
    };

    let tries = args.test_tries.unwrap_or(config.test_tries);

    let options = ProbeRunOptions {
        test_text,
        test_image: args.image,
        test_audio: args.audio,
        test_thinking: args.thinking,
        thinking_levels,
        test_tries: tries,
        only: filter_re,
    };

    let results = run_bench(config, http, &options)
        .await
        .context("benchmarking run failed")?;

    let has_failures = results
        .iter()
        .any(|m| m.probes.iter().any(|p| p.stats.failed_trials > 0));

    if args.json {
        let json_str =
            serde_json::to_string_pretty(&results).context("failed to serialize benchmark JSON")?;
        println!("{json_str}");
        return Ok(has_failures);
    }

    print_benchmark_table(&results, args.failures_only);
    Ok(has_failures)
}

fn print_benchmark_table(models: &[ModelBenchResult], failures_only: bool) {
    if models.is_empty() {
        println!("No models benchmarked.");
        return;
    }

    println!(
        "{:<32} {:<18} {:<10} {:>7} {:>9} {:>9} {:>8} {:>8} {:>8}",
        "MODEL",
        "PROBE",
        "THINKING",
        "SUCCESS",
        "LAT (P50)",
        "LAT (P95)",
        "TTFT(P50)",
        "TPOT",
        "TPS"
    );
    println!("{:-<120}", "");

    let mut printed_probes = 0;

    for model in models {
        for probe in &model.probes {
            if failures_only && probe.stats.failed_trials == 0 {
                continue;
            }
            printed_probes += 1;

            let success_str = format!(
                "{}/{}",
                probe.stats.successful_trials, probe.stats.total_trials
            );

            let lat_p50 = probe
                .stats
                .latency_stats
                .as_ref()
                .map_or("-".to_string(), |s| format!("{:.0}ms", s.p50));
            let lat_p95 = probe
                .stats
                .latency_stats
                .as_ref()
                .map_or("-".to_string(), |s| format!("{:.0}ms", s.p95));
            let ttft_p50 = probe
                .stats
                .ttft_stats
                .as_ref()
                .map_or("-".to_string(), |s| format!("{:.0}ms", s.p50));
            let tpot_mean = probe
                .stats
                .tpot_stats
                .as_ref()
                .map_or("-".to_string(), |s| format!("{:.1}ms", s.mean));
            let tps_mean = probe
                .stats
                .tps_stats
                .as_ref()
                .map_or("-".to_string(), |s| format!("{:.1}", s.mean));

            println!(
                "{:<32} {:<18} {:<10} {:>7} {:>9} {:>9} {:>8} {:>8} {:>8}",
                truncate(&model.model_id, 32),
                probe.name,
                probe.thinking_level.to_string(),
                success_str,
                lat_p50,
                lat_p95,
                ttft_p50,
                tpot_mean,
                tps_mean
            );

            if !probe.error_samples.is_empty() {
                for err in &probe.error_samples {
                    println!("    ↳ ❌ Error: {err}");
                }
            }
        }
    }

    if printed_probes == 0 && failures_only {
        println!("No failures recorded across all executed probes.");
    }
}

fn truncate(s: &str, max_width: usize) -> String {
    if s.chars().count() <= max_width {
        s.to_string()
    } else {
        let mut truncated: String = s.chars().take(max_width - 3).collect();
        truncated.push_str("...");
        truncated
    }
}
