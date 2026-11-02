//! Statistical aggregation and percentile calculations for benchmark trial metrics.

use serde::{Deserialize, Serialize};

/// Metric recorded from a single probe trial execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialMetric {
    /// Total wall-clock duration of the request in milliseconds.
    pub latency_ms: u64,

    /// Time to first token (TTFT) in milliseconds when streaming is enabled.
    pub ttft_ms: Option<u64>,

    /// Throughput in tokens per second (TPS).
    pub tps: Option<f64>,

    /// Total number of output tokens (or chunks) produced by the model.
    pub output_tokens: u64,

    /// Number of reasoning / thinking tokens produced, if reported by the backend.
    pub reasoning_tokens: Option<u64>,
}

/// Distribution statistics computed over a set of floating-point measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PercentileStats {
    /// Minimum value observed in the sample.
    pub min: f64,

    /// Arithmetic mean of the sample values.
    pub mean: f64,

    /// 50th percentile (median) of the distribution.
    pub p50: f64,

    /// 95th percentile of the distribution.
    pub p95: f64,

    /// Maximum value observed in the sample.
    pub max: f64,

    /// Standard deviation of the sample values.
    pub std_dev: f64,
}

/// Aggregated statistical summary across all repeat trials of a probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateStats {
    /// Total number of trial attempts dispatched.
    pub total_trials: usize,

    /// Number of successful trials.
    pub successful_trials: usize,

    /// Number of failed trials.
    pub failed_trials: usize,

    /// Distribution statistics for total request latency in milliseconds.
    pub latency_stats: Option<PercentileStats>,

    /// Distribution statistics for time to first token (TTFT) in milliseconds.
    pub ttft_stats: Option<PercentileStats>,

    /// Distribution statistics for token throughput (tokens per second).
    pub tps_stats: Option<PercentileStats>,

    /// Mean number of output tokens produced across successful trials.
    pub mean_output_tokens: f64,

    /// Mean number of reasoning tokens produced across trials reporting reasoning tokens.
    pub mean_reasoning_tokens: Option<f64>,
}

/// Compute percentile and summary statistics for a slice of numeric measurements.
///
/// Returns `None` if `values` contains zero finite values.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_percentiles(values: &[f64]) -> Option<PercentileStats> {
    let mut finite_vals: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_vals.is_empty() {
        return None;
    }

    finite_vals.sort_by(f64::total_cmp);
    let count = finite_vals.len();
    let min = finite_vals[0];
    let max = finite_vals[count - 1];

    let sum: f64 = finite_vals.iter().sum();
    let mean = sum / (count as f64);

    let variance = finite_vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count as f64);
    let std_dev = variance.sqrt();

    let p50 = quantile(&finite_vals, 0.50);
    let p95 = quantile(&finite_vals, 0.95);

    Some(PercentileStats {
        min,
        mean,
        p50,
        p95,
        max,
        std_dev,
    })
}

/// Interpolated quantile calculation using linear interpolation between closest ranks.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let index = q * ((n - 1) as f64);
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let weight = index - (lower as f64);

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

/// Aggregate individual trial results into a summary statistical report.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn aggregate_trials(trials: &[Result<TrialMetric, String>]) -> AggregateStats {
    let total_trials = trials.len();
    let ok_metrics: Vec<&TrialMetric> = trials.iter().filter_map(|r| r.as_ref().ok()).collect();
    let successful_trials = ok_metrics.len();
    let failed_trials = total_trials.saturating_sub(successful_trials);

    let latency_values: Vec<f64> = ok_metrics.iter().map(|m| m.latency_ms as f64).collect();
    let latency_stats = compute_percentiles(&latency_values);

    let ttft_values: Vec<f64> = ok_metrics
        .iter()
        .filter_map(|m| m.ttft_ms.map(|v| v as f64))
        .collect();
    let ttft_stats = compute_percentiles(&ttft_values);

    let tps_values: Vec<f64> = ok_metrics.iter().filter_map(|m| m.tps).collect();
    let tps_stats = compute_percentiles(&tps_values);

    let mean_output_tokens = if successful_trials > 0 {
        ok_metrics
            .iter()
            .map(|m| m.output_tokens as f64)
            .sum::<f64>()
            / (successful_trials as f64)
    } else {
        0.0
    };

    let reasoning_vals: Vec<f64> = ok_metrics
        .iter()
        .filter_map(|m| m.reasoning_tokens.map(|r| r as f64))
        .collect();
    let mean_reasoning_tokens = if reasoning_vals.is_empty() {
        None
    } else {
        Some(reasoning_vals.iter().sum::<f64>() / (reasoning_vals.len() as f64))
    };

    AggregateStats {
        total_trials,
        successful_trials,
        failed_trials,
        latency_stats,
        ttft_stats,
        tps_stats,
        mean_output_tokens,
        mean_reasoning_tokens,
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn compute_percentiles_empty() {
        assert_eq!(compute_percentiles(&[]), None);
        assert_eq!(compute_percentiles(&[f64::NAN]), None);
    }

    #[test]
    fn compute_percentiles_single_element() {
        let stats = compute_percentiles(&[100.0]).expect("single value stats");
        assert_eq!(stats.min, 100.0);
        assert_eq!(stats.max, 100.0);
        assert_eq!(stats.mean, 100.0);
        assert_eq!(stats.p50, 100.0);
        assert_eq!(stats.p95, 100.0);
        assert_eq!(stats.std_dev, 0.0);
    }

    #[test]
    fn compute_percentiles_multiple_elements() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = compute_percentiles(&values).expect("percentiles");
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 50.0);
        assert_eq!(stats.mean, 30.0);
        assert_eq!(stats.p50, 30.0);
        // p95 of [10, 20, 30, 40, 50]: index = 0.95 * 4 = 3.8 -> 40 + 0.8 * 10 = 48.0
        assert!((stats.p95 - 48.0).abs() < 1e-5);
        // variance: ((400 + 100 + 0 + 100 + 400) / 5) = 200.0 -> sqrt(200) = 14.1421356...
        assert!((stats.std_dev - 200.0_f64.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn aggregate_trials_all_failed() {
        let trials = vec![
            Err("timeout".to_string()),
            Err("500 internal server error".to_string()),
        ];
        let agg = aggregate_trials(&trials);
        assert_eq!(agg.total_trials, 2);
        assert_eq!(agg.successful_trials, 0);
        assert_eq!(agg.failed_trials, 2);
        assert_eq!(agg.latency_stats, None);
        assert_eq!(agg.ttft_stats, None);
        assert_eq!(agg.tps_stats, None);
        assert_eq!(agg.mean_output_tokens, 0.0);
        assert_eq!(agg.mean_reasoning_tokens, None);
    }

    #[test]
    fn aggregate_trials_mixed() {
        let trials = vec![
            Ok(TrialMetric {
                latency_ms: 500,
                ttft_ms: Some(100),
                tps: Some(25.0),
                output_tokens: 11,
                reasoning_tokens: Some(50),
            }),
            Ok(TrialMetric {
                latency_ms: 700,
                ttft_ms: Some(200),
                tps: Some(20.0),
                output_tokens: 11,
                reasoning_tokens: Some(100),
            }),
            Err("rate limit".to_string()),
        ];

        let agg = aggregate_trials(&trials);
        assert_eq!(agg.total_trials, 3);
        assert_eq!(agg.successful_trials, 2);
        assert_eq!(agg.failed_trials, 1);

        let latency = agg.latency_stats.expect("latency stats");
        assert_eq!(latency.min, 500.0);
        assert_eq!(latency.max, 700.0);
        assert_eq!(latency.mean, 600.0);

        let ttft = agg.ttft_stats.expect("ttft stats");
        assert_eq!(ttft.min, 100.0);
        assert_eq!(ttft.max, 200.0);
        assert_eq!(ttft.mean, 150.0);

        assert_eq!(agg.mean_output_tokens, 11.0);
        assert_eq!(agg.mean_reasoning_tokens, Some(75.0));
    }

    #[test]
    fn serde_roundtrip_trial_metric() {
        let metric = TrialMetric {
            latency_ms: 1234,
            ttft_ms: Some(120),
            tps: Some(64.5),
            output_tokens: 80,
            reasoning_tokens: Some(200),
        };
        let json = serde_json::to_string(&metric).expect("serialize metric");
        let parsed: TrialMetric = serde_json::from_str(&json).expect("deserialize metric");
        assert_eq!(metric, parsed);
    }
}
