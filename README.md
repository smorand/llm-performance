# llm-performance

A fast, flexible CLI tool in Rust to probe LLM model capabilities, evaluate generation performance, and measure the impact of reasoning/thinking levels.

## Features

- **Capability Testing**: Validates OpenAI chat completions, native Anthropic Messages API, vision (image), and audio capabilities.
- **Thinking / Reasoning Evaluation**: Measures latency and throughput changes across thinking levels (`off`, `minimal`, `low`, `medium`, `high`).
- **Statistical Rigor**: Aggregates multi-trial benchmarks (default 10 tries) providing Min, Mean, P50 (Median), P95, and Max for TTFT, TPOT, Latency, and TPS.
- **Layered Configuration**: Resolution hierarchy: Defaults < TOML config < Environment variables (`LLM_PERFM_*` with `IBM_ICA_*` fallbacks) < CLI arguments.
- **Embedded Payloads**: Self-contained test media (PNG 8x8, WAV 20ms) guaranteeing backend compatibility without external asset dependencies.

## Installation

```bash
make install
```

Or build directly with cargo:

```bash
cargo build --release -p llm-performance
```

## Configuration

Config file location: `~/.config/llm-performance/config.toml`

```toml
upstream_url = "https://api.nextgen-beta.ica.ibm.com/ica"
api_key = "your-api-key"
test_tries = 10
anthropic_patterns = ["^claude"]
vision_patterns = ["^claude-", "^gpt-", "^gemini-"]
audio_patterns = ["^gemini-"]
```

Environment variables:
- `LLM_PERFM_UPSTREAM_URL` or fallback `IBM_ICA_BASE_URL`
- `LLM_PERFM_API_KEY` or fallback `IBM_ICA_MODEL_KEY`
- `LLM_PERFM_TEST_TRIES`

## Usage

### List Available Models

```bash
llm-performance models
llm-performance models --model "^claude"
```

### Standard Performance Benchmark

Runs text tests (OpenAI chat and native Anthropic messages for eligible models) with 10 iterations:

```bash
llm-performance test
llm-performance test --model "^gpt-4o"
llm-performance test --test-tries 5
```

### Multimodal Probing

```bash
llm-performance test --image --audio
```

### Thinking / Reasoning Benchmark

Benchmark all thinking levels on selected models:

```bash
llm-performance test --thinking
llm-performance test --thinking --thinking-levels off,low,high
```

### JSON Output

```bash
llm-performance test --json
```
