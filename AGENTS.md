# llm-performance

## Overview
CLI benchmarking and capability testing tool for LLM models (OpenAI chat completions and native Anthropic messages), measuring latency, TTFT, throughput (TPS), multimodal support (vision, audio), and the impact of reasoning/thinking levels across multiple trials.

## Key Commands
make sync          # Fetch deps + build workspace
make run           # Run the binary (ARGS='...')
make check         # Full quality gate (format, clippy -D warnings, typecheck, security, tests+coverage, doc)
make bench         # Criterion benchmarks
make install       # Install to ~/.local/bin

## Project Structure
crates/core/            : library (config XDG, telemetry, error, domain, models, payloads, thinking, probe, stats, runner)
crates/llm-performance/ : binary (clap CLI with `models` and `test` subcommands, reporting tables, JSON output)

## Conventions
This project follows the `rust` skill.
- Logic in crates/core; binary is a thin entry point (sync `main -> ExitCode`, async `run`).
- thiserror in core (boxed large variants), anyhow in the binary.
- tracing + rolling file logs (~/.cache/llm-performance/logs/) for observability.
- Config via figment + etcetera (XDG dirs); precedence: defaults < TOML < env (LLM_PERFM_*) < CLI.
- Fallback env vars for compatibility: IBM_ICA_BASE_URL and IBM_ICA_MODEL_KEY.

## Quality Gate
Run `make check` before every commit: format-check, clippy (-D warnings), typecheck, security (audit + deny), test-cov (>= 80%), doc (broken intra-doc links).

## Auto-Evaluation Checklist
Before considering any task complete:
- [ ] `make check` passes
- [ ] No unwrap/expect/panic on fallible paths
- [ ] No blocking I/O / std Mutex across .await
- [ ] Boundary functions traced; no secrets in spans
- [ ] Config via Config struct (XDG dirs); state injected
- [ ] Functional e2e tests present; coverage >= 80%

## Documentation Index
- `.agent_docs/rust.md`         : Rust coding standards
- `.agent_docs/architecture.md` : Architecture and probing protocols
