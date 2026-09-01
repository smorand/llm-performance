# Rust Standards

This project adheres strictly to the `rust` skill standards:
- Edition 2024, pinned MSRV.
- Workspace layout: `crates/core` (logic) + `crates/llm-performance` (thin CLI binary).
- Config precedence: Defaults < TOML < Environment Variables (`LLM_PERFM_*`) < CLI flags.
- Error handling: `thiserror` in `core`, `anyhow` in CLI.
- Tracing + rolling file logs in `~/.cache/llm-performance/logs/`.
