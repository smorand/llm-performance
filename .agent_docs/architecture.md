# Architecture & Probing Protocols

## Probing Mechanisms

### Text (OpenAI and Anthropic)
- OpenAI-compatible endpoint: `POST {upstream_url}/v1/chat/completions` with streaming SSE enabled for TTFT and TPS measurement.
- Anthropic native endpoint: `POST {upstream_url}/v1/messages` with `anthropic-version: 2023-06-01`.

### Multimodal Payloads
- Image: Embedded 8x8 Crimson PNG base64 (`TEST_PNG_BASE64`). Hand-crafted valid PNG ensuring rejection immunity from picky cloud backends.
- Audio: Embedded 20ms silent 8kHz 16-bit PCM mono WAV base64 (`TEST_WAV_BASE64`).

### Thinking & Reasoning Levels
Supports 5 levels: `off`, `minimal`, `low`, `medium`, `high`.
- OpenAI: Mapped to `reasoning_effort` (`low`, `medium`, `high`).
- Anthropic: Mapped to `thinking: { type: "enabled", budget_tokens: N }` where N >= 1024.

### Statistical Metrics
Calculated across N independent trials (default 10):
- TTFT (Time to First Token) in ms.
- Latency (Total Wall-Clock Request Time) in ms.
- TPS (Output Tokens Per Second).
- Distribution metrics: Min, Mean, P50 (Median), P95, Max, Standard Deviation.
