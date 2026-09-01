//! Static payload constants and helpers for LLM capability probing.

use base64::Engine;
use base64::prelude::BASE64_STANDARD;

/// Standard minimal text prompt for latency / TTFT / TPOT probing.
pub const PROBE_PROMPT: &str = "ping";

/// Valid 8x8 crimson PNG image encoded as standard base64.
///
/// Hand-crafted minimal valid PNG ensuring rejection immunity from picky cloud backends.
pub const TEST_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEUlEQVR42mO4I2KDFTEMLQkAdntLAasbgAUAAAAASUVORK5CYII=";

/// Valid 20ms silent 8kHz 16-bit PCM mono WAV audio encoded as standard base64.
pub const TEST_WAV_BASE64: &str = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";

/// Decode [`TEST_PNG_BASE64`] into raw bytes.
///
/// # Errors
///
/// Returns a [`base64::DecodeError`] if the embedded base64 string fails to decode.
pub fn decode_test_png() -> Result<Vec<u8>, base64::DecodeError> {
    BASE64_STANDARD.decode(TEST_PNG_BASE64)
}

/// Decode [`TEST_WAV_BASE64`] into raw bytes.
///
/// # Errors
///
/// Returns a [`base64::DecodeError`] if the embedded base64 string fails to decode.
pub fn decode_test_wav() -> Result<Vec<u8>, base64::DecodeError> {
    BASE64_STANDARD.decode(TEST_WAV_BASE64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_prompt_not_empty() {
        assert_eq!(PROBE_PROMPT, "ping");
    }

    #[test]
    fn test_png_decoding() {
        let bytes = decode_test_png().expect("valid base64 png");
        assert!(!bytes.is_empty());
        // Standard PNG magic bytes: \x89 P N G \r \n \x1a \n
        assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn test_wav_decoding() {
        let bytes = decode_test_wav().expect("valid base64 wav");
        assert!(!bytes.is_empty());
        // Standard RIFF/WAVE header
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
    }
}
