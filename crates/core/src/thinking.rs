//! Reasoning and thinking budget configuration for LLM requests.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// Discrete thinking / reasoning level applied across model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    /// Thinking / reasoning disabled (default).
    #[default]
    Off,
    /// Minimal reasoning effort or budget (~1,024 tokens).
    Minimal,
    /// Low reasoning effort or budget (~2,048 tokens).
    Low,
    /// Medium reasoning effort or budget (~4,096 tokens).
    Medium,
    /// High reasoning effort or budget (~8,192 tokens).
    High,
}

impl ThinkingLevel {
    /// Map thinking level to OpenAI-compatible `reasoning_effort` string parameter.
    ///
    /// Returns `None` when thinking is disabled ([`ThinkingLevel::Off`]).
    #[must_use]
    pub fn to_openai_reasoning_effort(&self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Minimal | Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
        }
    }

    /// Map thinking level to Anthropic-compatible `thinking.budget_tokens` numeric parameter.
    ///
    /// Returns `None` when thinking is disabled ([`ThinkingLevel::Off`]).
    #[must_use]
    pub fn to_anthropic_budget_tokens(&self) -> Option<u32> {
        match self {
            Self::Off => None,
            Self::Minimal => Some(1024),
            Self::Low => Some(2048),
            Self::Medium => Some(4096),
            Self::High => Some(8192),
        }
    }

    /// Compute `max_tokens` for Anthropic requests given this thinking level.
    ///
    /// Anthropic requires `max_tokens > thinking.budget_tokens`.
    /// When thinking is disabled, a minimal default of `16` tokens is used.
    #[must_use]
    pub fn anthropic_max_tokens(&self) -> u32 {
        match self.to_anthropic_budget_tokens() {
            Some(budget) => budget + 1024,
            None => 16,
        }
    }
}

impl fmt::Display for ThinkingLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Minimal => write!(f, "minimal"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
        }
    }
}

impl FromStr for ThinkingLevel {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" | "false" | "0" => Ok(Self::Off),
            "minimal" | "min" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" | "med" => Ok(Self::Medium),
            "high" | "max" => Ok(Self::High),
            _ => Err(CoreError::InvalidThinkingLevel(s.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_from_str_roundtrip() {
        let levels = [
            (ThinkingLevel::Off, "off"),
            (ThinkingLevel::Minimal, "minimal"),
            (ThinkingLevel::Low, "low"),
            (ThinkingLevel::Medium, "medium"),
            (ThinkingLevel::High, "high"),
        ];

        for (level, expected_str) in levels {
            assert_eq!(level.to_string(), expected_str);
            assert_eq!(expected_str.parse::<ThinkingLevel>().unwrap(), level);
        }
    }

    #[test]
    fn from_str_aliases() {
        assert_eq!("none".parse::<ThinkingLevel>().unwrap(), ThinkingLevel::Off);
        assert_eq!(
            "disabled".parse::<ThinkingLevel>().unwrap(),
            ThinkingLevel::Off
        );
        assert_eq!(
            "min".parse::<ThinkingLevel>().unwrap(),
            ThinkingLevel::Minimal
        );
        assert_eq!(
            "med".parse::<ThinkingLevel>().unwrap(),
            ThinkingLevel::Medium
        );
        assert_eq!("max".parse::<ThinkingLevel>().unwrap(), ThinkingLevel::High);
    }

    #[test]
    fn from_str_invalid() {
        let err = "super-high".parse::<ThinkingLevel>().unwrap_err();
        assert!(matches!(err, CoreError::InvalidThinkingLevel(_)));
    }

    #[test]
    fn openai_reasoning_effort_mapping() {
        assert_eq!(ThinkingLevel::Off.to_openai_reasoning_effort(), None);
        assert_eq!(
            ThinkingLevel::Minimal.to_openai_reasoning_effort(),
            Some("low")
        );
        assert_eq!(ThinkingLevel::Low.to_openai_reasoning_effort(), Some("low"));
        assert_eq!(
            ThinkingLevel::Medium.to_openai_reasoning_effort(),
            Some("medium")
        );
        assert_eq!(
            ThinkingLevel::High.to_openai_reasoning_effort(),
            Some("high")
        );
    }

    #[test]
    fn anthropic_budget_and_max_tokens() {
        assert_eq!(ThinkingLevel::Off.to_anthropic_budget_tokens(), None);
        assert_eq!(ThinkingLevel::Off.anthropic_max_tokens(), 16);

        assert_eq!(
            ThinkingLevel::Minimal.to_anthropic_budget_tokens(),
            Some(1024)
        );
        assert_eq!(ThinkingLevel::Minimal.anthropic_max_tokens(), 2048);

        assert_eq!(ThinkingLevel::Low.to_anthropic_budget_tokens(), Some(2048));
        assert_eq!(ThinkingLevel::Low.anthropic_max_tokens(), 3072);

        assert_eq!(
            ThinkingLevel::Medium.to_anthropic_budget_tokens(),
            Some(4096)
        );
        assert_eq!(ThinkingLevel::Medium.anthropic_max_tokens(), 5120);

        assert_eq!(ThinkingLevel::High.to_anthropic_budget_tokens(), Some(8192));
        assert_eq!(ThinkingLevel::High.anthropic_max_tokens(), 9216);
    }

    #[test]
    fn serde_json_roundtrip() {
        let json = serde_json::to_string(&ThinkingLevel::Medium).unwrap();
        assert_eq!(json, "\"medium\"");
        let deserialized: ThinkingLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, ThinkingLevel::Medium);
    }
}
