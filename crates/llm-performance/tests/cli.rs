//! CLI integration tests using `assert_cmd` and `predicates`.

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn help_flag_succeeds() {
    Command::cargo_bin("llm-performance")
        .expect("binary exists")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "CLI tool to probe LLM model capabilities",
        ));
}

#[test]
fn models_help_shows_model_filter_flag() {
    Command::cargo_bin("llm-performance")
        .expect("binary exists")
        .args(["models", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"));
}

#[test]
fn test_help_shows_all_flags() {
    Command::cargo_bin("llm-performance")
        .expect("binary exists")
        .args(["test", "--help"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("--test-tries")
                .and(predicate::str::contains("--thinking"))
                .and(predicate::str::contains("--thinking-levels"))
                .and(predicate::str::contains("--image"))
                .and(predicate::str::contains("--audio")),
        );
}

#[test]
fn models_fails_gracefully_when_no_api_key() {
    Command::cargo_bin("llm-performance")
        .expect("binary exists")
        .env("LLM_PERFM_API_KEY", "")
        .env("IBM_ICA_MODEL_KEY", "")
        .args(["models"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no API key configured"));
}

#[test]
fn test_fails_gracefully_when_no_api_key() {
    Command::cargo_bin("llm-performance")
        .expect("binary exists")
        .env("LLM_PERFM_API_KEY", "")
        .env("IBM_ICA_MODEL_KEY", "")
        .args(["test"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no API key configured"));
}
