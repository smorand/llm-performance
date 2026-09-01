.PHONY: sync run run-dev build release release-dist test test-cov lint lint-fix format format-check \
        typecheck security check doc install uninstall bench bloat deps-dupes docker-build docker-push docker \
        run-up run-down clean clean-all info help

# ============================================================================
# Configuration
# ============================================================================
PROJECT_NAME := $(shell basename $(CURDIR))
# BIN: binary crate to run/install (override: make run BIN=other)
BIN          ?= llm-performance
VERSION      ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COV_MIN      ?= 80
MAKE_DOCKER_PREFIX ?=
DOCKER_TAG   ?= latest
DOCKER_IMAGE := $(MAKE_DOCKER_PREFIX)$(PROJECT_NAME):$(DOCKER_TAG)
ARGS         ?=

# Install prefix: /usr/local if root, ~/.local otherwise. Override for packaging.
PREFIX       ?= $(if $(filter 0,$(shell id -u)),/usr/local,$(HOME)/.local)
BINDIR       := $(PREFIX)/bin
UID_NUM      := $(shell id -u)
UNAME_S      := $(shell uname -s)

# Detect optional tools; degrade gracefully when absent.
HAS_NEXTEST  := $(shell command -v cargo-nextest >/dev/null 2>&1 && echo yes || echo no)
HAS_LLVMCOV  := $(shell command -v cargo-llvm-cov >/dev/null 2>&1 && echo yes || echo no)
# cargo-llvm-cov needs llvm-cov/llvm-profdata. The active rustc sysroot may lack
# them (e.g. a Homebrew rustc shadowing rustup in PATH). If so, borrow them from
# the rustup toolchain, which ships llvm-tools-preview.
LLVM_BIN     := $(shell rustc --print target-libdir 2>/dev/null | sed 's,/lib$$,/bin,')
HAS_LLVMTOOL := $(shell test -x "$(LLVM_BIN)/llvm-cov" && echo yes || echo no)
RUSTUP_LLVM  := $(shell rustup which --toolchain "$$(rustup show active-toolchain 2>/dev/null | awk '{print $$1}')" rustc 2>/dev/null | sed 's,/bin/rustc$$,/lib/rustlib/,')
RUSTUP_LLVMB := $(shell ls -d $(RUSTUP_LLVM)*/bin 2>/dev/null | head -1)
ifeq ($(HAS_LLVMTOOL),no)
ifneq ($(RUSTUP_LLVMB),)
export LLVM_COV      := $(RUSTUP_LLVMB)/llvm-cov
export LLVM_PROFDATA := $(RUSTUP_LLVMB)/llvm-profdata
endif
endif
HAS_AUDIT    := $(shell command -v cargo-audit >/dev/null 2>&1 && echo yes || echo no)
HAS_DENY     := $(shell command -v cargo-deny >/dev/null 2>&1 && echo yes || echo no)

## sync: Fetch dependencies and build the workspace
sync:
	cargo fetch
	cargo build --workspace

## run: Run the binary (make run BIN=ica-proxy ARGS='serve')
run:
	cargo run -p $(BIN) -- $(ARGS)

## run-dev: Run the binary with debug logging (RUST_LOG=debug)
run-dev:
	RUST_LOG=debug cargo run -p $(BIN) -- $(ARGS)

## build: Build the whole workspace (debug)
build:
	cargo build --workspace

## release: Build the whole workspace (optimized, unwinding — safe for tests/backtraces)
release:
	cargo build --workspace --release

## release-dist: Build the shipped binary (release + panic=abort, smallest/fastest)
release-dist:
	cargo build -p $(BIN) --profile release-dist

## test: Run tests (cargo-nextest if available, else cargo test)
test:
ifeq ($(HAS_NEXTEST),yes)
	cargo nextest run --workspace
	cargo test --workspace --doc
else
	@echo "cargo-nextest not found, using cargo test (install: cargo install cargo-nextest)"
	cargo test --workspace
endif

## test-cov: Run tests with coverage, fail under $(COV_MIN)% line coverage
test-cov:
ifeq ($(HAS_LLVMCOV),yes)
	cargo llvm-cov --workspace --fail-under-lines $(COV_MIN)
else
	@echo "cargo-llvm-cov not found (install: cargo install cargo-llvm-cov); running tests without coverage"
	$(MAKE) test
endif

## lint: Run clippy with warnings denied (the real lint/type gate)
lint:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

## lint-fix: Apply clippy and compiler auto-fixes
lint-fix:
	cargo clippy --workspace --all-targets --all-features --fix --allow-dirty --allow-staged
	cargo fix --workspace --allow-dirty --allow-staged

## format: Format all code with rustfmt
format:
	cargo fmt --all

## format-check: Verify formatting without modifying files
format-check:
	cargo fmt --all -- --check

## typecheck: Fast type/borrow check across the workspace
typecheck:
	cargo check --workspace --all-targets

## security: Audit dependencies for advisories and policy (audit + deny)
security:
ifeq ($(HAS_AUDIT),yes)
	cargo audit
else
	@echo "cargo-audit not found (install: cargo install cargo-audit)"
endif
ifeq ($(HAS_DENY),yes)
	cargo deny check
else
	@echo "cargo-deny not found (install: cargo install cargo-deny)"
endif

## check: Full quality gate (format-check, lint, typecheck, security, test-cov, doc)
check: format-check lint typecheck security test-cov doc
	@echo "All checks passed!"

## doc: Build docs, failing on broken intra-doc links
doc:
	RUSTDOCFLAGS="-D warnings -D rustdoc::broken_intra_doc_links" cargo doc --workspace --no-deps

## install: Build and install binary (shipped profile)
install: release-dist
	@mkdir -p "$(BINDIR)"
	@install -m 0755 "target/release-dist/$(BIN)" "$(BINDIR)/$(BIN)"
ifeq ($(UNAME_S),Darwin)
	@codesign --force --sign - "$(BINDIR)/$(BIN)" 2>/dev/null || true
endif
	@echo "Installed $(BIN) -> $(BINDIR)/$(BIN)"
	@case ":$$PATH:" in *":$(BINDIR):"*) ;; \
	  *) echo "NOTE: $(BINDIR) is not on PATH. Add it:"; \
	     echo "  export PATH=\"$(BINDIR):\$$PATH\"" ;; esac

## uninstall: Remove installed binary
uninstall:
	@rm -f "$(BINDIR)/$(BIN)"
	@echo "Removed $(BINDIR)/$(BIN)"

## status: Show daemon status (macOS)
status:
ifeq ($(UNAME_S),Darwin)
	@launchctl list net.semorand.ica-proxy 2>/dev/null || echo "Daemon not loaded"
else
	@echo "Not running on macOS"
endif

## bench: Run criterion benchmarks (results in target/criterion/)
bench:
	cargo bench --workspace

## bloat: Analyze binary size by crate (needs cargo-bloat)
bloat:
	@command -v cargo-bloat >/dev/null 2>&1 || { echo "install: cargo install cargo-bloat"; exit 1; }
	cargo bloat --release -p $(BIN) --crates

## deps-dupes: List duplicate dependency versions in the graph
deps-dupes:
	cargo tree -d

## docker-build: Build the Docker image
docker-build:
	docker build -t $(DOCKER_IMAGE) --build-arg APP_VERSION=$(VERSION) .

## docker-push: Push the Docker image to the registry
docker-push:
	docker push $(DOCKER_IMAGE)

## docker: Build and push the Docker image
docker: docker-build docker-push

## run-up: Build image and start services with docker compose
run-up: docker-build
	PROJECT_NAME=$(PROJECT_NAME) DOCKER_PREFIX=$(MAKE_DOCKER_PREFIX) DOCKER_TAG=$(DOCKER_TAG) docker compose up -d

## run-down: Stop docker compose services
run-down:
	PROJECT_NAME=$(PROJECT_NAME) DOCKER_PREFIX=$(MAKE_DOCKER_PREFIX) DOCKER_TAG=$(DOCKER_TAG) docker compose down

## clean: Remove build artifacts
clean:
	cargo clean

## clean-all: Remove build artifacts and uninstall the binary
clean-all: clean uninstall

## info: Show project info
info:
	@echo "Project:  $(PROJECT_NAME)"
	@echo "Version:  $(VERSION)"
	@echo "Binary:   $(BIN)"
	@echo "Install:  $(BINDIR)/$(BIN)"
	@echo "Image:    $(DOCKER_IMAGE)"
	@echo "nextest:  $(HAS_NEXTEST)   llvm-cov: $(HAS_LLVMCOV)   audit: $(HAS_AUDIT)   deny: $(HAS_DENY)"

## help: Show this help
help:
	@echo "$(PROJECT_NAME) Makefile"
	@echo "========================"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Examples:"
	@echo "  make sync                          # fetch deps + build"
	@echo "  make run ARGS='serve'              # start the proxy"
	@echo "  make check                         # full quality gate before commit"
	@echo "  make install                       # install to ~/.local/bin (or /usr/local/bin as root)"
