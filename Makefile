.PHONY: test lint clean build docker run help

# ── Default ──────────────────────────────────────────────────────
help:
	@echo "Kairos — Development Commands"
	@echo ""
	@echo "  make test        Run all tests (695)"
	@echo "  make test-cov    Run tests with coverage"
	@echo "  make lint        Run ruff linter"
	@echo "  make fmt         Format code with ruff"
	@echo "  make build       Build the package"
	@echo "  make docker      Build Docker image"
	@echo "  make run         Start gateway server"
	@echo "  make clean       Remove build artifacts"
	@echo "  make all         test + lint + build"

# ── Test ─────────────────────────────────────────────────────────
test:
	python -m pytest -v

test-cov:
	python -m pytest --cov=kairos --cov-report=term-missing

test-smoke:
	python tests/smoke_test.py

# ── Lint ─────────────────────────────────────────────────────────
lint:
	ruff check kairos/ tests/

fmt:
	ruff check --fix kairos/ tests/
	ruff format kairos/ tests/

# ── Build ────────────────────────────────────────────────────────
build:
	pip install --upgrade build
	python -m build

# ── Docker ───────────────────────────────────────────────────────
docker:
	docker build -t kairos-agent:latest .

docker-run:
	docker run --rm -it \
		-e DEEPSEEK_API_KEY=$${DEEPSEEK_API_KEY:-} \
		-p 8080:8080 \
		kairos-agent:latest

# ── Run ──────────────────────────────────────────────────────────
run:
	python -m kairos.gateway --port 8080

# ── Clean ────────────────────────────────────────────────────────
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── All ──────────────────────────────────────────────────────────
all: test lint build
