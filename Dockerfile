# Kairos Agent — Production Docker Image
# Multi-stage: build → runtime (minimal surface)

FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --user kairos-agent

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="Kairos Agent"
LABEL org.opencontainers.image.description="The right tool, at the right moment. Production-grade AI agent framework."
LABEL org.opencontainers.image.source="https://github.com/buer103/kairos"

RUN useradd --create-home --shell /bin/bash kairos \
    && mkdir -p /home/kairos/.kairos /home/kairos/.config/kairos \
    && chown -R kairos:kairos /home/kairos

COPY --from=builder /root/.local /home/kairos/.local
ENV PATH="/home/kairos/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy source for editable install / development
COPY --chown=kairos:kairos . /app
WORKDIR /app
RUN pip install --no-cache-dir -e .

USER kairos

# Gateway: HTTP API + Webhook server
EXPOSE 8080
# Optional: direct CLI port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Default: start gateway server
CMD ["python", "-m", "kairos.gateway"]
