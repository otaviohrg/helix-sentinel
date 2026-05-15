FROM ghcr.io/otaviohrg/helix-ml:latest

WORKDIR /workspace

COPY pyproject.toml .
COPY src/ src/

RUN --mount=type=ssh \
    mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    uv pip install -e ".[dev]"

