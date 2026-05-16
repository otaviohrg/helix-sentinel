FROM ghcr.io/otaviohrg/helix-ml:latest

WORKDIR /workspace

COPY pyproject.toml .
COPY src/ src/

RUN uv pip install \
    "git+https://github.com/otaviohrg/helix-core.git#subdirectory=shared/sdk" && \
    uv pip install -e ".[dev]"

