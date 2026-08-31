FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY llmcord.py context_management.py config.yaml ./

CMD ["uv", "run", "--locked", "--no-sync", "python", "llmcord.py"]
