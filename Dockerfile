FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY llmcord.py config.yaml ./

CMD ["uv", "run", "--no-sync", "python", "llmcord.py"]
