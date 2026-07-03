FROM ubuntu:26.04

RUN apt-get update && \
  apt-get install -y --no-install-recommends curl ca-certificates && \
  apt-get install -y git

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv

RUN useradd -m -u 10001 user
USER user
WORKDIR /home/user/disco-mimo

# Install uv for this user
COPY --from=ghcr.io/astral-sh/uv:latest --chown=user:user /uv /uvx /home/user/.local/bin/
ENV PATH="/home/user/.local/bin:$PATH"

# Install dependencies only
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --group torch --group agent

# Copy application code with matching user ownership
COPY --chown=user:user src/ /home/user/disco-mimo/src
COPY --chown=user:user pyproject.toml uv.lock README.md /home/user/disco-mimo/

# Sync the project itself
RUN uv sync --frozen --group torch --group agent
