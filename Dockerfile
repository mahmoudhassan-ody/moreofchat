# The image every process runs from — design §14, demo plan Task 39.
#
# One image, three commands. `api`, `worker-inbound` and `worker-outbound` are
# the same code with different entrypoints, so a version skew between them is
# not expressible: they are the same layer.
#
# Python 3.14 to match the host (CLAUDE.md). uv for installs, because the lock
# file is what makes a deploy reproducible and pip would ignore it.

FROM python:3.14-slim AS base

# uv, pinned by digest-free tag on purpose: this is the tool, not a dependency,
# and the lock file below is what actually pins the build.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies before source, so an edit to `src` does not reinstall the world.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

# Not root. Nothing here writes to the image, and the one process the internet
# can reach should not be able to either.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin moc
USER 10001

# Overridden per service in compose. Named here so `docker run` on the image
# does something honest rather than dropping into a shell.
CMD ["python", "-m", "moc.workers.run", "inbound"]
