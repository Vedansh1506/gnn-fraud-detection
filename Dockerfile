# Scoring API (SAD C8) — the serving container.
#
# Two things shape this file:
#
# 1. **It installs the `serving` dependency group, not the whole project.**
#    torch, torch-geometric, mlflow, feast and evidently are core dependencies
#    of the project but are never imported on the serving path — they belong to
#    offline training. Including them would add well over a gigabyte to an image
#    headed for a 1 GB t3.micro. tests/test_serving_deps.py enforces that the
#    group stays in step with the project's pins.
#
# 2. **Model artifacts are NOT baked in.** They are git-ignored, version-pinned,
#    and ~115 MB; an image rebuild per retrain would be the wrong unit of
#    change. The entrypoint pulls the pinned versions from S3 at start, which is
#    also what the SAD specifies.

FROM python:3.11-slim AS builder

# uv from its official image — pinned, and avoids a curl|sh in the build.
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Only the files that affect dependency resolution, so this layer caches across
# source edits.
COPY pyproject.toml uv.lock ./

# --frozen: fail if the lockfile disagrees with pyproject rather than quietly
# resolving something different from what CI tested.
RUN uv sync --frozen --only-group serving --no-install-project

# Strip the bundled CUDA runtime. xgboost's Linux wheel declares the nvidia-*
# packages so it can use a GPU; this image serves on a t3.micro that has none,
# and they weigh ~454 MB. Scoring is CPU-only and verified to work without them
# (see tests and the container smoke run) - but if xgboost ever starts linking
# them eagerly, this is the first place to look.
RUN rm -rf /app/.venv/lib/python3.11/site-packages/nvidia     && find /app/.venv -name "*.pyc" -delete     && find /app/.venv -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true


FROM python:3.11-slim AS runtime

# curl is for the container healthcheck; nothing else needs a shell tool.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Artifacts land here at runtime; created up front so the volume mount or the
# S3 sync writes as a non-root user.
RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app/artifacts

USER appuser
EXPOSE 8000

# Checks the app's own readiness, not just that a port is open: /health reports
# per-component status, and degrades rather than hanging when Postgres is down.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
