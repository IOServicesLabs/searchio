# searchio API image.
#
# Deliberately has NO browser in it. The stealth browser is heavy -- Chromium,
# Xvfb, fonts, hundreds of megabytes of X libraries -- and it is needed by a
# small fraction of requests. Baking it in here would mean every API replica
# carries that weight and that memory floor, so the two scale together whether
# or not they need to.
#
# Instead the browser lives in its own pool (SwarmIO's `browser-worker` target)
# and this image talks to it over HTTP. That is what makes the two independently
# scalable: a burst of ordinary searches scales the API replicas and leaves the
# expensive pool alone.
#
#   docker build -f docker/Dockerfile -t searchio .
#   docker run -p 8080:8080 -e SEARCHIO_SIDECAR_URL=http://sidecar:8077 searchio

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Container state lives on a volume; see compose. Keeping the learned
    # domain profiles and cache across restarts is worth a volume, because
    # both are how the ladder avoids re-paying for discovery.
    SEARCHIO_STATE_DIR=/var/lib/searchio

# curl for the healthcheck; libcurl for curl_cffi's TLS impersonation, which is
# the whole point of tier 1 and the one native dependency that matters here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first: source changes far more often than dependencies, and
# splitting them keeps rebuilds to seconds.
COPY pyproject.toml README.md ./
COPY src/searchio/__init__.py src/searchio/__init__.py
RUN pip install --no-cache-dir "." && pip uninstall -y searchio

COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

# Run unprivileged. Nothing here needs root, and the browser -- the part that
# does want --no-sandbox -- is in a different container entirely.
RUN useradd --create-home --uid 10001 searchio \
    && mkdir -p /var/lib/searchio \
    && chown -R searchio:searchio /var/lib/searchio /app
USER searchio

EXPOSE 8080

# Answers without touching providers or the browser pool, so it stays honest
# under load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

CMD ["uvicorn", "searchio.server:app", "--host", "0.0.0.0", "--port", "8080"]
