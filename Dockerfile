# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# searchio all-in-one image.
#
#   One container = the Python acquisition ladder + MCP server + the Rust
#   engine sidecar + a real Chromium (patchright) for the challenge tier.
#
#   stdio MCP (Claude Desktop, Cursor, any LLM client):
#       docker run -i --rm ghcr.io/ioserviceslabs/searchio-mcp
#
#   SSE/HTTP MCP (remote clients):
#       docker run -p 8080:8080 ghcr.io/ioserviceslabs/searchio-mcp \
#         searchio-mcp --transport sse --host 0.0.0.0 --port 8080
#
#   Keep clearance/cookies across restarts:
#       docker run -i --rm -v searchio-state:/root/.searchio ...
#
# The engine is built from the searchio-engine repo. The repo is private for
# now: mount a read token as the BuildKit secret `engine_token` and the clone
# authenticates via an http extraHeader (never a URL, never an image layer).
#   GitHub Actions: image.yml passes the ENGINE_CLONE_TOKEN repo secret.
#   Local: docker build --secret id=engine_token,env=ENGINE_CLONE_TOKEN ...
# Once the repo is public, drop the secret and the anonymous clone just works.
# ARG ENGINE_REF pins an engine tag/branch (a release tag once they exist).
#
ARG PYTHON_VERSION=3.12
ARG ENGINE_REF=main

# ── Stage 1: Rust engine sidecar ─────────────────────────────────────────────
FROM rust:1-bookworm AS engine
ARG ENGINE_REF
RUN --mount=type=secret,id=engine_token,required=false \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends git ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    if [ -s /run/secrets/engine_token ]; then \
      git -c "http.extraHeader=AUTHORIZATION: bearer $(cat /run/secrets/engine_token)" \
        clone --depth 1 --branch "${ENGINE_REF}" \
        https://github.com/IOServicesLabs/searchio-engine /engine; \
    else \
      git clone --depth 1 --branch "${ENGINE_REF}" \
        https://github.com/IOServicesLabs/searchio-engine /engine; \
    fi
WORKDIR /engine
RUN cargo build --release --bin se-serve

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Engine is the primary tier-2 backend; the container boots it lazily on
    # first escalation. The challenge tier (patchright) is configured via
    # SEARCHIO_SIDECAR_* env at run time if wanted.
    SEARCHIO_SIDECAR_ENGINE=1 \
    SEARCHIO_SIDECAR_AUTOSTART=1 \
    SEARCHIO_ENGINE_BIN=/usr/local/bin/se-serve \
    # The challenge sidecar's browser: headed (on Xvfb, see CMD), no
    # user-namespace sandbox in a container.
    SWARM_BROWSER_HEADLESS=0 \
    SWARM_BROWSER_NO_SANDBOX=1

# The challenge tier's Chromium system libraries are installed by patchright
# itself (--with-deps installs the distro set; bookworm is supported), so the
# list stays correct as Chromium's needs change. Fonts are for readable
# screenshots. The sidecar adds --no-sandbox/--disable-dev-shm-usage itself
# when it detects a container.
# Xvfb: the challenge tier runs Chromium HEADED on a virtual display.
# Cloudflare, DataDome and Akamai fingerprint headless Chromium; in a
# container A/B (bench/browserprobe.py, 6 walled pages x 2 passes) headed
# loaded 10/12 and headless 2/12 (both of those via tier 1, not the browser).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates fonts-liberation fonts-noto-color-emoji xvfb xauth tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=engine /engine/target/release/se-serve /usr/local/bin/se-serve

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[mcp]' patchright \
 && patchright install --with-deps chromium

# Clearance bank + tenant sessions live under ~/.searchio; mount a volume to
# keep them across container restarts.
VOLUME /root/.searchio

# stdio MCP by default (pair with `docker run -i`), on a virtual display so
# the challenge tier's Chromium can run headed. xvfb-run as PID 1 never sees
# Xvfb's ready signal and hangs (3/3 runs; 3/3 fine under an init), so tini
# is PID 1.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", "searchio-mcp"]
