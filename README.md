# searchio

searchio is a tool for search, page reading, and product lookup. Use it from
the command line, from Python, or over HTTP. Use it as a person or as an AI
agent. Search and reading need no API key.

This manual uses Simplified Technical English (ASD-STE100). Sentences are
short. Each step does one action.

## Quick start for AI agents

An AI agent can set up searchio for its user in one pass. Follow these
steps in order. Stop when the verify step passes.

**1. Use Docker when Docker runs.** Try:

```bash
docker run -i --rm ghcr.io/ioserviceslabs/searchio-mcp
```

The image holds the full system: fetch ladder, Rust engine sidecar, and
a Chromium for the challenge tier. No other install step. Go to step 3.

**2. No Docker: install from source.** Needs Python 3.10 or newer.

```bash
pip install "searchio[mcp] @ git+https://github.com/IOServicesLabs/searchio"
curl -fsSL https://raw.githubusercontent.com/IOServicesLabs/searchio-engine/main/scripts/install.sh | sh
```

The pip command installs the searchio package with the MCP server. The
curl command adds the Rust engine sidecar at ~/.searchio/bin/se-serve
(linux, macOS, Windows with Git Bash). The sidecar is the fast tier-2
backend. It is optional but recommended.

**3. Add the MCP server to the client.** Claude Desktop and Cursor use
this config shape.

Docker:

```json
{
  "mcpServers": {
    "searchio": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "ghcr.io/ioserviceslabs/searchio-mcp"]
    }
  }
}
```

Source install:

```json
{
  "mcpServers": {
    "searchio": { "command": "searchio-mcp" }
  }
}
```

**4. Verify.** Restart the client. Ask: "Use searchio to search the web
for the current time in Tokyo." Real results mean the setup works. An
error about a missing command means the config points at a path that
does not exist; fix the config and retry.

Note. `pip install "searchio[mcp]"` (PyPI) and
`npx @ioserviceslabs/searchio-mcp` activate when the registry packages
land. The steps above work today.
## 1. Install

You need Python 3.10 or newer.

**Path A -- pip (one command):**

```bash
pip install "searchio[mcp]"
searchio-mcp        # MCP server on stdio, ready for an LLM client
```

`searchio doctor` checks the install. The PyPI package activates when the
v0.1.0 publish lands; until then the source install above (quick start,
step 2) is the working pip path.

```
```

**Path B -- Docker (no Python needed):**

```bash
docker run -i --rm ghcr.io/ioserviceslabs/searchio-mcp              # stdio MCP
docker run -p 8080:8080 ghcr.io/ioserviceslabs/searchio-mcp \
  searchio-mcp --transport sse --host 0.0.0.0 --port 8080
```

No prebuilt image for your platform? Build from source:

```bash
docker build -t searchio https://github.com/IOServicesLabs/searchio.git
```

The image holds the full ladder: the Python tiers, the Rust engine sidecar,
and a Chromium for the challenge tier. State (clearance, sessions) persists
in a volume at `/root/.searchio`:

```bash
docker run -i --rm -v searchio-state:/root/.searchio ghcr.io/ioserviceslabs/searchio-mcp
```

**Path C -- from source:**

```bash
git clone https://github.com/IOServicesLabs/searchio.git
cd searchio
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e '.[mcp]'
searchio doctor
```

`searchio doctor` shows which fetch tiers work on your machine. Tier 0 is
plain HTTP. Tier 1 adds TLS impersonation. Tier 2 adds a real browser for
pages that block bots. Tier 2 starts on its own the first time it is needed.

### The engine tier (optional, recommended)

Tier 2 uses the Rust [searchio-engine](
https://github.com/IOServicesLabs/searchio-engine) sidecar when a `se-serve`
binary is present. It boots in milliseconds and has no Chromium. The Docker
image includes it. On a host install, one command adds it:

```bash
curl -fsSL https://raw.githubusercontent.com/IOServicesLabs/searchio-engine/main/scripts/install.sh | sh
```

The installer drops the binary at `~/.searchio/bin/se-serve` and searchio
finds it on its own. You can also set `SEARCHIO_ENGINE_BIN` to any binary.

### MCP client setup

Point any MCP client at the server:

```json
{
  "mcpServers": {
    "searchio": {
      "command": "searchio-mcp"
    }
  }
}
```

With Docker, use `"command": "docker", "args": ["run", "-i", "--rm", "searchio"]`.

No Python at all? The npm wrapper picks a backend for you (Docker when
present, else a local install):

```json
{
  "mcpServers": {
    "searchio": {
      "command": "npx",
      "args": ["-y", "@ioserviceslabs/searchio-mcp"]
    }
  }
}
```

## 2. Search

```bash
searchio search "best noise cancelling headphones 2026"
searchio search "rust async runtime comparison" --intent code
searchio search "artemis mission" --domains nasa.gov
searchio search "heat pump retrofit" --k 20 --json
```

You can set an intent: `web` (default), `news`, `video`, `shopping`,
`academic`, `code`, `forum`, `reference`, `local`. You can also put
`site:nasa.gov` in the query.

## 3. Read a page

```bash
searchio read https://example.com/some/article
searchio read https://example.com/paper.pdf --chars 20000
```

The command prints the readable text. It climbs walls, bot checks, and
JavaScript-only pages on its own. If a page cannot be read, you get a clear
reason. You do not get an empty page.

## 4. Products and listings

```bash
searchio items "sony wh-1000xm5"                    # one row per product across retailers
searchio items "dyson v15" --domains bestbuy.com --json
searchio marketplace "ford f-150" --city seattle    # Facebook Marketplace, browser-backed
```

## 5. Research with an AI swarm

This part costs money. Set one key:

```bash
set ANTHROPIC_API_KEY=...            # Windows
# export ANTHROPIC_API_KEY=...       # macOS / Linux
searchio research "What changed in the EU battery regulation this year, and who is affected?"
```

To use DeepSeek instead, set `SEARCHIO_LLM_PROVIDER=deepseek` and
`DEEPSEEK_API_KEY=...`. The answer comes back with findings. Each finding
cites pages the workers read.

## 6. Run it as an MCP server

searchio is also an MCP (Model Context Protocol) server. AI agents call it
over MCP. The same tools that the CLI uses are available to the agent.

Install the MCP extra:

```bash
pip install -e '.[mcp]'
```

Start the server over stdio:

```bash
searchio mcp
# or
python -m searchio.mcp
```

Start the server over HTTP:

```bash
searchio mcp --transport streamable-http --host 127.0.0.1 --port 8080 --path /mcp
```

### 6.1 MCP tools

The server has these tools:

| Tool | What it does |
|---|---|
| `search` | Fused, ranked web search across keyless providers. |
| `search_items` | Product listings with prices, merged across retailers. |
| `search_local` | Facebook Marketplace classifieds near a city (browser-backed). |
| `read_url` | Read one URL through the blocking-resistant ladder. Returns markdown. |
| `research` | Run the multi-step research swarm on a question (needs an LLM key). |
| `stats` | Ladder and provider health (per-domain learned tiers, block rates). |

The server also has one resource: `searchio://health` returns service health
as a small JSON object.

### 6.2 Payment for search and research (x402)

The MCP server has a payment gate. It uses the x402 shape. When the gate is
on, a billable tool returns a 402 (Payment Required) with an `X-Payment`
challenge until the caller pays.

Set the gate with `SEARCHIO_MCP_PAYMENT`:

| Value | Result |
|---|---|
| `` (empty, default) | Gate open. Nothing is charged. For local and dev use. |
| `static:<token>` | The caller must send `Authorization: Bearer <token>`. Without it the tool returns a 402 + X-Payment challenge. |

This is the shape a real verifier uses. It has no ledger. When you host the
server for others, put a real x402 verifier in front. The verifier checks
payment, then the tool runs. This is how searchio is hosted next to SwarmIO
for search and research credits.

## 7. Run it as an HTTP server

```bash
searchio serve --port 8080
```

| Endpoint | What it does |
|---|---|
| `GET /search?q=...&intent=web&k=10&domains=a.com,b.org&freshness=week&locale=de-DE` | Search results |
| `GET /read?url=...&session=me` | Readable text of one page |
| `GET /items?q=...&domains=...` | Product listings merged per product |
| `GET /marketplace?q=...&city=seattle` | Facebook Marketplace listings |
| `POST /research` `{"question": "..."}` | The AI research swarm |
| `GET /providers`, `GET /stats`, `GET /healthz` | Status |

Errors are honest status codes: 422 for a bad request, 400 for a target the
policy refuses, 451 for a page blocked by an anti-bot wall, 502 or 503 when
providers failed or are all cooling down after failures. A search answer
lists `providers_used`, `providers_failed` and `providers_skipped` (breakers
open, with the cooldown left).

If several people or agents share one server, give each its own `session` on
`/read`. Then cookies from one never ride along on another's requests.

## 8. Run it in Docker

The Dockerfile in this repo builds the full ladder: Python tiers, the Rust
engine sidecar, and a Chromium for the challenge tier. See Path B in section
1 for the build and run commands. `docker compose up` does the same with the
state volume wired up.

Inside the container, tier 2 is the engine sidecar first; the patchright
Chromium takes the pages that need a real browser (the challenge tier).
Cache, learned per-domain tiers, and clearance cookies live in the
`searchio-state` volume, so they survive restarts. Keys for the research
swarm go in as environment variables, the same names as in section 10.

## 9. Use it from Python

```python
import asyncio
from searchio.engine import Engine

async def main():
    eng = Engine()
    res = await eng.search("lithium iron phosphate battery lifespan", intent="web", k=5)
    for doc in res.docs:
        print(doc.title, doc.url)

    doc = await eng.read("https://example.com/article")
    print(doc.text[:500])

    items = await eng.find_items("sony wh-1000xm5")
    for it in items:
        print(it.title, it.price, it.url)

    await eng.close()

asyncio.run(main())
```

## 10. Settings

Everything is an environment variable with the `SEARCHIO_` prefix, or a line
in a `.env` file next to where you run it. The ones people change most:

| Variable | Default | Meaning |
|---|---|---|
| `SEARCHIO_MAX_TIER` | `2` | `0` HTTP only, `1` adds TLS impersonation, `2` adds the browser |
| `SEARCHIO_ROBOTS_POLICY` | `warn` | `enforce`, `warn`, or `off` |
| `SEARCHIO_PROXY` | none | Proxy URL for the HTTP tiers |
| `SEARCHIO_FACEBOOK_CITY` | none | Default city for Marketplace searches |
| `SEARCHIO_CACHE_ENABLED` | `true` | Cache fetched pages for an hour |
| `SEARCHIO_STATE_DIR` | `~/.searchio` | Where cache, profiles and clearance cookies live |
| `SEARCHIO_MCP_PAYMENT` | `` | MCP payment gate (see 6.2) |
| `SEARCHIO_MCP_BEARER` | none | Bearer token the MCP gate checks |

## 11. Good to know

- Search and reading need no keys and cost nothing. Only `research` calls a
  paid model.
- The tool never sends your credentials anywhere. Marketplace and other
  browser-backed lookups run anonymously.
- `searchio providers` lists every source and whether it is configured.
- `searchio doctor` is the first thing to run when something looks off.
