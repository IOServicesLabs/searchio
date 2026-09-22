# @ioserviceslabs/searchio-mcp

The [searchio](https://github.com/IOServicesLabs/searchio) MCP server, packaged
so any LLM client can launch it with zero Python setup.

```bash
npx @ioserviceslabs/searchio-mcp
```

The launcher picks a backend in this order:

1. `SEARCHIO_MCP_CMD` — your own command, verbatim, if set.
2. **Docker** — `docker run -i --rm ghcr.io/ioserviceslabs/searchio-mcp` when the
   daemon is up and the image is present or pullable; otherwise the launcher
   falls through to local install (a missing/private image is not fatal). The
   image is the full system: fetch ladder, Rust engine
   sidecar, and a Chromium for the challenge tier. Nothing else to install.
3. **Local install** — a `searchio-mcp` on your `PATH` (e.g. from
   `pip install "searchio[mcp]"`), falling back to `python -m searchio.mcp`.

## Client config

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

## Extra args

Anything after the package name reaches the server, e.g. an HTTP transport:

```bash
npx @ioserviceslabs/searchio-mcp --transport sse --host 0.0.0.0 --port 8080
```

(When using the Docker backend, publish the port yourself:
`docker run -p 8080:8080 ...` via `SEARCHIO_MCP_CMD`.)

## Env

| Variable | Default | Meaning |
|---|---|---|
| `SEARCHIO_MCP_CMD` | — | Operator override; replaces the whole backend choice |
| `SEARCHIO_MCP_IMAGE` | `ghcr.io/ioserviceslabs/searchio-mcp:latest` | Docker image the launcher runs |
| `SEARCHIO_DOCKER` | `auto` | Set `0` to skip the Docker backend |
