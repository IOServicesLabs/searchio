#!/usr/bin/env node
// searchio-mcp launcher for LLM clients (Claude Desktop, Cursor, ...).
//
// Resolution order:
//   1. SEARCHIO_MCP_CMD — run whatever the operator says, verbatim (sh -c).
//   2. Docker — `docker run -i --rm <image>` when the daemon responds AND the
//      image is present or pullable; a missing/private/unreachable image
//      falls through to local install instead of hard-failing.
//   3. Local install — `searchio-mcp` on PATH, else `python -m searchio.mcp`.
//
// Any extra args are forwarded to the server (e.g. --transport sse ...).
// Env: SEARCHIO_MCP_IMAGE (default ghcr.io/ioserviceslabs/searchio-mcp:latest),
//      SEARCHIO_DOCKER (default "auto"; "0" disables the Docker path).

import { spawn, spawnSync } from "node:child_process";

const args = process.argv.slice(2);
const IMAGE = process.env.SEARCHIO_MCP_IMAGE || "ghcr.io/ioserviceslabs/searchio-mcp:latest";

function fail(msg) {
  console.error(`searchio-mcp: ${msg}`);
  console.error("Install options: https://github.com/IOServicesLabs/searchio#1-install");
  process.exit(1);
}

function passthrough(cmd, cmdArgs, opts = {}) {
  const child = spawn(cmd, cmdArgs, { stdio: "inherit", ...opts });
  child.on("error", (err) => fail(`could not start ${cmd}: ${err.message}`));
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    process.exit(code ?? 1);
  });
  return child;
}

function dockerAvailable() {
  if ((process.env.SEARCHIO_DOCKER ?? "auto") === "0") return false;
  try {
    const r = spawnSync("docker", ["info", "--format", "{{.ServerVersion}}"], {
      stdio: "ignore",
      timeout: 8000,
    });
    return r.status === 0;
  } catch {
    return false;
  }
}

function dockerImagePresent() {
  const r = spawnSync("docker", ["image", "inspect", IMAGE], { stdio: "ignore", timeout: 8000 });
  return r.status === 0;
}

function dockerPull() {
  // Long timeout: first pull moves hundreds of MB. stdio inherited so the
  // registry's auth/error story is visible when this fails.
  const r = spawnSync("docker", ["pull", IMAGE], { stdio: "inherit", timeout: 600000 });
  return r.status === 0;
}

function localAvailable() {
  const r = spawnSync("searchio-mcp", ["--help"], { stdio: "ignore", timeout: 8000 });
  return r.status === 0;
}

function pythonLauncher() {
  for (const py of ["python3", "python", "py"]) {
    const r = spawnSync(py, ["-c", "import searchio.mcp"], { stdio: "ignore", timeout: 8000 });
    if (r.status === 0) return py;
  }
  return null;
}

function main() {
  // 1. explicit operator override
  if (process.env.SEARCHIO_MCP_CMD) {
    const cmd = process.env.SEARCHIO_MCP_CMD;
    if (process.platform === "win32") {
      const line = [cmd, ...args].join(" ");
      passthrough(process.env.ComSpec || "cmd", ["/d", "/s", "/c", line]);
    } else {
      passthrough(process.env.SHELL || "sh", ["-c", `${cmd} "$@"`, "-", ...args]);
    }
    return;
  }

  // 2. Docker (all-in-one image: python tiers + engine + Chromium). The image
  //    must be present or pullable; otherwise fall through to local install
  //    rather than dying on a bare `docker run`.
  if (dockerAvailable()) {
    if (dockerImagePresent() || dockerPull()) {
      // Image confirmed local; --pull=never so a registry hiccup can't kill a
      // working setup mid-start.
      passthrough("docker", ["run", "-i", "--rm", "--pull=never", IMAGE, "searchio-mcp", ...args]);
      return;
    }
    console.error(
      `searchio-mcp: docker image ${IMAGE} unavailable (missing, private, or registry unreachable); trying a local install`
    );
  }

  // 3. local installs
  if (localAvailable()) {
    passthrough("searchio-mcp", args);
    return;
  }
  const py = pythonLauncher();
  if (py) {
    passthrough(py, ["-m", "searchio.mcp", ...args]);
    return;
  }

  fail("no backend found (tried Docker, searchio-mcp on PATH, python -m searchio.mcp)");
}

main();
