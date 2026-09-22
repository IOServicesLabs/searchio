"""Command line interface.

The commands mirror the three price points in :mod:`searchio.engine`, and the
help text says which is which, because the most common way to misuse this tool
is to reach for ``research`` when ``search`` would have answered in a fifth of
a second for nothing.
"""

from __future__ import annotations

import asyncio
import json as jsonlib
import sys

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from .config import Settings, get_settings, set_settings
from .engine import Engine
from .errors import Blocked, ConfigError, SearchioError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LLM-assisted search over a blocking-resistant acquisition ladder.",
)

# Windows terminals default to cp1252, which raises on the first em dash in a
# page title. Force UTF-8 rather than mangling output.
console = Console(legacy_windows=False)
#: Diagnostics and errors (bug 90): stdout is the payload when --json is
#: on, so everything that is not the payload goes here.
err_console = Console(stderr=True, legacy_windows=False)


def _emit_json(text: str) -> None:
    """Raw JSON on stdout: rich's JSON renderable wraps at the terminal
    width (80 in a pipe), which split a long URL across lines and left the
    consumer with a document that was not JSON (bug 90)."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _settings(max_tier: int | None, no_cache: bool) -> Settings:
    try:
        s = get_settings()
    except ValidationError as exc:
        # A bad SEARCHIO_* value is an operator error, one line (bug 89).
        err_console.print(f"[bold red]config:[/] {_one_line(exc)}")
        raise typer.Exit(2)
    if max_tier is not None:
        s.max_tier = max_tier
    if no_cache:
        s.cache_enabled = False
    set_settings(s)
    return s


def _one_line(exc: ValidationError) -> str:
    """'intent: Input should be ...' -- the field and the reason, nothing else."""
    try:
        e = exc.errors()[0]
        loc = ".".join(str(p) for p in e.get("loc", ())) or "input"
        return f"{loc}: {e.get('msg', '')}"[:300]
    except Exception:  # noqa: BLE001
        return str(exc).splitlines()[0][:300]


def _cooling(eng, intent: str) -> dict[str, str]:
    """Configured providers for ``intent`` whose breaker is open.

    An empty item result is honest ("no listings") only when providers were
    actually asked. When every provider that serves the intent sits
    circuit-open from earlier failures, the emptiness is a cooldown, not an
    absence -- the same story /search names as ``providers_skipped`` (bug 164)
    and the tool bridge names as a circuit note (bug 129). The CLI item
    commands used to print a calm "No listings found." and exit 0 on it (the
    CLI twin of bug 75/164): now they say so and exit 1 (bug 182).
    """
    # Delegates to the router's canonical breaker read (bug 164), so the CLI,
    # /search and /items all name a cooldown the same way rather than three
    # copies drifting apart.
    router = getattr(eng, "router", None)
    cooling = getattr(router, "_cooling", None)
    return cooling(intent) if callable(cooling) else {}


def _report_no_items(eng, intent: str) -> None:
    """Print the empty-item verdict and set the exit code (bug 182)."""
    cooling = _cooling(eng, intent)
    if cooling:
        err_console.print("[bold red]error:[/] all providers cooling down: "
                          + "; ".join(f"{n}: {e}" for n, e in cooling.items()))
        raise typer.Exit(1)
    console.print("[yellow]No listings found.[/]")


def _run(coro):
    try:
        return asyncio.run(coro)
    except ValidationError as exc:
        # An invalid argument (--intent bogus) used to escape as a rich
        # traceback -- exit 1 with forty frames for an expected input
        # error (bug 89). Usage errors are exit 2, one line.
        err_console.print(f"[bold red]invalid:[/] {_one_line(exc)}")
        raise typer.Exit(2)
    except ConfigError as exc:
        err_console.print(f"[bold red]config:[/] {exc}")
        raise typer.Exit(2)
    except SearchioError as exc:
        err_console.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        raise typer.Exit(130)


@app.command()
def search(
    query: str = typer.Argument(..., help="What to search for."),
    intent: str = typer.Option("web", "--intent", "-i",
                               help="web|news|video|shopping|academic|code|local|reference|forum"),
    k: int = typer.Option(10, "--k", "-k", help="Number of results."),
    domains: str = typer.Option("", "--domains", "-d", help="Comma-separated domain allowlist."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON."),
    max_tier: int = typer.Option(None, "--max-tier", help="0=http 1=+TLS-impersonation 2=+browser"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show provider diagnostics."),
) -> None:
    """Fast multi-provider search. No model, no API key, no cost."""
    s = _settings(max_tier, no_cache)

    async def go():
        async with Engine(s) as eng:
            res = await eng.search(
                query, intent=intent, k=k,
                domains=[d.strip() for d in domains.split(",") if d.strip()],
            )
            if json_out:
                _emit_json(jsonlib.dumps(
                    {"query": query, "used": res.used, "failed": res.failed,
                     "skipped": dict(getattr(res, "skipped", None) or {}),
                     "elapsed_ms": res.elapsed_ms,
                     "results": [d.model_dump(exclude={"text"}) for d in res.docs]},
                    default=str))
                return
            skipped = dict(getattr(res, "skipped", None) or {})
            if not res.docs:
                if not res.used and (res.failed or skipped):
                    # Nobody answered: exit 1 with the reasons, not a calm
                    # "No results." and exit 0 (the CLI twin of bug 75); a
                    # fan-out where everyone sat circuit-open is the same
                    # story (bug 164).
                    reasons = {**res.failed, **skipped}
                    err_console.print("[bold red]error:[/] "
                                      + ("all providers failed: " if res.failed
                                         else "all providers cooling down: ")
                                      + "; ".join(f"{n}: {e}" for n, e in reasons.items()))
                    raise typer.Exit(1)
                console.print("[yellow]No results.[/]")
                for name, err in {**res.failed, **skipped}.items():
                    console.print(f"  [dim]{name}: {err}[/]")
                return
            table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
            table.add_column("#", width=3, style="dim")
            table.add_column("Result")
            for i, d in enumerate(res.docs, 1):
                agree = d.meta.get("agreement", 1)
                badge = f" [green]×{agree}[/]" if agree > 1 else ""
                srcs = ",".join(d.meta.get("sources", [d.source]))
                table.add_row(
                    str(i),
                    f"[bold]{d.title or '(untitled)'}[/]{badge}\n"
                    f"[blue]{d.url}[/]\n"
                    f"[dim]{d.snippet[:200]}[/]\n"
                    f"[dim italic]{srcs} · score {d.score:.4f}[/]",
                )
            console.print(table)
            foot = f"{len(res.docs)} results · {res.elapsed_ms}ms · providers: {', '.join(res.used)}"
            if res.failed:
                foot += f" · [red]failed: {', '.join(res.failed)}[/]"
            if skipped:
                foot += f" · [yellow]cooling down: {', '.join(skipped)}[/]"
            console.print(f"[dim]{foot}[/]")
            if verbose:
                for name, err in {**res.failed, **skipped}.items():
                    console.print(f"  [dim]{name}: {err}[/]")

    _run(go())


@app.command()
def items(
    query: str = typer.Argument(..., help="Product to look up."),
    k: int = typer.Option(10, "--k", "-k"),
    domains: str = typer.Option("", "--domains", "-d", help="Marketplace domains to restrict to."),
    browser: bool = typer.Option(False, "--browser", help="Allow the stealth browser tier."),
    json_out: bool = typer.Option(False, "--json"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Look up a product across marketplaces, merged into one row per product."""
    s = _settings(2 if browser else 1, no_cache)

    async def go():
        async with Engine(s) as eng:
            found = await eng.find_items(
                query, k=k, use_browser=browser,
                domains=[d.strip() for d in domains.split(",") if d.strip()] or None,
            )
            if json_out:
                _emit_json(jsonlib.dumps([i.model_dump() for i in found], default=str))
                return
            if not found:
                _report_no_items(eng, "shopping")
                return
            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("Price", justify="right", style="green", width=12)
            table.add_column("Product")
            table.add_column("Seller", width=18)
            table.add_column("Rating", width=12)
            for it in found:
                offers = it.meta.get("offer_count")
                name = it.title[:70] + (f"\n[dim]+{offers - 1} other offers[/]" if offers else "")
                rating = f"{it.rating:.1f} ({it.reviews or 0})" if it.rating is not None else "—"
                table.add_row(
                    str(it.price) if it.price.amount is not None else "[dim]—[/]",
                    name,
                    it.seller or it.domain,
                    rating,
                )
            console.print(table)
            console.print(f"[dim]{len(found)} products[/]")

    _run(go())


@app.command()
def marketplace(
    query: str = typer.Argument(..., help="What to look for."),
    city: str = typer.Option("", "--city", "-c",
                             help="Marketplace city slug, e.g. nyc, seattle, la. "
                                  "Without one, Facebook answers against your exit IP."),
    k: int = typer.Option(20, "--k", "-k"),
    json_out: bool = typer.Option(False, "--json"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Search Facebook Marketplace classifieds. Browser-backed, ~15-30s."""
    s = _settings(2, no_cache)  # browser-only; there is no cheaper tier here

    async def go():
        async with Engine(s) as eng:
            found = await eng.find_local_items(query, city=city, k=k)
            if json_out:
                _emit_json(jsonlib.dumps([i.model_dump() for i in found], default=str))
                return
            if not found:
                _report_no_items(eng, "local")
                return
            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("Price", justify="right", style="green", width=11)
            table.add_column("Was", justify="right", style="dim", width=6)
            table.add_column("Listing")
            table.add_column("Where", width=22)
            for it in found:
                table.add_row(
                    str(it.price) if it.price.amount is not None else "[dim]-[/]",
                    it.attributes.get("was_price", ""),
                    it.title[:64],
                    it.attributes.get("location") or it.attributes.get("delivery") or "[dim]-[/]",
                )
            console.print(table)
            if any(it.meta.get("session_gate") == "empty_feed" for it in found):
                console.print(
                    "[yellow]The saved Facebook session silently returned an empty search "
                    "feed (account gate) -- these listings were fetched anonymously. "
                    "Log in with a real, established account for full results, or set "
                    "SEARCHIO_FACEBOOK_HEADED=1 for roughly triple the anonymous "
                    "volume.[/]"
                )
            where = f"city:{city}" if city else "[yellow]exit-IP location[/]"
            console.print(f"[dim]{len(found)} listings · {where}[/]")
            if not city:
                console.print(
                    "[dim]Pass --city to pin the metro; otherwise these are "
                    "wherever your traffic egresses.[/]"
                )

    _run(go())


@app.command()
def read(
    url: str = typer.Argument(..., help="URL to fetch and extract."),
    max_tier: int = typer.Option(None, "--max-tier"),
    chars: int = typer.Option(4000, "--chars", help="Characters of text to print."),
) -> None:
    """Fetch one URL through the ladder and print its readable text."""
    s = _settings(max_tier, False)

    async def go():
        async with Engine(s) as eng:
            try:
                res = await eng.ladder.fetch(url)
            except Blocked as exc:
                err_console.print(f"[bold red]blocked[/] by {exc.vendor or 'unknown'}: {exc.reason}")
                raise typer.Exit(1)
            from .extract import to_markdown, title_of

            text = to_markdown(res.body, url)
            console.print(Panel(
                f"[bold]{title_of(res.body) or url}[/]\n"
                f"[dim]{res.final_url}\ntier {res.tier} ({res.via}) · {res.status} · "
                f"{res.elapsed_ms}ms · {len(text):,} chars"
                + (f"\nescalations: {res.escalations}" if res.escalations else "") + "[/]",
                expand=False))
            console.print(text[:chars])
            if len(text) > chars:
                console.print(f"[dim]... {len(text) - chars:,} more characters[/]")

    _run(go())


@app.command()
def research(
    question: str = typer.Argument(..., help="The research question."),
    json_out: bool = typer.Option(False, "--json"),
    max_tier: int = typer.Option(None, "--max-tier"),
) -> None:
    """Run the research swarm. Costs real money; needs ANTHROPIC_API_KEY."""
    s = _settings(max_tier, False)

    async def go():
        async with Engine(s) as eng:
            out = err_console if json_out else console

            def progress(event: str, data: dict) -> None:
                if event == "planning":
                    out.print("[dim]planning…[/]")
                elif event == "planned":
                    out.print(Panel(
                        f"[italic]{data['interpretation']}[/]\n\n"
                        + "\n".join(f"[cyan]{s_['id']}[/] {s_['objective']}"
                                    for s_ in data["subtasks"]),
                        title=f"plan · {len(data['subtasks'])} parallel workers", expand=False))
                elif event == "worker_start":
                    out.print(f"  [dim]▸ {data['id']} started[/]")
                elif event == "worker_done":
                    mark = "[red]✗[/]" if data["error"] else "[green]✓[/]"
                    out.print(
                        f"  {mark} [bold]{data['id']}[/] {data['findings']} findings, "
                        f"{data['sources']} sources, {data['tool_calls']} tool calls"
                        + (f" [red]{data['error']}[/]" if data["error"] else ""))
                elif event == "synthesising":
                    out.print("[dim]synthesising…[/]")

            result = await eng.research(question, progress=progress)
            if json_out:
                _emit_json(result.model_dump_json())
                return
            console.print()
            console.print(Panel(result.answer, title="answer", expand=False))
            if result.sources:
                console.print("\n[bold]Sources[/]")
                for d in result.sources[:20]:
                    console.print(f"  [blue]{d.url}[/] [dim]{d.title[:60]}[/]")
            # The cost comes from the run's own budget, which knows which
            # backend it used. Recomputing it here with Claude's list price
            # printed a confident, wrong dollar figure for every other backend.
            cost = (
                f" · ~${result.estimated_cost_usd:.3f}"
                if result.estimated_cost_usd is not None
                else " · cost not priced for this backend"
            )
            console.print(
                f"\n[dim]{result.model or 'model'} · {result.elapsed_s:.1f}s · "
                f"{result.input_tokens:,} in / {result.output_tokens:,} out{cost}"
                + (f" · stopped early: {result.stopped_early}" if result.stopped_early else "")
                + "[/]")

    _run(go())


@app.command()
def providers() -> None:
    """List providers, their capabilities, and whether they are configured."""
    s = _settings(None, False)  # guarded: a bad SEARCHIO_* value is a one-line exit 2

    async def go():
        async with Engine(s) as eng:
            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("Provider", width=18)
            table.add_column("Ready", width=6)
            table.add_column("Capabilities")
            table.add_column("Breaker", width=8)
            for row in eng.router.health_report():
                table.add_row(
                    row["provider"],
                    "[green]yes[/]" if row["configured"] else "[yellow]no[/]",
                    ", ".join(row["capabilities"]),
                    row["breaker"],
                )
            console.print(table)

    _run(go())


@app.command()
def doctor() -> None:
    """Check the environment: tiers, sidecar, keys, cache."""
    s = _settings(None, False)  # guarded: a bad SEARCHIO_* value is a one-line exit 2

    async def go():
        console.print(f"[bold]state dir[/]      {s.state_dir}")
        console.print(f"[bold]max tier[/]       {s.max_tier}")
        console.print(f"[bold]robots policy[/]  {s.robots_policy}")

        ok_key = bool(s.anthropic_key())
        console.print(
            f"[bold]anthropic key[/]  " + ("[green]set[/]" if ok_key else
                                           "[yellow]missing (research disabled)[/]"))

        try:
            from curl_cffi import requests  # noqa: F401
            console.print("[bold]tier 1[/]         [green]curl_cffi available[/]")
        except ImportError:
            console.print("[bold]tier 1[/]         [red]curl_cffi missing[/]")

        async with Engine(s) as eng:
            from .net.sidecar import default_script_path

            script = default_script_path()
            console.print(f"[bold]sidecar script[/]  {script or '[yellow]not found[/]'}")
            try:
                url = await eng.ladder.sidecar.ensure()
                console.print(f"[bold]tier 2[/]         [green]sidecar live at {url}[/]")
            except Exception as exc:
                console.print(f"[bold]tier 2[/]         [yellow]unavailable: {exc}[/]")

            console.print("\n[bold]probing tiers[/]")
            for url in ("https://example.com", "https://www.indeed.com/"):
                try:
                    r = await eng.ladder.fetch(url, use_cache=False)
                    console.print(f"  [green]ok[/]  tier {r.tier} ({r.via}) {r.elapsed_ms:5}ms  {url}")
                except Exception as exc:
                    console.print(f"  [red]fail[/] {type(exc).__name__}: {str(exc)[:60]}  {url}")

            profiles = eng.ladder.profiles.all()[:10]
            if profiles:
                console.print("\n[bold]learned domain tiers[/]")
                for p in profiles:
                    console.print(
                        f"  tier {p.min_tier}  {p.domain:24} "
                        f"[dim]{p.successes} ok / {p.blocks} blocked"
                        + (f" · {p.last_block_vendor}" if p.last_block_vendor else "") + "[/]")

    _run(go())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("searchio.server:app", host=host, port=port, log_level="info")


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport",
                                  help="stdio | sse | streamable-http"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    path: str = typer.Option("/mcp", "--path"),
) -> None:
    """Run the MCP server (Model Context Protocol) for search + research."""
    from .mcp import main as mcp_main

    mcp_main(["--transport", transport, "--host", host, "--port", port, "--path", path])


if __name__ == "__main__":
    app()
