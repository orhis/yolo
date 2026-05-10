"""Wspólne helpery do ładnego wyjścia w konsoli (rich-based).

Użycie:
    from src.console import console, phase, info, success, step, make_progress

    phase("Setup", "Instalacja zależności i sanity check")
    info("Ładuję model...")
    with step("Pobieranie wag"):
        ...
    success("Gotowe")
"""

import sys
from contextlib import contextmanager
from typing import Iterator

# Windows konsola domyślnie używa cp1250 → wywala się na znakach typu →, ✓.
# Wymuszamy UTF-8 zanim rich napisze cokolwiek.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

THEME = Theme({
    "phase": "bold magenta",
    "info": "cyan",
    "success": "bold green",
    "warning": "yellow",
    "error": "bold red",
    "metric": "bold cyan",
    "value": "white",
    "muted": "dim white",
})

console = Console(theme=THEME, highlight=False)


def phase(title: str, subtitle: str = "") -> None:
    """Duży nagłówek fazy pipeline'u (Setup, Training, Evaluation, ...)."""
    body = f"[phase]{title}[/phase]"
    if subtitle:
        body += f"\n[muted]{subtitle}[/muted]"
    console.print()
    console.print(Panel(body, expand=False, border_style="phase", padding=(0, 2)))


def info(msg: str) -> None:
    console.print(f"[info]>[/info] {msg}")


def success(msg: str) -> None:
    console.print(f"[success]✓[/success] {msg}")


def warning(msg: str) -> None:
    console.print(f"[warning]![/warning] {msg}")


def error(msg: str) -> None:
    console.print(f"[error]x[/error] {msg}")


def kv(key: str, value: object) -> None:
    """Pojedyncza linia metryka: wartość z wyrównaniem."""
    console.print(f"  [metric]{key:<28}[/metric] [value]{value}[/value]")


@contextmanager
def step(msg: str) -> Iterator[None]:
    """Spinner + opis kroku, na końcu zaznacza success."""
    with console.status(f"[info]{msg}...[/info]", spinner="dots"):
        yield
    success(msg)


def make_progress() -> Progress:
    """Standardowy progress bar dla długich operacji.

    Użycie:
        with make_progress() as progress:
            task = progress.add_task("Anotacja", total=120)
            for img in images:
                ...
                progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def stats_table(title: str, rows: list[tuple[str, object]]) -> Table:
    """Prosta tabela klucz: wartość (np. info o środowisku)."""
    table = Table(title=title, show_header=False, box=None, padding=(0, 2))
    table.add_column(style="metric")
    table.add_column(style="value")
    for k, v in rows:
        table.add_row(k, str(v))
    return table


def metrics_table(title: str, headers: list[str], rows: list[list[object]]) -> Table:
    """Tabela z nagłówkami — np. metryki per klasa."""
    table = Table(title=title, padding=(0, 1), header_style="bold magenta")
    for i, h in enumerate(headers):
        table.add_column(
            h,
            style="metric" if i == 0 else "value",
            justify="left" if i == 0 else "right",
        )
    for row in rows:
        table.add_row(*[str(c) for c in row])
    return table
