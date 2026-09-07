"""One table style for every `splat` command that prints a grid.

Two things matter here and neither is cosmetic. The table is sized to the
actual terminal, because a hardcoded width silently overflows and wraps every
border on a narrower one. And cells fold rather than truncate: an env var or a
cache path arriving as `SPLAT_EXTRACT_SURFACE_OPACIT...` is unusable, while the
same value wrapped over two lines is still copy-pasteable.
"""

from collections.abc import Iterable, Sequence

from rich import box
from rich.table import Table

from splat.cli._console import console


def render(
    columns: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    flex: int | None = None,
    group_by: int | None = None,
    title: str | None = None,
    caption: str | None = None,
) -> None:
    """`flex` names the column that absorbs the leftover width, so the others
    keep their natural size instead of every column shrinking proportionally
    and wrapping the short identifiers you actually type.

    `group_by` names a column whose value groups consecutive rows. The column
    is dropped and its value becomes a rule plus a heading row, so grouping
    costs one line per group instead of the widest label in every row.
    """
    table = Table(
        title=title,
        caption=caption,
        box=box.SQUARE,
        title_style="bold",
        caption_style="dim",
        header_style="bold",
    )
    for index, column in enumerate(columns):
        if index == group_by:
            continue
        table.add_column(column, overflow="fold", ratio=1 if index == flex else None)
    width = len(columns) - (1 if group_by is not None else 0)
    previous = None
    for row in rows:
        if group_by is not None and row[group_by] != previous:
            if previous is not None:
                table.add_section()
            previous = row[group_by]
            table.add_row(f"[bold]{previous}[/bold]", *[""] * (width - 1))
        table.add_row(*(cell for index, cell in enumerate(row) if index != group_by))
    console.print(table)
