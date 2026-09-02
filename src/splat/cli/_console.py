from rich.console import Console

console = Console()
err_console = Console(stderr=True)


def error(message: str) -> None:
    err_console.print(f"[bold red]error:[/bold red] {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]warning:[/yellow] {message}")
