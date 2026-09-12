from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import ClassVar, Literal

import click
import typer
from typer.core import TyperGroup

LazyKind = Literal["command", "typer"]


@dataclass(frozen=True)
class LazyCommandSpec:
    module: str
    attr: str
    kind: LazyKind
    help: str


class LazyCommand(click.Command):
    """A command placeholder that imports the real Typer command on demand."""

    def __init__(self, name: str, spec: LazyCommandSpec) -> None:
        super().__init__(name=name, help=spec.help, short_help=spec.help)
        self._spec = spec
        self._loaded: click.Command | None = None

    def load(self) -> click.Command:
        if self._loaded is None:
            module = importlib.import_module(self._spec.module)
            target = getattr(module, self._spec.attr)
            if self._spec.kind == "typer":
                command = typer.main.get_command(target)
                command.name = self.name
            else:
                app = typer.Typer()
                app.command(name=self.name)(target)
                command = typer.main.get_command(app)
            self._loaded = command
        return self._loaded

    @property
    def params(self) -> list[click.Parameter]:  # type: ignore[override]
        return self.load().params

    @params.setter
    def params(self, value: list[click.Parameter]) -> None:
        self.__dict__["params"] = value

    @property
    def commands(self) -> dict[str, click.Command]:
        return getattr(self.load(), "commands", {})

    @property
    def hidden(self) -> bool:  # type: ignore[override]
        return self.load().hidden

    @hidden.setter
    def hidden(self, value: bool) -> None:
        self.__dict__["hidden"] = value

    def invoke(self, ctx: click.Context) -> object:
        return self.load().invoke(ctx)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        return self.load().parse_args(ctx, args)

    def get_help(self, ctx: click.Context) -> str:
        return self.load().get_help(ctx)

    def get_short_help_str(self, limit: int = 45) -> str:
        return self.load().get_short_help_str(limit)


class LazyTyperGroup(TyperGroup):
    lazy_commands: ClassVar[dict[str, LazyCommandSpec]] = {}

    def __init__(self, *args: object, **kwargs: object) -> None:
        commands = dict(kwargs.pop("commands", {}) or {})
        commands.update(
            {
                name: LazyCommand(name, spec)
                for name, spec in self.lazy_commands.items()
                if name not in commands
            }
        )
        kwargs["commands"] = commands
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = self.commands.get(cmd_name)
        if isinstance(command, LazyCommand):
            loaded = command.load()
            self.commands[cmd_name] = loaded
            return loaded
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [
            (name, spec.help) for name, spec in self.lazy_commands.items() if name in self.commands
        ]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)
