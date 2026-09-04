from dataclasses import dataclass
from pathlib import Path

from splat.application.convert import ConvertResult, ConvertUseCase
from splat.registry.wiring import get_reader, get_writer


@dataclass(frozen=True)
class ConvertRequest:
    input_path: Path
    output_path: Path
    from_format: str | None = None
    to_format: str | None = None


def handle(request: ConvertRequest) -> ConvertResult:
    reader = get_reader(request.from_format or request.input_path.suffix)
    writer = get_writer(request.to_format or request.output_path.suffix)
    return ConvertUseCase(reader, writer).execute(request.input_path, request.output_path)
