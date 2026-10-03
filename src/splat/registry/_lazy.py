from functools import cache
from importlib import import_module


@cache
def load_backend(path: str) -> type:
    module, _, name = path.partition(":")
    return getattr(import_module(module), name)
