from splat.registry.wiring import get_client


def list_models() -> list[dict]:
    """List catalog models and whether their weights are cached locally."""
    return [
        {"name": row.name, "runtime": row.runtime, "license": row.license, "cached": row.cached}
        for row in get_client().models_list()
    ]


def pull(name: str) -> str:
    """Download a model's weights from HuggingFace Hub."""
    get_client().models_pull(name)
    return f"pulled {name}"


def info(name: str) -> dict:
    """Show a model's license, source repo, and expected input shape."""
    summary = get_client().models_info(name)
    return {
        "name": summary.name,
        "runtime": summary.runtime,
        "source": summary.source,
        "license": summary.license,
    }


def rm(name: str) -> str:
    """Remove a model's cached weights."""
    get_client().models_rm(name)
    return f"removed {name}"
