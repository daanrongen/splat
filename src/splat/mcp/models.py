from splat.handlers import models as models_handler


def list_models() -> list[dict]:
    """List catalog models and whether their weights are cached locally."""
    return [
        {
            "name": descriptor.name,
            "runtime": getattr(descriptor, "runtime", "-"),
            "license": str(descriptor.license),
            "cached": cached,
        }
        for descriptor, cached in models_handler.list_models()
    ]


def pull(name: str) -> str:
    """Download a model's weights from HuggingFace Hub."""
    models_handler.pull(name)
    return f"pulled {name}"


def info(name: str) -> dict:
    """Show a model's license, source repo, and expected input shape."""
    descriptor = models_handler.info(name)
    return {
        "name": descriptor.name,
        "runtime": getattr(descriptor, "runtime", "-"),
        "source": descriptor.hf_repo_id,
        "license": str(descriptor.license),
    }


def rm(name: str) -> str:
    """Remove a model's cached weights."""
    models_handler.rm(name)
    return f"removed {name}"
