from fastapi import APIRouter

from splat import __version__

router = APIRouter()


@router.get("/version")
def version() -> dict[str, str]:
    return {"version": __version__}
