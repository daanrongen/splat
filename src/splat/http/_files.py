"""Turning multipart uploads into on-disk files — the HTTP analogue of
cli/_pipeline_io.py's input resolution.
"""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import UploadFile


@contextmanager
def saved_upload(upload: UploadFile) -> Iterator[Path]:
    suffix = Path(upload.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.file.read())
        tmp_path = Path(tmp.name)
    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)
