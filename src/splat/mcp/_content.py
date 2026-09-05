"""Turning an Manifest into MCP content blocks — the MCP analogue of
http/_schemas.py's wire models.
"""

import base64

import mcp.types as types

from splat.domain.manifest import Manifest

_EXT_MIME = {
    "png": "image/png",
    "npy": "application/octet-stream",
    "glb": "model/gltf-binary",
    "txt": "text/plain",
    "ply": "application/octet-stream",
    "splat": "application/octet-stream",
    "obj": "text/plain",
}


def _mime_type(asset: Manifest) -> str:
    return _EXT_MIME.get(asset.content_path.suffix.lstrip("."), "application/octet-stream")


def image_content(asset: Manifest) -> types.ImageContent:
    data = base64.b64encode(asset.content_path.read_bytes()).decode("ascii")
    return types.ImageContent(data=data, mimeType=_mime_type(asset))


def resource_content(asset: Manifest) -> types.EmbeddedResource:
    data = base64.b64encode(asset.content_path.read_bytes()).decode("ascii")
    return types.EmbeddedResource(
        resource=types.BlobResourceContents(
            uri=f"asset://{asset.id}", mimeType=_mime_type(asset), blob=data
        )
    )


def text_content(text: str) -> types.TextContent:
    return types.TextContent(type="text", text=text)
