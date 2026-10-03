from splat.adapters.formats.ply import PlyReader, PlyWriter
from splat.adapters.formats.splat_fmt import SplatFormatReader, SplatFormatWriter
from splat.adapters.formats.spz import SpzReader, SpzWriter
from splat.ports.splat_io import SplatReader, SplatWriter

FORMAT_READERS: dict[str, type[SplatReader]] = {
    ".ply": PlyReader,
    ".splat": SplatFormatReader,
    ".spz": SpzReader,
}

FORMAT_WRITERS: dict[str, type[SplatWriter]] = {
    ".ply": PlyWriter,
    ".splat": SplatFormatWriter,
    ".spz": SpzWriter,
}
