from splat.adapters.formats.ply import PlyReader, PlyWriter
from splat.adapters.formats.splat_fmt import SplatFormatReader, SplatFormatWriter
from splat.ports.splat_io import SplatReader, SplatWriter

FORMAT_READERS: dict[str, type[SplatReader]] = {
    ".ply": PlyReader,
    ".splat": SplatFormatReader,
}

FORMAT_WRITERS: dict[str, type[SplatWriter]] = {
    ".ply": PlyWriter,
    ".splat": SplatFormatWriter,
}
