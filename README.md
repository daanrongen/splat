# splat

A universal converter for Gaussian Splatting files and models — the pandoc/ffmpeg of 3D Gaussian Splatting. Runs fully locally, no NVIDIA required (Apple Silicon MPS supported), no model weights bundled — they're pulled on demand from HuggingFace Hub.

## Features

- Convert between `.ply` and `.splat` (antimatter15) formats.
- Inspect (`info`) and validate (`validate`) splat files.
- Compress/prune splats for web delivery or archival (`compress`).
- Pull and manage reconstruction model weights (`models`).
- Image → splat reconstruction and per-scene training are designed (ports exist) but not yet wired to a real model — see [Architecture](#architecture).

## Installation

### Homebrew

```
brew install daanrongen/splat/splat
```

### From source

```
git clone https://github.com/daanrongen/splat.git
cd splat
mise install
uv sync
```

## Usage

```
splat convert scene.ply scene.splat
splat convert scene.ply scene.compressed.ply --profile web-delivery  # via `compress`
splat info scene.ply
splat validate scene.ply --strict
splat models list
splat models pull mvsplat
splat models info mvsplat
```

## Configuration

### Environment variables

Managed via `mise.toml`'s `[env]` and a gitignored `mise.local.toml` (copy `mise.local.toml.example`):

- `HF_TOKEN` — HuggingFace Hub token, for gated models.
- `HF_HOME` / `SPLAT_MODEL_CACHE_DIR` — where pulled model weights are cached.

### Model cache

Weights are never bundled with the package. `splat models pull <name>` downloads them into `SPLAT_MODEL_CACHE_DIR` on first use; `splat models rm <name>` removes them.

## Development

### Prerequisites

`mise`, `uv`, `lefthook` (all installed via Homebrew).

### Setup

```
mise trust && mise install
uv sync
lefthook install
```

### Running tests

```
mise run test   # or: uv run pytest
```

### Linting & formatting

```
mise run check  # lint + format-check + tests, the CI-equivalent
```

## Architecture

Domain-driven design, ports & adapters. A canonical `GaussianCloud` aggregate (`src/splat/domain/gaussians.py`) is the hub every format/model adapter reads and writes, avoiding an N² pairwise-converter explosion. See `src/splat/ports/` for the interfaces (`SplatReader`/`SplatWriter`, `ReconstructionBackend`, `TrainingBackend`, `Compressor`, `ModelSource`) and `src/splat/adapters/` for implementations. `src/splat/registry/` is the entire dependency-wiring layer — plain dict lookups, no framework.

Per-scene optimization training (`splat train`) and feed-forward image reconstruction (`splat convert --model ...`) are deliberately stubbed: every currently-available license-clean, pose-free, Apple-Silicon-ready model or trainer surveyed turned out to have a disqualifying issue (hard CUDA dependency, non-commercial license, or pre-alpha rasterizer) — see the docstring in `src/splat/adapters/reconstruction/mvsplat.py` for the full survey.

## Contributing

## License

MIT
