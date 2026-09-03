# splat

A universal converter for "3D image space" — from a text prompt to a
Gaussian Splat, and between Gaussian Splat file formats — built like
`pandoc`: every command reads and writes a small set of universal,
content-addressed representations instead of one converter per pair of
formats. Runs fully locally on Apple Silicon (MLX and CoreML backends, no
NVIDIA required); no model weights ship in the package, they're pulled from
HuggingFace Hub on first use and cached like `ollama pull`.

## What it does

Two halves, one architecture:

- **Splat file tooling** — convert, inspect, validate, and compress
  `.ply`/`.splat` Gaussian Splat files.
- **Generative pipeline** — `generate → segment → depth → mesh → …` turns a
  text prompt into an image, cuts a subject out as an RGBA "sticker",
  estimates its depth, and lifts it into a textured mesh. Each stage caches
  its output as an `Asset` and can read the previous stage's output
  directly, by cache ID, or piped from another `splat` command — the same
  "everything is one universal object" idea `pandoc` uses for documents.

Both halves are ports & adapters: a command talks to a `Protocol` port, and
a *runtime* (`mlx`, `coreml`, or `torch`) provides the adapter. The point of
splitting runtime from port is that adapters chain — a CoreML `generate`
can feed an MLX `segment`, or vice versa.

## Quick tour

```sh
# splat file tooling
splat convert scene.ply scene.splat
splat compress scene.ply scene.web.ply --profile web-delivery
splat info scene.ply
splat validate scene.ply --strict

# generative pipeline
splat generate "a small red toy robot, studio lighting" --model sdxl-turbo-mlx -o robot.png
splat segment robot.png --model sam-mlx -o stickers/ --max-stickers 5
splat depth stickers/sticker_000.png --model depth-pro -o depth.png
splat mesh stickers/sticker_000.png -o sticker.glb   # runs depth-pro internally if needed

# chained via Unix pipes — same pipeline, one line
splat generate "a small red toy robot" | splat segment - | splat depth - | splat mesh - -o robot.glb
```

## Installation

### Homebrew

```sh
brew install daanrongen/splat/splat
```

### From source

```sh
git clone https://github.com/daanrongen/splat.git
cd splat
mise install
uv sync
```

## CLI reference

| Command | Does | Key options |
|---|---|---|
| `generate PROMPT` | Text → image | `--model sdxl-turbo-mlx\|sd21-coreml`, `--negative`, `--steps`, `--seed`, `-o FILE` |
| `segment INPUT` | Image → RGBA sticker cutouts | `--model sam-mlx\|sam2-coreml`, `--max-stickers`, `-o DIR` |
| `depth INPUT` | Image → per-pixel metric depth | `--model depth-pro`, `-o FILE` (normalized preview PNG) |
| `mesh INPUT` | Image(+depth) → textured mesh; or splat → mesh export (not yet implemented) | `--model depth-heightfield`, `--depth-model`, `-t/--to glb\|obj\|ply`, `-o FILE` |
| `gaussian INPUT...` | Image(s) → Gaussian splat (feed-forward reconstruction) | `--model mvsplat`, `--device`, `-o FILE` |
| `convert INPUT... [OUTPUT]` | Splat↔splat format conversion | `-f/--from`, `-t/--to`, `-o FILE` |
| `info PATH` | Point count, SH degree, bounding box | |
| `validate PATH` | Check domain invariants, exit non-zero on failure | `--strict` |
| `compress INPUT OUTPUT` | Prune + quantize for delivery | `--profile web-delivery\|archival` |
| `train DATASET_DIR` | Per-scene optimization | not yet implemented, stubbed intentionally |
| `models list\|pull\|info\|rm NAME` | Manage cached model weights | |
| `env` | Show every `SPLAT_*` default and where it resolved from | |

`INPUT` on `segment`/`depth`/`mesh` accepts a file path, `@<asset-id>` to
address a cached asset directly, or `-` to read piped NDJSON asset records
from an earlier `splat` command — except a path with a recognized splat
extension (`.ply`/`.splat`), which routes `mesh` to the (not yet
implemented) gaussian → mesh export instead.

### Piping contract

Every pipeline command prints one NDJSON record per output asset when
stdout is piped (`{"id", "kind", "path", "metadata", ...}`), or a
human-readable summary in an interactive terminal — the same
machine/human duality `jq`, `ripgrep --json`, and friends use. `-o` writes
a plain file alongside the cache entry; it never replaces caching.

## Model catalog

| Model | Task | Runtime | License |
|---|---|---|---|
| `sdxl-turbo-mlx` | generate | MLX | StabilityAI-NC-Community (non-commercial) |
| `sd21-coreml` | generate | CoreML | OpenRAIL-M |
| `sam-mlx` | segment | MLX | Apache-2.0 |
| `sam2-coreml` | segment | CoreML | Apache-2.0 |
| `depth-pro` | depth | PyTorch/MPS | Apple-ASCL |
| `depth-heightfield` | mesh | geometry (no model weights) | MIT |
| `mvsplat` | image → splat | PyTorch/MPS | MIT (stub — see below) |

`splat models info <name>` prints a model's exact source repo and license.
Non-commercial licenses print a loud warning when used.

`mvsplat` (feed-forward image → Gaussian reconstruction) is deliberately
left as a stub raising `NotImplementedError`: every currently-available
license-clean, pose-free, Apple-Silicon-ready alternative surveyed turned
out to have a disqualifying issue (hard CUDA dependency, non-commercial
license, or a pre-alpha rasterizer). See the docstring in
`src/splat/adapters/reconstruction/mvsplat.py` for the full survey. Per-scene
optimization training (`splat train`) is stubbed for the same reason: the
reference 3DGS rasterizer is CUDA-only with no settled MPS equivalent yet.

## Caching

Nothing ships in the package. Everything lands under one root:

```
$XDG_CACHE_HOME/splat/        (defaults to ~/.cache/splat)
├── huggingface/               HF_HOME — raw HF downloads
├── models/                    SPLAT_MODEL_CACHE_DIR — converted/compiled weights
└── assets/                    SPLAT_ASSET_CACHE_DIR — pipeline outputs (images, stickers, depth maps)
```

`splat models pull <name>` / `splat models rm <name>` manage the model
cache explicitly; pipeline commands populate the asset cache automatically
as they run. Override any of the three via the matching env var (set in
`mise.toml`'s `[env]`, or a gitignored `mise.local.toml` for `HF_TOKEN`).

Every command's `--model`/`--device`/parameter flags also fall back to a
`SPLAT_<COMMAND>_<PARAM>` env var (e.g. `SPLAT_GENERATE_MODEL`,
`SPLAT_SEGMENT_DEVICE`) before their built-in default — set globally via
shell `export`, or per-project via `mise.toml`'s `[env]`. Precedence:
CLI flag > `os.environ` > `mise env --json` (queried lazily when the var
isn't in `os.environ` and `mise` is on `PATH`) > built-in default. Run
`splat env` to see every setting's resolved value and source.

## Architecture

Domain-driven design, ports & adapters, two aggregates at the hub instead
of one converter per format/model pair:

- `GaussianCloud` (`domain/gaussians.py`) — the splat file side. Every
  format reader/writer and reconstruction backend reads or writes this.
- `Asset` (`domain/asset.py`) — the generative pipeline side. A typed,
  content-addressed envelope (`kind`, `content_path`, `parent_ids`,
  `metadata`) that every `generate`/`segment`/`depth` command reads and
  writes, cached by `FilesystemAssetCache`.

`ports/` defines the `Protocol` interfaces (`SplatReader`/`SplatWriter`,
`ReconstructionBackend`, `ImageGenerationBackend`, `SegmentationBackend`,
`DepthEstimationBackend`, `MeshPredictionBackend`, `MeshExporter`,
`ModelSource`, `Compressor`); `adapters/` implements them per runtime;
`registry/` is the entire dependency-wiring
layer — plain dict catalogs, no framework. `cli/` is a thin Typer
presentation layer that resolves adapters through `registry/wiring.py` and
never imports a concrete adapter directly.

## Development

```sh
mise trust && mise install   # python, uv, lefthook
uv sync
lefthook install

mise run test    # uv run pytest
mise run check   # lint + format-check + tests, the CI-equivalent
```

## License

MIT
