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
- **Generative pipeline** — `diffuse → segment → depth → mesh → …` turns a
  text prompt into an image, cuts a subject out as an RGBA "sticker",
  estimates its depth, and predicts a mesh from it. Each stage caches its
  output as an `Asset` and can read the previous stage's output directly,
  by cache ID, or piped from another `splat` command — the same
  "everything is one universal object" idea `pandoc` uses for documents.

Both halves are ports & adapters: a command talks to a `Protocol` port, and
a *runtime* (`mlx`, `coreml`, or `torch`) provides the adapter. The point of
splitting runtime from port is that adapters chain — a CoreML `diffuse`
can feed an MLX `segment`, or vice versa.

Sitting alongside that pipeline is **`splat tools`** — deterministic,
non-ML operators that compose between the stochastic ML stages via the
same `Asset`/cache/pipe contract, but carry none of the swappable-backend
machinery (no `Protocol` port, no model catalog, no license) since there's
nothing to pick between. `diffuse`/`segment`/`depth`/`mesh` all imply
"pick a backend, get non-deterministic output"; a `tools` command is
always the same pure function.

## Quick tour

```sh
# splat file tooling
splat convert scene.ply scene.splat
splat compress scene.ply scene.web.ply --profile web-delivery
splat info scene.ply
splat validate scene.ply --strict

# generative pipeline
splat diffuse "a small red toy robot, studio lighting" --model sdxl-turbo-mlx -o robot.png
splat segment robot.png --model sam-mlx -o stickers/ --max-stickers 5
splat depth stickers/sticker_000.png --model depth-pro -o depth.png
splat mesh stickers/sticker_000.png -o sticker.glb   # not yet implemented, see below

# a deterministic `tools` operator spliced into the same pipeline
splat depth stickers/sticker_000.png | splat tools displace.height - -o sticker.glb

# chained via Unix pipes — same pipeline, one line
splat diffuse "a small red toy robot" | splat segment - | splat depth - \
  | splat tools displace.height - -o robot.glb
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
| `diffuse PROMPT` | Text → image | `--model sdxl-turbo-mlx\|sd21-coreml`, `--negative`, `--steps`, `--seed`, `-o FILE` |
| `segment INPUT` | Image → RGBA sticker cutouts | `--model sam-mlx\|sam2-coreml`, `--max-stickers`, `-o DIR` |
| `depth INPUT` | Image → per-pixel metric depth | `--model depth-pro`, `-o FILE` (normalized preview PNG) |
| `mesh INPUT` | Image → mesh, via a learned model | not yet implemented, stubbed intentionally; `--model triposr`, `--device`, `-o FILE` |
| `gaussian INPUT...` | Image(s) → Gaussian splat (feed-forward reconstruction) | `--model mvsplat`, `--device`, `-o FILE` |
| `convert INPUT... [OUTPUT]` | Splat↔splat format conversion | `-f/--from`, `-t/--to`, `-o FILE` |
| `info PATH` | Point count, SH degree, bounding box | |
| `validate PATH` | Check domain invariants, exit non-zero on failure | `--strict` |
| `compress INPUT OUTPUT` | Prune + quantize for delivery | `--profile web-delivery\|archival` |
| `train DATASET_DIR` | Per-scene optimization | not yet implemented, stubbed intentionally |
| `tools displace.height INPUT` | Depth map → triangulated, textured mesh | `-t/--to glb\|obj\|ply`, `-o FILE` |
| `tools extract.surface INPUT OUTPUT` | Gaussian splat → mesh export (SuGaR-style) | not yet implemented, stubbed intentionally; `-t/--to`, `--device` |
| `models list\|pull\|info\|rm NAME` | Manage cached model weights | |
| `http` | Start splat's HTTP server | `--host`, `--port` (default `8000`) |
| `env` | Show every `SPLAT_*` default and where it resolved from | |

`INPUT` on `segment`/`depth`/`mesh`/`tools displace.height` accepts a file
path, `@<asset-id>` to address a cached asset directly, or `-` to read
piped NDJSON asset records from an earlier `splat` command.
`displace.height` specifically requires that asset to be a `depth_map` —
pipe it through `splat depth` first; it fetches the source image for
texturing via the depth asset's own provenance (`parent_ids`), it never
invokes `depth` itself.

### Piping contract

Every pipeline command prints one NDJSON record per output asset when
stdout is piped (`{"id", "kind", "path", "metadata", ...}`), or a
human-readable summary in an interactive terminal — the same
machine/human duality `jq`, `ripgrep --json`, and friends use. `-o` writes
a plain file alongside the cache entry; it never replaces caching.

## HTTP server

`splat http` exposes the same pipeline over a REST API — one route per CLI
command, synchronous (a request blocks until the result is ready, then
returns the asset bytes directly):

| Route | Mirrors |
|---|---|
| `POST /diffuse` | `splat diffuse` — JSON body, response body is the image |
| `POST /segment` | `splat segment` — multipart upload, JSON manifest of sticker asset ids |
| `POST /depth` | `splat depth` — multipart upload, response body is the depth `.npy` |
| `POST /mesh` | `splat mesh` — multipart upload, response body is the `.glb` |
| `POST /gaussian` | `splat gaussian` — multipart upload(s), response body is the splat file |
| `POST /convert` | `splat convert` — multipart upload + `to`/`from_format`, response body is the converted file |
| `POST /compress` | `splat compress` — multipart upload + `profile`, response body is the compressed file |
| `POST /info` | `splat info` — multipart upload, JSON |
| `POST /validate` | `splat validate` — multipart upload + `strict`, JSON |
| `GET/POST/DELETE /models...` | `splat models list\|pull\|info\|rm` — JSON |
| `GET /assets/{id}` | fetch a cached asset's raw bytes by id |

**No authentication in v1** — `splat http` is intended for a trusted local
network (e.g. running it on one machine and driving it from another on the
same LAN); don't expose it beyond that without adding your own auth layer
in front of it. It always executes locally and never proxies elsewhere.

## Model catalog

| Model | Task | Runtime | License |
|---|---|---|---|
| `sdxl-turbo-mlx` | diffuse | MLX | StabilityAI-NC-Community (non-commercial) |
| `sd21-coreml` | diffuse | CoreML | OpenRAIL-M |
| `sam-mlx` | segment | MLX | Apache-2.0 |
| `sam2-coreml` | segment | CoreML | Apache-2.0 |
| `depth-pro` | depth | PyTorch/MPS | Apple-ASCL |
| `triposr` | mesh | PyTorch/MPS | MIT (stub — see below) |
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

`triposr` (single-image → mesh) is stubbed the same way: no Apple-native or
depth-conditioned image-to-mesh model exists anywhere surveyed, and
TripoSR's own `tsr` package isn't vendored yet. See the docstring in
`src/splat/adapters/mesh/triposr.py` for the full survey. Depth-map → mesh
via pure geometry (no model, no license, no download) is available today
as `splat tools displace.height` instead.

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
`SPLAT_<COMMAND>_<PARAM>` env var (e.g. `SPLAT_DIFFUSE_MODEL`,
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
  `metadata`) that every `diffuse`/`segment`/`depth`/`mesh`/`tools`
  command reads and writes, cached by `FilesystemAssetCache`.

`ports/` defines the `Protocol` interfaces (`SplatReader`/`SplatWriter`,
`ReconstructionBackend`, `DiffusionBackend`, `SegmentationBackend`,
`DepthEstimationBackend`, `MeshPredictionBackend`, `MeshExporter`,
`ModelSource`, `Compressor`); `adapters/` implements them per runtime;
`registry/` is the entire dependency-wiring
layer — plain dict catalogs, no framework. `splat tools` operators skip
this entirely — no port, no registry, no `--model` — since a deterministic
function has nothing to swap; `application/tools/` holds the plain
functions directly.

`handlers/` sits one level above `application/`: one module per capability,
each resolving a backend/cache from `registry/wiring.py` and invoking the
matching use case — the one place that logic lives, so it isn't duplicated
per driving adapter. `cli/` and `http/` are both thin driving adapters on
top of `handlers/`: `cli/` maps argv to a request and renders the result as
console output or a written file; `http/` maps a REST request to the same
request type and renders the result as an HTTP response. Neither imports a
concrete adapter directly.

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
