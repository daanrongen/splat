# splat

`splat` is a local image-to-3D asset pipeline for Apple Silicon. It generates,
reads, transforms, reconstructs, renders, and serves visual assets through one
typed manifest graph.

The core idea is simple:

```text
text prompt -> image -> depth / caption / embedding / sticker
image       -> gaussian_cloud -> render -> image
gaussian_cloud -> convert / compress / declutter / extract.surface
depth_map + source image -> shape_3d
```

Every pipeline stage reads and writes a `Manifest`. Every splat file tool reads
and writes a `GaussianCloud`. Those two hub types keep the project from turning
into an N-by-N pile of one-off converters.

## Install

```sh
git clone https://github.com/daanrongen/splat.git
cd splat
mise install
uv sync
uv run splat --help
```

Released builds can also be installed with Homebrew:

```sh
brew install daanrongen/splat/splat
```

Model weights are downloaded on demand into the local cache. To inspect or pull
them explicitly:

```sh
splat models list
splat models info sharp
splat models pull sharp
splat models rm sharp
```

## Command Taxonomy

`splat` deliberately separates expensive model work from cheap deterministic
work and cache inspection.

| Surface | Commands | Contract |
|---|---|---|
| Model-backed stages | `diffuse`, `caption`, `embed`, `segment`, `upscale`, `depth`, `gaussian`, `render` | May load model/runtime backends and use substantial compute. |
| Deterministic tools | `tools convert`, `tools compress`, `tools declutter`, `tools normalize.color`, `tools displace.height`, `tools extract.surface` | Fixed transforms for a given input and option set. No model selection. |
| Inspection/admin | `info`, `validate`, `manifest`, `models`, `env` | Should be fast and should not import ML runtimes unless explicitly needed. |
| Services | `http`, `mcp` | Expose the same handler layer over HTTP or MCP stdio. |

Cold-start is part of the architecture. Commands such as
`splat manifest list` and `splat manifest get <id>` read local metadata only and
must not import Torch, CoreML, MLX, Transformers, Open3D, Trimesh, or OpenCV.

## Manifests

A `Manifest` is the pipeline currency:

- `id`
- `kind`
- `content_path`
- typed `metadata`
- `params`
- `parent_ids`
- `created_by`
- `content_size`
- `content_sha256`
- `created_at`

Pipeline-produced manifests are invocation-addressed: the id is derived from
stage, model, parameters, and parent manifest ids. External files are
content-addressed: registering the same bytes resolves to the same cached
content when possible.

The cache defaults to:

```text
$XDG_CACHE_HOME/splat/
|-- models/     # converted or downloaded model artifacts
`-- manifests/  # cached content + .meta.json records
```

Manifest commands always operate on the local cache:

```sh
splat manifest list
splat manifest list --kind image --created-by diffuse
splat manifest get <id>
splat manifest rm <id>
splat manifest clear --kind image --yes
```

Inputs accepted by pipeline stages:

- a filesystem path
- `@<manifest-id>`
- `-` for NDJSON manifests from stdin

Manifest-producing commands print NDJSON when stdout is piped, so stages can be
chained directly:

```sh
splat diffuse "a small red toy robot, studio lighting" \
  | splat gaussian - --model sharp \
  | splat render - -o robot.png
```

## Asset Kinds

`ManifestKind` is intentionally flat. Capability tags express what each kind can
do without creating a rigid class hierarchy.

| Kind | Tags | Common producers | Storage |
|---|---|---|---|
| `image` | `raster`, `rgb`, `colorlike` | `diffuse`, `upscale`, `render`, `tools normalize.color` | `.png` |
| `sticker` | `raster`, `rgba`, `colorlike` | `segment` | `.png` |
| `caption` | `text` | `caption` | `.txt` |
| `embedding` | `vector` | `embed` | `.npy` |
| `depth_map` | `raster`, `single_channel` | `depth` | `.npy` |
| `shape_3d` | `mesh_3d` | `tools displace.height`, `tools extract.surface` | `.glb`, `.obj`, `.ply` |
| `gaussian_cloud` | `splat_3d` | `gaussian` | `.ply` |

Every stage declares a `StageContract` over kinds or tags. For example,
`caption`, `embed`, `depth`, `segment`, and `gaussian` can all accept any
`colorlike` input.

## Common Workflows

### Text To Rendered Gaussian Splat

```sh
splat diffuse "beautiful romantic Turner painting of a landscape" \
  | splat gaussian - --model sharp -o scene.ply

splat render scene.ply -o scene.png --width 1280 --height 720
```

`sharp` is the default single-image Gaussian backend. It is fast enough for the
headline chain and records that its output has metric scale.

Use `mlx3d-capture` when you have 3 or more real overlapping views of one
physical scene:

```sh
splat gaussian frame-*.png --model mlx3d-capture --quality balanced -o scene.ply
```

Prompting a diffusion model for "front", "side", and "back" views does not make
multi-view-consistent input. Use real views today; future multi-view generation
belongs in a dedicated `views` stage.

### Image To Depth To Mesh

```sh
splat depth photo.png --model depth-pro \
  | splat tools displace.height - -o relief.glb
```

`depth-pro` produces metric depth with a focal-length estimate. Relative
disparity backends are useful for visualization and ranking, but should not be
treated as metric geometry input.

### Gaussian File Delivery

```sh
splat tools declutter scene.ply scene.clean.ply
splat tools convert scene.clean.ply scene.sog
splat tools compress scene.clean.ply scene.web.sog --profile web-delivery
```

`.ply` is the canonical lossless interchange format. `.splat` is the classic
32-byte-per-point web-viewer format. `.sog` is a compact, spatially sorted,
image-codec-compressed bundle intended for delivery.

### Caption And Embed

```sh
splat caption photo.png -o caption.txt
splat embed photo.png -o image.embedding.npy
splat embed --text "red toy robot" -o text.embedding.npy
```

Embeddings are normalized vectors. Captions and embeddings are regular manifests
and can be chained with the rest of the graph.

## Backends

Backends are loaded lazily. A catalog entry is pure metadata until a
model-backed command actually runs.

| Stage | Models | Notes |
|---|---|---|
| `diffuse` | `sdxl-turbo-mlx`, `sd21-coreml` | Text/image to raster image. |
| `caption` | `fastvlm-0.5b` | Image/sticker to text. |
| `embed` | `mobileclip2-s0` | Image/text to normalized vector. |
| `segment` | `sam-mlx`, `sam2-coreml` | Image/sticker to RGBA stickers. |
| `depth` | `depth-pro`, `depth-anything-v2-coreml` | Metric depth or relative disparity, depending on backend. |
| `upscale` | `realesrgan-mlx` | 2x or 4x raster upscaling. |
| `gaussian` | `sharp`, `mlx3d-capture` | Single-image feed-forward or multi-view optimization. |
| `render` | `blender` | Local Blender-backed rendering. |

Licenses are part of catalog metadata. Non-commercial or research-only models
are reported in `splat models list` and warned at invocation time.

## Remote And Programmatic Use

Run a trusted local HTTP server:

```sh
splat http --host 127.0.0.1:8000
```

Route remote-capable CLI and SDK calls through it:

```sh
SPLAT_URL=http://macbook:8000 splat diffuse "dog" -o dog.png
```

There is no authentication in v1. Bind only to trusted interfaces or put
authentication in front of the server.

Python SDK:

```python
import splat

image = splat.diffuse("a small red boat").asset
cloud = splat.gaussian(image, model="sharp")[0]
render = splat.render(cloud, width=1280, height=720)[0]
pixels = render.as_image()
```

MCP:

```sh
splat mcp
```

## Environment

Every option that can be defaulted from the environment declares its own
`SPLAT_*` variable. Inspect the effective values:

```sh
splat env
splat env --export > .env.example
```

Precedence is:

```text
CLI flag > environment > built-in default
```

Important non-option settings:

| Variable | Purpose |
|---|---|
| `SPLAT_MODEL_CACHE_DIR` | Model artifact cache. |
| `SPLAT_MANIFEST_CACHE_DIR` | Manifest cache. |
| `SPLAT_URL` | Remote HTTP server base URL. |
| `SPLAT_BLENDER_BIN` | Blender executable path. |
| `SPLAT_RENDER_TIMEOUT` | Render subprocess timeout in seconds. |

## Architecture

The dependency rule is ports-and-adapters:

```text
cli/http/mcp/sdk -> handlers -> application -> ports/domain
registry -> ports/domain
adapters -> ports/domain
```

The important boundaries:

- `domain/` defines stable data and invariants.
- `ports/` defines backend, repository, file IO, and client protocols.
- `application/` performs cache-aware orchestration.
- `registry/` stores pure catalog metadata and lazy factories.
- `adapters/` own ML runtimes, file formats, external processes, cache, and clients.
- `handlers/` are transport-neutral request handlers.
- `cli/`, `http/`, `mcp/`, and `api.py` are driving adapters.

See [docs/architecture.md](docs/architecture.md) for the detailed contributor
architecture and [docs/README.md](docs/README.md) for the documentation map.
