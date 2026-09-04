# splat

`splat` is a local 3D image-space toolkit for generating, segmenting, estimating depth, predicting meshes, reconstructing Gaussian splats, and transforming Gaussian splat files. It is designed for Apple Silicon first, with MLX, CoreML, and PyTorch/MPS backends, and it downloads model weights on demand instead of shipping them in the package.

The CLI is split by behavior:

- **Model-backed stages** live at the top level: `diffuse`, `segment`, `depth`, `gaussian`, `mesh`, and `train`. These commands choose a model/runtime or run model optimization, may use substantial compute, and can produce backend-dependent results.
- **Deterministic tools** live under `splat tools`: `convert`, `compress`, `displace.height`, and `extract.surface`. These commands are pure transforms for a given input and option set; they do not select models, devices, or licenses.
- **Inspection, services, and administration** stay separate: `info`, `validate`, `models`, `http`, `mcp`, and `env`.

## Getting Started

```sh
git clone https://github.com/daanrongen/splat.git
cd splat
mise install
uv sync
uv run splat --help
```

Install from Homebrew when using a released build:

```sh
brew install daanrongen/splat/splat
```

Check the model catalog and pull weights explicitly:

```sh
splat models list
splat models info sdxl-turbo-mlx
splat models pull sdxl-turbo-mlx
splat models rm sdxl-turbo-mlx
```

## Model-Backed Stages

### diffuse

`splat diffuse` turns a text prompt into an image asset.

```sh
splat diffuse "a small red toy robot, studio lighting" --model sdxl-turbo-mlx -o robot.png
```

Key options: `--model sdxl-turbo-mlx|sd21-coreml`, `--negative`, `--steps`, `--seed`, `--device`, `-o/--output`.

### segment

`splat segment` turns an image asset into RGBA sticker cutouts.

```sh
splat segment robot.png --model sam-mlx --max-stickers 5 -o stickers/
```

Key options: `--model sam-mlx|sam2-coreml`, `--max-stickers`, `--device`, `-o/--output`.

### depth

`splat depth` turns an image or sticker into a metric depth-map asset stored losslessly as `.npy`, with `-o` writing a normalized preview PNG.

```sh
splat depth stickers/sticker_000.png --model depth-pro -o depth.png
```

Key options: `--model depth-pro`, `--device`, `-o/--output`.

### gaussian

`splat gaussian` reconstructs a Gaussian splat from input images via a feed-forward reconstruction backend.

```sh
splat gaussian view-a.png view-b.png --model mvsplat -o scene.ply
```

Key options: `--model mvsplat`, `--device`, `-o/--output`. `mvsplat` is currently a deliberate stub because the available license-clean, Apple-native feed-forward options are not ready for this package.

### mesh

`splat mesh` predicts a mesh from an image via a learned image-to-mesh backend.

```sh
splat mesh sticker.png --model triposr -o sticker.glb
```

Key options: `--model triposr`, `--device`, `-o/--output`. `triposr` is currently a deliberate stub; use `splat tools displace.height` for the deterministic depth-map-to-mesh path available today.

### train

`splat train` is reserved for per-scene Gaussian splat optimization and is currently a stub.

```sh
splat train dataset/
```

## Deterministic Tools

`splat tools` contains transforms whose output is fully determined by their inputs and options. These commands do not have `--model`, `--device`, model downloads, or model-license warnings.

```sh
splat tools --help
```

### tools convert

`splat tools convert` rewrites Gaussian splat files between registered file formats.

```sh
splat tools convert scene.ply scene.splat
splat tools convert scene.ply -o scene.splat
```

Key options: `-f/--from`, `-t/--to`, `-o/--output`.

### tools compress

`splat tools compress` prunes and quantizes a Gaussian splat for a named delivery profile.

```sh
splat tools compress scene.ply scene.web.ply --profile web-delivery
```

Key options: `--profile web-delivery|archival`.

### tools displace.height

`splat tools displace.height` turns a depth-map asset into a triangulated, textured mesh. It reads the source image through the depth asset's provenance.

```sh
splat depth sticker.png | splat tools displace.height - -o sticker.glb
splat tools displace.height @depth_asset_id -o sticker.obj
```

Key options: `-t/--to glb|obj|ply`, `-o/--output`.

### tools extract.surface

`splat tools extract.surface` is reserved for deterministic Gaussian-splat-to-surface extraction and is currently a stub.

```sh
splat tools extract.surface scene.ply scene.obj
```

## Inspection

`info` and `validate` remain top-level debug commands because they inspect data rather than transform it.

```sh
splat info scene.ply
splat validate scene.ply --strict
```

## Piping And Assets

Pipeline commands accept a file path, `@<asset-id>`, or `-` for NDJSON records from stdin when the command works with assets. Every asset-producing stage writes to the content-addressed cache and prints NDJSON when stdout is piped.

```sh
splat diffuse "dog" | splat segment - | splat depth - | splat tools displace.height - -o test.obj
```

`-o/--output` writes a convenient copy to the path you choose; it does not replace the cache entry. Cached assets keep provenance so downstream tools can retrieve parents, such as `displace.height` loading the image that produced a depth map.

## HTTP Server

`splat http` exposes the same work over a trusted-local-network REST API. Requests execute on the machine running the server.

```sh
splat http --host 127.0.0.1:8000
SPLAT_HOST=0.0.0.0:8000 splat http
```

| Route | CLI surface | Response |
|---|---|---|
| `POST /diffuse` | `splat diffuse` | image bytes |
| `POST /segment` | `splat segment` | sticker asset manifest |
| `POST /depth` | `splat depth` | depth `.npy` bytes |
| `POST /mesh` | `splat mesh` | mesh bytes |
| `POST /gaussian` | `splat gaussian` | Gaussian splat bytes |
| `POST /convert` | `splat tools convert` | converted file bytes |
| `POST /compress` | `splat tools compress` | compressed file bytes |
| `POST /info` | `splat info` | JSON summary |
| `POST /validate` | `splat validate` | JSON summary |
| `GET /assets/{id}` | asset fetch | raw asset bytes |
| `GET/POST/DELETE /models...` | `splat models list|pull|info|rm` | JSON |

There is no authentication in v1. Bind it only on trusted interfaces or put authentication in front of it.

## Remote Execution

Set `SPLAT_URL` on a client machine to send remote-capable CLI commands to a running `splat http` server. Outputs are written locally, and asset-producing results are mirrored into the local asset cache under the same id.

```sh
# on the compute machine
SPLAT_HOST=0.0.0.0:8000 splat http

# on another machine
SPLAT_URL=http://macbook:8000 splat diffuse "dog" -o test.png
```

`SPLAT_HOST` controls where `splat http` binds. `SPLAT_URL` controls where client commands execute. The two settings are intentionally separate.

## MCP Server

`splat mcp` exposes the same command taxonomy over stdio for MCP clients. Model-backed operations use top-level tool names such as `diffuse`, `segment`, `depth`, `mesh`, and `gaussian`; deterministic operations use `tools_convert`, `tools_compress`, and `tools_displace_height`.

```sh
splat mcp
```

When `SPLAT_URL` is set, remote-capable MCP tools route through the configured `splat http` server just like the CLI.

## Environment

Cache roots default under `$XDG_CACHE_HOME/splat` or `~/.cache/splat`:

```text
$XDG_CACHE_HOME/splat/
|-- huggingface/
|-- models/
`-- assets/
```

Important settings:

| Env var | Purpose |
|---|---|
| `SPLAT_CACHE_ROOT` | Base cache directory |
| `SPLAT_MODEL_CACHE_DIR` | Converted or compiled model cache |
| `SPLAT_ASSET_CACHE_DIR` | Pipeline asset cache |
| `SPLAT_URL` | Remote `splat http` base URL for client commands |
| `SPLAT_HOST` | Bind host and port for `splat http` |
| `SPLAT_<COMMAND>_<PARAM>` | Default value for supported command options |

Run `splat env` to inspect every resolved setting, its source, and whether `SPLAT_URL` is reachable.

Precedence for command defaults is CLI flag, `os.environ`, `mise env --json`, then built-in default.

## Architecture

The implementation follows a ports-and-adapters layout with a small transport-neutral handler layer.

| Layer | Role |
|---|---|
| `domain/` | Core value objects such as `GaussianCloud`, `Asset`, `DepthMap`, and errors |
| `ports/` | Protocols for model backends, file IO, compression, cache, and client dispatch |
| `application/` | Model-backed use cases and cache-aware orchestration |
| `application/tools/` | Deterministic tool use cases |
| `adapters/` | ML runtimes, file formats, cache implementation, local client, and HTTP client |
| `registry/` | Model and format catalogs plus simple factory wiring |
| `handlers/` | Transport-neutral request handlers for model-backed commands |
| `handlers/tools/` | Transport-neutral request handlers for deterministic tools |
| `cli/` | Top-level Typer commands for model-backed stages and service/debug commands |
| `cli/tools/` | Typer subcommands for deterministic tools |
| `http/` | REST routes that call handlers directly and always execute locally |
| `http/tools/` | REST route modules backed by deterministic tool handlers |
| `mcp/` | MCP stdio tools using the same command taxonomy |

The key boundary is intentional: choosing a model/runtime belongs to top-level model-backed commands; deterministic transforms belong under `tools`; inspection commands stay top-level because they report facts without transforming data.
