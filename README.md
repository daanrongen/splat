# splat

`splat` is a local 3D image-space toolkit for generating, captioning, embedding, segmenting, upscaling, estimating depth, predicting meshes, reconstructing Gaussian splats, and transforming Gaussian splat files. It is designed for Apple Silicon first, with MLX, CoreML, PyTorch/MPS, and OpenCV-backed image I/O, and it downloads model weights on demand instead of shipping them in the package.

The CLI is split by behavior:

- **Model-backed stages** live at the top level: `diffuse`, `caption`, `embed`, `segment`, `upscale`, `depth`, `gaussian`, `mesh`, and `train`. These commands choose a model/runtime or run model optimization, may use substantial compute, and can produce backend-dependent results.
- **Deterministic tools** live under `splat tools`: `convert`, `compress`, `declutter`, `normalize.color`, `displace.height`, and `extract.surface`. These commands are pure transforms for a given input and option set; they do not select models, devices, or licenses.
- **Inspection, services, and administration** stay separate: `info`, `validate`, `manifest`, `models`, `http`, `mcp`, and `env`.

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

### caption

`splat caption` turns an image or sticker into a UTF-8 text caption asset.

```sh
splat diffuse "dog" | splat caption - --model fastvlm-0.5b -o dog.txt
```

Key options: `--model fastvlm-0.5b`, `--prompt`, `--max-tokens`, `--temperature`, `--device`, `-o/--output`. FastVLM weights are cataloged as research/non-commercial.

### embed

`splat embed` turns an image, sticker, or text string into a normalized `.npy` embedding vector asset.

```sh
splat embed robot.png --model mobileclip2-s0 -o robot.embedding.npy
splat embed --text "red toy robot" --model mobileclip2-s0
```

Key options: `--model mobileclip2-s0`, `--text`, `--device`, `-o/--output`. `mobileclip2-s0` is cataloged as research/non-commercial under Apple's model license. It produces vectors only; search, deduplication, indexes, and retrieval UI are out of scope.

### depth

`splat depth` turns an image or sticker into a depth-map asset stored losslessly as `.npy`, with `-o` writing a normalized preview PNG.

```sh
splat depth stickers/sticker_000.png --model depth-pro -o depth.png
splat depth stickers/sticker_000.png --model depth-anything-v2-coreml -o depth.png
```

Key options: `--model depth-pro|depth-anything-v2-coreml`, `--device`, `-o/--output`. `depth-pro` produces metric depth in meters with a focal-length estimate; `depth-anything-v2-coreml` (Apple's CoreML export of Depth Anything V2 Small) produces relative inverse depth (disparity, higher value = closer) on an arbitrary per-image scale, with no focal length - `tools displace.height` requires a known focal length, so it only works with `depth-pro` output.

### upscale

`splat upscale` turns an image or sticker asset into a higher-resolution image asset with a super-resolution backend.

```sh
splat diffuse "dog" | splat upscale - --factor 4 -o dog-4x.png
```

Key options: `--model realesrgan-mlx`, `--factor 2|4`, `--tile`, `-o/--output`.

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

`splat tools convert` rewrites Gaussian splat files between registered file formats: `.ply` (lossless), `.splat` (antimatter15's 32-byte-per-point web-viewer format), and `.sog` (a spatially-sorted, image-codec-compressed bundle - see below).

```sh
splat tools convert scene.ply scene.splat
splat tools convert scene.ply -o scene.splat
splat tools convert scene.ply scene.sog
```

Key options: `-f/--from`, `-t/--to`, `-o/--output`.

### tools compress

`splat tools compress` prunes and quantizes a Gaussian splat for a named delivery profile. Since the output format is inferred from the output path (any registered format), writing to `.sog` combines pruning with the much higher compression ratio described below.

```sh
splat tools compress scene.ply scene.web.ply --profile web-delivery
splat tools compress scene.ply scene.web.sog --profile web-delivery
```

Key options: `--profile web-delivery|archival`.

`.sog` (SOG-inspired, #65) is a spatially-sorted, image-codec-compressed bundle: points are reordered along a 3D Morton (Z-order) curve so spatially nearby Gaussians land near each other in raster order, each attribute (position, scale, rotation, SH-degree-0 color/opacity) is packed into an 8/16-bit-per-channel grid, and each grid is PNG-encoded - the sort is what lets PNG's DEFLATE compress far better than on unsorted data. This is a simplified, license-clean take on Self-Organizing Gaussians (SOG, ECCV'24) and PlayCanvas's real `.sog` format: Morton order stands in for PLAS's differentiable grid optimization, and plain PNG stands in for WebP plus per-attribute codebooks - not bit-compatible with either, but on a real 38k-point capture it took a 9.5MB `.ply` down to 535KB (~18x), beating `.splat`'s already-lossy 1.2MB. Higher-order SH is always dropped, like `web-delivery`.

### tools declutter

`splat tools declutter` removes isolated floater Gaussians via neighbor-density outlier detection - a visual-cleanliness pass, distinct from `compress`'s delivery-size pruning.

```sh
splat tools declutter scene.ply scene.clean.ply
```

Key options: `--k` (neighbors considered per point, default 16), `--std-ratio` (outlier threshold in standard deviations, default 2.0).

### tools normalize.color

`splat tools normalize.color` corrects per-view exposure/white-balance drift across a multi-photo capture, ahead of `gaussian` reconstruction, so it doesn't bake into per-Gaussian SH color as spurious view-dependent noise. Matches each image's per-channel mean to the cohort's per-channel median (a gray-world-style gain correction) - the cheapest per-view color-correction model in the appearance-embedding/tone-curve literature, jointly across every image passed in one call.

```sh
splat tools normalize.color capture/*.jpg | splat gaussian -
```

Takes 2 or more image/sticker paths or `@<asset-id>`s and produces one corrected image manifest per input, so it composes with piping like any other pipeline stage.

### tools displace.height

`splat tools displace.height` turns a depth-map asset into a triangulated, textured mesh. It reads the source image through the depth asset's provenance.

```sh
splat depth sticker.png | splat tools displace.height - -o sticker.glb
splat tools displace.height @depth_asset_id -o sticker.obj
```

Key options: `-t/--to glb|obj|ply`, `-o/--output`.

### tools extract.surface

`splat tools extract.surface` extracts a textured triangle mesh from a `GaussianCloud` via screened Poisson surface reconstruction (Kazhdan & Hoppe) over Open3D - Gaussians below `--opacity-threshold` are dropped first (they're usually background/floaters, not surface), remaining means become an oriented point cloud, and low-density Poisson vertices are trimmed to cut the characteristic "bubble" artifact. Output format is inferred from the output path's extension (`.obj`, `.glb`, `.gltf`); `.usdz` isn't supported yet.

```sh
splat tools extract.surface scene.ply scene.glb
splat tools extract.surface scene.ply scene.obj --depth 10 --opacity-threshold 0.2
```

Key options: `-t/--to obj|glb|gltf`, `--depth` (Poisson octree depth, default 9), `--opacity-threshold` (default 0.1).

## Inspection

`info` and `validate` remain top-level debug commands because they inspect data rather than transform it.

```sh
splat info scene.ply
splat validate scene.ply --strict
```

`splat manifest` reads and manages the local manifest cache directly (`~/.cache/splat/assets` by default) — a manifest already lives wherever it was produced, so this always operates on the local cache rather than routing through `SPLAT_URL`.

```sh
splat manifest list --kind image --created-by diffuse
splat manifest get <id>
splat manifest rm <id>
```

## Piping And Manifests

Every stage consumes and produces a `Manifest` — the pipeline's universal currency, and its "one canonical shape every stage consumes and produces, so stages chain without knowing about each other." A `Manifest` is a cached, content-addressed record: `id`, `kind`, `content_path`, typed `metadata` (facts about the content itself), `params` (the stage invocation that produced it), `parent_ids`, `created_by`, plus cache-wide provenance every kind gets for free — `content_size`, `content_sha256`, and `created_at`.

Pipeline commands accept a file path, `@<manifest-id>`, or `-` for NDJSON records from stdin. Every manifest-producing stage writes to the content-addressed cache and prints NDJSON when stdout is piped.

The cache itself sits behind `ManifestRepository` (`ports/manifest_repository.py`), with `FilesystemManifestRepository` as its one adapter today. Besides the `find`/`get`/`put`/`put_external` every stage uses to read and write manifests, it exposes `list` (filter by `kind` or a `created_by` substring, most recent first) and `delete` — the CRUD surface `splat manifest`/`GET /manifests`/`manifest_list` MCP tool sit on top of, documented under Inspection and HTTP Server below.

```sh
splat diffuse "dog" | splat caption - -o dog.txt
splat diffuse "dog" | splat upscale - --factor 2 | splat segment - | splat depth - | splat tools displace.height - -o test.obj
```

`-o/--output` writes a convenient copy to the path you choose; it does not replace the cache entry. Cached manifests keep provenance so downstream tools can retrieve parents, such as `displace.height` loading the image that produced a depth map.

### Kind taxonomy

`ManifestKind` stays a flat set — no class hierarchy — but each kind carries composable capability tags (`domain/manifest.py::KIND_TAGS`), so a stage declares "accepts any color image" once instead of repeating `(image, sticker)` tuples across handlers:

| Kind | Tags | Produced by | On-disk format |
|---|---|---|---|
| `image` | `raster`, `rgb`, `colorlike` | diffuse, upscale | `.png` |
| `sticker` | `raster`, `rgba`, `colorlike` | segment | `.png` |
| `caption` | `text` | caption | `.txt` |
| `embedding` | `vector` | embed | `.npy` |
| `depth_map` | `raster`, `single_channel`, `metric` | depth | `.npy` |
| `shape_3d` | `mesh_3d` | mesh, tools displace.height | `.glb` / `.obj` |
| `gaussian_cloud` | `splat_3d` | gaussian | `.ply` |

Every model-backed stage declares what it needs as a `StageContract` (`domain/contracts.py`): named input slots, accepted kinds/tags, and a min/max count, checked by one shared validator instead of ad hoc kind checks. `splat gaussian`'s contract is built per-request from the chosen backend's `required_image_count()` (mlx3d-capture needs 3+); piping the wrong kind in fails with a message naming both sides:

```
$ splat depth photo.png | splat gaussian -
error: gaussian (mlx3d-capture) requires at least 3 image or sticker, got 1 depth_map. gaussian has no
text-to-3D or depth-only reconstruction path — pipe image/sticker assets in instead, e.g.
`splat diffuse ... | splat segment - | splat gaussian -`.
```

### Stage flow

Every stage that accepts a "colorlike" manifest (`image` or `sticker`) can consume the output of every stage that produces one — that fan-out/fan-in is what the tag system buys, instead of thirteen separately-remembered producer/consumer pairs:

```mermaid
flowchart LR
    diffuse([diffuse]) -->|image| colorlike{{image / sticker}}
    segment([segment]) -->|sticker xN| colorlike
    upscale([upscale]) -->|image| colorlike
    colorlike -->|colorlike| segment
    colorlike -->|colorlike| upscale
    colorlike -->|colorlike| depth([depth])
    colorlike -->|colorlike| caption([caption])
    colorlike -->|colorlike| embed([embed])
    colorlike -->|colorlike| mesh([mesh])
    colorlike -->|colorlike xN| gaussian([gaussian])
    colorlike -->|colorlike xN| normalize([tools normalize.color])
    normalize -->|image xN| colorlike
    caption -->|caption text| embed
    depth -->|depth_map| displace([tools displace.height])
    gaussian -->|gaussian_cloud .ply| convert([tools convert])
    gaussian -->|gaussian_cloud .ply| compress([tools compress])
    gaussian -->|gaussian_cloud .ply| declutter([tools declutter])
    gaussian -->|gaussian_cloud .ply| extract([tools extract.surface])
```

`tools convert`/`tools compress`/`tools declutter`/`tools extract.surface`/`info`/`validate` sit outside the `Manifest` system by design — they're deterministic file-in/file-out transforms over `GaussianCloud`, not cached pipeline stages. `tools normalize.color` and `tools displace.height` are still cached pipeline stages like any model-backed command — "tools" means "no swappable model catalog," not "no `Manifest`."

## HTTP Server

`splat http` exposes the same work over a trusted-local-network REST API. Requests execute on the machine running the server.

```sh
splat http --host 127.0.0.1:8000
SPLAT_HOST=0.0.0.0:8000 splat http
```

| Route | CLI surface | Response |
|---|---|---|
| `POST /diffuse` | `splat diffuse` | image bytes |
| `POST /caption` | `splat caption` | text bytes |
| `POST /embed` | `splat embed` | embedding `.npy` bytes |
| `POST /segment` | `splat segment` | list of sticker manifest summaries |
| `POST /depth` | `splat depth` | depth `.npy` bytes |
| `POST /upscale` | `splat upscale` | image bytes |
| `POST /mesh` | `splat mesh` | mesh bytes |
| `POST /gaussian` | `splat gaussian` | Gaussian splat bytes |
| `POST /convert` | `splat tools convert` | converted file bytes |
| `POST /compress` | `splat tools compress` | compressed file bytes |
| `POST /declutter` | `splat tools declutter` | decluttered file bytes |
| `POST /extract-surface` | `splat tools extract.surface` | mesh file bytes |
| `POST /info` | `splat info` | JSON summary |
| `POST /validate` | `splat validate` | JSON summary |
| `GET /assets/{id}` | asset fetch | raw asset bytes |
| `GET /manifests` | `splat manifest list` | list of manifest summaries |
| `GET /manifests/{id}` | `splat manifest get` | full manifest detail |
| `DELETE /manifests/{id}` | `splat manifest rm` | JSON |
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

`splat mcp` exposes the same command taxonomy over stdio for MCP clients. Model-backed operations use top-level tool names such as `diffuse`, `caption`, `embed`, `segment`, `upscale`, `depth`, `mesh`, and `gaussian`; deterministic operations use `tools_convert`, `tools_compress`, `tools_declutter`, `tools_normalize_color`, `tools_displace_height`, and `tools_extract_surface`; manifest CRUD uses `manifest_list`, `manifest_get`, and `manifest_delete`.

```sh
splat mcp
```

When `SPLAT_URL` is set, remote-capable MCP tools route through the configured `splat http` server just like the CLI.

## Python SDK

`import splat` gives plain Python callables over the same handlers the CLI and `splat http` use - no subprocess, no argument parsing.

```python
import splat

result = splat.diffuse("a small red boat", steps=4)
stickers = splat.segment(result.asset, max_stickers=5)
cloud_manifest = splat.gaussian(stickers)

pixels = result.asset.as_image()          # numpy RGB/RGBA array
cloud = cloud_manifest[0].as_gaussian_cloud()  # domain.gaussians.GaussianCloud
```

Every function takes a file path, `@<asset-id>`, or a `Manifest` object (or a list of any mix) wherever the CLI accepts `INPUT` positionally, and raises `SplatDomainError`/`ValueError` directly instead of exiting the process. `Manifest.load(id)` fetches a cached manifest by id, and `.as_image()`/`.as_text()`/`.as_array()`/`.as_gaussian_cloud()` decode its content per kind. Setting `SPLAT_URL` redirects remote-capable calls exactly like the CLI - `render()` is the one exception, always executing locally (see `ports/client.py`).

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
| `SPLAT_UPSCALE_MODEL` | Default model for `splat upscale` |
| `SPLAT_UPSCALE_FACTOR` | Default factor for `splat upscale` |
| `SPLAT_UPSCALE_TILE` | Default tile size for `splat upscale` |
| `SPLAT_<COMMAND>_<PARAM>` | Default value for supported command options |

Run `splat env` to inspect every resolved setting, its source, and whether `SPLAT_URL` is reachable.

Precedence for command defaults is CLI flag, `os.environ`, `mise env --json`, then built-in default.

## Architecture

The implementation follows a ports-and-adapters layout with a small transport-neutral handler layer.

| Layer | Role |
|---|---|
| `domain/` | Core value objects such as `GaussianCloud`, `Manifest`, `DepthMap`, contracts, and errors |
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
