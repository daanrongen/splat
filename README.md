# splat

`splat` is a local-first asset pipeline for Apple Silicon. Prompts, photos and captures go in; images, depth maps, captions, embeddings, Gaussian splats and meshes come out. Every stage reads and writes one typed, cached `Manifest`, so any command's output pipes into the next and every file keeps its lineage.

```sh
splat diffuse "a stainless steel kettle, studio lighting" \
  | splat gaussian - \
  | splat manifest label - kettle \
  | splat render - -o kettle.png
```

[docs/pipeline.md](docs/pipeline.md) walks one image through every stage, with the commands, numbers and pictures.

```text
prompt ─ diffuse ─▶ image ─┬─ caption ─▶ caption
                           ├─ embed ───▶ embedding
                           ├─ segment ─▶ sticker
                           ├─ upscale ─▶ image
                           ├─ depth ───▶ depth_map ─ mesh ─▶ shape_3d
                           └─ gaussian ▶ gaussian_cloud ─┬─ render ─▶ image
                                                         └─ mesh ───▶ shape_3d
any manifest ─ export ─▶ file + .manifest.json sidecar
```

## Install

```sh
brew trust daanrongen/splat
brew install daanrongen/splat/splat
```

Recent Homebrew refuses third-party taps until they are trusted. Without Homebrew, unpack `splat-<version>-macos-arm64.tar.gz` from a [release](https://github.com/daanrongen/splat/releases) anywhere and run its `bin/splat`; it carries its own Python and needs nothing else installed.

From source:

```sh
git clone https://github.com/daanrongen/splat.git && cd splat
mise install && uv sync
uv run splat --help
```

Model weights download on first use. `render` needs [Blender](https://www.blender.org) on `PATH` (or `SPLAT_BLENDER_BIN`).

## How it works

**Manifests.** Each stage output is a manifest: content in its kind's format plus typed metadata, params, parent ids, the producing model and its license. Ids are invocation keys (stage, model, semantic params, parents), so re-running anything is a cache hit. A stage whose output changes between releases bumps its `STAGE_REVISION`, so only that stage's results are recomputed. External files are content-addressed.

**Inputs.** Every stage takes a file path, `@<manifest-id>`, or `-` for NDJSON records piped from another command. When stdout is piped, commands print one NDJSON record per output; in a terminal they print a summary.

**Files that keep their lineage.** `-o` writes the file plus `<file>.manifest.json`. Feeding that file back into any command, in any cache, restores its manifest; a converted file links back to its source. `splat manifest export` copies a manifest with all its ancestors.

**Kinds.** Each kind has a contract that the cache checks whenever a stage stores an output, so a backend that hands off anything else fails loudly instead of leaving the next stage to patch it. Files imported from outside are stored as they are.

| Kind | Produced by | Contract |
|---|---|---|
| `image` | `diffuse`, `upscale`, `render` | RGB or RGBA `.png`, size matches its metadata |
| `sticker` | `segment` | RGBA `.png` plus its `(x, y, width, height)` box in the source image |
| `caption` | `caption` | non-empty UTF-8 `.txt` |
| `embedding` | `embed` | `.npy` whose dtype and shape match its metadata |
| `depth_map` | `depth` | 2D float32 `.npy` in `metres`, or relative `disparity`, plus focal length when known |
| `gaussian_cloud` | `gaussian` | `.ply` in the OpenGL convention (+Y up), SH degree 0 to 3, point count matching its metadata, plus the source cameras. The PLY header stores the capture camera as `comment capture_camera_intrinsics [fx, fy, cx, cy, width, height]` (a JSON array, stable), and `splat info --json` prints it as `intrinsics` next to `coordinate_convention` and `up_axis` |
| `shape_3d` | `mesh` | `.glb`, `.obj`, `.ply` or `.gltf` with typed vertex and face counts |

## Command reference

Most options can also come from a `SPLAT_*` environment variable (`splat env` lists them). Precedence: flag, then environment, then default. `--device` takes `auto`, `cpu` or `mps`.

### Stages

**`splat diffuse SOURCE [PROMPT]`** generates an image from a text prompt, or edits an image (path or `@id`) with PROMPT as the instruction.

| Option | Default | |
|---|---|---|
| `--model` | `sdxl-turbo-mlx` | see [Models](#models) |
| `--negative` | | negative prompt |
| `--steps` | model default | denoising steps |
| `--strength` | | image-to-image change, 0 to 1 |
| `--seed` | | |
| `--width`, `--height` | 512 | multiples of 64, MLX backend only |
| `--device`, `-o` | | |

**`splat caption INPUT`** writes a one-sentence caption: `--model` (`fastvlm-0.5b`), `--prompt`, `--max-tokens` (80), `--temperature` (0.0), `--device`, `-o`.

**`splat embed [INPUT] [--text TEXT]`** embeds images or a text query as a normalized vector: `--model` (`mobileclip2-s0`), `--device`, `-o` (`.npy`).

**`splat segment INPUT`** cuts an image into RGBA stickers: `--model` (`sam-mlx`), `--max-stickers` (20), `--device`, `--point x,y` and `--not-point x,y` (repeatable) or `--box x0,y0,x1,y1` to cut out one prompted object (`sam2-coreml` takes a box or up to 2 points), `--foreground` (one cutout of the main subject), `--drop-background` (drop masks that cover the frame or its border), `-o DIR` or `-o file.png` for a single sticker.

**`splat depth INPUT`** estimates per-pixel depth: `--model` (`depth-pro`, metric), `--device`, `-o` (a viewable PNG, near is bright).

**`splat upscale INPUT`** super-resolves images or stickers: `--factor` (2 or 4, default 4), `--model` (`realesrgan-mlx`), `--tile` (0 disables tiling), `-o`.

**`splat gaussian INPUTS...`** reconstructs a Gaussian splat. `sharp` takes one image; `mlx3d-capture` takes 3 or more overlapping photos of one scene.

| Option | Default | |
|---|---|---|
| `--model` | `sharp` | `sharp`, `mlx3d-capture` |
| `--focal-35mm` | 30 | `sharp`: focal length for images without EXIF |
| `--quality`, `--iters`, `--max-dim`, `--sh-degree` | `fast` | `mlx3d-capture` training |
| `--poses`, `--refine-poses` | `auto` | `mlx3d-capture` pose estimation |
| `--normalize-color` / `--no-normalize-color` | on | `mlx3d-capture`: even out exposure across views |
| `--low-mem`, `--seed` | | `mlx3d-capture` |
| `--declutter` | off | drop isolated floater Gaussians |
| `--mask STICKER` | | keep only Gaussians that project inside a sticker of the first image, as a cached child cloud |
| `--score` | off | render source view 0 with Blender and record PSNR and SSIM against its photo |
| `--min-registered` | | fail if fewer than this fraction of inputs registered |
| `--orbit-frames`, `--orbit-degrees` | 30° | also render N synthetic views across the sweep |
| `--verbose`, `--device`, `-o` | | `-o` format follows the extension |

Every cloud records a `quality` report in its metadata (opacity histogram, near-transparent, needle and out-of-view ratios), which `splat manifest get` prints.

**`splat render INPUT`** renders a splat to a PNG with Blender.

| Option | Default | |
|---|---|---|
| `--engine` | `cycles` | `cycles` (accurate) or `eevee` (fast preview) |
| `--width`, `--height`, `--samples` | 1280, 720, 32 | |
| `--background` | `black` | `transparent`, `black`, `white`, `grey` or a hex colour |
| `--view N` | | render from source camera N of the cloud |
| `--azimuth`, `--elevation` | 25°, 20° | orbit camera instead of the capture pose |
| `--zoom`, `--distance`, `--fov`, `--look-at x,y,z` | fit to cloud | `--zoom 1` fits the whole subject at any scale; `--distance` is absolute, in the cloud's units |
| `-o` | | |

**`splat mesh INPUT`** turns a metric depth map (`heightfield`, textured by its source image) or a Gaussian cloud (`isosurface`) into a mesh. The backend follows the input kind unless `--model` says otherwise. Options: `--to` (`glb`, `obj`, `ply`, `gltf`; default from `-o`, else `glb`), `--resolution` (voxels along the longest axis, 192), `--opacity-threshold` (0.1), `-o`.

### Files

**`splat export INPUT -o FILE`** writes any asset to a file in the format of its extension, with a lineage sidecar. Gaussian clouds convert between `.ply` (lossless), `.spz` (Niantic, about 7x smaller, keeps SH) and `.splat` (antimatter15 web format, SH dropped). `--profile web-delivery|archival` compresses a cloud first; `--pruning blue-noise --target-count N` thins it to N points.

**`splat info PATH`** prints point count, SH degree, bounds, size and cloud metadata. **`splat validate PATH [--strict]`** checks a splat's invariants and exits non-zero on failure.

### Manifests

| Command | |
|---|---|
| `splat manifest list [--kind] [--created-by] [--label] [--limit]` | most recent first, with label, prompt and size |
| `splat manifest get ID` | full detail, children, and the lineage tree with each model's license |
| `splat manifest label ID\|- TEXT` | name manifests; piped records pass through |
| `splat manifest export ID DIR` | copy a manifest and its ancestors, each with a sidecar |
| `splat manifest rm ID [--cascade]` | refuses while other manifests derive from it, unless `--cascade` |
| `splat manifest clear [--kind] [--created-by] [-y]` | delete a selection |
| `splat manifest gc` | remove cache files no valid manifest owns |

### Models

| Command | |
|---|---|
| `splat models list` | every catalogued model, its stage, runtime, license and whether it's cached |
| `splat models info NAME` | license, source repo, input counts and notes |
| `splat models pull NAME` / `rm NAME` | download or remove weights |
| `splat models prune [--yes]` | report, or delete, cached weights no model uses |

| Stage | Model | Runtime | License |
|---|---|---|---|
| `diffuse` | `sdxl-turbo-mlx` | MLX | Stability AI non-commercial |
| `diffuse` | `sd21-coreml` | Core ML | OpenRAIL-M |
| `caption` | `fastvlm-0.5b` | torch | Apple ML Research (research only) |
| `embed` | `mobileclip2-s0` | torch | Apple ML Research (research only) |
| `segment` | `sam-mlx` | MLX | Apache-2.0 |
| `segment` | `sam2-coreml` | Core ML | Apache-2.0 |
| `depth` | `depth-pro` | torch | Apple ASCL |
| `depth` | `depth-anything-v2-coreml` | Core ML | Apache-2.0 (relative disparity) |
| `upscale` | `realesrgan-mlx` | MLX | BSD-3-Clause |
| `gaussian` | `sharp` | torch | Apple ML Research (research only) |
| `gaussian` | `mlx3d-capture` | MLX | MIT |

### Services and settings

**`splat mcp`** runs an MCP server over stdio. Tools: `diffuse`, `caption`, `embed`, `segment`, `depth`, `upscale`, `gaussian`, `render` (returns the image), `mesh`, `export`, `info`, `validate`, `models_list`, `models_info`, `models_pull`, `models_rm`, `manifest_list`, `manifest_get`, `manifest_label`, `manifest_delete`.

**`splat http [--host 127.0.0.1:8000]`** serves the same operations over HTTP, with no authentication, so bind it only to trusted interfaces. Routes: `POST /diffuse /caption /embed /segment /depth /upscale /gaussian /render /mesh /export /info /validate`, `GET /models`, `GET|DELETE /models/{name}`, `POST /models/{name}/pull`, `GET /version`, `GET /assets/{id}`, `GET /manifests`, `GET|DELETE /manifests/{id}`.

**`SPLAT_URL=http://host:8000`** makes the model-backed stages in the CLI, SDK and MCP server run on that `splat http` server. `render`, `mesh` and `export` always run locally. The client checks the server version once per process: a minor mismatch warns, a major one fails.

**`splat doctor`** checks the install: the splat and Python versions, whether torch, torchvision, mlx, coremltools and transformers import, the device (`mps` or `cpu`) and how many catalog models are cached. With `SPLAT_URL` set it also checks the server's version and an authenticated route, without running a model. It prints one line per check and exits non-zero if any fail, and a broken import does not stop the others.

**`splat env [--export]`** prints every `SPLAT_*` setting with its resolved value and source, or a `.env` template. Settings that aren't command options:

| Variable | |
|---|---|
| `SPLAT_MANIFEST_CACHE_DIR` | manifest cache, default `$XDG_CACHE_HOME/splat/manifests` |
| `SPLAT_MODEL_CACHE_DIR` | model cache, default `$XDG_CACHE_HOME/splat/models` |
| `SPLAT_URL` | remote `splat http` server |
| `SPLAT_BLENDER_BIN` | Blender executable |
| `SPLAT_RENDER_TIMEOUT` | render timeout in seconds, 0 disables it |
| `SPLAT_NO_MANIFEST` | write `-o` files without a `.manifest.json` sidecar, same as `splat --no-manifest`; the cache still records the manifest |
| `SPLAT_DEBUG` | show the full traceback on failure, same as `splat --debug` |

`splat --version` prints the version.

## Python SDK

The same operations, returning `Manifest` objects with `.as_image()`, `.as_text()`, `.as_array()` and `.as_gaussian_cloud()`. Stage options are keyword arguments named after their flags (`--focal-35mm` is `focal_35mm=`), and a test keeps them in step with the CLI:

```python
import splat

image = splat.diffuse("a small red boat").asset
cloud = splat.gaussian(image, score=True)[0]
print(cloud.metadata.quality)  # psnr, ssim, needle_ratio, ...
frame = splat.render(cloud, view=0)[0].as_image()
mesh = splat.mesh(cloud, format="glb")[0]
splat.export(cloud, "boat.spz")
```

Also: `caption`, `embed`, `segment`, `depth`, `upscale`, `info`, `validate`, `Manifest.load("<id>")`.

## Development

```sh
uv run pytest                                                    # unit and contract tests
SPLAT_INTEGRATION_TESTS=1 uv run pytest tests/test_model_smoke.py  # every model for real
```

The smoke tests run each catalogued model once and check its output against its kind's contract; a model added to a catalog fails the suite until it has a smoke case. They download weights on first use and take about 15 minutes on an M1 Pro.

## Architecture

Ports and adapters: `cli/`, `http/`, `mcp/` and `api.py` drive transport-neutral `handlers/`, which run cache-aware `application/` code against `ports/` and `domain/`; `adapters/` own the ML runtimes, file formats, cache and external processes, and `registry/` holds the lazy model catalog. See [docs/architecture.md](docs/architecture.md) and [docs/README.md](docs/README.md).
