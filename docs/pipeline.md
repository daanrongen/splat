# Pipeline guide

This guide shows how the shipped stages compose. It focuses on command shape,
manifest flow, and output expectations rather than research notes.

## The Universal Flow

Every manifest-producing command stores a cached asset and prints an NDJSON
manifest when stdout is piped.

```sh
splat diffuse "a small red toy robot, studio lighting" \
  | splat caption -
```

Every stage can also consume a path or an existing manifest id:

```sh
splat caption photo.png
splat caption @fe3af96cea08ba36
```

## Text To Image

```sh
splat diffuse "beautiful romantic Turner painting of a landscape" \
  --model sdxl-turbo-mlx \
  --seed 42 \
  -o landscape.png
```

Output:

- kind: `image`
- storage: `.png`
- metadata: output dimensions, optional source dimensions for image-to-image use
- parent ids: empty for pure text-to-image, one image parent for image-to-image

`sdxl-turbo-mlx` is the fast default. `sd21-coreml` exists as a CoreML backend
with different quality/performance tradeoffs.

## Image To Caption Or Embedding

```sh
splat caption landscape.png -o caption.txt
splat embed landscape.png -o landscape.embedding.npy
splat embed --text "romantic landscape painting" -o text.embedding.npy
```

Captions are text manifests. Embeddings are normalized `.npy` vector manifests.
Embedding metadata is the reference quality bar for other stages: it records
input type, dtype, shape, dimension, normalization, model, and text hashes where
applicable.

## Image To Stickers

```sh
splat segment landscape.png --model sam-mlx --max-stickers 5 -o stickers/
```

Output:

- one `sticker` manifest per cutout;
- RGBA PNG payloads;
- bounding box, score, area, width, and height metadata.

Segmentation is a fan-out stage. Its cache entry records children so reruns can
short-circuit without recomputing masks.

## Image To Depth To Mesh

```sh
splat depth landscape.png --model depth-pro \
  | splat mesh - -o relief.glb
```

`depth-pro` is the metric-depth path and is the correct backend for geometry.
Relative disparity backends are useful for visual depth previews but should not
be consumed as metric depth.

Output:

- `depth` produces a lossless `.npy` `depth_map` manifest;
- `mesh` (heightfield) resolves the source image through provenance and
  writes a `shape_3d` manifest.

## Image To Gaussian Splat

Single image:

```sh
splat gaussian landscape.png --model sharp -o landscape.ply
```

Piped from generation:

```sh
splat diffuse "a small ceramic fox on a table" \
  | splat gaussian - --model sharp -o fox.ply
```

`sharp` reconstructs visible 3D structure from one image. Its output is a
`gaussian_cloud` manifest stored as `.ply`.

Multi-view capture:

```sh
splat gaussian frame-*.png --model mlx3d-capture --quality balanced -o scene.ply
```

`mlx3d-capture` needs genuinely overlapping, multi-view-consistent photos or
video frames of one physical scene. Several independently generated prompts do
not satisfy that requirement.

## Gaussian Files And Meshes

```sh
splat gaussian frame-*.png --model mlx3d-capture --declutter -o scene.ply
splat export scene.ply -o scene.spz
splat export scene.ply -o scene.web.splat --profile web-delivery
splat mesh scene.ply -o scene.glb
```

| Command | Input | Output | Purpose |
|---|---|---|---|
| `export` | any manifest | file + sidecar | Write in the format of the extension; `--profile` compresses clouds. |
| `gaussian --declutter` | images | `gaussian_cloud` | Remove isolated floater Gaussians after reconstruction. |
| `mesh` | `gaussian_cloud` | `shape_3d` | Poisson surface extraction, in its own worker process. |

## Render Back To Image

```sh
splat render scene.ply -o scene.png \
  --width 1280 \
  --height 720 \
  --samples 32 \
  --background black
```

Render output is an `image` manifest, so it can be captioned, embedded, upscaled,
or segmented like any other image.

Camera controls:

```sh
splat render scene.ply -o side.png --azimuth 90 --elevation 10
splat render scene.ply -o close.png --distance 2.5 --fov 45
```

## Inspect And Validate

```sh
splat info scene.ply
splat validate scene.ply --strict
splat manifest list --kind gaussian_cloud
splat manifest get <id>
```

Inspection commands should be cheap. Manifest listing reads metadata only and
does not import model backends.

## Recommended Debug Loop

1. Generate or register assets.
2. Use `splat manifest list` to find ids.
3. Use `splat manifest get <id>` to inspect provenance and metadata.
4. Use `splat info` for Gaussian geometry statistics.
5. Use `splat render` to visually inspect a cloud.
6. Use `splat export` to package the cloud, or `splat mesh` for a surface.

When output looks wrong, inspect metadata before rerunning heavy stages. Parent
ids, model name, effective parameters, depth semantics, camera count, coordinate
convention, and license often explain the result.
