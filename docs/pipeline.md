# Walkthrough

One image goes through every stage of `splat`: a diffused kettle becomes a caption, an embedding, stickers, depth maps, a sharper image, a Gaussian splat, renders, meshes and exports. Every number and picture below comes from running these commands in order, on an M-series Mac with the weights already downloaded (first use of a model downloads it, which takes minutes).

Each command stores its output as a manifest in the cache and prints one NDJSON record when piped. `-o` also writes the file, with a `.manifest.json` sidecar so the lineage travels with it. Later commands take a file path, an `@<manifest-id>`, or `-` for piped records.

## 1. Diffuse

```sh
splat diffuse "a single stainless steel kettle with a black handle, centered, plain mid-gray studio backdrop, soft even lighting, 3/4 view" --seed 8 -o kettle.png
```

![The diffused kettle](images/01-diffuse.png)

The default model is `sd15-coreml`, a 6-bit palettized SD 1.5 that renders 512 px in about 30 s here, model load included, and supports image-to-image (`sdxl-turbo-mlx` is the alternative, with a non-commercial license that `splat` prints on every run). A plain backdrop and a single centred subject matter: every later stage does better on an image like this, and the rest of this page keeps using this one file.

## 2. Caption and embed

```sh
splat caption kettle.png -o caption.txt
splat embed kettle.png -o kettle.npy
```

The caption is `A shiny silver tea kettle with a black handle and lid.` (`fastvlm-0.5b`, a `caption` manifest of 54 characters). The embedding is a normalized 512 dimensional float32 vector from `mobileclip2-s0`, stored as `.npy`. `embed --text "..."` produces the same kind of vector for a text query, and a dot product compares two of them.

## 3. Segment

```sh
splat segment kettle.png --max-stickers 8 -o stickers/
splat segment kettle.png --model sam-mlx --max-stickers 8 -o stickers-mlx/
```

![Automatic stickers from sam-mlx (top) and sam2-coreml (bottom)](images/02-segment-auto.png)

Automatic segmentation fans out into one RGBA `sticker` per mask, largest first. `sam2-coreml` is the default: it is about twice as fast as `sam-mlx` and a fifth of the size. In the picture it (bottom) starts with the frame sized mask, shown as the backdrop with the kettle cut out, then finds the kettle, the handle, the body, the spout and the lid. `sam-mlx` (top) finds the handle and the backdrop inside it, then slices of the body's highlights and shadows. `--drop-background` removes the masks that cover most of the frame or span part of its edge.

To say which object you mean, prompt the model instead:

```sh
splat segment kettle.png --point 92,330 -o spout.png
splat segment kettle.png --box 55,100,410,495 -o kettle-box.png
splat segment kettle.png --foreground -o kettle-cutout.png
```

![A point on the spout, a box around the kettle, and --foreground](images/03-segment-prompts.png)

`--point x,y` (and `--not-point x,y` for background, both repeatable) and `--box x0,y0,x1,y1` take pixel coordinates in the source image and return one sticker. `--foreground` needs no prompt: it finds the box around the masks that are not background and box-prompts the model with it, which gives one cutout of the kettle (347 by 409 px). For the box prompt, `sam-mlx` returns `(58, 102, 345, 390)` and `sam2-coreml` `(57, 103, 347, 400)` as `x, y, width, height`, within ten pixels of each other. `sam2-coreml` takes a box or up to two points, because its converted prompt encoder takes exactly two; use `--model sam-mlx` for more points.

A sticker is a tight crop. Its `bbox` in the manifest places it back in the source image, which is how a sticker works as a mask for the next stages.

## 4. Depth

```sh
splat depth kettle.png -o depth-pro-2.png
splat depth kettle.png --model depth-anything-v2-coreml -o depth-anything.png
```

![The image, depth-pro-2, and depth-anything-v2-coreml (near is bright)](images/04-depth.png)

`depth-pro-2` is the default. It is metric: the `depth_map` is in metres and records the focal length (971 px, a 29.5 degree field of view). Its weights have a research-only license, and `splat models pull depth-pro-2` fetches them. `depth-anything-v2-coreml` is faster but only relative disparity, with no focal length. Both are cached losslessly as `.npy`, and `-o` writes a viewable PNG of them. Only the metric one can become geometry, and `mesh` says so if you hand it the other:

```text
error: heightfield needs metric depth in metres, but 9e92f96c07db3f60 is relative disparity (depth:depth-anything-v2-coreml); use a metric model such as depth-pro-2.
```

## 5. Upscale

```sh
splat upscale kettle.png -o kettle-x4.png
```

![A crop of the lid, bicubic resize of the original (left) and realesrgan-mlx (right)](images/05-upscale.png)

`realesrgan-mlx` makes a 2048 px `image` out of the 512 px one, and the result is an ordinary image manifest that any stage can take.

## 6. Gaussian splat

```sh
splat gaussian kettle.png --score -o kettle.ply
```

`sharp` reconstructs a Gaussian splat from the single image: 1,179,648 Gaussians, an 80 MB `.ply` in the OpenGL convention (+Y up), SH degree 0, in metres. It takes about 50 s here, model load included. `--score` renders source view 0 with Blender and records the comparison against the photo in the cloud's `quality`: PSNR 38.09 and SSIM 0.9719, plus floater statistics. `splat manifest get <id>` prints them.

A single photo has a backdrop, and SHARP reconstructs that too, as a wall behind the subject. Hand `gaussian` the cutout from step 3 and it keeps only the Gaussians that project inside it:

```sh
splat gaussian kettle.png --mask kettle-cutout.png -o kettle-masked.ply
```

That is 335,586 Gaussians, a 23 MB file. Gaussians that project inside the mask but sit more than half a subject size behind its nearest surface are dropped too, since they belong to the wall in line with the subject. The masked cloud is a cached child of the full one, so changing the mask does not repeat the reconstruction.

## 7. Render

```sh
splat render kettle.ply --width 512 --height 512 -o full.png
splat render kettle-masked.ply --width 512 --height 512 -o view0.png --view 0
splat render kettle-masked.ply --width 512 --height 512 -o orbit.png --azimuth 25 --elevation 8
splat render kettle-masked.ply --width 512 --height 512 -o side.png --azimuth 45 --elevation 8
```

![The full cloud, the masked cloud from the capture camera, and the masked cloud orbited by 25 and 45 degrees](images/06-gaussian.png)

`render` runs Blender (set `SPLAT_BLENDER_BIN` if it is not on `PATH`) and produces an `image` manifest. The default frame is 1280x720, which crops the top of a square capture, so these use `--width 512 --height 512`. The first cell is the full cloud with the backdrop wall behind the kettle. `--view 0` renders from the source camera and reproduces the photo. Orbiting shows the limit of one image: SHARP only knows the visible side, so the kettle gets thin as you turn away. `--zoom` scales the framing independent of the cloud's units (`1` fits the subject, `2` is twice as close), `--distance` is absolute in the cloud's units, and `--fov` and `--look-at x,y,z` adjust the rest.

## 8. Mesh

```sh
splat mesh @<depth-id> -o relief.glb
splat mesh kettle-masked.ply -o kettle.glb
```

![The heightfield from depth-pro-2 (left) and the isosurface of the masked cloud (right)](images/07-mesh.png)

`mesh` has two paths. A metric depth map becomes a textured heightfield, a relief of everything in the photo (262,144 vertices, 512,782 faces, 11.6 MB). A Gaussian cloud becomes an isosurface of its density field with per-vertex colours (68,904 vertices, 137,612 faces, 2.8 MB), which is the kettle alone because the backdrop was masked out. Both are `shape_3d` manifests with typed counts, written as `.glb`, `.obj`, `.ply` or `.gltf` by the extension.

## 9. Export

```sh
splat export kettle-masked.ply -o kettle.spz
splat export kettle-masked.ply -o kettle.web.splat --profile web-delivery
```

`export` writes any manifest in the format of the extension, with a sidecar. For the masked cloud: `.ply` 22.8 MB, `.spz` 3.3 MB, `.web.splat` 10.7 MB (335,346 points after compression). `splat validate kettle-masked.ply --strict` checks a cloud's invariants and `splat info` prints its count, SH degree, bounds and cameras.

## Lineage

Every output remembers its parents, so `splat manifest get <id>` on the mesh shows how it was made, down to the prompt, with each model's license:

```text
44797c671f155897  shape_3d  mesh:isosurface
└── 5217881b851af4c9  gaussian_cloud  mask
    ├── e5b2e55cd308fdc6  gaussian_cloud  gaussian:sharp  Apple-ML-Research
    │   └── fe498117241c9aec  image  diffuse:sd15-coreml  OpenRAIL-M
    └── 0c9722fff85ac5a1-000  sticker  segment:sam2-coreml  Apache-2.0
        └── fe498117241c9aec  image  diffuse:sd15-coreml  OpenRAIL-M
```

Re-running any command with the same inputs is a cache hit. `splat manifest list`, `get`, `export`, `rm` and `gc` manage the cache.

## Not in this flow

`gaussian --model mlx3d-capture` is the multi-view path. It needs three or more genuinely overlapping photos of one scene, so it has no place in a single-image walkthrough. Several independently generated images do not satisfy that.

## Debug loop

1. Find ids with `splat manifest list`.
2. Read parameters, model, license and lineage with `splat manifest get <id>`.
3. Check a cloud with `splat info` and `splat validate`.
4. Look at it with `splat render`.

When output looks wrong, read the metadata before rerunning a heavy stage: parent ids, effective parameters, depth units, camera count and coordinate convention explain most surprises.
