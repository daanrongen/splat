# Roadmap

The roadmap is organized around the converter graph, not around model names.
Models change quickly; the stable question is what typed asset transformation
the project should support.

## Core Graph

Current graph:

```text
text -> image
image -> caption
image -> embedding
image -> sticker
image -> depth_map
image -> gaussian_cloud
gaussian_cloud -> image
gaussian_cloud -> gaussian_cloud
gaussian_cloud -> shape_3d
depth_map + image -> shape_3d
```

Target graph:

```text
text -> image -> views -> gaussian_cloud -> render -> image
image -> views -> gaussian_cloud
images -> poses -> gaussian_cloud
gaussian_cloud + style image -> gaussian_cloud
gaussian_cloud + sticker mask -> gaussian_cloud
gaussian_cloud -> web delivery bundle
```

## Near Term

### Cold-start and backend isolation

Cheap commands should remain cheap even as more backends are added.

Done:

- lazy CLI root command resolution;
- lazy top-level package SDK exports;
- catalog descriptors store backend import paths;
- manifest listing reads metadata only;
- import regression tests for `manifest` and `models`.

Next:

- split `application.pipeline` into per-stage modules;
- add one operation inventory for CLI/HTTP/MCP/SDK parity;
- add explicit job runner ports for subprocess-isolated work.

### Output quality

Priority fixes:

- strip FastVLM chat-template scaffolding and record caption truncation;
- rank segmentation output so background masks do not win by default;
- type depth semantics as metric depth versus relative disparity;
- preserve camera/provenance metadata through deterministic Gaussian tools;
- ensure render defaults are visually useful and reproducible.

## New Converter Stages

### `views`: one image to N consistent views

Tracked by #95.

Purpose: turn one image into multiple views of the same object or scene so
multi-view reconstruction and evaluation become possible.

Candidate backends:

| Backend family | Why it fits |
|---|---|
| Zero123++ | Simple image-to-six-views shape and permissive code license. |
| MV-Adapter | Reuses existing diffusion base weights instead of adding a parallel stack. |
| SV3D | High-quality orbit generation, but non-commercial license class. |
| Era3D | Higher-resolution multi-view output, heavier research path. |

Contract:

```text
1 colorlike manifest -> N image manifests
```

Each output should carry known camera pose metadata when the backend provides a
fixed view schedule.

### `poses`: learned pose and geometry front-end

Tracked by #94.

Purpose: replace brittle classical SfM for real multi-view captures.

Preferred direction:

- add VGGT-style pose/intrinsics estimation;
- write a COLMAP-compatible sparse model;
- feed that into `mlx3d-capture` with `poses=existing`;
- report registered/usable view count loudly.

This should be an internal backend option first unless a standalone pose asset
becomes useful to users.

### Feed-forward multi-view Gaussian reconstruction

Tracked by #61.

Purpose: produce Gaussians from multiple unposed or sparsely posed images without
a minutes-long optimizer loop.

Preferred direction: evaluate AnySplat before reviving MVSplat. MVSplat needs
calibrated poses; AnySplat is closer to the shape `splat` can provide.

Contract:

```text
N colorlike manifests -> gaussian_cloud manifest
```

## Training And Refinement

Tracked by #62.

The unresolved design question is what `train` means:

- raw dataset directory to optimized `gaussian_cloud`; or
- existing `gaussian_cloud` plus extra views to refined `gaussian_cloud`.

The second shape composes better with the manifest graph and future stylization,
but the first is familiar to 3DGS users. Pick one before adding a backend.

Candidate implementation styles:

- external process adapter for mature native tools such as Brush;
- in-process MLX adapter if quality and performance are proven;
- avoid copyleft runtime dependencies unless explicitly accepted.

## Styling And Editing

Tracked by #67 and #68.

Whole-cloud style transfer:

```text
gaussian_cloud + style image -> gaussian_cloud
```

Object-level restyle:

```text
gaussian_cloud + sticker mask + camera pose -> gaussian_cloud
```

Both require multi-slot `StageContract` support and typed camera/pose metadata.
Object-level editing additionally needs a projection from 2D mask pixels to a
3D Gaussian subset.

## Backend Acceptance Criteria

Before a backend is cataloged:

- descriptor metadata is complete enough for `models list/info`;
- license metadata is explicit;
- import is lazy;
- invalid inputs fail before model loading;
- one opt-in integration test runs a tiny inference;
- output metadata records dimensions, semantics, effective parameters, and
  provenance;
- runtime warnings are scoped to the command that uses the runtime.
