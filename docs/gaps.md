# Active gaps

This file summarizes the active architecture and quality gaps. GitHub issues are
the source of truth for task tracking; this page keeps the project-level shape
easy to see.

## Tier 0: Architecture And Cold Start

### Import isolation for cheap commands

Status: implemented for CLI root command resolution, `manifest`, and `models`
metadata paths.

Invariant:

```text
splat manifest list
splat manifest get <id>
splat models list
splat models info <name>
```

must not import `torch`, `coremltools`, `mlx`, `transformers`, `trimesh`, `cv2`,
`open3d`, or `splat.application.pipeline`.

Regression coverage lives in CLI tests.

### Backend loading descriptors

Status: implemented for model catalogs.

Catalog entries are pure metadata until execution. Concrete backend classes are
resolved from import paths only after input validation.

Remaining work:

- move more command metadata into generated docs;
- add a single operation inventory for transport parity;
- split `application.pipeline` into per-stage orchestration modules.

## Tier 1: Correctness Bugs

| Issue | Gap | Why it matters |
|---|---|---|
| #93 | FastVLM captions include chat-template scaffolding and truncation. | Caption quality pollutes downstream embeddings. |
| #92 | Fan-out markers are typed as `sticker`; backend exceptions leak. | Violates manifest contracts and user-facing error boundaries. |
| #91 | `depth_map` cannot distinguish metric depth from relative disparity. | Future geometry consumers can silently invert or mis-scale results. |
| #85 | `tools extract.surface` can hang due to Open3D/Torch OpenMP runtime clashes. | Needs subprocess isolation, timeout, and progress. |
| #84 | `sam2-coreml` crashes on first inference. | `models list` says cached, but the backend is not usable. |
| #83 | Render camera controls and turntable output need completion. | Presentation quality depends on predictable viewpoint control. |

## Tier 2: Pipeline Quality

| Issue | Gap | Direction |
|---|---|---|
| #95 | Need a `views` stage for multi-view-consistent generated views. | Add image-to-N-views model-backed stage; do not pretend prompted views are real views. |
| #94 | Classical SfM drops too many views. | Add VGGT-style learned pose/geometry front-end for `mlx3d-capture`. |
| #61 | Feed-forward multi-view backend should target AnySplat over the old MVSplat stub path. | Prefer backends that accept the data `splat` can actually provide. |
| #62 | `train` is a stub and overlaps conceptually with optimization-based `gaussian`. | Decide whether `train` is dataset optimization or refinement. |
| #68 | Object-level Gaussian restyle needs mask-to-cloud projection. | Requires multi-slot contracts and camera pose handling. |
| #67 | Reference-image Gaussian style transfer needs feature representation research. | Decide whether `GaussianCloud` needs extra learned features. |

## Tier 3: Tooling And Docs

| Issue | Gap | Direction |
|---|---|---|
| #89 | README drifted from code. | Keep README current, principle-first, and generated where possible. |
| #88 | Tests cover plumbing more than real payloads. | Add opt-in backend integration tests. |
| #70 | License lineage for candidate splat dependencies needs tracking. | Preserve explicit `ModelLicense` metadata and document gated/non-commercial models. |
| #55 | Stage contracts support only one input slot today. | Add named multi-slot validation when the next composite stage lands. |
| #34 | Bytecode/cache artifacts accumulate during local checks. | Keep repository hygiene lightweight and explicit. |
| #4 | CI/release workflows are missing. | Add local-check-equivalent CI before release automation. |

## Quality Principles

- Wrong inputs fail before backend import or weight loading.
- Runtime warnings appear only when the runtime is actually used.
- Metadata records effective parameters and semantics, not raw unresolved flags.
- Deterministic tools preserve camera/provenance metadata unless documented.
- Every cataloged backend needs one opt-in integration test.
- Docs describe shipped behavior; future work lives in roadmap or issues.
