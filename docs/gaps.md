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

## Tier 1: Correctness

| Issue | Gap | Why it matters |
|---|---|---|
| #156 | `to_convention` flips positions and rotations but not higher-order SH. | Colmap-frame clouds with `sh_degree > 0` get mirrored view-dependent colour. |
| #130 | Manifest kinds have no enforced canonical invariants. | Backends can hand off output another stage must patch before using. |

## Tier 2: Reconstruction Quality

| Issue | Gap | Direction |
|---|---|---|
| #135 | SHARP has no intrinsics for images without EXIF and no small-input upscaling. | Estimate focal length (MoGe-2), judge pre-processing against #138. |
| #136 | `train` is a stub; feed-forward clouds are never refined. | msplat-backed `splat refine` to SH3. |
| #137 | Few-view input relies on classical SfM. | Depth Anything 3 multi-view backend. |
| #94 | SfM registers too few views. | Learned pose front-end (DA3, MapAnything, VGGT). |
| #95 | Single-image clouds can't show unseen sides. | Generative orbit views via video diffusion. |
| #138 | No quality signal to compare backends. | Held-out-view metrics in cloud metadata. |

## Tier 3: Backends, Formats And Tests

| Issue | Gap | Direction |
|---|---|---|
| #139, #140, #141, #142 | Stage catalogs lag the state of the art. | MoGe-2/DA3 depth, SAM 3, mflux diffuse and upscale, Qwen3-VL and SigLIP2. |
| #143 | No OpenUSD export. | `ParticleField3DGaussianSplat` writer behind `-o .usd`. |
| #88 | Only some catalogued models have a real-inference test. | One opt-in smoke test per model, asserting canonical invariants. |

## Quality Principles

- Wrong inputs fail before backend import or weight loading.
- Runtime warnings appear only when the runtime is actually used.
- Metadata records effective parameters and semantics, not raw unresolved flags.
- Exports keep provenance: every `-o` file gets a `.manifest.json` sidecar.
- Every cataloged backend needs one opt-in integration test.
- Docs describe shipped behavior; future work lives in roadmap or issues.
