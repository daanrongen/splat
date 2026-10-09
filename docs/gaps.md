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
`scipy`, `huggingface_hub`, or `splat.application.pipeline`.

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
| #130 | Contracts are enforced at `put`, but `image` has no intrinsics and `depth-anything-v2-coreml` is still relative disparity. | Every image and depth map should carry a camera so 3D stages need no guesses. |
| #163 | mlx3d-capture training is aborted by the Metal GPU watchdog on an M1 Pro. | The only multi-view backend fails its smoke test there. |

## Tier 2: Reconstruction Quality

| Issue | Gap | Direction |
|---|---|---|
| #136 | `train` is a stub; feed-forward clouds are never refined. | msplat-backed `splat refine` to SH3. |
| #137 | Few-view input relies on classical SfM. | Depth Anything 3 multi-view backend. |
| #94 | SfM registers too few views. | Learned pose front-end (DA3, MapAnything, VGGT). |
| #95 | Single-image clouds can't show unseen sides. | Generative orbit views via video diffusion. |
| #138 | Clouds record stats and, with `--score`, PSNR/SSIM against a training view. | Add LPIPS and a true held-out view. |

## Tier 3: Backends, Formats And Tests

| Issue | Gap | Direction |
|---|---|---|
| #139, #140, #141, #142 | Stage catalogs lag the state of the art. | MoGe-2/DA3 depth, SAM 3, mflux diffuse and upscale, Qwen3-VL and SigLIP2. |
| #143 | No OpenUSD export. | `ParticleField3DGaussianSplat` writer behind `-o .usd`. |

## Quality Principles

- Wrong inputs fail before backend import or weight loading.
- Runtime warnings appear only when the runtime is actually used.
- Metadata records effective parameters and semantics, not raw unresolved flags.
- Exports keep provenance: every `-o` file gets a `.manifest.json` sidecar.
- Every cataloged backend needs one opt-in integration test.
- Docs describe shipped behavior; future work lives in roadmap or issues.
