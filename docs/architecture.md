# Architecture

`splat` is a ports-and-adapters pipeline. The architecture is built around two
hub types:

- `Manifest`: the typed, cached asset that flows between pipeline stages.
- `GaussianCloud`: the canonical in-memory splat representation used by format,
  compression, cleanup, surface extraction, and rendering code.

The goal is not just neat folders. The goal is a converter graph where new
visual capabilities can be added without every stage knowing about every other
stage.

## Dependency Rule

```text
cli/http/mcp/sdk -> handlers -> application -> ports/domain
registry -> ports/domain
adapters -> ports/domain
```

| Layer | Role |
|---|---|
| `domain/` | Core data, invariants, metadata, contracts, and domain errors. |
| `ports/` | Protocols for backends, repositories, file IO, clients, and services. |
| `application/` | Use cases and cache-aware orchestration. |
| `registry/` | Lightweight backend/format descriptors and lazy factories. |
| `adapters/` | ML runtimes, file formats, cache, HTTP client, and external processes. |
| `handlers/` | Transport-neutral request handlers. |
| `cli/`, `http/`, `mcp/`, `api.py` | Driving adapters. |

Driving adapters do not call ML/runtime adapters directly. They call handlers.
Handlers validate request shapes and obtain ports from `registry.wiring`.
Adapters translate runtime-specific failures into domain errors.

## Import Boundary

Cold-start is a first-class architectural constraint.

Cheap commands must not import heavy runtime modules:

- `splat manifest list`
- `splat manifest get <id>`
- `splat models list`
- `splat models info <name>`
- root command help
- deterministic commands until their implementation actually needs a format or
  backend-specific dependency

The forbidden imports for those paths are:

```text
torch
coremltools
mlx
transformers
open3d
trimesh
cv2
splat.application.pipeline
```

The root CLI uses a lazy Typer group. It lists commands from metadata and imports
the real command module only when that command is resolved. The top-level
`splat` package also exposes SDK functions lazily so importing `splat.cli.main`
does not import `splat.api`.

Backend catalogs store import paths, not class objects:

```python
backend = "splat.adapters.gaussian.sharp:SharpBackend"
```

Factories resolve the class only when a model-backed command is about to run.
Model administration uses descriptor metadata and cache state, so listing models
does not import backend adapters.

## Backend Lifecycle

A model-backed command follows this order:

1. Parse transport-specific input.
2. Resolve input paths or manifest ids.
3. Validate `StageContract` from lightweight metadata.
4. Validate output paths before expensive compute.
5. Resolve model source and backend adapter.
6. Pull/load weights only if needed.
7. Run the use case.
8. Store a manifest with effective params, metadata, parent ids, size, hash, and
   creation time.

The important rule is that invalid input fails before backend import or model
weight loading whenever the required information is available from descriptors.

## Manifests And Cache

The filesystem cache stores two files per manifest:

```text
<id>.<ext>
<id>.meta.json
```

Normal manifest listing reads metadata only. Payload reads and hash checks belong
in explicit verification/repair operations, not in table rendering.

There are two id schemes in one namespace:

- Pipeline output ids are invocation hashes: stage, model, params, and parents.
- External file ids are content hashes: the bytes themselves.

Both schemes are intentional, but docs and code must be explicit about which one
is being used. Provenance-sensitive flows should prefer `@<manifest-id>` or
piped NDJSON so parentage is preserved.

## Contracts

Every stage declares a `StageContract` over one or more requirements:

- accepted kinds
- accepted capability tags
- min/max input count
- result kind
- user-facing hint for wrong-kind errors

`ManifestKind` stays flat. Tags express capabilities such as `colorlike`,
`raster`, `text`, `vector`, `mesh_3d`, and `splat_3d`.

The next contract evolution is multi-slot input validation. That is needed for
stages such as:

- `gaussian_cloud + sticker mask -> object restyle`
- `gaussian_cloud + camera pose -> render from known camera`
- `image + pose bundle -> posed reconstruction`
- `depth_map + source image -> shape_3d`

Until multi-slot validation exists, avoid hiding second inputs inside free-form
`params`.

## GaussianCloud

`GaussianCloud` is the canonical splat aggregate:

- means
- scales
- rotations
- opacities
- SH DC color
- optional higher-order SH
- activation conventions
- metadata

File readers should normalize only representation details, not silently change
geometry. Coordinate-system conversions must be explicit and reflected in
metadata. Camera and pose data should use domain value objects rather than raw
nested lists.

## Transport Parity

The CLI, HTTP, MCP, and SDK surfaces should be derived from the same operation
inventory where possible. Drift is a bug.

Current intentional differences:

- `render` executes locally because Blender is a local external process.
- `manifest` operates on the local cache by default.
- HTTP has no authentication in v1 and should be treated as trusted-LAN only.

Any other missing operation in one transport should either be added or documented
as deliberately unsupported.

## Job Delegation

Long or fragile operations should sit behind an explicit job boundary.

Recommended shape:

- `JobRunner` port with local synchronous execution as the default.
- Subprocess runner for dependency-isolated jobs.
- Timeout and cancellation support for every external process.
- Progress callback contract shared by CLI, HTTP, MCP, and SDK.
- Adapter-level exception translation into domain errors.

Use subprocess isolation when runtimes conflict in one process. The known case is
Open3D Poisson surface extraction interacting badly with Torch/OpenMP imports.

## Quality Bar

Before adding or cataloging a backend:

- descriptor metadata must be complete enough for `models list/info`;
- wrong inputs must fail before loading weights;
- one opt-in integration test must run the backend on a tiny fixture;
- license metadata must be explicit;
- runtime warnings should appear only on commands that actually use that runtime;
- output metadata must record effective parameters, not raw unresolved CLI input.
