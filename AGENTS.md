# AGENTS.md

`splat` is a local-first asset pipeline for Apple Silicon built as ports and adapters around a typed `Manifest`. [docs/architecture.md](docs/architecture.md) is the full reference; this page is the short list of rules to follow when changing it.

## Principles

1. Dependency injection over conditional logic.
2. Single state machines.
3. Domain driven design, hexagonal architecture, ports and adapters.
4. Atomic tests.
5. Conventional commits, no attribution.
6. Radical simplicity: the least code that solves the problem in front of you.

## Dependency injection over conditional logic

Choose behavior by wiring in a different implementation, not by branching on a name, kind or flag.

- Backends, formats, clients and sources are looked up in `registry/` (plain dicts and factories, no container, no decorator magic) and handed to the code that needs them.
- Adding a backend is one dict entry plus one adapter file. Never add an `if model == ...` to a handler or use case.
- Use cases take their ports in `__init__`; handlers get them from `registry.wiring`.
- `SPLAT_URL` swaps `LocalSplatClient` for `RemoteSplatClient` behind one `SplatClient` port. Commands never check where they run.

## Single state machines

Each lifecycle has exactly one explicit state machine. Do not spread it over booleans and flags.

- Model a state as one enum or value object with named transitions, owned by the domain.
- Invalid transitions fail with a domain error at the transition, not later at a use site.
- A manifest's kind and its `StageContract` are the pipeline's state: a stage accepts some kinds and produces one, and the contract is checked in one shared validator.
- A model-backed command follows one fixed order: parse input, resolve inputs, validate the contract, validate output paths, resolve the backend, load weights, run, store the manifest. Invalid input fails before any backend import or weight load.

## Domain, hexagon, ports and adapters

```text
cli/http/mcp/api -> handlers -> application -> ports/domain
registry -> ports/domain
adapters -> ports/domain
```

| Layer | Role |
|---|---|
| `domain/` | Data, invariants, contracts, domain errors. Imports nothing from the other layers. |
| `ports/` | `Protocol`s for backends, repositories, file IO and clients. |
| `application/` | Use cases and cache-aware orchestration. |
| `registry/` | Descriptors holding import paths, plus lazy factories. |
| `adapters/` | Runtimes (mlx, coreml, torch), formats, cache, HTTP client, external processes. |
| `handlers/` | Transport-neutral request handlers. |
| `cli/`, `http/`, `mcp/`, `api.py` | Driving adapters. |

- Driving adapters call handlers, never runtimes directly.
- Adapters are interchangeable and agnostic: a backend knows its runtime and its port, nothing about other stages or transports. They translate runtime failures into `SplatDomainError` subclasses.
- Cold start is a constraint. `--help`, `manifest list/get`, `models list/info` must not import `torch`, `mlx`, `transformers`, `coremltools`, `trimesh`, `cv2`, `scipy` or `huggingface_hub`. Import heavy runtimes inside the function that needs them, and keep catalogs as import-path strings.
- CLI, HTTP, MCP and SDK expose the same operations. Drift is a bug unless documented as deliberate.
- Before cataloging a backend: complete descriptor metadata, explicit license, wrong inputs fail before weights load, one opt-in integration test on a tiny fixture, and metadata that records effective params.

## Atomic tests

- One behavior per test, named as a sentence (`test_find_does_not_delete`), with one reason to fail.
- `tests/` mirrors `src/splat`, so each layer is tested at its own level.
- Test use cases against small hand-written `Fake*` ports, and adapters with the runtime mocked (`pytest-mock`). No test depends on another.
- The cache is isolated per test (`tmp_path` via the autouse fixture). Never touch the real cache, network or model weights in a unit test.
- Real model runs are opt-in: `@pytest.mark.integration` with `SPLAT_INTEGRATION_TESTS=1`.
- A fix lands with the test that failed before it.

## Conventional commits, no attribution

- Format: `type(scope): message`, imperative and lowercase, for example `fix(segment): handle empty mask batch`. Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `chore`.
- No attribution anywhere: no `Co-Authored-By` trailer, and no mention of Claude or any agent or assistant in a commit message or a PR description.
- One concern per commit. The body is a sentence or two, only when the why is not obvious.
- Branches are `<issue#>-<slug>` and PRs are merged with a merge commit. Releases are cut by the release workflow (`chore(release): vX.Y.Z`), never by hand.

## Workflow

- `mise run check` runs `ruff check`, `ruff format --check` and `pytest`, the same as CI. lefthook runs ruff on commit and pytest on push.
- Ruff: line length 100, `_vendor/` excluded.

## Style

- Comments are one short line and only explain a non-obvious why.
- Prefer short, isolated functions. Avoid indirection, redundancy and building for hypothetical requirements. When in doubt, cut.
- In prose, write each paragraph on one line and use `~` for the home directory.
