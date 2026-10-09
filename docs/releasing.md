# Releasing

Releases are tag-driven. Any `vX.Y.Z` tag works: the release workflow bumps `pyproject.toml` and `uv.lock` to match and commits that to `main` as `github-actions[bot]`, so there is nothing to bump by hand.

```sh
git tag v0.1.0 && git push origin v0.1.0
```

| Workflow | Trigger | Does |
| --- | --- | --- |
| `test.yml` | push, pull request | `uv sync --frozen`, `ruff check`, `ruff format --check`, `pytest` (same as `mise run check`) |
| `build.yml` | push to `main` | `uv build`, uploads the sdist and wheel as an artifact |
| `release.yml` | `v*.*.*` tag | bumps the version on `main`, runs test, build and bundle from that commit, creates the GitHub Release with the sdist, wheel and bundle |
| `publish.yml` | `Release` succeeds, or manual dispatch with a tag | sends the bundle's version, url and sha256 to the tap as a `repository_dispatch` |

## The bundle

`scripts/bundle.sh` (`mise run bundle`) builds `dist/splat-<version>-macos-arm64.tar.gz` and its `.sha256` on Apple Silicon, using only uv:

1. `uv build --wheel` builds splat.
2. `uv python install 3.12` fetches the latest standalone CPython 3.12, which links only against macOS system libraries.
3. `uv export --frozen` pins every dependency from `uv.lock`, and `uv pip install` puts them and the wheel straight into that Python, with no venv.
4. `bin/splat` runs `../python/bin/python3.12 -I` relative to itself, so the folder works from any path.

The script then unpacks the tarball elsewhere and runs `splat --version` and `splat doctor` with an empty environment, so a bundle that depends on anything outside itself fails the release.

## Homebrew

The tap [`daanrongen/homebrew-splat`](https://github.com/daanrongen/homebrew-splat) owns `Formula/splat.rb`, which installs the bundle. On the `release` dispatch its workflow updates the formula's url and sha256, installs and tests it, and commits. This repo holds no formula code.

One-time setup:

1. Create a fine-grained personal access token with **Contents: read and write** on the tap repo only.
2. In this repo's settings, add the secret `HOMEBREW_TAP_TOKEN` (the token) and the variables `HOMEBREW_TAP_OWNER` (`daanrongen`) and `HOMEBREW_TAP_REPO` (`homebrew-splat`).
3. Pushing workflow files needs a GitHub token with the `workflow` scope: `gh auth refresh -s workflow`.

To resend a release to the tap by hand, run the **Publish Homebrew formula** workflow with the tag.
