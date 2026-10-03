# Releasing

Releases are tag-driven. Any `vX.Y.Z` tag works: the release workflow bumps `pyproject.toml` and `uv.lock` to match and commits that to `main` as `github-actions[bot]`, so there is nothing to bump by hand.

```sh
git tag v0.1.0 && git push origin v0.1.0
```

| Workflow | Trigger | Does |
| --- | --- | --- |
| `test.yml` | push, pull request | `uv sync --frozen`, `ruff check`, `ruff format --check`, `pytest` (same as `mise run check`) |
| `build.yml` | push to `main` | `uv build`, uploads the sdist and wheel as an artifact |
| `release.yml` | `v*.*.*` tag | bumps the version on `main`, runs test and build from that commit, creates the GitHub Release with the sdist, wheel and a source archive |
| `publish.yml` | `Release` succeeds, or manual dispatch with a tag | renders `Formula/splat.rb` and pushes it to the tap |

The formula installs the release's source archive with `uv sync --frozen` into a private venv, so it needs Apple Silicon macOS (mlx, CoreML).

## One-time setup

1. Create the tap repository `daanrongen/homebrew-splat` (public, with a `Formula/` directory or an empty initial commit).
2. Create a fine-grained personal access token with **Contents: read and write** on that repo only.
3. In this repo's settings, add the secret `HOMEBREW_TAP_TOKEN` (the token) and the variables `HOMEBREW_TAP_OWNER` (`daanrongen`) and `HOMEBREW_TAP_REPO` (`homebrew-splat`).
4. Pushing workflow files needs a GitHub token with the `workflow` scope: `gh auth refresh -s workflow`.

Then users install with:

```sh
brew tap daanrongen/splat
brew install splat
```

To republish a release by hand, run the **Publish Homebrew formula** workflow with the tag.
