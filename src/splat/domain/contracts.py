"""Declarative per-stage input contracts, checked by one shared validator
instead of ad hoc `if manifest.kind not in (...)` blocks scattered across
handlers. A stage names what it needs (`Requirement`) instead of every
handler re-deriving it from a hand-written kind tuple.
"""

from dataclasses import dataclass

from splat.domain.errors import WrongManifestCount, WrongManifestKind
from splat.domain.manifest import KIND_TAGS, Manifest, ManifestKind

_TAG_LABELS = {"colorlike": "image or sticker"}


@dataclass(frozen=True)
class Requirement:
    """One named input slot a stage needs.

    `any_of_kinds`/`any_of_tags` are OR'd together — a manifest satisfies the
    requirement if its kind is in `any_of_kinds` OR any of its `KIND_TAGS` is
    in `any_of_tags`.
    """

    name: str
    any_of_kinds: frozenset[ManifestKind] = frozenset()
    any_of_tags: frozenset[str] = frozenset()
    min_count: int = 1
    max_count: int | None = 1
    hint: str | None = None  # "why not" / "do this instead" clause for error messages


@dataclass(frozen=True)
class StageContract:
    stage: str
    inputs: tuple[Requirement, ...]  # length 1 today; >1 for future composite stages
    produces: ManifestKind


def _satisfies(requirement: Requirement, manifest: Manifest) -> bool:
    if manifest.kind in requirement.any_of_kinds:
        return True
    return bool(KIND_TAGS.get(manifest.kind, frozenset()) & requirement.any_of_tags)


def _describe_requirement(requirement: Requirement) -> str:
    labels = sorted(
        {k.value for k in requirement.any_of_kinds}
        | {_TAG_LABELS.get(tag, tag) for tag in requirement.any_of_tags}
    )
    kinds_desc = " or ".join(labels) if labels else "any kind"
    if requirement.max_count is None:
        count_desc = f"at least {requirement.min_count}"
    elif requirement.min_count == requirement.max_count:
        count_desc = f"exactly {requirement.min_count}"
    else:
        count_desc = f"{requirement.min_count} to {requirement.max_count}"
    return f"{count_desc} {kinds_desc}"


def _describe_given(manifests: list[Manifest]) -> str:
    if not manifests:
        return "nothing"
    counts: dict[ManifestKind, int] = {}
    for manifest in manifests:
        counts[manifest.kind] = counts.get(manifest.kind, 0) + 1
    return ", ".join(f"{count} {kind.value}" for kind, count in counts.items())


def validate_inputs(contract: StageContract, provided: list[Manifest]) -> None:
    """Check `provided` against every `Requirement` in `contract.inputs`.

    Every contract today has exactly one slot, so "does this manifest match
    the slot" is the same question as "is this manifest valid input" — a
    future multi-slot contract (e.g. a stage needing both a sticker and a
    depth map) would need to partition `provided` by which slot each manifest
    satisfies before applying per-slot counts; that partitioning isn't built
    yet since nothing uses more than one slot.
    """
    for requirement in contract.inputs:
        if any(not _satisfies(requirement, manifest) for manifest in provided):
            hint = f" {requirement.hint}" if requirement.hint else ""
            raise WrongManifestKind(
                f"{contract.stage} requires {_describe_requirement(requirement)}, "
                f"got {_describe_given(provided)}.{hint}"
            )
        count = len(provided)
        if count < requirement.min_count or (
            requirement.max_count is not None and count > requirement.max_count
        ):
            hint = f" {requirement.hint}" if requirement.hint else ""
            raise WrongManifestCount(
                f"{contract.stage} requires {_describe_requirement(requirement)}, "
                f"got {_describe_given(provided)}.{hint}"
            )
