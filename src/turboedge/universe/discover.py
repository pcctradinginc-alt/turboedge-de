"""Merge product snapshots from multiple sources into one deduplicated universe.

Different adapters (Deutsche Boerse, Boerse Stuttgart, issuer feeds) can and
do return the same ISIN. This module dedupes per ISIN, picks a winning
observation (freshest quote, tie-broken by quality), and separately reports
any *master-data* disagreement between sources (financing level, barrier,
ratio) as a :class:`FieldConflict` so it can be surfaced as DATA_QUALITY
rather than silently resolved.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from turboedge.storage.schemas import ProductSnapshot

# Static/master-data fields that should agree across sources for the same
# ISIN within a small relative tolerance; a larger disagreement indicates a
# stale or mis-parsed source rather than a legitimate difference.
_STATIC_FIELDS: tuple[str, ...] = ("financing_level", "knockout_barrier", "ratio")

DEFAULT_STATIC_FIELD_REL_TOL = 1e-4


@dataclass(frozen=True)
class FieldConflict:
    """A disagreement between sources about a supposedly static field for one ISIN."""

    isin: str
    field: str
    values: tuple[Any, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class MergeResult:
    products: list[ProductSnapshot]
    conflicts: list[FieldConflict]


def _pick_winner(group: list[ProductSnapshot]) -> ProductSnapshot:
    """Prefer the freshest quote_timestamp, tie-broken by higher quality_score."""

    def sort_key(snapshot: ProductSnapshot) -> tuple[float, float]:
        if snapshot.quote_timestamp is not None:
            ts = snapshot.quote_timestamp.timestamp()
        else:
            ts = float("-inf")
        return (ts, snapshot.quality_score)

    return max(group, key=sort_key)


def _detect_conflicts(
    isin: str, group: list[ProductSnapshot], rel_tol: float
) -> list[FieldConflict]:
    if len(group) < 2:
        return []
    conflicts: list[FieldConflict] = []
    for field in _STATIC_FIELDS:
        values = [getattr(snapshot, field) for snapshot in group]
        non_null = [v for v in values if v is not None]
        if len(non_null) < 2:
            continue
        scale = max(abs(v) for v in non_null) or 1.0
        spread = max(non_null) - min(non_null)
        if spread / scale > rel_tol:
            conflicts.append(
                FieldConflict(
                    isin=isin,
                    field=field,
                    values=tuple(values),
                    sources=tuple(snapshot.source for snapshot in group),
                )
            )
    return conflicts


def merge_snapshots(
    snapshot_lists: Iterable[Iterable[ProductSnapshot]],
    *,
    static_field_rel_tol: float = DEFAULT_STATIC_FIELD_REL_TOL,
) -> MergeResult:
    """Merge and deduplicate product snapshots from multiple source lists.

    Args:
        snapshot_lists: One iterable of :class:`ProductSnapshot` per source
            (e.g. ``[deutsche_boerse_snapshots, boerse_stuttgart_snapshots]``).
        static_field_rel_tol: Relative tolerance (fraction of the larger
            magnitude) allowed between sources for financing_level,
            knockout_barrier and ratio before it is reported as a conflict.

    Returns:
        A :class:`MergeResult` with one winning :class:`ProductSnapshot` per
        ISIN and the list of detected :class:`FieldConflict` records (which
        do not prevent a winner from being chosen - the caller decides how
        to react, typically by flagging the candidate DATA_QUALITY).
    """
    by_isin: dict[str, list[ProductSnapshot]] = defaultdict(list)
    for source_list in snapshot_lists:
        for snapshot in source_list:
            by_isin[snapshot.isin].append(snapshot)

    products: list[ProductSnapshot] = []
    conflicts: list[FieldConflict] = []
    for isin, group in by_isin.items():
        conflicts.extend(_detect_conflicts(isin, group, static_field_rel_tol))
        products.append(_pick_winner(group))

    return MergeResult(products=products, conflicts=conflicts)
