"""Every research read of external data must carry a point-in-time cutoff.

`Store.list_external_observations(series_id=..., available_at_max=None)`
returns everything stored, including rows whose `available_at` lies after
the moment a study is reasoning about. That default is correct for the one
place that uses it today -- `external/ingest.py` counts accumulated evidence
and must see all of it -- and it is a loaded gun for the next caller, which
will be a feature builder or a backtest.

`external_series_aggregate` and `latest_external_observation_time` are the
same gun without even a safety: they take no cutoff at all, by design, and
compute over every stored vintage. They were added on 2026-10-24 to stop
the readiness evaluation from materialising a million rows per series, and
they must stay confined to the ingestion loop. A guard that only watched
`list_external_observations` would have let that optimisation quietly move
the unfiltered read somewhere a backtest could reach it.

CLAUDE.md rule 5 is `available_at <= prediction_time`, and the repository
already has the machinery to honour it: `features/availability.py` for
in-memory filtering and `research/archive.py:read_as_of` for the Parquet
archive. Nothing currently bypasses them. These tests exist so that nothing
starts to.

Deliberately a static check on call sites rather than a runtime assertion.
Making the cutoff a required argument would be stronger, but it would change
a shipped signature used by the ingestion loop, which is outside what this
pass is allowed to touch. A test that fails the moment an unguarded consumer
appears buys the same protection at zero behavioural risk.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path("src/turboedge")

#: Modules permitted to read external observations without a cutoff, with
#: the reason. `external/ingest.py` measures how much evidence has
#: accumulated; a cutoff there would make a series look permanently short of
#: its own readiness threshold.
CUTOFF_EXEMPT: dict[str, str] = {
    "external/ingest.py": "measures accumulated evidence for readiness, not a study",
}

#: Reads that return every stored vintage unless told otherwise. The value
#: is the keyword that makes the read point-in-time safe, or None when the
#: method takes no cutoff at all and is therefore never safe for a study.
UNFILTERED_READS: dict[str, str | None] = {
    "list_external_observations": "available_at_max",
    "external_series_aggregate": None,
    "latest_external_observation_time": None,
}


def _unguarded_call_sites() -> list[tuple[str, int]]:
    """Reads of stored external data that carry no point-in-time cutoff."""
    offenders: list[tuple[str, int]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - would fail the lint gate first
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in UNFILTERED_READS:
                continue
            # The definitions themselves are not call sites.
            if rel.endswith("storage/duckdb.py"):
                continue
            cutoff_kwarg = UNFILTERED_READS[name]
            if cutoff_kwarg is None or cutoff_kwarg not in {kw.arg for kw in node.keywords}:
                offenders.append((rel, node.lineno))
    return offenders


def test_only_documented_modules_read_external_data_without_a_cutoff() -> None:
    offenders = _unguarded_call_sites()
    undocumented = [(m, line) for m, line in offenders if m not in CUTOFF_EXEMPT]

    assert undocumented == [], (
        "these call sites read external observations with no point-in-time "
        f"cutoff: {undocumented}. CLAUDE.md rule 5 requires "
        "`available_at <= prediction_time`. Either pass `available_at_max`, or "
        "filter through features/availability.py, or -- if the call genuinely "
        "needs every stored row the way the ingestion loop does -- add it to "
        "CUTOFF_EXEMPT with the reason."
    )


def test_the_exemptions_are_still_real() -> None:
    # An exemption for a module that no longer makes such a call is stale,
    # and a stale exemption is a hole waiting for an unrelated caller.
    offending_modules = {m for m, _ in _unguarded_call_sites()}
    stale = sorted(set(CUTOFF_EXEMPT) - offending_modules)

    assert stale == [], (
        f"these modules are exempted but no longer read external observations "
        f"without a cutoff: {stale}. Remove the exemption."
    )


def test_the_cutoff_parameter_still_exists_and_filters_availability() -> None:
    # If the parameter were renamed or removed, the static check above would
    # pass vacuously while the protection was gone.
    import inspect

    from turboedge.storage.duckdb import Store

    signature = inspect.signature(Store.list_external_observations)
    assert "available_at_max" in signature.parameters

    source = inspect.getsource(Store.list_external_observations)
    assert "available_at <= ?" in source, (
        "list_external_observations no longer filters on available_at. The "
        "point-in-time guarantee for external data lived in that clause."
    )
    assert "observation_time <=" not in source, (
        "list_external_observations filters on observation_time. That is the "
        "substitution CLAUDE.md rule 5 exists to forbid: a value's period is "
        "not the moment it became knowable."
    )


def test_the_cutoff_free_reads_are_still_cutoff_free() -> None:
    # The two aggregate reads are exempted on the grounds that they take no
    # cutoff. If one ever grows a cutoff parameter, the exemption above stops
    # being an accurate description of it and the stricter rule should apply.
    import inspect

    from turboedge.storage.duckdb import Store

    for name, cutoff_kwarg in UNFILTERED_READS.items():
        if cutoff_kwarg is not None:
            continue
        method = getattr(Store, name)
        parameters = inspect.signature(method).parameters
        assert "available_at_max" not in parameters, (
            f"Store.{name} now takes a point-in-time cutoff. Record it in "
            "UNFILTERED_READS so call sites are required to pass it, instead "
            "of being exempted as inherently unfiltered."
        )
