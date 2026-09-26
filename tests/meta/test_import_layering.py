"""Import-order regression tests for the meta package.

`storage.duckdb` imports the research schemas from `turboedge.meta`, so
everything `turboedge/meta/__init__.py` pulls in is loaded before `Store`
exists. A module in that chain that reaches back into `storage.duckdb` or
`turboedge.learning` therefore closes a cycle.

The reason this needs its own test rather than a comment: the cycle is
invisible to both the CLI and the rest of the suite, because both load
`turboedge.learning` before `turboedge.storage.duckdb` for unrelated reasons.
Only a fresh interpreter whose *first* turboedge import is the storage module
sees it -- which is exactly what an analysis script or a notebook does.
"""

from __future__ import annotations

import subprocess
import sys

_FIRST_IMPORTS = [
    "from turboedge.storage.duckdb import Store",
    "import turboedge.meta",
    "from turboedge.meta.research_queue import ResearchQueue",
    "from turboedge.meta.research_memory import enrich",
    "from turboedge.meta.catalog import CATALOG",
    "from turboedge.meta import render_research_queue",
]


def test_each_module_imports_cleanly_as_the_first_turboedge_import() -> None:
    failures: list[str] = []
    for statement in _FIRST_IMPORTS:
        proc = subprocess.run(
            [sys.executable, "-c", statement],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            failures.append(f"{statement!r} -> {proc.stderr.strip().splitlines()[-1]}")

    assert not failures, "import cycle(s):\n" + "\n".join(failures)


def test_meta_package_init_does_not_pull_in_the_store() -> None:
    """The invariant behind the test above, stated directly.

    Asserted on a fresh interpreter because `sys.modules` in the test process
    already holds everything.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import turboedge.meta; print('turboedge.storage.duckdb' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", (
        "turboedge.meta's __init__ now imports storage.duckdb; that reverses the "
        "dependency storage.duckdb itself relies on and reintroduces the cycle"
    )
