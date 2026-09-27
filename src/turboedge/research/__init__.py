"""Research infrastructure: data partitions, archive, system generations.

Deliberately imports nothing that imports `storage.duckdb`. That cycle broke
`from turboedge.storage.duckdb import Store` twice in one day via
`meta/__init__`; see `tests/meta/test_import_layering.py`.
"""

from turboedge.research.archive import (
    ParquetResearchArchive,
    PointInTimeUnavailable,
    ResearchArchive,
    UnknownTable,
)
from turboedge.research.generations import (
    GenerationStatus,
    SystemGeneration,
)
from turboedge.research.partitions import (
    PartitionAction,
    PartitionType,
    ResearchPartition,
)

__all__ = [
    "GenerationStatus",
    "ParquetResearchArchive",
    "PartitionAction",
    "PartitionType",
    "PointInTimeUnavailable",
    "ResearchArchive",
    "ResearchPartition",
    "SystemGeneration",
    "UnknownTable",
]
