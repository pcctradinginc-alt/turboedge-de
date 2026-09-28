"""External Data Factory: point-in-time ingestion of economic and
alternative data, with automatic determination of research readiness.

Deliberately imports nothing that imports `storage.duckdb` at module scope.
That cycle broke `from turboedge.storage.duckdb import Store` twice; see
`tests/meta/test_import_layering.py`.
"""
