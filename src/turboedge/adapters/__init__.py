"""Every external data source is accessed exclusively through an adapter here.

No module outside ``adapters/`` may call an external HTTP endpoint directly
(CLAUDE.md rule 27: "Every source behind adapter").
"""

from __future__ import annotations
