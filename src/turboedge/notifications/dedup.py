"""Notification deduplication via hash-based tracking in DuckDB."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime

from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import NotificationRecord


def notification_hash(
    candidate_id: str | None,
    category: str,
    key_values: Mapping[str, float | str | None],
    decimals: int = 3,
) -> str:
    """Generate a SHA-256 hash for deduplication.

    Hashes over a canonical JSON representation of:
    - candidate_id (or null)
    - category
    - key_values with floats rounded to `decimals` decimal places

    Args:
        candidate_id: Product/candidate identifier or None.
        category: Notification category (ACTIONABLE, WATCH, REJECT, DATA_QUALITY).
        key_values: Core values to hash (e.g., leverage, spread). Floats are
            rounded to `decimals` places for stability across runs.
        decimals: Number of decimal places to round floats to (default 3).

    Returns:
        Lowercase hex SHA-256 digest.
    """
    # Canonicalize key_values: round floats, preserve everything else
    canonical_values: dict[str, object] = {}
    for k, v in key_values.items():
        if isinstance(v, float):
            canonical_values[k] = round(v, decimals)
        else:
            canonical_values[k] = v

    # Build canonical document
    doc = {
        "candidate_id": candidate_id,
        "category": category,
        "key_values": canonical_values,
    }

    # Hash over sorted JSON
    json_str = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class NotificationDeduplicator:
    """Tracks sent notifications in DuckDB to avoid duplicates."""

    def __init__(self, store: Store) -> None:
        """Initialize the deduplicator.

        Args:
            store: An open Store instance (used for database access).
        """
        self.store = store

    def should_send(self, hash_value: str) -> bool:
        """Check if a notification with this hash has already been sent.

        Args:
            hash_value: The notification hash.

        Returns:
            True if this notification has NOT yet been sent, False if already sent.
        """
        return not self.store.notification_already_sent(hash_value)

    def mark_sent(
        self,
        hash_value: str,
        candidate_id: str | None,
        category: str,
        subject: str,
        sent_at: datetime | None = None,
    ) -> None:
        """Record that a notification was sent.

        Args:
            hash_value: The notification hash.
            candidate_id: Product/candidate ID or None.
            category: Notification category.
            subject: Email subject line.
            sent_at: When it was sent (default now in UTC).
        """
        if sent_at is None:
            sent_at = datetime.now(UTC)

        record = NotificationRecord(
            notification_hash=hash_value,
            candidate_id=candidate_id,
            category=category,
            sent_at=sent_at,
            subject=subject,
        )
        self.store.record_notification(record)


__all__ = [
    "NotificationDeduplicator",
    "notification_hash",
]
