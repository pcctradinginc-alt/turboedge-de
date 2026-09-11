"""Tests for notifications/dedup.py."""

from __future__ import annotations

from datetime import UTC, datetime

from turboedge.notifications.dedup import NotificationDeduplicator, notification_hash
from turboedge.storage.duckdb import Store


class TestNotificationHash:
    """Test notification_hash()."""

    def test_same_inputs_same_hash(self) -> None:
        """Same inputs produce the same hash."""
        hash1 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5, "spread_pct": 0.015},
        )
        hash2 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5, "spread_pct": 0.015},
        )

        assert hash1 == hash2

    def test_different_candidate_different_hash(self) -> None:
        """Different candidate_id produces different hash."""
        hash1 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5},
        )
        hash2 = notification_hash(
            candidate_id="cand_002",
            category="WATCH",
            key_values={"leverage": 5.5},
        )

        assert hash1 != hash2

    def test_different_category_different_hash(self) -> None:
        """Different category produces different hash."""
        hash1 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5},
        )
        hash2 = notification_hash(
            candidate_id="cand_001",
            category="REJECT",
            key_values={"leverage": 5.5},
        )

        assert hash1 != hash2

    def test_different_values_different_hash(self) -> None:
        """Different key_values produces different hash."""
        hash1 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5, "spread_pct": 0.015},
        )
        hash2 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 6.0, "spread_pct": 0.015},
        )

        assert hash1 != hash2

    def test_float_rounding(self) -> None:
        """Floats are rounded to specified decimals for stability."""
        # 5.5001 and 5.5002 should round to same value with decimals=2
        hash1 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5001},
            decimals=2,
        )
        hash2 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5002},
            decimals=2,
        )

        assert hash1 == hash2

    def test_float_different_with_higher_precision(self) -> None:
        """Higher decimals distinguish close values."""
        hash1 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5001},
            decimals=4,
        )
        hash2 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5002},
            decimals=4,
        )

        assert hash1 != hash2

    def test_none_candidate_id(self) -> None:
        """Hashing with None candidate_id works."""
        hash1 = notification_hash(
            candidate_id=None,
            category="WATCH",
            key_values={"score": 0.75},
        )
        hash2 = notification_hash(
            candidate_id=None,
            category="WATCH",
            key_values={"score": 0.75},
        )

        assert hash1 == hash2

    def test_none_candidate_vs_value_different(self) -> None:
        """None candidate_id is different from a value."""
        hash1 = notification_hash(
            candidate_id=None,
            category="WATCH",
            key_values={"score": 0.75},
        )
        hash2 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"score": 0.75},
        )

        assert hash1 != hash2

    def test_none_values_in_key_values(self) -> None:
        """None values in key_values are handled correctly."""
        hash1 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5, "spread_pct": None},
        )
        hash2 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5, "spread_pct": None},
        )

        assert hash1 == hash2

    def test_hash_is_hex_string(self) -> None:
        """Hash is a valid hex string."""
        hash_val = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5},
        )

        # Should be valid hex (lowercase)
        assert len(hash_val) == 64  # SHA-256 is 64 hex chars
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_order_independence_in_key_values(self) -> None:
        """Key order doesn't affect hash (canonical JSON)."""
        hash1 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"leverage": 5.5, "spread_pct": 0.015},
        )
        hash2 = notification_hash(
            candidate_id="cand_001",
            category="WATCH",
            key_values={"spread_pct": 0.015, "leverage": 5.5},
        )

        assert hash1 == hash2


class TestNotificationDeduplicator:
    """Test NotificationDeduplicator."""

    def test_should_send_new_hash(self, tmp_path):
        """should_send returns True for a new hash."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        dedup = NotificationDeduplicator(store)
        hash_val = "test_hash_001"

        assert dedup.should_send(hash_val) is True

        store.close()

    def test_should_send_after_mark_sent(self, tmp_path):
        """should_send returns False after mark_sent."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        dedup = NotificationDeduplicator(store)
        hash_val = "test_hash_002"
        now = datetime.now(UTC)

        # First check: should send
        assert dedup.should_send(hash_val) is True

        # Mark as sent
        dedup.mark_sent(
            hash_val,
            candidate_id="cand_001",
            category="WATCH",
            subject="Test",
            sent_at=now,
        )

        # Second check: should not send
        assert dedup.should_send(hash_val) is False

        store.close()

    def test_mark_sent_creates_record(self, tmp_path):
        """mark_sent creates a record in the database."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        dedup = NotificationDeduplicator(store)
        hash_val = "test_hash_003"
        now = datetime(2026, 9, 11, 10, 30, tzinfo=UTC)

        dedup.mark_sent(
            hash_val,
            candidate_id="cand_001",
            category="WATCH",
            subject="Test Subject",
            sent_at=now,
        )

        # Verify by checking should_send
        assert dedup.should_send(hash_val) is False

        store.close()

    def test_mark_sent_with_none_candidate(self, tmp_path):
        """mark_sent works with None candidate_id."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        dedup = NotificationDeduplicator(store)
        hash_val = "test_hash_004"

        dedup.mark_sent(
            hash_val,
            candidate_id=None,
            category="DATA_QUALITY",
            subject="Health Alert",
        )

        # Verify
        assert dedup.should_send(hash_val) is False

        store.close()

    def test_mark_sent_default_time(self, tmp_path):
        """mark_sent uses current time when sent_at is None."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        dedup = NotificationDeduplicator(store)
        hash_val = "test_hash_005"

        # No sent_at provided
        dedup.mark_sent(
            hash_val,
            candidate_id="cand_001",
            category="WATCH",
            subject="Test",
        )

        # Should still be marked
        assert dedup.should_send(hash_val) is False

        store.close()

    def test_multiple_hashes_independent(self, tmp_path):
        """Different hashes are tracked independently."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        dedup = NotificationDeduplicator(store)

        # Mark first hash
        dedup.mark_sent("hash_001", candidate_id="c1", category="WATCH", subject="S1")

        # Second hash should still be sendable
        assert dedup.should_send("hash_001") is False
        assert dedup.should_send("hash_002") is True

        store.close()
