"""Tests for positions/ledger.py."""

from __future__ import annotations

import re
from datetime import date

import pytest

from turboedge.positions.ledger import PositionLedger
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import PositionStatus


class TestWknValidation:
    """Test WKN validation."""

    def test_valid_wkn_uppercase(self, tmp_path):
        """Valid WKN is normalized to uppercase."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        pos = ledger.add(
            wkn="abc123",
            qty=100,
            price=4.86,
            entry_date=date(2026, 9, 10),
        )

        assert pos.wkn == "ABC123"
        store.close()

    def test_invalid_wkn_too_short(self, tmp_path):
        """WKN with < 6 characters raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)

        with pytest.raises(ValueError, match="invalid WKN"):
            ledger.add(
                wkn="ABC12",
                qty=100,
                price=4.86,
                entry_date=date(2026, 9, 10),
            )

        store.close()

    def test_invalid_wkn_too_long(self, tmp_path):
        """WKN with > 6 characters raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)

        with pytest.raises(ValueError, match="invalid WKN"):
            ledger.add(
                wkn="ABC1234",
                qty=100,
                price=4.86,
                entry_date=date(2026, 9, 10),
            )

        store.close()

    def test_invalid_wkn_non_alphanumeric(self, tmp_path):
        """WKN with non-alphanumeric characters raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)

        with pytest.raises(ValueError, match="invalid WKN"):
            ledger.add(
                wkn="ABC-123",
                qty=100,
                price=4.86,
                entry_date=date(2026, 9, 10),
            )

        store.close()

    def test_wkn_trimmed(self, tmp_path):
        """WKN with leading/trailing whitespace is trimmed."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        pos = ledger.add(
            wkn="  ABC123  ",
            qty=100,
            price=4.86,
            entry_date=date(2026, 9, 10),
        )

        assert pos.wkn == "ABC123"
        store.close()


class TestPositionAdd:
    """Test PositionLedger.add()."""

    def test_add_position(self, tmp_path):
        """Add a new position successfully."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        pos = ledger.add(
            wkn="ABC123",
            qty=100,
            price=4.86,
            entry_date=date(2026, 9, 10),
            isin="DE000ABC1234",
        )

        assert pos.wkn == "ABC123"
        assert pos.qty == 100
        assert pos.entry_price == 4.86
        assert pos.entry_date == date(2026, 9, 10)
        assert pos.isin == "DE000ABC1234"
        assert pos.status == PositionStatus.OPEN
        assert pos.exit_price is None
        assert pos.exit_date is None

        store.close()

    def test_add_position_no_isin(self, tmp_path):
        """Add a position without ISIN."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        pos = ledger.add(
            wkn="XYZ999",
            qty=50,
            price=2.50,
            entry_date=date(2026, 9, 11),
        )

        assert pos.isin is None
        store.close()

    def test_add_duplicate_open_position_raises(self, tmp_path):
        """Cannot add a second open position with same WKN."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add(
            wkn="ABC123",
            qty=100,
            price=4.86,
            entry_date=date(2026, 9, 10),
        )

        with pytest.raises(ValueError, match="already exists"):
            ledger.add(
                wkn="ABC123",
                qty=50,
                price=5.00,
                entry_date=date(2026, 9, 11),
            )

        store.close()

    def test_add_qty_zero_raises(self, tmp_path):
        """Adding with qty=0 raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)

        with pytest.raises(ValueError, match="qty must be > 0"):
            ledger.add(
                wkn="ABC123",
                qty=0,
                price=4.86,
                entry_date=date(2026, 9, 10),
            )

        store.close()

    def test_add_negative_qty_raises(self, tmp_path):
        """Adding with negative qty raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)

        with pytest.raises(ValueError, match="qty must be > 0"):
            ledger.add(
                wkn="ABC123",
                qty=-100,
                price=4.86,
                entry_date=date(2026, 9, 10),
            )

        store.close()

    def test_add_zero_price_raises(self, tmp_path):
        """Adding with price=0 raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)

        with pytest.raises(ValueError, match="price must be > 0"):
            ledger.add(
                wkn="ABC123",
                qty=100,
                price=0,
                entry_date=date(2026, 9, 10),
            )

        store.close()

    def test_add_negative_price_raises(self, tmp_path):
        """Adding with negative price raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)

        with pytest.raises(ValueError, match="price must be > 0"):
            ledger.add(
                wkn="ABC123",
                qty=100,
                price=-4.86,
                entry_date=date(2026, 9, 10),
            )

        store.close()


class TestPositionList:
    """Test PositionLedger.list()."""

    def test_list_empty(self, tmp_path):
        """list() returns empty list when no positions."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        positions = ledger.list()

        assert positions == []
        store.close()

    def test_list_all_positions(self, tmp_path):
        """list() returns all positions when no filter."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))
        ledger.add("XYZ999", qty=50, price=2.50, entry_date=date(2026, 9, 11))

        positions = ledger.list()

        assert len(positions) == 2
        wkns = {p.wkn for p in positions}
        assert wkns == {"ABC123", "XYZ999"}

        store.close()

    def test_list_by_status_open(self, tmp_path):
        """list(OPEN) returns only open positions."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))
        ledger.add("XYZ999", qty=50, price=2.50, entry_date=date(2026, 9, 11))

        # Close one
        ledger.close("ABC123", price=5.50, exit_date=date(2026, 9, 15))

        open_positions = ledger.list(status=PositionStatus.OPEN)

        assert len(open_positions) == 1
        assert open_positions[0].wkn == "XYZ999"

        store.close()

    def test_list_by_status_closed(self, tmp_path):
        """list(CLOSED) returns only closed positions."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))
        ledger.add("XYZ999", qty=50, price=2.50, entry_date=date(2026, 9, 11))

        # Close one
        ledger.close("ABC123", price=5.50, exit_date=date(2026, 9, 15))

        closed_positions = ledger.list(status=PositionStatus.CLOSED)

        assert len(closed_positions) == 1
        assert closed_positions[0].wkn == "ABC123"

        store.close()


class TestPositionClose:
    """Test PositionLedger.close()."""

    def test_close_position(self, tmp_path):
        """Close an open position successfully."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))

        closed = ledger.close("ABC123", price=5.50, exit_date=date(2026, 9, 15))

        assert closed.wkn == "ABC123"
        assert closed.exit_price == 5.50
        assert closed.exit_date == date(2026, 9, 15)
        assert closed.status == PositionStatus.CLOSED

        store.close()

    def test_close_nonexistent_position_raises(self, tmp_path):
        """Closing a non-existent position raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)

        with pytest.raises(ValueError, match="no open position found"):
            ledger.close("NOTFND", price=5.50, exit_date=date(2026, 9, 15))

        store.close()

    def test_close_zero_price_raises(self, tmp_path):
        """Closing with price=0 raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))

        with pytest.raises(ValueError, match="price must be > 0"):
            ledger.close("ABC123", price=0, exit_date=date(2026, 9, 15))

        store.close()

    def test_close_exit_before_entry_raises(self, tmp_path):
        """Closing with exit_date < entry_date raises ValueError."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))

        with pytest.raises(
            ValueError, match=re.escape("exit_date") + r".*cannot be before entry_date"
        ):
            ledger.close("ABC123", price=5.50, exit_date=date(2026, 9, 5))

        store.close()

    def test_close_same_day(self, tmp_path):
        """Closing on the same day as entry is allowed."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))

        closed = ledger.close("ABC123", price=5.50, exit_date=date(2026, 9, 10))

        assert closed.exit_date == date(2026, 9, 10)

        store.close()


class TestRealizedPnL:
    """Test PositionLedger.realized_pnl()."""

    def test_realized_pnl_closed_position(self, tmp_path):
        """Calculate P&L for a closed position."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))
        closed = ledger.close("ABC123", price=5.50, exit_date=date(2026, 9, 15))

        pnl = ledger.realized_pnl(closed)

        assert pnl == (5.50 - 4.86) * 100  # 64.0

        store.close()

    def test_realized_pnl_open_position(self, tmp_path):
        """realized_pnl returns None for open position."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        pos = ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))

        pnl = ledger.realized_pnl(pos)

        assert pnl is None

        store.close()

    def test_realized_pnl_loss(self, tmp_path):
        """Calculate negative P&L for a closed loss."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=5.50, entry_date=date(2026, 9, 10))
        closed = ledger.close("ABC123", price=4.86, exit_date=date(2026, 9, 15))

        pnl = ledger.realized_pnl(closed)

        assert pnl == (4.86 - 5.50) * 100  # -64.0

        store.close()


class TestRealizedReturn:
    """Test PositionLedger.realized_return()."""

    def test_realized_return_closed_position(self, tmp_path):
        """Calculate return % for a closed position."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))
        closed = ledger.close("ABC123", price=5.50, exit_date=date(2026, 9, 15))

        ret = ledger.realized_return(closed)

        expected = (5.50 - 4.86) / 4.86
        assert abs(ret - expected) < 0.0001

        store.close()

    def test_realized_return_open_position(self, tmp_path):
        """realized_return returns None for open position."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        pos = ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))

        ret = ledger.realized_return(pos)

        assert ret is None

        store.close()

    def test_realized_return_loss(self, tmp_path):
        """Calculate negative return % for a loss."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=5.50, entry_date=date(2026, 9, 10))
        closed = ledger.close("ABC123", price=4.86, exit_date=date(2026, 9, 15))

        ret = ledger.realized_return(closed)

        expected = (4.86 - 5.50) / 5.50
        assert abs(ret - expected) < 0.0001
        assert ret < 0

        store.close()

    def test_realized_return_break_even(self, tmp_path):
        """Return is zero when exit price = entry price."""
        db_path = tmp_path / "test.duckdb"
        store = Store(str(db_path))
        store.init_schema()

        ledger = PositionLedger(store)
        ledger.add("ABC123", qty=100, price=5.00, entry_date=date(2026, 9, 10))
        closed = ledger.close("ABC123", price=5.00, exit_date=date(2026, 9, 15))

        ret = ledger.realized_return(closed)

        assert ret == 0.0

        store.close()
