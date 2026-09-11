"""Manual position ledger: add, list, close positions via CLI.

This milestone only stores and retrieves manually-entered positions.
It does not perform any re-evaluation, P&L modeling, or reevaluation.
All positions are stored in the DuckDB positions_manual table.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, date, datetime

from turboedge.storage.duckdb import Store, StoreError
from turboedge.storage.schemas import ManualPosition, PositionStatus


def _validate_wkn(wkn: str) -> str:
    """Validate and normalize a WKN.

    Args:
        wkn: Warrant Kennnummer (German securities identifier).

    Returns:
        Normalized WKN (uppercase, stripped).

    Raises:
        ValueError: If WKN is not valid (must be 6 alphanumeric characters).
    """
    normalized = wkn.strip().upper()
    if not re.match(r"^[A-Z0-9]{6}$", normalized):
        raise ValueError(
            f"invalid WKN {wkn!r}: expected 6 alphanumeric characters, got {normalized!r}"
        )
    return normalized


class PositionLedger:
    """Manual position ledger: add, list, and close positions.

    Only stores user-entered positions (via CLI). Does not perform
    re-evaluation, modeling, or margin calculations.

    Usage::

        ledger = PositionLedger(store)
        pos = ledger.add("ABC123", qty=100, price=4.86, entry_date=date(2026, 9, 10))
        positions = ledger.list()
        closed_pos = ledger.close("ABC123", price=5.42, exit_date=date(2026, 9, 15))
    """

    def __init__(
        self, store: Store, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        """Initialize the ledger.

        Args:
            store: An open Store instance (used for persistence).
            clock: Callable returning current datetime (injectable for testing).
        """
        self.store = store
        self.clock = clock

    def add(
        self,
        wkn: str,
        qty: float,
        price: float,
        entry_date: date,
        isin: str | None = None,
    ) -> ManualPosition:
        """Add a new long or short position.

        Args:
            wkn: Warrant Kennnummer (6 alphanumeric characters, case-insensitive).
            qty: Number of units (must be > 0).
            price: Entry price (must be > 0).
            entry_date: Entry date.
            isin: Optional ISIN (12 alphanumeric characters).

        Returns:
            The stored ManualPosition.

        Raises:
            ValueError: If WKN is invalid, quantity/price <= 0, or an open
                position with the same WKN already exists.
        """
        wkn_normalized = _validate_wkn(wkn)

        if qty <= 0:
            raise ValueError(f"qty must be > 0, got {qty!r}")
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price!r}")

        # Check for duplicate open position with same WKN
        open_positions = self.store.list_positions(status=PositionStatus.OPEN)
        for pos in open_positions:
            if pos.wkn.upper() == wkn_normalized:
                raise ValueError(
                    f"an open position for WKN {wkn_normalized!r} already exists "
                    f"(position_id={pos.position_id}); "
                    "close it before opening another"
                )

        now = self.clock()
        position = ManualPosition(
            wkn=wkn_normalized,
            isin=isin,
            qty=qty,
            entry_price=price,
            entry_date=entry_date,
            status=PositionStatus.OPEN,
            created_at=now,
            updated_at=now,
        )

        self.store.insert_position(position)
        return position

    def list(self, status: PositionStatus | None = None) -> list[ManualPosition]:
        """List positions, optionally filtered by status.

        Args:
            status: Filter by OPEN or CLOSED (None = all).

        Returns:
            List of ManualPosition records, ordered by creation time.
        """
        return self.store.list_positions(status=status)

    def close(self, wkn: str, price: float, exit_date: date) -> ManualPosition:
        """Close an open position.

        Args:
            wkn: Warrant Kennnummer (case-insensitive).
            price: Exit price (must be > 0).
            exit_date: Exit date.

        Returns:
            The closed ManualPosition.

        Raises:
            ValueError: If price <= 0, exit_date < entry_date, or no open
                position with this WKN exists.
            StoreError: If the WKN is ambiguous (multiple open positions).
        """
        wkn_normalized = _validate_wkn(wkn)

        if price <= 0:
            raise ValueError(f"price must be > 0, got {price!r}")

        # Fetch the open position to validate exit_date
        open_positions = self.store.list_positions(status=PositionStatus.OPEN)
        matching = [p for p in open_positions if p.wkn.upper() == wkn_normalized]

        if not matching:
            raise ValueError(f"no open position found for WKN {wkn_normalized!r}; nothing to close")

        if len(matching) > 1:
            ids = [p.position_id for p in matching]
            raise StoreError(
                f"ambiguous close: {len(matching)} open positions for "
                f"WKN {wkn_normalized!r} ({ids}); "
                "this ledger only supports closing by WKN when exactly one lot is open"
            )

        position = matching[0]
        if exit_date < position.entry_date:
            raise ValueError(
                f"exit_date {exit_date} cannot be before entry_date {position.entry_date}"
            )

        # Close via store (it updates the record)
        closed = self.store.close_position(wkn_normalized, price, exit_date)
        return closed

    def realized_pnl(self, position: ManualPosition) -> float | None:
        """Calculate realized P&L for a closed position.

        For a closed position: (exit_price - entry_price) * qty.
        Returns None if the position is not closed.

        Args:
            position: A ManualPosition (should be closed).

        Returns:
            Realized P&L in EUR (or product currency), or None if not closed.
        """
        if position.status != PositionStatus.CLOSED or position.exit_price is None:
            return None
        return (position.exit_price - position.entry_price) * position.qty

    def realized_return(self, position: ManualPosition) -> float | None:
        """Calculate realized return (%) for a closed position.

        For a closed position: (exit_price - entry_price) / entry_price.
        Returns None if the position is not closed.

        Args:
            position: A ManualPosition (should be closed).

        Returns:
            Realized return as a decimal (e.g., 0.1 for +10%), or None if not closed.
        """
        if position.status != PositionStatus.CLOSED or position.exit_price is None:
            return None
        return (position.exit_price - position.entry_price) / position.entry_price


__all__ = ["PositionLedger"]
