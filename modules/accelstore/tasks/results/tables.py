"""Shared table helpers for the bench, ycsb and redis result parsers.

The CSV writer, the fixed-digit cell formatter, the per-second rate and the
acknowledgement summary. The ablation and MLPerf tables keep their own
``_fmt``, which passes non-numeric values through. Read-side only.
"""

import csv
from pathlib import Path
from typing import List, Optional, Sequence


def write_csv(path: Path, columns: List[str], rows: List[dict]) -> None:
    """Rows as CSV: keys outside `columns` are dropped, missing ones blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, columns, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ {len(rows)} rows → {path}")


def fmt(value: Optional[float], digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def rate(amount: float, wall_ms: float) -> float:
    """Per-second rate of something measured over `wall_ms`; 0 for no wall."""
    return amount / (wall_ms / 1000.0) if wall_ms > 0 else 0.0


def ack_summary(record: dict, client_keys: Sequence[str]) -> str:
    """What an acknowledged operation guarantees on this arm, in one token.

    A client on a block stack has two promises, its own durability call and the
    stack's underneath it (named by `client_keys`), and both go in the row.
    """
    ack = record.get("ack_semantics")
    if isinstance(ack, dict) and "summary" in ack:
        return str(ack["summary"])
    if isinstance(ack, dict):
        parts = []
        for key in client_keys:
            value = ack.get(key)
            if isinstance(value, dict):
                value = value.get("summary")
            if value:
                parts.append(str(value))
        return "+".join(parts)
    return str(ack) if ack else ""
