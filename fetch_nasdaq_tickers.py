#!/usr/bin/env python3
"""
Fetch NASDAQ symbol directory files and produce a clean ticker dataset.

Sources (official Nasdaq Trader symbol directories):
- nasdaqlisted.txt
- otherlisted.txt
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List

import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def fetch_pipe_file(url: str) -> List[Dict[str, str]]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = [line.strip() for line in r.text.splitlines() if line.strip()]

    # Last line is usually "File Creation Time:..." metadata
    if lines and lines[-1].startswith("File Creation Time"):
        lines = lines[:-1]

    reader = csv.DictReader(lines, delimiter="|")
    return [dict(row) for row in reader]


def normalize_nasdaq_listed(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        symbol = (r.get("Symbol") or "").strip()
        if not symbol:
            continue
        out.append(
            {
                "symbol": symbol,
                "name": (r.get("Security Name") or "").strip(),
                "exchange": "NASDAQ",
                "market_category": (r.get("Market Category") or "").strip(),
                "test_issue": (r.get("Test Issue") or "").strip(),
                "financial_status": (r.get("Financial Status") or "").strip(),
                "round_lot_size": (r.get("Round Lot Size") or "").strip(),
                "etf": (r.get("ETF") or "").strip(),
                "nextshares": (r.get("NextShares") or "").strip(),
                "source_file": "nasdaqlisted.txt",
            }
        )
    return out


def normalize_other_listed(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        exchange = (r.get("Exchange") or "").strip()
        if exchange != "Q":  # Q = Nasdaq from this file
            continue

        symbol = (r.get("ACT Symbol") or "").strip()
        if not symbol:
            continue

        out.append(
            {
                "symbol": symbol,
                "name": (r.get("Security Name") or "").strip(),
                "exchange": "NASDAQ",
                "market_category": "",
                "test_issue": (r.get("Test Issue") or "").strip(),
                "financial_status": "",
                "round_lot_size": (r.get("Round Lot Size") or "").strip(),
                "etf": (r.get("ETF") or "").strip(),
                "nextshares": "",
                "source_file": "otherlisted.txt",
            }
        )
    return out


def dedupe(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = {}
    for r in rows:
        seen[r["symbol"]] = r
    return sorted(seen.values(), key=lambda x: x["symbol"])


def write_csv(path: Path, rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol",
        "name",
        "exchange",
        "market_category",
        "test_issue",
        "financial_status",
        "round_lot_size",
        "etf",
        "nextshares",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull NASDAQ ticker dataset")
    parser.add_argument("--out-dir", default="data/nasdaq", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Also write JSON output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    now_utc = dt.datetime.now(dt.UTC)
    run_day = now_utc.strftime("%Y-%m-%d")

    nasdaq_listed = fetch_pipe_file(NASDAQ_LISTED_URL)
    other_listed = fetch_pipe_file(OTHER_LISTED_URL)

    combined = dedupe(
        normalize_nasdaq_listed(nasdaq_listed)
        + normalize_other_listed(other_listed)
    )

    csv_path = out_dir / f"nasdaq_tickers_{run_day}.csv"
    write_csv(csv_path, combined)

    summary = {
        "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "run_day": run_day,
        "rows": len(combined),
        "output_csv": str(csv_path),
        "sources": [NASDAQ_LISTED_URL, OTHER_LISTED_URL],
    }

    summary_path = out_dir / f"summary_{run_day}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.json:
        json_path = out_dir / f"nasdaq_tickers_{run_day}.json"
        json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
