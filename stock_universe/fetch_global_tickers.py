#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

BASE_URL = "https://eodhd.com/api"

# Pragmatic "major exchange" starter set.
# You can add/remove codes as needed.
DEFAULT_MAJOR_EXCHANGES = [
    "US", "NYSE", "NASDAQ", "AMEX", "TSX", "LSE", "XETRA", "F", "PA", "AS",
    "SW", "MC", "MI", "BR", "STU", "TO", "HK", "T", "SHE", "SHG", "BSE", "NSE",
    "AX", "NZ", "KQ", "KO", "SA", "MX", "TA", "BUD", "WAR",
]


def api_get(path: str, params: Dict[str, str], timeout: int = 45, retries: int = 3):
    url = f"{BASE_URL}{path}"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed GET {url}: {last_err}")


def normalize_symbol(exchange_code: str, row: Dict) -> Dict[str, str]:
    return {
        "symbol": str(row.get("Code") or row.get("code") or "").strip(),
        "name": str(row.get("Name") or row.get("name") or "").strip(),
        "exchange": exchange_code,
        "country": str(row.get("Country") or row.get("country") or "").strip(),
        "currency": str(row.get("Currency") or row.get("currency") or "").strip(),
        "type": str(row.get("Type") or row.get("type") or "").strip(),
        "isin": str(row.get("ISIN") or row.get("isin") or "").strip(),
        "figi": str(row.get("FIGI") or row.get("figi") or "").strip(),
        "mic": str(row.get("MIC") or row.get("mic") or "").strip(),
        "active": str(row.get("IsActive") or row.get("is_active") or "").strip(),
    }


def read_prev_symbols(prev_csv: Path) -> set:
    if not prev_csv.exists():
        return set()
    out = set()
    with prev_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.add((row.get("exchange", ""), row.get("symbol", "")))
    return out


def find_previous_snapshot(root: Path, today_folder: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and p.name < today_folder], reverse=True)
    for c in candidates:
        f = c / "master_tickers.csv"
        if f.exists():
            return f
        legacy = c / "normalized_symbols.csv"
        if legacy.exists():
            return legacy
    return None


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Fetch global ticker symbol universe (major exchanges) daily")
    parser.add_argument("--api-key", default=os.getenv("EODHD_API_KEY"), help="EODHD API key (or set EODHD_API_KEY)")
    parser.add_argument("--out", default="stock_universe/data", help="Output folder")
    parser.add_argument("--major-only", action="store_true", default=True, help="Fetch only default major exchanges")
    parser.add_argument("--all-exchanges", action="store_true", help="Fetch all exchanges from provider")
    parser.add_argument("--exchanges", default="", help="Comma-separated exchange codes override")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Missing API key. Set EODHD_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(2)

    out_root = Path(args.out)
    snapshot_day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    snapshot_dir = out_root / "snapshots" / snapshot_day
    raw_dir = snapshot_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    params = {"api_token": args.api_key, "fmt": "json"}

    exchanges = api_get("/exchanges-list/", params=params)
    (raw_dir / "exchanges.json").write_text(json.dumps(exchanges, indent=2), encoding="utf-8")

    if args.exchanges.strip():
        target = [x.strip().upper() for x in args.exchanges.split(",") if x.strip()]
    elif args.all_exchanges:
        target = sorted({str(x.get("Code", "")).upper() for x in exchanges if x.get("Code")})
    else:
        discovered = {str(x.get("Code", "")).upper() for x in exchanges if x.get("Code")}
        target = [x for x in DEFAULT_MAJOR_EXCHANGES if x in discovered]

    all_rows: List[Dict[str, str]] = []
    errors = []

    for code in target:
        try:
            symbols = api_get(f"/exchange-symbol-list/{code}", params=params)
            (raw_dir / f"symbols_{code}.json").write_text(json.dumps(symbols, indent=2), encoding="utf-8")
            for row in symbols:
                norm = normalize_symbol(code, row)
                if norm["symbol"]:
                    all_rows.append(norm)
            print(f"Fetched {code}: {len(symbols)}")
            time.sleep(0.2)
        except Exception as e:
            errors.append({"exchange": code, "error": str(e)})
            print(f"WARN {code}: {e}", file=sys.stderr)

    # Deduplicate by (exchange, symbol)
    dedup = {}
    for r in all_rows:
        dedup[(r["exchange"], r["symbol"])] = r
    normalized = list(dedup.values())
    normalized.sort(key=lambda x: (x["exchange"], x["symbol"]))

    fields = ["symbol", "name", "exchange", "country", "currency", "type", "isin", "figi", "mic", "active"]

    # Daily snapshot master file
    master_csv = snapshot_dir / "master_tickers.csv"
    write_csv(master_csv, normalized, fields)

    # Legacy filename kept for backward compatibility
    normalized_csv = snapshot_dir / "normalized_symbols.csv"
    write_csv(normalized_csv, normalized, fields)

    # Rolling latest master file for easy downstream consumption
    latest_csv = out_root / "master_tickers_latest.csv"
    write_csv(latest_csv, normalized, fields)

    prev_csv = find_previous_snapshot(out_root / "snapshots", snapshot_day)
    prev_set = read_prev_symbols(prev_csv) if prev_csv else set()
    cur_set = {(r["exchange"], r["symbol"]) for r in normalized}

    added = sorted(cur_set - prev_set)
    removed = sorted(prev_set - cur_set)

    summary = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_day": snapshot_day,
        "exchanges_requested": target,
        "exchange_count": len(target),
        "symbol_count": len(normalized),
        "master_snapshot_csv": str(master_csv),
        "master_latest_csv": str(latest_csv),
        "errors": errors,
        "prev_snapshot_csv": str(prev_csv) if prev_csv else None,
        "added_count": len(added),
        "removed_count": len(removed),
    }
    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = []
    report.append(f"# Daily Global Ticker Pull — {snapshot_day}")
    report.append("")
    report.append(f"- Exchanges requested: **{len(target)}**")
    report.append(f"- Symbols collected: **{len(normalized)}**")
    report.append(f"- Added vs prior snapshot: **{len(added)}**")
    report.append(f"- Removed vs prior snapshot: **{len(removed)}**")
    report.append(f"- Errors: **{len(errors)}**")
    if errors:
        report.append("")
        report.append("## Errors")
        for e in errors[:20]:
            report.append(f"- {e['exchange']}: {e['error']}")
    (snapshot_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
