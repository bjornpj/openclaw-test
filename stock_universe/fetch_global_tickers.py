#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import string
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}

# Regions chosen to cover major global markets in a no-key setup.
DEFAULT_REGIONS = ["US", "CA", "GB", "DE", "FR", "IT", "ES", "AU", "NZ", "JP", "HK", "IN", "SG", "BR", "MX"]
DEFAULT_LANG = "en-US"


def yahoo_search(query: str, region: str, count: int = 100, retries: int = 4) -> Dict:
    params = {
        "q": query,
        "quotesCount": str(count),
        "newsCount": "0",
        "listsCount": "0",
        "enableFuzzyQuery": "false",
        "region": region,
        "lang": DEFAULT_LANG,
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(YAHOO_SEARCH_URL, params=params, headers=HTTP_HEADERS, timeout=30)
            if r.status_code == 429 and attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Yahoo search failed ({region}:{query}): {last_err}")


def normalize_quote(region: str, q: Dict) -> Optional[Dict[str, str]]:
    symbol = str(q.get("symbol") or "").strip()
    if not symbol:
        return None

    quote_type = str(q.get("quoteType") or "").upper()
    if quote_type not in {"EQUITY", "ETF", "MUTUALFUND", "INDEX", "FUTURE", "CRYPTOCURRENCY", "CURRENCY"}:
        return None

    description = str(q.get("shortname") or q.get("longname") or "").strip()

    return {
        "exchange": str(q.get("exchange") or "").strip(),
        "ticker": symbol,  # Yahoo Finance symbol naming convention
        "description": description,
        "exch_disp": str(q.get("exchDisp") or "").strip(),
        "quote_type": quote_type,
        "region": region,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_prev_symbols(prev_csv: Path) -> Set[Tuple[str, str]]:
    if not prev_csv.exists():
        return set()
    out: Set[Tuple[str, str]] = set()
    with prev_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker") or row.get("symbol") or ""
            out.add((row.get("exchange", ""), ticker))
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


def build_prefixes(two_char: bool = False, max_prefixes: int = 0) -> List[str]:
    base = list(string.ascii_uppercase) + list(string.digits)
    if two_char:
        pfx = base + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    else:
        pfx = base
    if max_prefixes > 0:
        pfx = pfx[:max_prefixes]
    return pfx


def main():
    parser = argparse.ArgumentParser(description="Fetch global ticker universe from Yahoo Finance (no API key)")
    parser.add_argument("--out", default="stock_universe/data", help="Output folder")
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS), help="Comma-separated Yahoo regions")
    parser.add_argument("--two-char-prefix", action="store_true", help="Expand search with AA..ZZ prefixes (larger coverage, slower)")
    parser.add_argument("--max-prefixes", type=int, default=0, help="Limit number of prefixes for faster test runs")
    parser.add_argument("--sleep-ms", type=int, default=120, help="Sleep between requests")
    args = parser.parse_args()

    out_root = Path(args.out)
    snapshot_day = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    snapshot_dir = out_root / "snapshots" / snapshot_day
    raw_dir = snapshot_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    prefixes = build_prefixes(two_char=args.two_char_prefix, max_prefixes=args.max_prefixes)

    all_rows: List[Dict[str, str]] = []
    errors = []

    for region in regions:
        for p in prefixes:
            try:
                payload = yahoo_search(p, region=region, count=100)
                (raw_dir / f"search_{region}_{p}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                for q in payload.get("quotes", []) or []:
                    row = normalize_quote(region, q)
                    if row:
                        all_rows.append(row)
                time.sleep(max(0, args.sleep_ms) / 1000.0)
            except Exception as e:
                errors.append({"region": region, "prefix": p, "error": str(e)})

    dedup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in all_rows:
        key = (r["exchange"], r["ticker"])
        dedup[key] = r

    normalized = sorted(dedup.values(), key=lambda x: (x["exchange"], x["ticker"]))

    # Requested primary format:
    # exchange, ticker (Yahoo symbol), description
    fields = ["exchange", "ticker", "description", "exch_disp", "quote_type", "region"]

    master_csv = snapshot_dir / "master_tickers.csv"
    write_csv(master_csv, normalized, fields)

    # Strict 3-column output requested by user.
    minimal_rows = [
        {
            "exchange": r.get("exchange", ""),
            "ticker": r.get("ticker", ""),
            "description": r.get("description", ""),
        }
        for r in normalized
    ]
    master_min_csv = snapshot_dir / "master_tickers_minimal.csv"
    write_csv(master_min_csv, minimal_rows, ["exchange", "ticker", "description"])

    normalized_csv = snapshot_dir / "normalized_symbols.csv"
    write_csv(normalized_csv, normalized, fields)

    latest_csv = out_root / "master_tickers_latest.csv"
    write_csv(latest_csv, normalized, fields)
    latest_min_csv = out_root / "master_tickers_latest_minimal.csv"
    write_csv(latest_min_csv, minimal_rows, ["exchange", "ticker", "description"])

    prev_csv = find_previous_snapshot(out_root / "snapshots", snapshot_day)
    prev_set = read_prev_symbols(prev_csv) if prev_csv else set()
    cur_set = {(r["exchange"], r["ticker"]) for r in normalized}

    added = sorted(cur_set - prev_set)
    removed = sorted(prev_set - cur_set)

    summary = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "snapshot_day": snapshot_day,
        "regions": regions,
        "prefix_count": len(prefixes),
        "records_collected": len(all_rows),
        "symbol_count": len(normalized),
        "master_snapshot_csv": str(master_csv),
        "master_snapshot_minimal_csv": str(master_min_csv),
        "master_latest_csv": str(latest_csv),
        "master_latest_minimal_csv": str(latest_min_csv),
        "errors": errors,
        "prev_snapshot_csv": str(prev_csv) if prev_csv else None,
        "added_count": len(added),
        "removed_count": len(removed),
    }
    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        f"# Daily Global Ticker Pull (Yahoo, no-key) — {snapshot_day}",
        "",
        f"- Regions: **{len(regions)}** ({', '.join(regions)})",
        f"- Prefixes queried: **{len(prefixes)}**",
        f"- Raw records collected: **{len(all_rows)}**",
        f"- Unique symbols: **{len(normalized)}**",
        f"- Minimal file columns: **exchange, ticker, description**",
        f"- Added vs prior snapshot: **{len(added)}**",
        f"- Removed vs prior snapshot: **{len(removed)}**",
        f"- Errors: **{len(errors)}**",
        "",
        "> Note: Yahoo search is best-effort discovery and may not be a complete official listing for each exchange.",
    ]
    if errors:
        report.append("")
        report.append("## Errors")
        for e in errors[:30]:
            report.append(f"- {e['region']}:{e['prefix']}: {e['error']}")
    (snapshot_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
