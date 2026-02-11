#!/usr/bin/env python3
"""Build a large global equity ticker universe via Yahoo screener pagination.

This is more complete than prefix-search discovery because it pages through
exchange-level screen results (offset/count) instead of taking top search hits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

# Yahoo exchange codes (best-effort). Add/remove as needed for your workflows.
MAJOR_EXCHANGES: Dict[str, List[str]] = {
    "US": ["NMS", "NYQ", "ASE", "NGM", "NCM"],
    "CA": ["TOR", "VAN", "CNQ", "NEO"],
    "GB": ["LSE", "IOB"],
    "DE": ["GER", "FRA", "DUS", "MUN", "BER", "HAM", "HAN", "STU"],
    "FR": ["PAR"],
    "IT": ["MIL"],
    "ES": ["MCE"],
    "NL": ["AMS"],
    "CH": ["EBS"],
    "JP": ["JPX"],
    "HK": ["HKG"],
    "IN": ["NSI", "BSE"],
    "AU": ["ASX"],
    "SG": ["SES"],
    "BR": ["SAO"],
    "MX": ["MEX"],
}


def parse_regions(regions_csv: str) -> List[str]:
    out = [r.strip().upper() for r in regions_csv.split(",") if r.strip()]
    return out


def fetch_exchange(exchange_code: str, page_size: int, pause_ms: int, max_pages: int) -> List[Dict]:
    q = yf.EquityQuery("eq", ["exchange", exchange_code])

    rows: List[Dict] = []
    offset = 0
    pages = 0
    total = None

    while True:
        pages += 1
        if max_pages > 0 and pages > max_pages:
            break

        payload = yf.screen(
            q,
            offset=offset,
            size=page_size,
            sortField="ticker",
            sortAsc=True,
        )

        quotes = payload.get("quotes") or []
        if total is None:
            total = int(payload.get("total") or 0)

        for qq in quotes:
            symbol = str(qq.get("symbol") or "").strip()
            if not symbol:
                continue

            quote_type = str(qq.get("quoteType") or "").upper()
            if quote_type and quote_type != "EQUITY":
                continue

            rows.append(
                {
                    "exchange": str(qq.get("exchange") or exchange_code),
                    "ticker": symbol,
                    "description": str(qq.get("shortName") or qq.get("longName") or "").strip(),
                    "quote_type": quote_type,
                    "currency": str(qq.get("currency") or "").strip(),
                    "region_hint": str(qq.get("region") or "").strip(),
                    "source": "yahoo_screener",
                }
            )

        offset += len(quotes)
        if not quotes or (total is not None and offset >= total):
            break

        time.sleep(max(0, pause_ms) / 1000.0)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect thousands of tickers by paging Yahoo screener per exchange")
    parser.add_argument("--out", default="stock_universe/data", help="Output root folder")
    parser.add_argument(
        "--regions",
        default=",".join(MAJOR_EXCHANGES.keys()),
        help="Comma-separated region keys from built-in map",
    )
    parser.add_argument("--page-size", type=int, default=250, help="Rows per Yahoo screen page")
    parser.add_argument("--pause-ms", type=int, default=200, help="Pause between API pages")
    parser.add_argument("--max-pages", type=int, default=0, help="Safety cap per exchange (0 = unlimited)")
    args = parser.parse_args()

    out_root = Path(args.out)
    snapshot_day = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    snapshot_dir = out_root / "snapshots" / snapshot_day
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    regions = parse_regions(args.regions)

    all_rows: List[Dict] = []
    errors: List[Dict] = []
    stats: List[Dict] = []

    for region in regions:
        exchanges = MAJOR_EXCHANGES.get(region, [])
        for exch in exchanges:
            try:
                before = len(all_rows)
                rows = fetch_exchange(
                    exchange_code=exch,
                    page_size=max(25, args.page_size),
                    pause_ms=args.pause_ms,
                    max_pages=args.max_pages,
                )
                all_rows.extend(rows)
                stats.append({"region": region, "exchange": exch, "rows": len(rows)})
                print(f"{region}:{exch} -> {len(rows)}")
            except Exception as e:
                errors.append({"region": region, "exchange": exch, "error": str(e)})
                print(f"ERROR {region}:{exch} -> {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(columns=["exchange", "ticker", "description", "quote_type", "currency", "region_hint", "source", "region"])

    # Map region by exchange list membership
    exch_to_region = {
        exch: region
        for region, exchanges in MAJOR_EXCHANGES.items()
        for exch in exchanges
    }
    df["region"] = df["exchange"].map(exch_to_region).fillna("")

    df = (
        df.drop_duplicates(subset=["exchange", "ticker"], keep="first")
        .sort_values(["exchange", "ticker"], kind="stable")
        .reset_index(drop=True)
    )

    full_cols = ["exchange", "ticker", "description", "quote_type", "currency", "region", "region_hint", "source"]
    df = df.reindex(columns=full_cols)

    master_csv = snapshot_dir / "master_tickers.csv"
    latest_csv = out_root / "master_tickers_latest.csv"
    minimal_csv = snapshot_dir / "master_tickers_minimal.csv"
    latest_min_csv = out_root / "master_tickers_latest_minimal.csv"

    df.to_csv(master_csv, index=False)
    df.to_csv(latest_csv, index=False)
    df[["exchange", "ticker", "description"]].to_csv(minimal_csv, index=False)
    df[["exchange", "ticker", "description"]].to_csv(latest_min_csv, index=False)

    summary = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "snapshot_day": snapshot_day,
        "regions": regions,
        "exchanges_total": sum(len(MAJOR_EXCHANGES.get(r, [])) for r in regions),
        "symbol_count": int(len(df)),
        "master_snapshot_csv": str(master_csv),
        "master_latest_csv": str(latest_csv),
        "errors": errors,
        "stats": stats,
    }
    (snapshot_dir / "summary_full.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
