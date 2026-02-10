#!/usr/bin/env python3
import argparse
import datetime as dt
import inspect
import json
import string
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yfinance as yf

# Regions chosen to cover major global markets in a no-key setup.
DEFAULT_REGIONS = ["US", "CA", "GB", "DE", "FR", "IT", "ES", "AU", "NZ", "JP", "HK", "IN", "SG", "BR", "MX"]
DEFAULT_LANG = "en-US"

VALID_QUOTE_TYPES = {"EQUITY", "ETF", "MUTUALFUND", "INDEX", "FUTURE", "CRYPTOCURRENCY", "CURRENCY"}


def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


def _build_search_kwargs(query: str, region: str, count: int) -> Dict:
    """Build kwargs compatible with the installed yfinance.Search signature."""
    sig = inspect.signature(yf.Search)
    params = set(sig.parameters.keys())

    kwargs: Dict = {}
    if "query" in params:
        kwargs["query"] = query
    elif "q" in params:
        kwargs["q"] = query

    if "max_results" in params:
        kwargs["max_results"] = count
    elif "quotes_count" in params:
        kwargs["quotes_count"] = count

    if "news_count" in params:
        kwargs["news_count"] = 0
    if "lists_count" in params:
        kwargs["lists_count"] = 0
    if "enable_fuzzy_query" in params:
        kwargs["enable_fuzzy_query"] = False
    if "include_news" in params:
        kwargs["include_news"] = False

    # Optional region/lang arguments (only if supported by installed yfinance version)
    if "region" in params:
        kwargs["region"] = region
    if "lang" in params:
        kwargs["lang"] = DEFAULT_LANG

    return kwargs


def yf_search(query: str, region: str, count: int = 100, retries: int = 4) -> Dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            kwargs = _build_search_kwargs(query=query, region=region, count=count)
            search_obj = yf.Search(**kwargs)
            quotes = getattr(search_obj, "quotes", []) or []
            return {"quotes": quotes}
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"yfinance search failed ({region}:{query}): {last_err}")


def normalize_quote(region: str, q: Dict) -> Optional[Dict[str, str]]:
    symbol = str(q.get("symbol") or "").strip()
    if not symbol:
        return None

    quote_type = str(q.get("quoteType") or "").upper()
    if quote_type not in VALID_QUOTE_TYPES:
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


def read_prev_symbols(prev_csv: Path) -> Set[Tuple[str, str]]:
    if not prev_csv.exists():
        return set()
    try:
        prev_df = pd.read_csv(prev_csv)
    except Exception:
        return set()

    if prev_df.empty:
        return set()

    ticker_col = "ticker" if "ticker" in prev_df.columns else "symbol"
    if ticker_col not in prev_df.columns or "exchange" not in prev_df.columns:
        return set()

    prev_df[ticker_col] = prev_df[ticker_col].fillna("").astype(str)
    prev_df["exchange"] = prev_df["exchange"].fillna("").astype(str)
    return set(zip(prev_df["exchange"], prev_df[ticker_col]))


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
        prefixes = base + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    else:
        prefixes = base
    if max_prefixes > 0:
        prefixes = prefixes[:max_prefixes]
    return prefixes


def main():
    parser = argparse.ArgumentParser(description="Fetch global ticker universe from Yahoo Finance using yfinance + pandas")
    parser.add_argument("--out", default="stock_universe/data", help="Output folder")
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS), help="Comma-separated Yahoo regions")
    parser.add_argument("--two-char-prefix", action="store_true", help="Expand search with AA..ZZ prefixes (larger coverage, slower)")
    parser.add_argument("--max-prefixes", type=int, default=0, help="Limit number of prefixes for faster test runs")
    parser.add_argument("--sleep-ms", type=int, default=150, help="Sleep between requests")
    args = parser.parse_args()

    out_root = Path(args.out)
    snapshot_day = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    snapshot_dir = out_root / "snapshots" / snapshot_day
    raw_dir = snapshot_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    prefixes = build_prefixes(two_char=args.two_char_prefix, max_prefixes=args.max_prefixes)

    all_rows: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    for region in regions:
        for p in prefixes:
            try:
                payload = yf_search(p, region=region, count=100)
                (raw_dir / f"search_{region}_{p}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                for q in payload.get("quotes", []) or []:
                    row = normalize_quote(region, q)
                    if row:
                        all_rows.append(row)
                time.sleep(max(0, args.sleep_ms) / 1000.0)
            except Exception as e:
                errors.append({"region": region, "prefix": p, "error": str(e)})

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(columns=["exchange", "ticker", "description", "exch_disp", "quote_type", "region"])

    # Deduplicate and sort with pandas
    df = (
        df.drop_duplicates(subset=["exchange", "ticker"], keep="last")
        .sort_values(["exchange", "ticker"], kind="stable")
        .reset_index(drop=True)
    )

    # Requested primary format
    full_cols = ["exchange", "ticker", "description", "exch_disp", "quote_type", "region"]
    df = df.reindex(columns=full_cols)

    master_csv = snapshot_dir / "master_tickers.csv"
    df.to_csv(master_csv, index=False)

    minimal_cols = ["exchange", "ticker", "description"]
    master_min_csv = snapshot_dir / "master_tickers_minimal.csv"
    df[minimal_cols].to_csv(master_min_csv, index=False)

    normalized_csv = snapshot_dir / "normalized_symbols.csv"
    df.to_csv(normalized_csv, index=False)

    latest_csv = out_root / "master_tickers_latest.csv"
    df.to_csv(latest_csv, index=False)

    latest_min_csv = out_root / "master_tickers_latest_minimal.csv"
    df[minimal_cols].to_csv(latest_min_csv, index=False)

    prev_csv = find_previous_snapshot(out_root / "snapshots", snapshot_day)
    prev_set = read_prev_symbols(prev_csv) if prev_csv else set()
    cur_set = set(zip(df["exchange"].fillna("").astype(str), df["ticker"].fillna("").astype(str)))

    added = sorted(cur_set - prev_set)
    removed = sorted(prev_set - cur_set)

    summary = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "snapshot_day": snapshot_day,
        "regions": regions,
        "prefix_count": len(prefixes),
        "records_collected": int(len(all_rows)),
        "symbol_count": int(len(df)),
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
        f"# Daily Global Ticker Pull (yfinance + pandas) — {snapshot_day}",
        "",
        f"- Regions: **{len(regions)}** ({', '.join(regions)})",
        f"- Prefixes queried: **{len(prefixes)}**",
        f"- Raw records collected: **{len(all_rows)}**",
        f"- Unique symbols: **{len(df)}**",
        "- Minimal file columns: **exchange, ticker, description**",
        f"- Added vs prior snapshot: **{len(added)}**",
        f"- Removed vs prior snapshot: **{len(removed)}**",
        f"- Errors: **{len(errors)}**",
        "",
        "> Note: Yahoo discovery is best-effort and may not be a complete official listing for every exchange.",
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
