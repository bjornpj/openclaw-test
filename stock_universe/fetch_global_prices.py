#!/usr/bin/env python3
"""Fetch historical price data for a global ticker universe using yfinance.

Typical workflow:
1) Build/refresh the global ticker universe:
   python fetch_global_tickers.py
2) Pull price history for that universe:
   python fetch_global_prices.py --years 2 --interval 1d

Outputs are written under stock_universe/data/prices/<run_day>/
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


OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def load_tickers(ticker_file: Path, limit: int = 0) -> pd.DataFrame:
    if not ticker_file.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_file}")

    df = pd.read_csv(ticker_file)
    required = {"exchange", "ticker"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Ticker file missing required columns {required}. Got: {list(df.columns)}"
        )

    out = df[["exchange", "ticker"]].dropna().copy()
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["exchange"] = out["exchange"].astype(str).str.strip()
    out = out[out["ticker"] != ""].drop_duplicates(subset=["exchange", "ticker"]) \
        .sort_values(["exchange", "ticker"], kind="stable")

    if limit and limit > 0:
        out = out.head(limit)

    out = out.reset_index(drop=True)
    return out


def _flatten_download(df: pd.DataFrame, requested_tickers: List[str]) -> pd.DataFrame:
    """Normalize yf.download output into long format:

    date, ticker, Open, High, Low, Close, Adj Close, Volume
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "ticker", *OHLCV_COLS])

    # yfinance may return either:
    # - single-level columns (single ticker)
    # - MultiIndex columns where one level is fields and the other is tickers
    if isinstance(df.columns, pd.MultiIndex):
        level_0_vals = set(str(v) for v in df.columns.get_level_values(0))
        level_1_vals = set(str(v) for v in df.columns.get_level_values(1))

        if any(c in level_0_vals for c in OHLCV_COLS):
            # columns like (field, ticker)
            out = (
                df.stack(level=1, future_stack=True)
                .rename_axis(index=["date", "ticker"])
                .reset_index()
            )
        elif any(c in level_1_vals for c in OHLCV_COLS):
            # columns like (ticker, field)
            out = (
                df.stack(level=0, future_stack=True)
                .rename_axis(index=["date", "ticker"])
                .reset_index()
            )
        else:
            # Fallback path
            out = (
                df.stack(future_stack=True)
                .reset_index()
                .rename(columns={"level_0": "date", "level_1": "ticker"})
            )
    else:
        # Single ticker case.
        ticker = requested_tickers[0] if requested_tickers else "UNKNOWN"
        out = df.copy().reset_index()
        date_col = "Date" if "Date" in out.columns else out.columns[0]
        out = out.rename(columns={date_col: "date"})
        out["ticker"] = ticker

    # Keep expected columns where present.
    keep = [c for c in ["date", "ticker", *OHLCV_COLS] if c in out.columns]
    out = out[keep].copy()

    # Ensure all OHLCV columns exist for stable downstream processing.
    for col in OHLCV_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    # Normalize date type.
    out["date"] = pd.to_datetime(out["date"], utc=False).dt.tz_localize(None)

    return out[["date", "ticker", *OHLCV_COLS]]


def download_prices_batch(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=False,
        threads=True,
    )
    return _flatten_download(raw, requested_tickers=tickers)


def chunked(items: List[str], n: int) -> List[List[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect historical price data for a global ticker universe with yfinance"
    )
    parser.add_argument(
        "--tickers-file",
        default="stock_universe/data/master_tickers_latest.csv",
        help="CSV containing at least exchange,ticker columns",
    )
    parser.add_argument("--years", type=int, default=2, help="How many years of history to fetch")
    parser.add_argument("--interval", default="1d", help="yfinance interval (1d, 1wk, 1h, etc.)")
    parser.add_argument("--batch-size", type=int, default=100, help="Tickers per download request")
    parser.add_argument("--pause-ms", type=int, default=300, help="Pause between batch requests")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for quick testing")
    parser.add_argument(
        "--retries", type=int, default=2, help="Retries per failed batch"
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Use adjusted prices from yfinance",
    )
    parser.add_argument(
        "--out",
        default="stock_universe/data/prices",
        help="Output root folder",
    )
    args = parser.parse_args()

    tickers_file = Path(args.tickers_file)
    ticker_df = load_tickers(tickers_file, limit=args.limit)

    tickers = ticker_df["ticker"].tolist()
    exchange_map: Dict[str, str] = dict(zip(ticker_df["ticker"], ticker_df["exchange"]))

    end_dt = dt.datetime.now(dt.UTC)
    start_dt = end_dt - dt.timedelta(days=365 * max(1, args.years))
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    run_day = end_dt.strftime("%Y-%m-%d")
    out_root = Path(args.out) / run_day
    out_root.mkdir(parents=True, exist_ok=True)

    all_prices: List[pd.DataFrame] = []
    errors: List[Dict[str, str]] = []

    for idx, batch in enumerate(chunked(tickers, max(1, args.batch_size)), start=1):
        batch_ok = False
        last_error = None

        for attempt in range(1, args.retries + 2):
            try:
                print(f"[{idx}] Fetching batch size={len(batch)} attempt={attempt}")
                batch_df = download_prices_batch(
                    tickers=batch,
                    start_date=start_date,
                    end_date=end_date,
                    interval=args.interval,
                    auto_adjust=args.auto_adjust,
                )

                if not batch_df.empty:
                    all_prices.append(batch_df)
                batch_ok = True
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(min(3 * attempt, 10))

        if not batch_ok:
            errors.append(
                {
                    "batch_index": str(idx),
                    "size": str(len(batch)),
                    "first_ticker": batch[0] if batch else "",
                    "last_ticker": batch[-1] if batch else "",
                    "error": last_error or "unknown",
                }
            )

        time.sleep(max(0, args.pause_ms) / 1000.0)

    if all_prices:
        prices = pd.concat(all_prices, ignore_index=True)
    else:
        prices = pd.DataFrame(columns=["date", "ticker", *OHLCV_COLS])

    # Attach exchange, sort, dedupe.
    prices["exchange"] = prices["ticker"].map(exchange_map).fillna("")
    prices = (
        prices.drop_duplicates(subset=["date", "ticker"], keep="last")
        .sort_values(["ticker", "date"], kind="stable")
        .reset_index(drop=True)
    )

    # Save long table.
    prices_csv = out_root / "global_prices.csv"
    prices.to_csv(prices_csv, index=False)

    # Save per-ticker files (helpful for model training pipelines).
    per_ticker_dir = out_root / "by_ticker"
    per_ticker_dir.mkdir(parents=True, exist_ok=True)

    non_empty_tickers = 0
    for ticker, g in prices.groupby("ticker", sort=True):
        if g.empty:
            continue
        non_empty_tickers += 1
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        g.to_csv(per_ticker_dir / f"{safe_ticker}.csv", index=False)

    summary = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "tickers_file": str(tickers_file),
        "requested_tickers": int(len(tickers)),
        "tickers_with_data": int(non_empty_tickers),
        "rows": int(len(prices)),
        "start_date": start_date,
        "end_date": end_date,
        "years": int(args.years),
        "interval": args.interval,
        "batch_size": int(args.batch_size),
        "errors": errors,
        "global_prices_csv": str(prices_csv),
        "per_ticker_dir": str(per_ticker_dir),
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
