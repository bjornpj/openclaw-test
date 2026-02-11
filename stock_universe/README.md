# Global Daily Ticker Pull (pandas + yfinance)

This setup builds a global master ticker file using:
- import pandas as pd
- import yfinance as yf

No paid API key required.

## Install dependencies

Use a Python environment with pip available:

```bash
cd stock_universe
python3 -m pip install -r requirements.txt
```

## Run

### Fast (discovery / smaller universe)
```bash
cd stock_universe
python3 fetch_global_tickers.py
```

### Larger universe (recommended)
Uses Yahoo screener pagination per exchange and usually returns thousands of symbols.

```bash
cd stock_universe
python3 fetch_global_tickers_full.py
```

## Output

Each run writes:

- `data/snapshots/YYYY-MM-DD/master_tickers.csv` (full columns)
- `data/snapshots/YYYY-MM-DD/master_tickers_minimal.csv` (exchange,ticker,description)
- `data/master_tickers_latest.csv` (rolling latest full)
- `data/master_tickers_latest_minimal.csv` (rolling latest minimal)
- `data/snapshots/YYYY-MM-DD/summary.json`
- `data/snapshots/YYYY-MM-DD/report.md`
- `data/snapshots/YYYY-MM-DD/raw/*.json`

## Collect historical prices for the global universe

After generating `data/master_tickers_latest.csv`, run:

```bash
cd stock_universe
python3 fetch_global_prices.py --years 2 --interval 1d
```

This writes:

- `data/prices/YYYY-MM-DD/global_prices.csv` (long table: date,ticker,exchange,OHLCV)
- `data/prices/YYYY-MM-DD/by_ticker/*.csv` (one file per symbol)
- `data/prices/YYYY-MM-DD/summary.json`

Useful options:

```bash
# Quick test (first 200 symbols)
python3 fetch_global_prices.py --limit 200

# Weekly bars, 5 years
python3 fetch_global_prices.py --years 5 --interval 1wk

# Smaller request chunks (helps with rate-limit/network issues)
python3 fetch_global_prices.py --batch-size 50 --pause-ms 500
```

## Tuning

Faster test run:

```bash
python fetch_global_tickers.py --regions "US" --max-prefixes 5
```

Higher coverage (slower):

```bash
python fetch_global_tickers.py --two-char-prefix
```

Custom regions:

```bash
python fetch_global_tickers.py --regions "US,CA,GB,DE,FR,JP,HK,IN,AU,BR"
```

For the larger screener-based pull:

```bash
# All configured major regions
python fetch_global_tickers_full.py

# Only US + Canada (quick sanity run)
python fetch_global_tickers_full.py --regions "US,CA"

# Add pacing if Yahoo starts rate-limiting
python fetch_global_tickers_full.py --pause-ms 400
```

## Daily cron (6:00 AM ET)

```cron
0 6 * * * cd /home/bjorn/.openclaw/workspace/stock_universe && /usr/bin/python3 fetch_global_tickers.py >> data/cron.log 2>&1
```

## Note

Yahoo discovery is best-effort and may still hit temporary rate-limits depending on host/network reputation.
