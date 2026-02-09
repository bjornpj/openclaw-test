# Global Daily Ticker Pull

This setup pulls ticker symbol metadata daily from major global exchanges and writes dated snapshots.

## 1) Install

```bash
cd stock_universe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure API key

```bash
cp .env.example .env
# edit .env and set EODHD_API_KEY
```

Then run with env loaded:

```bash
set -a; source .env; set +a
python fetch_global_tickers.py
```

## 3) Output

Each run writes to:

- `data/snapshots/YYYY-MM-DD/master_tickers.csv` (master file for all fetched exchanges)
- `data/master_tickers_latest.csv` (rolling latest master file)
- `data/snapshots/YYYY-MM-DD/normalized_symbols.csv` (legacy compatibility copy)
- `data/snapshots/YYYY-MM-DD/summary.json`
- `data/snapshots/YYYY-MM-DD/report.md`
- `data/snapshots/YYYY-MM-DD/raw/*.json`

## 4) Schedule daily at 6:00 AM ET (system cron)

```cron
0 6 * * * cd /home/bjorn/.openclaw/workspace/stock_universe && /usr/bin/bash -lc 'set -a; source .env; set +a; ./.venv/bin/python fetch_global_tickers.py >> data/cron.log 2>&1'
```

## Notes

- By default it fetches a pragmatic major-exchange list.
- Use all exchanges when needed:

```bash
python fetch_global_tickers.py --all-exchanges
```

- Or custom list:

```bash
python fetch_global_tickers.py --exchanges "US,NYSE,NASDAQ,LSE,HK,T"
```
