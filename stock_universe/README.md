# Global Daily Ticker Pull (Yahoo, No API Key)

This setup builds a global *master ticker file* using Yahoo Finance search endpoints (free, no key).

## Run

```bash
cd stock_universe
python3 fetch_global_tickers.py
```

## Output

Each run writes to:

- `data/snapshots/YYYY-MM-DD/master_tickers.csv` (daily master file)
- `data/master_tickers_latest.csv` (rolling latest master file)
- `data/snapshots/YYYY-MM-DD/summary.json`
- `data/snapshots/YYYY-MM-DD/report.md`
- `data/snapshots/YYYY-MM-DD/raw/*.json`

## Tuning

Faster test run:

```bash
python fetch_global_tickers.py --max-prefixes 10
```

Higher coverage (slower):

```bash
python fetch_global_tickers.py --two-char-prefix
```

Custom regions:

```bash
python fetch_global_tickers.py --regions "US,CA,GB,DE,FR,JP,HK,IN,AU,BR"
```

## Daily cron (6:00 AM ET)

```cron
0 6 * * * cd /home/bjorn/.openclaw/workspace/stock_universe && /usr/bin/python3 fetch_global_tickers.py >> data/cron.log 2>&1
```

## Note

Yahoo discovery is best-effort (search-index based), not a guaranteed official complete listing for every exchange.
