NASDAQ RL Agent (PyTorch)

What this is
- A PyTorch RL training pipeline for stock price-action decisions.
- Trains across many NASDAQ tickers (from your local NASDAQ ticker file or fallback list).
- Compares policy performance vs buy-and-hold and SPY benchmark.

Important
- No model can guarantee outperformance.
- This is a research framework, not investment advice.

Setup
1) Create venv and install deps
   cd rl_nasdaq_agent
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt

2) Train
   python train_rl_agent.py \
     --start 2016-01-01 \
     --end 2025-12-31 \
     --episodes 120 \
     --tickers-per-episode 64

Outputs
- artifacts/metrics.json
- artifacts/policy.pt
- artifacts/run_config.json

Ticker source
- Script first tries: ../data/nasdaq/nasdaq_tickers_YYYY-MM-DD.csv (latest date)
- Fallback: a built-in NASDAQ large-cap list

Design notes
- Action space: 0=short, 1=flat, 2=long
- Reward: next-day return * position minus turnover penalty
- State features: momentum, moving-average spread, volatility, RSI-like signal, volume z-score
- Policy: actor-critic MLP in PyTorch with clipped PPO-style objective
