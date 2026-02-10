#!/usr/bin/env python3
import argparse
import dataclasses
import datetime as dt
import glob
import json
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def latest_nasdaq_csv(base_dir: str = "../data/nasdaq") -> str | None:
    files = glob.glob(os.path.join(base_dir, "nasdaq_tickers_*.csv"))
    return max(files) if files else None


def load_tickers() -> List[str]:
    p = latest_nasdaq_csv()
    if p and os.path.exists(p):
        df = pd.read_csv(p)
        col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
        if col:
            vals = df[col].dropna().astype(str).str.strip().unique().tolist()
            vals = [v for v in vals if v and "^" not in v and "/" not in v]
            if len(vals) > 100:
                return vals
    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","COST","NFLX",
        "AMD","INTC","QCOM","ADBE","CSCO","PEP","AMGN","TXN","INTU","BKNG",
        "MU","PANW","ADP","LRCX","KLAC","MRVL","MELI","SNPS","CDNS","ASML",
    ]


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in data.columns:
                s = data[(t, "Close")].rename(t)
                frames.append(s)
    else:
        if "Close" in data.columns and len(tickers) == 1:
            frames.append(data["Close"].rename(tickers[0]))
    if not frames:
        raise RuntimeError("No price data downloaded.")
    px = pd.concat(frames, axis=1).sort_index().ffill().dropna(axis=1, thresh=100)
    return px


def feature_engineering(px: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ret1 = px.pct_change()
    mom5 = px.pct_change(5)
    mom20 = px.pct_change(20)
    ma10 = px.rolling(10).mean()
    ma30 = px.rolling(30).mean()
    ma_spread = (ma10 / ma30) - 1.0
    vol20 = ret1.rolling(20).std()
    rsi_num = ret1.clip(lower=0).rolling(14).mean()
    rsi_den = (-ret1.clip(upper=0)).rolling(14).mean() + 1e-12
    rsi = 100 - (100 / (1 + (rsi_num / rsi_den)))

    feats = {
        "ret1": ret1,
        "mom5": mom5,
        "mom20": mom20,
        "ma_spread": ma_spread,
        "vol20": vol20,
        "rsi": (rsi / 100.0) - 0.5,
    }
    return feats


def build_panel(px: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    feats = feature_engineering(px)
    common = px.index
    for f in feats.values():
        common = common.intersection(f.index)
    common = common.sort_values()

    X_list = []
    Y_list = []
    valid_dates = []

    tickers = list(px.columns)
    for i in range(len(common) - 1):
        d = common[i]
        d_next = common[i + 1]
        rows = []
        y = []
        for t in tickers:
            vals = [feats[k].at[d, t] if d in feats[k].index and t in feats[k].columns else np.nan for k in feats.keys()]
            if any(pd.isna(v) for v in vals):
                vals = [0.0 if pd.isna(v) else float(v) for v in vals]
            rows.append(vals)
            r = (px.at[d_next, t] / px.at[d, t]) - 1.0 if (pd.notna(px.at[d_next, t]) and pd.notna(px.at[d, t]) and px.at[d, t] != 0) else 0.0
            y.append(float(r))
        X_list.append(np.array(rows, dtype=np.float32))
        Y_list.append(np.array(y, dtype=np.float32))
        valid_dates.append(d)

    X = np.stack(X_list, axis=0)  # [T, N, F]
    Y = np.stack(Y_list, axis=0)  # [T, N]
    return X, Y, valid_dates, tickers


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, n_actions: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.actor = nn.Linear(hid, n_actions)
        self.critic = nn.Linear(hid, 1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


@dataclasses.dataclass
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logp_old: torch.Tensor
    returns: torch.Tensor
    adv: torch.Tensor


def collect_rollout(model: PolicyNet, X: np.ndarray, Y: np.ndarray, tick_idx: np.ndarray, device: str, turnover_penalty: float = 0.0005):
    model.eval()
    T = X.shape[0]
    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []

    prev_pos = np.zeros(len(tick_idx), dtype=np.int64) + 1  # start flat(1)

    for t in range(T):
        obs_np = X[t, tick_idx, :]  # [M,F]
        ret_np = Y[t, tick_idx]     # [M]

        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        logits, values = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)

        pos = actions.detach().cpu().numpy() - 1  # map {0,1,2}->{-1,0,1}
        turn = (actions.detach().cpu().numpy() != prev_pos).astype(np.float32)
        reward = pos * ret_np - turnover_penalty * turn
        prev_pos = actions.detach().cpu().numpy()

        obs_buf.append(obs.detach().cpu())
        act_buf.append(actions.detach().cpu())
        logp_buf.append(logp.detach().cpu())
        rew_buf.append(torch.tensor(reward, dtype=torch.float32))
        val_buf.append(values.detach().cpu())

    rewards = torch.stack(rew_buf)   # [T,M]
    values = torch.stack(val_buf)    # [T,M]

    gamma, lam = 0.99, 0.95
    Tn, Mn = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(Mn)
    next_value = torch.zeros(Mn)
    for t in reversed(range(Tn)):
        delta = rewards[t] + gamma * next_value - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
        next_value = values[t]
    returns = adv + values

    batch = TransitionBatch(
        obs=torch.cat(obs_buf, dim=0),
        actions=torch.cat(act_buf, dim=0),
        logp_old=torch.cat(logp_buf, dim=0),
        returns=returns.reshape(-1),
        adv=adv.reshape(-1),
    )
    return batch


def ppo_update(model, optimizer, batch: TransitionBatch, device: str, epochs=4, minibatch=2048, clip=0.2, vf_coef=0.5, ent_coef=0.01):
    model.train()
    obs = batch.obs.to(device)
    actions = batch.actions.to(device)
    logp_old = batch.logp_old.to(device)
    returns = batch.returns.to(device)
    adv = batch.adv.to(device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = torch.arange(obs.size(0), device=device)
    for _ in range(epochs):
        perm = idx[torch.randperm(idx.numel())]
        for i in range(0, perm.numel(), minibatch):
            b = perm[i:i + minibatch]
            logits, values = model(obs[b])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions[b])
            ratio = torch.exp(logp - logp_old[b])

            surr1 = ratio * adv[b]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((returns[b] - values) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate(model: PolicyNet, X: np.ndarray, Y: np.ndarray, tick_idx: np.ndarray, device: str):
    model.eval()
    T = X.shape[0]
    strat_daily = []
    bh_daily = []

    with torch.no_grad():
        for t in range(T):
            obs_np = X[t, tick_idx, :]
            ret_np = Y[t, tick_idx]
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            logits, _ = model(obs)
            actions = torch.argmax(logits, dim=-1).cpu().numpy() - 1  # {-1,0,1}
            strat_daily.append(float(np.mean(actions * ret_np)))
            bh_daily.append(float(np.mean(ret_np)))

    def stats(r):
        r = np.array(r, dtype=np.float64)
        ann_ret = (1 + r.mean()) ** 252 - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-12)
        eq = np.cumprod(1 + r)
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak) - 1
        max_dd = float(dd.min()) if len(dd) else 0.0
        return {
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "final_equity": float(eq[-1]) if len(eq) else 1.0,
        }

    return stats(strat_daily), stats(bh_daily)


def spy_benchmark(start: str, end: str):
    try:
        s = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"].dropna()
        r = s.pct_change().dropna().values
        ann_ret = (1 + r.mean()) ** 252 - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-12)
        eq = np.cumprod(1 + r)
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak) - 1
        return {
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(dd.min()) if len(dd) else 0.0,
            "final_equity": float(eq[-1]) if len(eq) else 1.0,
        }
    except Exception:
        return {"error": "SPY benchmark fetch failed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tickers-per-episode", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs("artifacts", exist_ok=True)

    tickers = load_tickers()
    px = download_prices(tickers, args.start, args.end)
    X, Y, dates, cols = build_panel(px)

    if X.shape[0] < 50 or X.shape[1] < 10:
        raise RuntimeError("Insufficient panel size for training.")

    model = PolicyNet(in_dim=X.shape[2]).to(args.device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    n_tickers = X.shape[1]
    m = min(args.tickers_per_episode, n_tickers)

    for ep in range(args.episodes):
        idx = np.random.choice(np.arange(n_tickers), size=m, replace=False)
        batch = collect_rollout(model, X, Y, idx, args.device)
        ppo_update(model, opt, batch, args.device)
        if (ep + 1) % 10 == 0:
            s, b = evaluate(model, X, Y, idx, args.device)
            print(f"episode={ep+1} strat_sharpe={s['sharpe']:.3f} bh_sharpe={b['sharpe']:.3f}")

    full_idx = np.arange(n_tickers)
    strat, bh = evaluate(model, X, Y, full_idx, args.device)
    spy = spy_benchmark(args.start, args.end)

    metrics = {
        "period": {"start": args.start, "end": args.end},
        "n_tickers": int(n_tickers),
        "n_days": int(X.shape[0]),
        "strategy": strat,
        "buy_hold_equal_weight": bh,
        "spy": spy,
        "note": "Research only. No guarantee of outperformance.",
    }

    torch.save(model.state_dict(), "artifacts/policy.pt")
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("artifacts/run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
