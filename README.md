# openclaw-test

Workspace repository for OpenClaw setup/testing, memory files, and small Python projects (especially around stock universe data and RL experiments).

## Repository Contents

### Core OpenClaw Workspace Files
- `AGENTS.md` — workspace behavior, safety, memory, and heartbeat conventions.
- `SOUL.md` — assistant tone/persona guidance.
- `USER.md` — user profile notes.
- `TOOLS.md` — local environment/tooling notes.
- `IDENTITY.md` — assistant identity metadata.
- `HEARTBEAT.md` — heartbeat task checklist file.
- `BOOTSTRAP.md` — initial first-run bootstrap instructions.

### Memory
- `memory/` — dated memory logs.
  - `memory/2026-02-09.md` — current daily memory note in repo.

### Market / Trading Data Utilities
- `fetch_nasdaq_tickers.py` — script for fetching/storing NASDAQ ticker data.
- `data/nasdaq/` — NASDAQ-related data output directory.

### RL Project
- `rl_nasdaq_agent/` — reinforcement-learning experiment scaffold.
  - `train_rl_agent.py` — training script.
  - `requirements.txt` — Python dependencies for the RL project.
  - `README.md` — project-specific setup/usage notes.
  - `.venv/` — local Python virtual environment.

### Stock Universe Project
- `stock_universe/` — scripts/data for broader ticker universe collection.
  - `fetch_global_tickers.py` — global ticker fetcher script.
  - `requirements.txt` — Python dependencies.
  - `.env.example` — environment variable template.
  - `README.md` — project-specific usage notes.
  - `data/` — generated/output data files.

### Skill Development
- `skills/` — local skill folders.
  - `skills/mercedes-w204-mechanic/` — W204-focused automotive troubleshooting skill.

### Misc Test Folder
- `github-setup-test/`
  - `hello.py` — minimal test script.

## Notes
- This repo appears to be a mixed personal workspace (assistant config + experiments), not a single-purpose application.
- Generated artifacts (`.venv`, `__pycache__`, data outputs) may be environment-specific and may not be required for all users.
