# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository overview
- This repo is a scaffolded Python project organized around an ML/decisioning pipeline. Source code is expected under `src/` with these high-level layers:
  - `src/app/`: orchestration or application entrypoints (CLI/API) to run pipelines or serve results
  - `src/ingest/`: data acquisition/connectors
  - `src/etl/`: batch transforms/cleaning
  - `src/features/`: feature engineering
  - `src/models/`: model training/inference
  - `src/risk_engine/`: decision/risk logic combining models and rules
  - `src/execution/`: workflow/pipeline definitions or schedulers
  - `src/utils/`: shared helpers
- `data/` holds data artifacts by lifecycle (`raw/`, `clean/`, `features/`, `models/`).
- `config/` is for project configuration.
- `notebooks/` contains exploratory notebooks (`experiments/` and per-member folders).
- Current state: code and configs are placeholders (mostly `.txt` files). `README.md` and `requirements.txt` are present but empty.

Quickstart (Python environment)
- Create and activate a virtualenv, then install dependencies (none pinned yet, but this sets up the environment):
  - macOS/Linux (zsh/bash)
    - python3 -m venv .venv
    - source .venv/bin/activate
    - pip install -r requirements.txt
  - Deactivate when finished: deactivate

Common commands
- Install dependencies: pip install -r requirements.txt
- Freeze currently installed packages (if you add some): pip freeze > requirements.txt
- Run a module or script (once implemented): python -m src.<package>.<module>
- Launch Jupyter (if you choose to use notebooks in this repo’s environment):
  - pip install jupyterlab
  - jupyter lab

Build, lint, and test
- Build: no build step is configured (pure Python layout without `pyproject.toml` or a build system).
- Lint/format: no linter/formatter configuration found (e.g., ruff/black/flake8 not configured).
- Tests: no test framework configured (e.g., pytest/unittest not present) and no test files detected.
  - Running a single test: not applicable until a test framework is added.

Working conventions (current repo)
- Source lives under `src/` and is organized by pipeline phase, not by framework.
- Data subfolders (`raw`, `clean`, `features`, `models`) mirror the ML data lifecycle to keep artifacts separated.
- Notebooks are organized by `experiments/` and per-member folders to isolate exploratory work.

Notes for future agents
- Before implementing features, decide on and add concrete tooling (e.g., pytest for tests, ruff/black for lint/format) and pin dependencies in `requirements.txt` (or migrate to `pyproject.toml`).
- If you add entrypoints (CLI or services), place them under `src/app/` and document the commands in `README.md` so they can be surfaced here.
