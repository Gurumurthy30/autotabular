# mlagent — Autonomous Multi-Agent Machine Learning Engineer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/llm-Ollama-purple.svg)](https://ollama.com)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-green.svg)](https://github.com)

**`mlagent`** is a production-grade, CLI multi-agent machine learning engineer built on **LangGraph**. It autonomously explores, codes, evaluates, diagnoses, tunes, and ensembles machine learning models for tabular datasets and Kaggle competitions.

Driven by a single local or remote LLM (default `gemma4:31b` or any compatible model via Ollama) and governed by a **deterministic controller**, `mlagent` prevents hallucinations, maintains strictly honest cross-validation folds, self-heals from runtime and memory errors, and generates competitive submission files.

---

## Architecture & How It Works

`mlagent` orchestrates 8 specialized agents around a shared, immutable experiment ledger:

```
                      ┌──────────────────────┐
                      │    START (Dataset)   │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │    PROFILER AGENT    │  (Schema, distributions, leak risks)
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │   STRATEGIST AGENT   │  (Hypothesis queue generation)
                      └──────────┬───────────┘
                                 │
                                 ▼
            ┌─────────► ┌──────────────────┐
            │           │   CONTROLLER     │ ◄─────────────────────────┐
            │           └────────┬─────────┘                           │
            │                    │ Batches ≤ workers                   │
            │                    ▼                                     │
            │           ┌──────────────────┐                           │
            │           │   EXPERIMENTER   │                           │
            │           │  & CODER AGENT   │                           │
            │           └────────┬─────────┘                           │
            │                    │                                     │
            │                    ├─► [Crashes]                         │
            │                    │     ├─► API Error ──► Docs Agent ───┤
            │                    │     ├─► OOM Error ──► Downgrade ────┤
            │                    │     └─► Logic Err ──► Analyzer ─────┤
            │                    ▼                                     │
            │           ┌──────────────────┐                           │
            │           │ VALIDATOR AGENT  │                           │
            │           └────────┬─────────┘                           │
            │                    │                                     │
            │                    ▼ (Every K runs)                      │
            │           ┌──────────────────┐                           │
            │           │  ANALYZER AGENT  ├───────────────────────────┘
            │           └──────────────────┘
            │
            │ (Budget / Target reached)
            ▼
┌──────────────────────┐
│     TUNER AGENT      │  (Optuna hyperparameter study on top performers)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    FINISHER AGENT    │  (Single / Greedy / Voting / Stacking on exact folds)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   KAGGLE LB CHECK    │  (Detect CV/LB drift, upload final submission)
└──────────┬───────────┘
           │
           ▼
         [ END ]
```

---

## Key Features

- **Autonomous End-to-End Pipeline**: From raw CSV to cross-validated feature engineering, model training, Optuna tuning, and stacking.
- **Strictly Honest Cross-Validation**: Fixed `folds.json` and `holdout.json` partition created up-front. All candidates, feature steps, and ensemble layers are evaluated strictly out-of-fold.
- **Self-Healing Execution Loop**:
  - **API / Syntax Errors**: Triggers the **Docs Agent** (local introspection first, allowlisted official doc search fallback).
  - **OOM / Resource Limits**: Applies automatic level downgrades (Level 1: lighter hyperparameters; Level 2: CPU fallback).
  - **Logic Errors**: The **Analyzer Agent** triages failure causes and dynamically queues counter-hypotheses.
- **Cross-Platform & Windows Native**: Full UTF-8 support across all file I/O and terminal outputs (`safe_box=True`, `sys.stdout.reconfigure`), eliminating Windows `cp1252` encoding pitfalls.
- **`<think>` Tag Stripping & Robust JSON Extraction**: Seamless compatibility with reasoning models (DeepSeek-R1, Gemma 4, Qwen).
- **Ensemble Engine**: Evaluates Single Best, Greedy Forward Selection, Voting, and Ridge Stacking. A more complex ensemble is rejected unless it demonstrably outperforms the best single model on identical rows.
- **Built-in System Diagnostics (`mlagent doctor`)**: Validates Python environment, dependencies, dataset schema, and live LLM round-trip prior to running.

---

## Quickstart

### 1. Installation

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
# Clone the repository
git clone https://github.com/Gurumurthy30/ML_agent.git
cd ML_agent

# Create and activate virtual environment
python -m venv .venv

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install mlagent in editable mode with boost & dev dependencies
pip install -e ".[boost,dev]"
```

### 2. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure your LLM provider. `mlagent` supports both local and cloud-hosted Ollama instances:

```env
# For Local Ollama (default: http://localhost:11434)
OLLAMA_MODEL=gemma4:31b

# For Remote / Cloud Ollama
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key_here
OLLAMA_MODEL=gemma4:31b
```

*(You can also override the model via `MLAGENT_MODEL=qwen2.5:32b` or `--model` CLI flag).*

### 3. Verify Setup with `mlagent doctor`

Ensure your environment, libraries, and LLM backend are healthy:

```bash
mlagent doctor path/to/dataset.csv --check-json
```

---

## Usage Guide

### One-Shot Autonomous Run

Run an end-to-end ML exploration on your dataset:

```bash
mlagent run data/train.csv --target survived --id-col passenger_id --yes --workers 2
```

Flags:
- `--target <column>`: Target variable to predict (inferred automatically if omitted).
- `--id-col <column>`: Row identifier column (excluded from training features).
- `--test <path>`: Optional test CSV for final submission predictions.
- `--workers <int>`: Concurrency level for parallel script execution (default: 1).
- `--yes`, `-y`: Non-interactive mode (auto-approves initial strategist plan).
- `--max-experiments <int>`: Cap on total experiment hypotheses to try.

### Resuming an Interrupted Run

If an experiment run was halted via `Ctrl+C` or a rate limit, resume exactly where it stopped:

```bash
mlagent resume data/train.csv
```

To re-open a completed workspace for additional exploration:

```bash
mlagent resume data/train.csv --more 5
```

### Kaggle Public Leaderboard Sync

Record a public leaderboard score to compute the CV/LB gap:

```bash
mlagent lb data/train.csv --score 0.785
```

Submit directly to a Kaggle competition via the Kaggle CLI:

```bash
mlagent lb data/train.csv --submit --kaggle titanic
```

---

## Workspace Directory Structure

All run artifacts, scripts, and logs are kept isolated next to your data:

```
data/
├── train.csv
└── mlagent_train/                     <-- Isolated workspace
    ├── config.yaml                    # Frozen configuration for this run
    ├── ledger.json                    # Complete ledger of all hypotheses & scores
    ├── control.json                   # Deterministic state machine checkpointer
    ├── folds.json                     # Fixed CV fold split indices
    ├── holdout.json                   # Fixed holdout split indices
    ├── mlkit.py                       # Standalone reproducibility library
    ├── report.md                      # Final Markdown summary & leaderboard
    ├── submission.csv                 # Final test predictions (if test provided)
    ├── scripts/                       # Executable Python scripts for each experiment
    │   ├── e001_baseline.py
    │   ├── e002_family_size.py
    │   └── t_e002_optuna.py
    ├── artifacts/                     # OOF predictions, models, metrics
    └── logs/                          # Full stdout/stderr logs per run
```

---

## Configuration Hierarchy

Configuration settings follow a strict precedence:

$$\text{CLI flags} > \text{Environment variables} > \text{Workspace config.yaml} > \text{Global mlagent.yaml}$$

To generate a sample global config:

```bash
mlagent config init
```

Available keys include:
- `model`: LLM model name (`gemma4:31b`, `qwen2.5:32b`, etc.).
- `seed`: Random seed for reproducibility.
- `budget.max_experiments`: Maximum number of experiments before stopping.
- `parallel.workers`: Concurrency limit for execution.
- `hardware.oom_downgrade`: Enable/disable automatic level degradation on memory failure.
- `finisher.methods`: Enabled ensemble algorithms (`["single", "greedy", "voting", "stacking"]`).

---

## Testing

The test suite runs with a mocked LLM driver and requires no external API credits:

```bash
# Run full test suite (23 tests: unit + end-to-end integration)
pytest -q tests/
```

Test coverage includes:
- Cross-platform UTF-8 file handling and Windows path resilience.
- `<think>` tag extraction and nested JSON resilience.
- `to_submission` edge cases and ID column fallback.
- `mlagent doctor` diagnostic checks.
- End-to-end LangGraph graph loops, Optuna tuning, and Ensembling.

---

## License

This project is licensed under the Apache 2.0 License.
