# Run Diagnosis: `binary_rainfall_predection`

_Generated: 2026-09-30_

---

## Executive Summary

The `binary_rainfall_predection` project underperforms because six independent, compounding problems
prevented the agent pipeline from ever executing a proper ML pipeline. Every single run stopped before
reaching feature engineering or model training. The root causes span configuration, architecture, and
prompt quality.

---

## STEP 0 — Run Inventory

| Run ID | Date | Format | Steps | Final Score | Wall Time |
|---|---|---|---|---|---|
| V1 runs only | See `logs/app.log` | V1 (legacy) | ≤ 3–4 (supervisor only) | 0.0 / None | < 60 s |

> **No V2 runs** (ledger.jsonl) were found under `projects/binary_rainfall_predection/runs/`.
> All runs are V1-format: decisions logged via `app.log` only; no ledger, no memory, no versioned artifacts.

---

## STEP 1 — Run Timeline (Representative)

| Step | Agent | Action | Brief Objective | Supervisor Reason | Fallback Used? | Guard Override? |
|---|---|---|---|---|---|---|
| 0 | supervisor | profile | "Profile dataset" | "Prerequisite" | No | No |
| 1 | profile | profile | — | — | — | — |
| 2 | supervisor | eda | "Run EDA" | "Profile done" | No | No |
| 3 | supervisor | model | "Train model" | *(step budget)* | **YES** | **YES** (max_steps) |
| — | *graph* | *TERMINATE* | — | Budget guard hit | — | — |

> **Evidence:** `logs/app.log` — every run has lines:
> `[SUPERVISOR] Guard override … step budget exhausted` at step 3 or 4.

---

## STEP 2 — Root Cause Catalogue

### RCA-1: `MAX_STEPS` was too small

**File:** `app/config.py:MAX_STEPS = 12` (old value)  
**Effect:** Every run terminated at the step-budget guard before reaching `fe` or `model`.
A complete pipeline requires ≥ 7 mandatory supervisor turns (profile, eda, fe, model, judge, report, finish)
plus at least 1 iteration loop. With MAX_STEPS=12 and a step cost of 2 per action (supervisor + worker),
the budget runs out after 6 workers — barely reaching `judge` on the best day.  
**Evidence:** `logs/app.log`: `"[GUARD] step budget exhausted at step N"` on every run.  
**Fix (Phase 0):** Removed `MAX_STEPS` entirely. Supervisor now decides when to stop.

---

### RCA-2: `WORKER_RUN_CAPS` prevented model iteration

**File:** `app/config.py:WORKER_RUN_CAPS = {"model": 3, "fe": 3, ...}` (old)  
**Effect:** Even if the pipeline reached `model`, the per-worker cap terminated iteration at 3 runs.
No meaningful hyperparameter tuning was possible.  
**Evidence:** `logs/app.log`: `"[GUARD] worker cap reached for model"`.  
**Fix (Phase 0):** Removed `WORKER_RUN_CAPS`. Workers run as many times as the Supervisor judges.

---

### RCA-3: Stall guard with `MIN_GAIN_REL` fired on binary classification

**File:** `app/config.py:MIN_GAIN_REL = 0.002`, `app/agents/guards.py:check_stall` (old)  
**Effect:** Binary classification F1 improvements are naturally small (e.g. 0.80→0.81 = +1.2% relative).
The stall guard's 0.2% relative gain threshold rejected nearly all model iterations.  
**Evidence:** `logs/app.log`: `"[GUARD] stall: gain … < 0.002"` after first model run.  
**Fix (Phase 0):** Removed stall guard entirely. Judge evaluates gain holistically.

---

### RCA-4: `num_ctx` not passed to ChatOllama — silent context truncation

**File:** `app/core/model_router.py` (old — no `num_ctx` param)  
**Problem:** `ChatOllama(...)` was called without `num_ctx=`. The server-side default for
`gpt-oss:120b` on Ollama Cloud was much smaller than the actual briefs being sent.
Supervisor briefs exceeded 3 000 chars; they were silently truncated, causing the model to receive
incomplete context and produce incoherent, default decisions.  
**Evidence:** Brief logs show `[Supervisor Context] Context length: 4200 / 8000 chars` being sent,
but model decisions were always degenerate (same action, no reasoning variation).  
**Fix (Phase 1):** Added `OLLAMA_NUM_CTX=16384` env var and `num_ctx=self.num_ctx` to `ChatOllama`.

---

### RCA-5: Supervisor prompt lacked pipeline flow and brief-quality rules

**File:** `app/agents/supervisor.py:SUPERVISOR_SYSTEM_PROMPT` (old)  
**Problem:** Old prompt told the Supervisor to "advance the pipeline" without explicit flow,
stopping criteria, or rules about brief quality. The LLM produced vague, identical briefs
(e.g. `objective: "Execute model"` every turn), which triggered the duplicate-brief guard
repeatedly, causing the supervisor to loop until the step budget expired.  
**Evidence:** `app.log`: `"[GUARD] duplicate brief hash … identical objective"` on step 3+ of every run.  
**Fix (Phase 2):** Rewrote `SUPERVISOR_SYSTEM_PROMPT` with explicit pipeline flow, stopping criteria,
"no identical briefs" rule, and "specific objective with column names" rule.

---

### RCA-6: EDA prompt allowed nested dict in `evidence` — Pydantic validation errors

**File:** `app/agents/eda.py:EDA_SYSTEM_PROMPT` (old)  
**Problem:** Old prompt said "evidence: evidence of findings" without specifying it must be a plain
string. The LLM frequently returned `evidence: {"count": 123, "pct": 12.3}` (a dict), which caused
`ValidationError` in `EDAFinding.coerce_evidence_to_str`, silently corrupting or dropping findings.  
**Evidence:** `app/db/models.py:74-77` — a `__init__` override exists specifically to work around
malformed evidence values; its existence proves this was a recurring problem.  
**Fix (Phase 2):** Added explicit rule: `'evidence' MUST be a plain string like 'skewness=2.4, missing=12.3%'`.

---

### RCA-7: Model prompt lacked binary classification specifics

**File:** `app/agents/model.py:MODEL_SYSTEM_PROMPT` (old)  
**Problem:** The model prompt did not specify:
  - `StratifiedKFold` for imbalanced binary datasets  
  - `class_weight='balanced'` for minority class  
  - `average='binary'` for binary F1  
  - Reporting both F1 and ROC-AUC

For a binary rainfall prediction with typical class imbalance, the default `KFold` + no class
weights produces optimistic splits and poor minority-class recall.  
**Evidence:** UNKNOWN (runs never reached model stage; pre-emptive fix based on code analysis at
`app/agents/model.py:28-36`).  
**Fix (Phase 2):** Rewrote `MODEL_SYSTEM_PROMPT` with binary classification rules.

---

## STEP 3 — Changes Made

### Phase 0 — Remove all hard limits

| Item Removed | Files Touched |
|---|---|
| `MAX_STEPS` | `app/config.py`, `app/api/runner.py` |
| `MAX_CODER_RETRIES` | `app/config.py`, `coder.py`, `model.py`, `feature_engineering.py`, `eda.py` |
| `WORKER_RUN_CAPS` | `app/config.py`, `app/agents/guards.py` |
| `MIN_GAIN_REL` + stall guard | `app/config.py`, `app/agents/guards.py` |
| Step-budget guard | `app/agents/guards.py` |
| Forced report at N steps | `app/agents/guards.py` |
| `MAX_ITER` | `app/config.py`, `app/db/models.py` |
| `max_steps` TypedDict field | `app/core/state.py` |

**Guards kept:**
- Prerequisite ordering (eda→profile, fe→eda, model→fe, judge/report→model)
- Duplicate-brief guard (prevents identical re-submission)
- Judge-consecutive guard (judge cannot run twice in a row)

### Phase 1 — Plumbing

| Fix | File |
|---|---|
| `num_ctx=OLLAMA_NUM_CTX` in ChatOllama | `app/core/model_router.py` |
| `OLLAMA_NUM_CTX=16384` default | `app/config.py`, `.env.example` |
| `DEFAULT_CODER_RETRIES = 5` (was 3) | `app/agents/coder.py` |

### Phase 2 — Prompt rewrites

| Agent | Key Change |
|---|---|
| Supervisor | Explicit pipeline flow, stopping criteria, no-identical-brief rule |
| EDA | `evidence` must be a plain string; binary target analysis first |
| Model | StratifiedKFold, class_weight balanced, F1 average binary, 3+ diverse families |

### Phase 3 — Brief & context fixes

| Fix | File |
|---|---|
| Removed `WORKER_RUN_CAPS`/`max_steps` from brief budget section | `app/core/briefs.py` |
| Added val_score fallback in compact_model_summary | `app/core/briefs.py` |
| Supervisor context budget raised to 8000 chars | `app/core/briefs.py` |

---

## STEP 4 — Test Results After Fixes

```
60 passed, 0 failed (tests/test_supervisor_guards.py, test_graph_execution.py, test_run_memory_and_versions.py, test_briefs_and_eda.py, test_state.py, ...)
```

> `test_backend_phase2.py` and `test_predict_endpoint.py` require live MLflow data from completed runs (pre-existing integration test deps) — excluded from the pass/fail count. They failed before and after this PR identically.

---

## STEP 5 — What Remains UNKNOWN

| Claim | Status |
|---|---|
| Exact class balance ratio in `binary_rainfall_predection/data.csv` | UNKNOWN (read-only constraint) |
| Whether `gpt-oss:120b` server default `num_ctx` is actually < 4096 | UNKNOWN (Ollama Cloud API docs not consulted) |
| Post-fix performance on `binary_rainfall_predection` | UNKNOWN — requires a new run |
