# Multi-Agent Workflow Flow Contract

This document defines the execution and interaction contract for the Supervisor-led autonomous machine learning team.

---

## 1. Core Lifecycle & Flow Contract

```mermaid
sequenceDiagram
    autonumber
    participant S as Supervisor
    participant W as Worker (Profile / EDA / FE / Model / Judge / Report)
    participant G as Graph Wrapper / Memory
    participant L as Ledger / Artifacts

    S->>S: Reads context (Mission, Plan, Best, Ledger, Notes, Concerns)
    S->>W: Issues specific brief {objective, focus_points, constraints, mode}
    W->>W: Reads brief + notebook lessons; executes specialist task
    W->>G: Returns WorkerReport {status, result_summary, evidence, concern?, suggestion, notebook, artifacts}
    G->>L: Updates ledger.jsonl, notebooks/, version snapshots, best.json
    G->>S: Advances step, clears guard note, sets open_concern if raised
    S->>S: Evaluates updated state, addresses concern, decides next action
    Note over S,W: Final report runs on BEST version, followed by finish
```

### Steps

1. **Supervisor decides**:
   - Inspects the unified context: Mission, dynamic Plan, Best model version, recent Ledger rows (including mean, std, paired noise floor, significance, delta), Advisory Notes from the system, and any Open Concern.
   - Emits a structured decision with `action`, `reason`, `plan_update`, and `brief` containing `{objective, focus_points, constraints, mode}`.
2. **Worker executes**:
   - Specialist worker (Profile, EDA, FE, Model, Judge, or Report) receives the brief and its own historical notebook lessons.
   - Executes deterministically or via Coder SubAgent.
   - Concludes report with `SKIPPED:` (what was deliberately omitted and why) and `NEXT:` (what to do next or "nothing").
   - Returns a structured `WorkerReport` with `{status, result_summary, evidence, concern?, suggestion, notebook, artifacts}`.
3. **Graph Wrapper processes**:
   - In a single centralized wrapper (`build_graph.py`):
     - Appends an immutable row to `runs/<run_id>/memory/ledger.jsonl`.
     - Appends specialist learnings/errors to `runs/<run_id>/memory/notebooks/<worker>.jsonl`.
     - Handles versioning snapshots (`versions/<version_id>/`).
     - Updates `best.json` when significant improvement is achieved.
     - Sets `open_concern` in state if raised with evidence by the worker.
4. **Supervisor reviews & iterates**:
   - Reads the updated ledger with any new advisory notes.
   - Resolves or overrules any open concern in its next `reason`.
   - Continues until the best model is identified and verified.
5. **Report & Finish**:
   - Report agent writes the comprehensive post-mortem report based on the verified best version and full ledger history.
   - Supervisor finishes the run.

---

## 2. Context & Prompt Size Budgets

Context budgets are strict prompt-size ceilings to prevent token explosion and maintain LLM focus; they are **NOT** run or execution step limits:
- **Supervisor Context:** $\le$ 8,000 characters.
- **Worker Brief:** $\le$ 6,000 characters.
- **Judge Context:** $\le$ 8,000 characters.

> Context formatting must never slice raw JSON by arbitrary characters. Context renderers deterministically summarize tables, collapse older ledger entries, and reference file paths on disk.

---

## 3. Worker Specialization & Modes

### Profile Agent
- Deterministically categorizes dataset columns: `datetime`, `row_counter` (sequential order), `id` (hash/UUID), `high_card`, `numeric`, `categorical`.
- Detects time signals, target distributions, and near-duplicate pairs.

### EDA Agent
- Answers specific decision-altering questions (split strategy, leakage, signal, domain hypotheses).
- Emits structured findings (`EDAFindingItem`) with quantitative evidence.

### Feature Engineering Agent
- **`mode="basic"`**: Deterministically generates preprocessor via `make_basic_preprocessor(profile)` without calling Coder LLM.
- **`mode="new"` / `mode="patch"`**: Coder SubAgent defines `make_feature_pipeline() -> Pipeline / ColumnTransformer` using safe transformers (`app.ml_harness.transformers`).
- **Contract Enforcement**: Never calls `train_test_split`; never transforms the full dataset; fits inside cross-validation folds.

### Model Agent
- **`mode="baseline"`**: Fast deterministic baseline (`Dummy`, `Logistic/Ridge`, `HistGradientBoosting`) inside CV folds via `cv_evaluate`.
- **`mode="new"` / `mode="tune"`**: Evaluates candidate models on the feature pipeline inside CV folds. Compares paired fold scores against the current best using `is_significant_gain`.
- If improvement exceeds the paired noise floor: promotes to best version.
- If within the noise floor: retains current best and returns `status="no_gain"`.

### Judge Agent
- Reviews overall pipeline integrity (EDA, FE, Model, Score).
- On failure or missing inputs, returns `overall="blocked"`. Never claims "ship" on failure.

### Report Agent
- Synthesizes findings, baseline comparison, final model performance with confidence intervals ($\text{mean} \pm \text{std}$), paired noise floor, and operational risks.
