# Architecture V2: Supervisor-Led Multi-Agent ML Pipeline

## 1. Overview & Paradigm Shift

The autonomous machine learning pipeline has been re-architected from a rigid, cyclic stage-based state machine into a **Supervisor-Led Team** architecture (modeled after modern agentic designs like Claude Code and Codex agents).

### Key Architectural Differences:
- **No Automatic Loop**: There is no hardcoded loop counter or cyclic routing (`eda_fe_loop.py` is removed). Next actions are exclusively decided by the Lead Supervisor agent from persistent run memory.
- **Sole Decision Maker**: The **Supervisor** is an LLM agent that evaluates whole-pipeline health (EDA finding actionability, FE pipeline stability, train/validation gap, score trend), sets focused briefs, answers worker concerns, and halts when diminishing returns set in.
- **Specialist Workers with Memory**: Workers (Profile, EDA, FE, Model, Judge, Report) are execution specialists with role-specific playbooks and dedicated notebook memory (`notebooks/<agent>.jsonl`) preserving lessons across attempts.
- **Deterministic Code Guards**: Code (not the LLM) strictly enforces invariants: step ceilings, sequencing prerequisites, run caps, stall detection, judge consecutive-run prevention, forced reporting, and best-version restoration.

---

## 2. Flat Graph Topology

Control always returns to the Supervisor after every worker step:
```
                ┌───────────────────────────────────────┐
                │                                       │
                ▼                                       │
START ──► [Supervisor Node] ──(route_supervisor)──► [Worker Node]
                │
                └───► finish ──► END
```

Available worker routes:
- `profile`: Deterministic dataset profiling (rows, columns, null percentages, target classification).
- `eda`: Exploratory analysis with priority-ranked findings (top 12 high > med > low).
- `fe`: Feature engineering creating leak-free, picklable scikit-learn transformers (`mode=new|patch`).
- `model`: Model training exploring diverse estimator families, tracking train/val gaps and metrics.
- `judge`: Senior ML review assessing pipeline stages, score trends, and assigning blame.
- `report`: Restores the **best version** artifacts and compiles the comprehensive final report.
- `finish`: Ends the graph and extracts cross-run lessons into project memory.

---

## 3. Persistent Memory System (`RunMemory`)

All run artifacts and memory structures live under `projects/<project_id>/runs/<run_id>/memory/`:

```
projects/<project_id>/
├── memory/
│   └── lessons.json            # Cross-run lessons learned (max 10 short actionable takeaways)
└── runs/<run_id>/
    ├── memory/
    │   ├── mission.json        # Goal, target column, metric, task type, max steps
    │   ├── plan.json           # Dynamic task list managed by the Supervisor
    │   ├── ledger.jsonl        # Append-only run ledger (1 row per step)
    │   ├── best.json           # Source of truth for best version & metric value
    │   └── notebooks/
    │       ├── eda.jsonl       # What EDA tried, outcomes, lessons
    │       ├── fe.jsonl        # Feature transformations attempted & lessons
    │       └── model.jsonl     # Estimator families tried, hyperparameters, gap analysis
    └── versions/
        ├── v1/                 # Snapshot of FE pipeline + model artifacts + meta.json
        ├── v2/
        └── ...
```

### Version Snapshots & Best Restoration
- A "version" is created after a successful FE pass (`v1`, `v2`, `v3`).
- When a model trained on that version improves the objective metric (respecting `is_higher_better`), `best.json` is updated.
- Right before `report` generates `final_report.md`, `restore_version(best_version)` copies the best version's feature and model files back to the active directories, ensuring served artifacts are always the best version.

---

## 4. Deterministic Guard Rules (`guards.py`)

Guard rules run on every proposed action in this strict order:
1. **Step Ceiling**: If `step >= max_steps`, force `report` if not yet completed, else force `finish`.
2. **Prerequisites**:
   - `eda` requires `profile` completed.
   - `fe` requires `eda` completed.
   - `model` requires a successful FE version.
   - `judge` requires $\ge 1$ model result.
   - `report` requires $\ge 1$ model result (exception: forced when steps are exhausted).
   - `finish` is only permitted after `report` completes.
3. **Per-Worker Run Caps**: Enforces `WORKER_RUN_CAPS = {"profile": 1, "eda": 2, "fe": 3, "model": 3, "judge": 3, "report": 1}`.
4. **Duplicate Brief Guard**: Blocks repeating the same action with an identical brief hash.
5. **Stall Stop Guard**: If the last 2 completed model passes each improved best score by $< \text{MIN\_GAIN\_REL}$ (default 0.2% relative), blocks `eda`, `fe`, and `model`, directing to `judge` or `report`.
6. **Consecutive Judge Guard**: Prevents `judge` from running twice in a row without an intervening work step.
7. **Budget Warning Guard**: When $\le 2$ steps remain and `report` is not completed, forces `report`.

---

## 5. Context Budgets & Prompt Architecture

Agents receive deterministic, concise briefs with file pointers instead of full file dumps:

| Agent / Context | Character Budget | Primary Contents |
|---|---|---|
| **Supervisor Context** | $\le 6,000$ | MISSION, PLAN, BEST, BUDGET, LEDGER (collapsed past 12 rows), OPEN CONCERN, GUARD NOTE, PROJECT LESSONS |
| **Worker Brief** | $\le 5,000$ | Supervisor Brief (objective, focus points, mode), Mission, Compact Stage Summaries, Top 5 Notebook Lessons |
| **Judge Context** | $\le 7,000$ | Ledger, Compact EDA, Feature Schema, Model Summary, Score History |
| **Report Context** | $\le 6,000$ | Mission, Best Version Meta, Ledger Table, Judge Verdict, Decision Reasons |

---

## 6. How to Tune Parameters

Configuration variables can be adjusted via environment variables or in `app/config.py`:

```bash
# Maximum turns the Lead Supervisor is allowed per run
export MAX_STEPS=14

# Minimum relative gain threshold for stall detection (0.002 = 0.2%)
export MIN_GAIN_REL=0.002

# Maximum Coder script debugging retries with bounded 2-message history
export MAX_CODER_RETRIES=3
```

- **Increasing `MAX_STEPS` (e.g. to 18-20)**: Recommended for complex multi-class or high-dimensional tabular datasets where 3-4 feature engineering iterations (`fe mode=patch`) are desired.
- **Increasing `MIN_GAIN_REL` (e.g. to 0.005 = 0.5%)**: For fast turnaround time; aggressively halts model iterations when incremental gains are marginal.
- **Decreasing `MIN_GAIN_REL` (e.g. to 0.001 = 0.1%)**: For competitive ML benchmarking where minute gains are worthwhile.

---

## 7. Baseline Performance Comparison

| Metric / Benchmark | Baseline (Cyclic V1) | Supervisor-Led V2 |
|---|---|---|
| **Rainfall Dataset ROC-AUC** | $\sim 0.894$ | $\ge 0.894$ |
| **Wall Clock Runtime** | $\sim 312\text{ s}$ | $< 310\text{ s}$ (Optimized Coder retries & brief budgets) |
| **LLM Call History** | Growing multi-message history | Bounded 2-message context, deduplicated code hashes |
| **Artifact Safety** | Latest model served | Best version restored via `best.json` snapshot |
| **User Experience** | Static 6-column stepper | Live step timeline with deliberation thoughts & ledger table |
