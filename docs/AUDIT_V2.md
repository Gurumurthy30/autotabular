# Master Audit Report: Architecture V2 Alignment

## 1. Summary

- **Totals:** 20 Findings (5 P0, 7 P1, 5 P2, 3 P3) | 6 Items on DELETE LIST | 11 Items to FIX | 1 FIXED (`init_mission`) | 1 Item UNKNOWN (E2E live run benchmark numbers pending LLM run).
- **Top 5 Critical Risks:**
  1. **Deterministic Guard Deadlock (P0):** `guards.py` causes an infinite loop between `fe` and `model` fallback when `fe` reaches its cap without a valid version.
  2. **Uncaught Worker Exceptions (P0):** `build_graph.py` worker wrapper lacks `try...except`, causing unexpected agent exceptions to crash the entire LangGraph execution stream.
  3. **Concurrent Run File Collisions (P0):** Workers write active parquet/pipeline/model files to shared project directories (`projects/<project_id>/features/`) rather than run-isolated working folders.
  4. **Stall Guard Premature Trigger (P1):** `guards.py` checks relative gain after a single model run instead of requiring 2 passes, halting prematurely.
  5. **Incomplete Project Purge on UI Delete (P1):** Clicking Delete Project in the UI leaves MLflow run files on disk, skips locked files silently, and fails to abort in-flight background execution tasks.

---

## 2. Findings Table

| ID | Severity | Area | file:line | Problem | Evidence | Proposed Fix | Action |
|---|---|---|---|---|---|---|---|
| F-01 | **P0** | Backend | `app/agents/guards.py:122`, `66-83` | Fallback deadlock when FE cap reached without version | When `fe` reaches run cap (3) and `model` is proposed without version, fallback picks `model`, which immediately redirects to `fe`, looping infinitely. | In fallback, check prerequisites; if `fe` is capped and no version exists, redirect to `report` or `finish`. | FIX |
| F-02 | **P0** | Backend | `app/graph/build_graph.py:41-45` | Uncaught exceptions inside worker node crash the graph stream | `fn(state, router, registry, brief=brief)` is not wrapped in `try...except`. Unexpected worker errors terminate the pipeline instead of logging a failed WorkerReport. | Wrap worker invocation in `try...except Exception as exc`, record `status="failed"`, and append error to notebook. | FIX |
| F-03 | **P0** | Backend | `app/agents/feature_engineering.py:248,326`, `app/core/versions.py:26,36` | Concurrent run filesystem collisions on shared project directories | FE and Model agents write directly to `projects/<project_id>/features/` and `models/`. Concurrent runs overwrite each other's in-flight artifacts. | Scope active working directories under `projects/<project_id>/runs/<run_id>/workspace/`; copy to project root only upon `restore_version`. | FIX |
| F-04 | **P0** | API | `app/api/routes.py:937-955` | Predict endpoint loads candidate models from unversioned directory | `/models/{experiment_id}/predict` falls back to `projects/<project_id>/models/*.pkl` without verifying version tag, risking serving non-best model. | Resolve model file directly from `best.json` version snapshot path (`versions/<best_version>/models/`). | FIX |
| F-05 | **P1** | Backend | `app/agents/guards.py:59-60` | Stall guard triggers after only 1 model pass | `elif len(gains) == 1: return gains[0] < MIN_GAIN_REL` triggers stall guard on the very first gain calculation, violating the 2-pass specification. | Change condition to require `len(gains) >= 2` before comparing against `MIN_GAIN_REL`. | FIX |
| F-06 | **P1** | Backend | `app/agents/eda.py:277`, `feature_engineering.py:410`, `model.py:279` | Worker notebooks discard Coder error summaries | `coder.run_task` produces `error_summaries`, but worker wrappers hardcode `notebook={"errors": []}`. | Populate `notebook["errors"]` with `res.get("error_summaries", [])` to retain learning context across turns. | FIX |
| F-07 | **P1** | API | `app/api/routes.py:383-470` | Missing REST endpoints for persistent V2 run inspection | UI cannot fetch ledger, plan, supervisor decisions, versions, or agent notebooks for historical/completed runs. | Add `GET /projects/{project_id}/runs/{run_id}/{ledger\|plan\|decisions\|versions\|best\|notebooks}`. | FIX |
| F-08 | **P1** | API/Core | `app/core/events.py:29-71`, `app/api/routes.py:401-415` | SSE event stream lacks monotonic sequence numbering | Reconnect replays all events from index 0; no event sequence id or `since_seq` filtering exists to prevent duplicate UI rendering. | Add integer `seq` field to `Event` table/dict; support `since_seq` query param on `/events`. | FIX |
| F-09 | **P1** | UI | `frontend/src/pages/WorkspacePage.tsx:129` | Workspace navigation handler drops `"judge"` stage | `handleStageSelect` checks `"evaluator"` and `"evaluation"`, but omits `"judge"`, falling back to `"overview"`. | Add `case "judge": setActiveTab("evaluation"); break;`. | FIX |
| F-10 | **P1** | API/DB | `app/api/runner.py:149, 402, 437` | Run failure mid-workflow drops best metric from DB | `run_record.best_metric_value` is only written when `status == "SUCCESS"`. If a run fails later, DB value remains `None` despite `best.json` existing. | In runner error/interruption handlers, sync `best_metric_value` and `best_experiment_id` from state before saving. | FIX |
| F-11 | **P2** | Backend | `app/core/versions.py:128` | `restore_version` crashes with `TypeError` if version_id is None | `version_dir = PROJECTS_DIR / ... / version_id` fails when `version_id` is `None`. | Add guard: `if not version_id: return False`. | FIX |
| F-12 | **P2** | Backend | `app/agents/evaluator.py:1-11` | Dead forwarding shim for removed Evaluator agent | File merely re-exports `run_judge` from `app.agents.judge`. | Remove `evaluator.py` and update imports to `app.agents.judge`. | DELETE |
| F-13 | **P2** | DB/Core | `app/db/models.py:121-128`, `app/api/routes.py:34, 153` | Unused `SupervisorMemoryRecord` SQLModel table | Table is never read or written in V2 pipeline (RunMemory uses filesystem JSON). | Remove SQLModel model and references (stop writing/referencing, do not drop table in DB). | DELETE |
| F-14 | **P2** | Backend | `app/config.py:80,85`, `app/db/models.py:6,48` | Deprecated `MAX_ITER` constant and field alias | Legacy alias for `MAX_STEPS` remains in config and `WorkflowRun` default factory. | Clean up `MAX_ITER` alias; use `MAX_STEPS` explicitly. | DELETE |
| F-15 | **P2** | Tests | `tests/test_backend_phase2.py:28`, `test_predict_endpoint.py:30,117,152` | Tests assert `len(runs) > 0` on unseeded DB | Clearing DB tables causes these tests to fail because they rely on pre-existing database rows. | Seed mock project and run fixtures inside the test functions. | FIX |
| F-16 | **P2** | UI | `frontend/src/App.css` | Unused CSS file | File is imported nowhere and contains obsolete Vite demo styles. | Remove `frontend/src/App.css`. | DELETE |
| F-17 | **P3** | Docs | `README.md:3, 10, 14, 15, 34` | Outdated documentation referencing Evaluator loop | README claims "Model <-> Evaluator feedback loop" and displays an Evaluator agent in Mermaid diagram. | Rewrite README to reflect Supervisor-Led team and Judge specialist. | FIX |
| F-18 | **P3** | Docs | `docs/ARCHITECTURE_V2.md:122-131` | Hypothetical benchmark numbers in Section 7 | Section 7 lists unverified baseline performance numbers (~312s, ROC-AUC 0.894). | Mark Section 7 as UNVERIFIED until measured in Part B E2E testing. | FIX |
| F-19 | **P1** | API/Backend | `app/api/routes.py:143-180` | Incomplete project purge on UI delete: artifacts, MLflow runs, active tasks leak | `routes.py:143-180`: `mlflow.delete_experiment` leaves physical MLflow runs on disk; `ignore_errors=True` leaves Windows-locked files; active runner threads are not cancelled and continue writing. | In `delete_project`: cancel in-flight runner tasks, clean SSE queues, purge physical MLflow runs directory, robustly remove `projects/<project_id>` with file-lock retries, and cascade DB rows. | FIX |
| F-20 | **P0** | Core/Backend | `app/core/run_memory.py:34`, `app/api/runner.py:90` | `RunMemory.init_mission()` rejected keyword arguments from `runner.py` | `runner.py:90` calls `mem.init_mission(user_goal=..., ...)` with kwargs, but `init_mission` only accepted `mission_data: dict`, crashing run initialization. | Updated `init_mission(mission_data: dict | None = None, **kwargs)` to accept both dictionaries and keyword arguments. | FIXED |

---

## 3. Delete List

| Target Item | Type | Location | Proof of Zero External References (Grep Verification) | Guarding Test |
|---|---|---|---|---|
| `app/agents/evaluator.py` | Dead Shim File | `app/agents/evaluator.py` | Grep across `app/` and `tests/` shows only forwarding alias in `evaluator.py` itself and deprecated import in `tests/test_backend_phase2.py`. | `tests/test_supervisor_guards.py` |
| `SupervisorMemoryRecord` | SQLModel Class | `app/db/models.py:121-128` | Grep shows no `session.add`, `session.exec(select(SupervisorMemoryRecord))` anywhere; only imported in `runner.py:9` (unused) and `routes.py:153` (cascade delete). | `tests/test_backend_phase2.py` |
| `MAX_ITER` | Config Constant | `app/config.py:80,85` | Grep confirms all runtime scheduling uses `MAX_STEPS`. `MAX_ITER` only referenced in `models.py:48` default factory. | `tests/test_supervisor_guards.py` |
| `frontend/src/App.css` | Unused Asset | `frontend/src/App.css` | Knip report (`docs/audit_raw/frontend_knip.txt`) confirms zero imports across all frontend TSX/CSS files. | `npm run build` |
| `EvaluationIssue` | Unused TS Type | `frontend/src/types/index.ts:126` | Knip report confirms unused export; no component or hook references `EvaluationIssue`. | `npx tsc --noEmit` |
| `tailwindcss`, `postcss`, `autoprefixer` | Unused devDeps | `frontend/package.json` | Depcheck report (`docs/audit_raw/frontend_depcheck.txt`) confirms zero references in project (built with pure vanilla CSS). | `npm run build` |

---

## 4. Route Table & UI Event Table

### 4.1 Backend Route Table (`app/api/routes.py`)

| Method | Path | Handler | Reads What | Writes What | Used by UI? | Still Valid Under V2? |
|---|---|---|---|---|---|---|
| POST | `/projects` | `create_project` | `ProjectCreate` body | `Project` DB, directory tree | Yes (`api.createProject`) | VALID |
| GET | `/projects` | `list_projects` | `Project` DB table | None | Yes (`api.getProjects`) | VALID |
| GET | `/projects/{project_id}` | `get_project` | `Project` DB or directory | None | Yes (`api.getProject`) | VALID |
| DELETE | `/projects/{project_id}` | `delete_project` | Project DB, disk, MLflow | Cascades DB, deletes folder | Yes (`api.deleteProject`) | VALID |
| GET | `/projects/{project_id}/code-executions` | `get_code_executions` | `code_executions/*.json` | None | Yes (`api.getCodeExecutions`) | VALID |
| POST | `/projects/{project_id}/datasets` | `upload_dataset` | Multipart CSV file | Saves `data.csv`, `Dataset` DB | Yes (`api.uploadDataset`) | VALID |
| GET | `/projects/{project_id}/datasets` | `list_datasets` | `Dataset` DB table | None | Yes (`api.getDatasets`) | VALID |
| POST | `/projects/{project_id}/runs` | `trigger_run` | `RunCreate` payload | `WorkflowRun` DB, starts async | Yes (`api.triggerRun`) | VALID |
| GET | `/projects/{project_id}/runs` | `list_runs` | `WorkflowRun` DB table | None | Yes (`api.getRuns`) | VALID |
| GET | `/projects/{project_id}/runs/{run_id}` | `get_run` | `WorkflowRun` DB row | None | Yes (`api.getRun`) | VALID |
| GET | `/projects/{project_id}/events` | `stream_events` | `Event` DB & async queue | None | Yes (`useSSE`) | VALID (Needs seq ID) |
| GET | `/projects/{project_id}/artifacts` | `get_artifacts` | `ArtifactIndex` DB table | None | Yes (`api.getArtifacts`) | VALID |
| GET | `/projects/{project_id}/artifacts/{artifact_id}/content` | `get_artifact_content` | Disk artifact file | None | Yes (`api.getArtifactContent`) | VALID |
| GET | `/projects/{project_id}/datasets/{version}/preview` | `preview_dataset` | Dataset CSV on disk | None | Yes (`api.previewDataset`) | VALID |
| GET | `/projects/{project_id}/models` | `get_leaderboard` | MLflow tracking store | None | Yes (`api.getLeaderboard`) | VALID |
| GET | `/projects/{project_id}/experiments` | `get_experiments` | MLflow tracking store | None | Yes (`api.getExperiments`) | VALID |
| POST | `/projects/{project_id}/models/{experiment_id}/predict` | `predict_model` | Uploaded CSV, model pkl | Generates prediction CSV | Yes (`api.predictModel`) | BUG (F-04: unversioned model path) |
| GET | `/projects/{project_id}/evaluation` | `get_evaluation` | `evaluations/*.json, *.md` | None | Yes (`api.getEvaluation`) | VALID |
| GET | `/projects/{project_id}/report` | `get_report` | `reports/summary.json, *.md` | None | Yes (`api.getReport`) | VALID |

#### Proposed Missing Endpoints for V2 (to be added in Part B):
- `GET /projects/{project_id}/runs/{run_id}/ledger`: Returns raw ledger JSON rows from `RunMemory`.
- `GET /projects/{project_id}/runs/{run_id}/plan`: Returns current plan items from `RunMemory`.
- `GET /projects/{project_id}/runs/{run_id}/decisions`: Returns chronological supervisor deliberations.
- `GET /projects/{project_id}/runs/{run_id}/versions`: Returns all version snapshots with metadata.
- `GET /projects/{project_id}/runs/{run_id}/best`: Returns current best model version and score.
- `GET /projects/{project_id}/runs/{run_id}/notebooks/{agent}`: Returns specialist worker notebook entries.

### 4.2 UI Event Handling Table (`frontend/src`)

| Event Type | Emitted by Backend? | Handled in UI? | UI Component / Behavior |
|---|---|---|---|
| `WORKFLOW_STARTED` | Yes (`runner.py:45`) | Yes | Appears in Raw Stream & status indicators |
| `SUPERVISOR_DECISION` / `supervisor_decision` | Yes (`runner.py:72, 176, 193`) | Yes | Renders step header, target action, reasoning, and brief in `AgentActivityFeed` |
| `guard_override` | Yes (`runner.py:212`) | Yes | Renders red guard intervention banner in step timeline |
| `AGENT_COMPLETED` | Yes (`runner.py:223`) | Yes | Handled as completion marker |
| `worker_report` | Yes (`runner.py:240`) | Yes | Renders specialist worker report card, status badge, evidence |
| `concern` | Yes (`runner.py:257`) | Yes | Renders amber concern card; Supervisor ruling badge currently missing |
| `EXPERIMENT_COMPLETED` | Yes (`runner.py:300`) | Yes | Handled in feed |
| `EVALUATION_FAILED` | Yes (`runner.py:312`) | Yes | Handled in feed |
| `WORKFLOW_COMPLETED` | Yes (`runner.py:80, 158, 415, 446`) | Yes | Terminates live SSE stream and triggers final state invalidation |

---

## 5. Claim Check: `docs/ARCHITECTURE_V2.md`

| Section / Claim | Status | Verification Evidence / Finding |
|---|---|---|
| **Section 1**: `eda_fe_loop.py` removed; no cyclic hardcoded loop | **VERIFIED** | Grep confirms `eda_fe_loop.py` does not exist; graph uses flat supervisor routing. |
| **Section 1**: Supervisor is sole decision maker with persistent memory | **VERIFIED** | `app/agents/supervisor.py` invokes LLM decision model with `RunMemory` context. |
| **Section 1**: Specialist workers with dedicated notebooks (`notebooks/<agent>.jsonl`) | **VERIFIED** | `app/core/run_memory.py:121` appends worker execution logs to JSONL notebooks. |
| **Section 2**: Flat Graph Topology (`START -> supervisor -> worker -> supervisor -> END`) | **VERIFIED** | `app/graph/build_graph.py:229-242` configures strict star topology around supervisor. |
| **Section 3**: RunMemory layout (`mission.json`, `plan.json`, `ledger.jsonl`, `best.json`) | **VERIFIED** | `app/core/run_memory.py:28-36` initializes these exact file paths. |
| **Section 3**: `restore_version` restores best version artifacts before report | **VERIFIED** | `app/graph/build_graph.py:223` executes `restore_version(best_version)` before report node. |
| **Section 4**: Guard order (Ceiling -> Prereqs -> Caps -> Dupe -> Stall -> Judge -> Budget) | **VERIFIED** | `app/agents/guards.py:17-86` executes checks in this exact sequential order. |
| **Section 4**: Stall guard detects `< MIN_GAIN_REL` over last 2 model passes | **FALSE** | `guards.py:59` triggers on `len(gains) == 1`, halting prematurely on single pass (F-05). |
| **Section 5**: Context char budgets ($\le 6000$ sup, $\le 5000$ worker, $\le 7000$ judge, $\le 6000$ rep) | **VERIFIED** | `app/core/briefs.py:15-18` defines and enforces character limits with truncation warnings. |
| **Section 6**: Config options (`MAX_STEPS=14`, `MIN_GAIN_REL=0.002`, `MAX_CODER_RETRIES=3`) | **VERIFIED** | Defined in `app/config.py` with environment variable overrides. |
| **Section 7**: Performance comparison table (Rainfall ROC-AUC $\ge 0.894$, runtime $< 310$s) | **UNVERIFIED** | Numbers represent targets; real execution measurements pending Part B testing. |

---

## 6. Proposed PART B Plan

| Step | Scope & Description | Estimated Impact |
|---|---|---|
| **B1: P0/P1 Fixes** | Fix fallback deadlock in `guards.py`, wrap worker execution in `build_graph.py` with `try...except`, scope worker filesystem paths to run directory to prevent collisions, fix predict model loading from version path, and harden project deletion (`delete_project`) to cancel running tasks, purge MLflow artifacts, and handle Windows file locks. | ~180 lines changed across 5 files |
| **B2: API/UI Contract** | Add missing endpoints (`/ledger`, `/plan`, `/decisions`, `/versions`, `/best`, `/notebooks/{agent}`), add monotonic `seq` to SSE events and DB, sync best metric on mid-run failures. | ~200 lines backend, ~50 lines types |
| **B3: Dead Code Removal** | Delete `app/agents/evaluator.py`, remove `SupervisorMemoryRecord` SQLModel code, remove `MAX_ITER` alias, delete `frontend/src/App.css`, remove unused npm deps. | ~80 lines deleted across backend & UI |
| **B4: Prompt Cleanup** | Remove references to loops, Evaluators, and iteration from system prompts; eliminate broken string slicing on JSON in `briefs.py`. | ~60 lines modified |
| **B5: UI Enhancements** | Add Plan checklist component, Versions list with BEST badge, steps budget indicator in header, Supervisor ruling badge on concerns, fix stage routing for `judge`. | ~250 lines frontend TSX |
| **B6: Tests & Validation** | Add regression tests for guards, worker exception wrapper, version restoration, and new API endpoints; seed test DB fixtures. | ~200 lines test code |
| **B7: Documentation** | Update `README.md` and `docs/ARCHITECTURE_V2.md` to remove legacy Evaluator loop diagrams and verify Section 7 with measured test runs. | ~100 lines docs |
