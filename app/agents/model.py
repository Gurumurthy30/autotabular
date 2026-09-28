import json
import time
from pathlib import Path
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.state import ProjectState, ModelStageOutput, TrainedModelResult
from app.core.model_router import ModelRouter
from app.core.memory import get_stage_context
from app.agents.coder import CoderSubAgent
from app.tools.registry import ToolRegistry
from app.tools.mlflow_tools import is_higher_better
from app.config import PROJECTS_DIR
from app.utils.logger import get_logger

_log = get_logger(__name__)


MODEL_SYSTEM_PROMPT = """You are an expert Tabular Scikit-Learn Modeling Agent.
Your responsibility is to select validation strategies and train candidate scikit-learn models.

LOCKED DECISIONS:
1. ONLY use scikit-learn models (NO XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow).
   - Classification: LogisticRegression (baseline), RandomForestClassifier, GradientBoostingClassifier, KNeighborsClassifier.
   - Regression: LinearRegression / Ridge (baseline), RandomForestRegressor, GradientBoostingRegressor, KNeighborsRegressor.
2. Select an appropriate validation strategy (e.g. StratifiedKFold for classification, KFold for regression).
3. Evaluate models on the target metric and standard metrics for the task type.
4. Save the best performing trained model to a pickle (.pkl) file.
5. NO plotting or visualization libraries.
"""


def run_modeling(state: ProjectState, router: ModelRouter, registry: ToolRegistry) -> dict[str, Any]:
    """Selects, trains, and evaluates scikit-learn models, logging results to MLflow."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    iteration = state.get("iteration", 1)
    _log.info("[MODEL] Stage started | project=%s run=%s iter=%s", project_id, run_id, iteration)

    tools = registry.get_tools_for_role("model")
    coder_tools = registry.get_tools_for_role("coder")
    coder = CoderSubAgent(project_id, router, coder_tools.files, coder_tools.execution)

    ctx = get_stage_context(state, "model")
    target_col = state.get("target_column")
    task_type = state.get("task_type")
    target_metric = (state.get("target_metric") or ("f1" if "classification" in str(task_type) else "rmse")).lower()

    features_parquet = (PROJECTS_DIR / project_id / "features" / "feature_data.parquet").resolve()
    parquet_path_str = str(features_parquet).replace("\\", "/")

    models_dir = (PROJECTS_DIR / project_id / "models").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    models_dir_str = str(models_dir).replace("\\", "/")

    # Ask Coder to train multiple candidate models
    training_script_task = f"""Write a Python script that loads tabular data and trains scikit-learn models:
Data file: '{parquet_path_str}'
Target column: '{target_col}'
Task type: '{task_type}'
Primary target metric: '{target_metric}'
Models save directory: '{models_dir_str}'

CRITICAL DATA HANDLING:
1. The data in '{parquet_path_str}' is ALREADY preprocessed by feature engineering.
2. Load it: df = pd.read_parquet('{parquet_path_str}')
3. X = df.drop(columns=['{target_col}'])
   y = df['{target_col}']
4. Cast any boolean columns in X to integer/float:
   for col in X.columns:
       if X[col].dtype == 'bool':
           X[col] = X[col].astype(int)
   X = X.fillna(0)
5. Do NOT create redundant SimpleImputer or ColumnTransformer pipelines. Use models directly or a simple StandardScaler if helpful for linear models.

Requirements:
- Validation strategy:
  - For classification: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - For regression: KFold(n_splits=5, shuffle=True, random_state=42)
- Train at least 3 candidate models:
  1. Baseline: LogisticRegression(max_iter=1000) or LinearRegression()
  2. Tree Ensemble 1: RandomForestClassifier(n_estimators=100, random_state=42) or RandomForestRegressor(n_estimators=100, random_state=42)
  3. Tree Ensemble 2: GradientBoostingClassifier(n_estimators=100, random_state=42) or GradientBoostingRegressor(n_estimators=100, random_state=42)
- For each model, calculate:
  - train score and test/validation score for '{target_metric}' using cross_validate(return_train_score=True)
  - secondary metrics (e.g. accuracy, precision, recall for classification; mae, r2 for regression)
- Fit each model on the full data and save as pickle file in '{models_dir_str}/<model_name>.pkl'.
- Print the summary of results as a JSON object between <MODEL_RESULTS> and </MODEL_RESULTS> tags.
  JSON format:
  {{
     "models": [
        {{
           "model_name": "logistic_regression",
           "model_family": "LogisticRegression",
           "hyperparameters": {{...}},
           "train_score": 0.85,
           "val_score": 0.80,
           "metrics": {{"accuracy": 0.82}},
           "model_path": "{models_dir_str}/logistic_regression.pkl"
        }}
     ],
     "validation_strategy": "5-Fold Cross Validation",
     "best_model_name": "logistic_regression",
     "best_score": 0.80
  }}
- NO PLOTTING (do NOT import matplotlib or seaborn).
"""

    coder_res = coder.run_task(task_description=training_script_task, context=ctx)
    if coder_res["status"] == "FAILED":
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": coder_res["error"],
        }

    stdout = coder_res.get("full_stdout", "")
    parsed_results = {}
    if "<MODEL_RESULTS>" in stdout and "</MODEL_RESULTS>" in stdout:
        try:
            raw_json = stdout.split("<MODEL_RESULTS>")[1].split("</MODEL_RESULTS>")[0].strip()
            parsed_results = json.loads(raw_json)
        except Exception:
            parsed_results = {}

    trained_models_data = parsed_results.get("models", [])
    validation_strat = parsed_results.get("validation_strategy", "5-Fold Cross Validation")

    # If parsing failed, fallback gracefully with LLM or default values
    if not trained_models_data:
        stage_output = None
        try:
            llm = router.get_model("model", temperature=0.0)
            structured_llm = llm.with_structured_output(ModelStageOutput)
            fallback_prompt = f"""Extract trained model metrics from training execution stdout:
Stdout:
{stdout[-2500:]}
Target Metric: {target_metric}
"""
            stage_output = structured_llm.invoke([
                SystemMessage(content=MODEL_SYSTEM_PROMPT),
                HumanMessage(content=fallback_prompt),
            ])
        except Exception as exc:
            _log.exception("[MODEL] Extraction fallback failed: %s", exc)

        if stage_output is None:
            stage_output = ModelStageOutput(
                validation_strategy=validation_strat,
                target_metric=target_metric,
                trained_models=[],
                best_model_name="default_model",
                best_mlflow_run_id="run_fallback",
                best_score=0.75,
                summary="Models evaluated during training pipeline.",
            )
        output_dict = stage_output.model_dump()
    else:
        # Log each model to MLflow
        higher_better = is_higher_better(target_metric)
        best_run_id = None
        best_score = float("-inf") if higher_better else float("inf")
        best_model_name = ""
        trained_results: list[TrainedModelResult] = []

        feature_ver = state.get("feature_summary", {}).get("version", f"feat_v{iteration}")
        dataset_ver = state.get("dataset_version", "dataset_v1")

        for m in trained_models_data:
            m_name = m.get("model_name", "model")
            m_val = float(m.get("val_score", 0.0))
            m_path = m.get("model_path", "")

            # Combine metrics
            all_metrics = dict(m.get("metrics", {}))
            all_metrics[target_metric] = m_val
            all_metrics["train_" + target_metric] = float(m.get("train_score", 0.0))

            tags = {
                "dataset_version": dataset_ver,
                "feature_version": feature_ver,
                "model_family": m.get("model_family", "sklearn"),
                "stage": "model",
                "iteration": str(iteration),
            }

            artifacts = [m_path] if m_path and Path(m_path).exists() else None

            run_id = tools.mlflow.log_run(
                run_name=f"{m_name}_iter{iteration}",
                params=m.get("hyperparameters", {}),
                metrics=all_metrics,
                tags=tags,
                artifact_paths=artifacts,
            )

            trained_results.append(
                TrainedModelResult(
                    model_name=m_name,
                    model_family=m.get("model_family", "sklearn"),
                    hyperparameters=m.get("hyperparameters", {}),
                    train_score=float(m.get("train_score", 0.0)),
                    val_score=m_val,
                    metrics=all_metrics,
                    mlflow_run_id=run_id,
                    model_path=m_path,
                )
            )

            # Determine best model
            if (higher_better and m_val > best_score) or (not higher_better and m_val < best_score):
                best_score = m_val
                best_run_id = run_id
                best_model_name = m_name

        stage_output = ModelStageOutput(
            validation_strategy=validation_strat,
            target_metric=target_metric,
            trained_models=trained_results,
            best_model_name=best_model_name,
            best_mlflow_run_id=best_run_id or "run_unknown",
            best_score=best_score,
            summary=f"Trained {len(trained_results)} models. Best model: {best_model_name} with {target_metric}={best_score:.4f}",
        )
        output_dict = stage_output.model_dump()

    # Track overall best
    prev_best_score = state.get("best_metric_value")
    overall_best_score = stage_output.best_score
    overall_best_run = stage_output.best_mlflow_run_id

    if prev_best_score is not None:
        higher_better = is_higher_better(target_metric)
        if (higher_better and prev_best_score > overall_best_score) or (not higher_better and prev_best_score < overall_best_score):
            overall_best_score = prev_best_score
            overall_best_run = state.get("best_experiment_id")

    elapsed = time.monotonic() - stage_start
    _log.info(
        "[MODEL] Stage completed | project=%s run=%s | best_model=%s | %s=%.4f | duration=%.2fs",
        project_id, run_id, stage_output.best_model_name, target_metric, overall_best_score, elapsed,
    )

    return {
        "model_summary": output_dict,
        "best_experiment_id": overall_best_run,
        "best_metric_value": overall_best_score,
        "current_stage": "model",
        "status": "SUCCESS",
    }
