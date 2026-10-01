from typing import Any

import mlflow
import mlflow.tracking
from mlflow.exceptions import MlflowException

from app.config import MLFLOW_TRACKING_URI
from app.utils.logger import get_logger

_log = get_logger(__name__)

# Explicit metric direction map (PROJECT_SPEC.md §11)
METRIC_DIRECTIONS = {
    "rmse": "min",
    "mae": "min",
    "mse": "min",
    "log_loss": "min",
    "loss": "min",
    "f1": "max",
    "f1_score": "max",
    "roc_auc": "max",
    "accuracy": "max",
    "precision": "max",
    "recall": "max",
    "r2": "max",
}


def is_higher_better(metric_name: str) -> bool:
    direction = METRIC_DIRECTIONS.get(metric_name.lower().strip(), "max")
    return direction == "max"


class MLflowTools:
    """Wrapper around MLflow for local experiment and run tracking."""

    def __init__(self, project_id: str, tracking_uri: str = MLFLOW_TRACKING_URI):
        self.project_id = project_id
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_id = self._ensure_active_experiment()

    def _ensure_active_experiment(self) -> str:
        """
        Get or create the MLflow experiment for this project, restoring it if it
        was soft-deleted.  Returns the active experiment_id as a string.

        MLflow soft-deletes experiments (lifecycle_stage='deleted') when you call
        mlflow.delete_experiment().  get_experiment_by_name() still returns them,
        but start_run() raises MlflowException if the experiment is not 'active'.
        We restore them instead of creating a duplicate name.
        """
        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.project_id)

        if experiment is None:
            # Brand-new project — create the experiment
            exp_id = mlflow.create_experiment(self.project_id)
            _log.info("[MLFLOW] Created new experiment | project=%s exp_id=%s", self.project_id, exp_id)
            return exp_id

        if experiment.lifecycle_stage == "deleted":
            # Experiment was soft-deleted (e.g. project was previously deleted then re-run).
            # Restore it so we can log new runs without losing the experiment ID linkage.
            try:
                client.restore_experiment(experiment.experiment_id)
                _log.info(
                    "[MLFLOW] Restored deleted experiment | project=%s exp_id=%s",
                    self.project_id, experiment.experiment_id,
                )
            except MlflowException as exc:
                # If restore fails for any reason, create a fresh experiment with a suffixed name
                _log.warning(
                    "[MLFLOW] Could not restore experiment %s (%s) — creating fresh one",
                    experiment.experiment_id, exc,
                )
                suffix = 1
                while True:
                    new_name = f"{self.project_id}_v{suffix}"
                    if mlflow.get_experiment_by_name(new_name) is None:
                        break
                    suffix += 1
                exp_id = mlflow.create_experiment(new_name)
                _log.info("[MLFLOW] Created fresh experiment | name=%s exp_id=%s", new_name, exp_id)
                return exp_id

        return experiment.experiment_id


    def log_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        artifact_paths: list[str] | None = None,
    ) -> str:
        """Starts an MLflow run under the project experiment, logs all data, and returns the run_id."""
        # Self-healing: if the experiment was deleted between init and now, restore it and retry once.
        try:
            ctx = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        except MlflowException as exc:
            if "active" in str(exc).lower() or "deleted" in str(exc).lower():
                _log.warning(
                    "[MLFLOW] Experiment %s is not active (%s) — attempting restore before retry",
                    self.experiment_id, exc,
                )
                self.experiment_id = self._ensure_active_experiment()
                ctx = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
            else:
                raise
        with ctx as run:

            run_id = run.info.run_id

            # Log params (ensure stringified or primitive)
            for k, v in params.items():
                mlflow.log_param(k, str(v)[:250])

            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # Log tags
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, str(v))

            # Log artifacts
            if artifact_paths:
                for path in artifact_paths:
                    mlflow.log_artifact(path)

            return run_id

    def get_leaderboard(self, target_metric: str = "f1") -> list[dict[str, Any]]:
        """Queries runs for this experiment and returns them sorted by the target metric."""
        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        runs = client.search_runs(experiment_ids=[self.experiment_id])

        leaderboard = []
        higher_better = is_higher_better(target_metric)

        for r in runs:
            m_val = r.data.metrics.get(target_metric)
            tags = r.data.tags or {}
            
            # calculate training duration if available
            duration = None
            if r.info.start_time and r.info.end_time:
                duration = max(0.0, (r.info.end_time - r.info.start_time) / 1000.0)

            model_name = tags.get("mlflow.runName") or r.info.run_name or r.info.run_id

            leaderboard.append({
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "model_name": model_name,
                "model_type": tags.get("model_family"),
                "experiment_id": self.experiment_id,
                "status": r.info.status,
                "target_metric": target_metric,
                "metric_value": m_val,
                "score": m_val,
                "feature_version": tags.get("feature_version", "feat_v1"),
                "dataset_version": tags.get("dataset_version", "dataset_v1"),
                "duration_seconds": duration,
                "metrics": r.data.metrics,
                "params": r.data.params,
                "tags": tags,
                "artifact_uri": r.info.artifact_uri,
                "start_time": r.info.start_time,
            })

        # Sort runs: valid numbers first (ranked per direction), followed by None/missing always at bottom
        def sort_key(item: dict[str, Any]):
            val = item["score"]
            if val is None:
                # With reverse=True, -inf always places None/missing runs at the very bottom
                return float("-inf")
            return val if higher_better else -val

        leaderboard.sort(key=sort_key, reverse=True)
        return leaderboard
