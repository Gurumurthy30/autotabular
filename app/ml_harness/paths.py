"""Run-scoped paths ensuring isolation between workflow runs (Phase 3.8)."""

from pathlib import Path

from app.config import PROJECTS_DIR


class RunPaths:
    """Provides run-scoped directory and file locations."""

    def __init__(self, project_id: str, run_id: str):
        self.project_id = project_id
        self.run_id = run_id
        self.run_dir = PROJECTS_DIR / project_id / "runs" / run_id
        self.features_dir = self.run_dir / "features"
        self.models_dir = self.run_dir / "models"
        self.versions_dir = self.run_dir / "versions"
        self.memory_dir = self.run_dir / "memory"
        self.notebooks_dir = self.run_dir / "notebooks"
        self.reports_dir = self.run_dir / "reports"

        # Ensure directories exist
        for d in (
            self.run_dir,
            self.features_dir,
            self.models_dir,
            self.versions_dir,
            self.memory_dir,
            self.notebooks_dir,
            self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def version_dir(self, version_id: str) -> Path:
        vd = self.versions_dir / version_id
        vd.mkdir(parents=True, exist_ok=True)
        return vd

    @property
    def feature_pipeline_py(self) -> Path:
        return self.features_dir / "feature_pipeline.py"

    @property
    def feature_pipeline_pkl(self) -> Path:
        return self.features_dir / "feature_pipeline.pkl"

    @property
    def feature_schema_json(self) -> Path:
        return self.features_dir / "feature_schema.json"

    @property
    def best_model_pkl(self) -> Path:
        return self.models_dir / "best_model.pkl"

    @property
    def model_summary_json(self) -> Path:
        return self.models_dir / "model_summary.json"
