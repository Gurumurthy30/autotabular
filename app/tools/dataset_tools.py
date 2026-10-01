import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import PROJECTS_DIR


class DatasetTools:
    """Manages versioned tabular datasets for a project."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.datasets_dir = PROJECTS_DIR / project_id / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def register_dataset_from_file(self, source_path: str, version: str = "dataset_v1") -> str:
        """Copies an external tabular CSV into the project's versioned dataset directory."""
        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source dataset not found: {source_path}")

        dest_dir = self.datasets_dir / version
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "data.csv"
        shutil.copy2(src, dest_file)
        return str(dest_file)

    def get_dataset_path(self, version: str = "dataset_v1") -> Path:
        """Returns the path to the versioned dataset."""
        path = self.datasets_dir / version / "data.csv"
        if not path.exists():
            # Check for parquet alternative
            parquet_path = self.datasets_dir / version / "data.parquet"
            if parquet_path.exists():
                return parquet_path
            raise FileNotFoundError(f"Dataset for version '{version}' not found at {path}")
        return path

    def load_dataset(self, version: str = "dataset_v1") -> pd.DataFrame:
        """Loads a versioned dataset into a Pandas DataFrame."""
        path = self.get_dataset_path(version)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def get_schema(self, version: str = "dataset_v1") -> dict[str, str]:
        """Returns column names and string representations of their dtypes."""
        df = self.load_dataset(version)
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def get_sample(self, version: str = "dataset_v1", n: int = 5) -> list[dict[str, Any]]:
        """Returns the first n rows as dictionaries."""
        df = self.load_dataset(version)
        return df.head(n).to_dict(orient="records")
