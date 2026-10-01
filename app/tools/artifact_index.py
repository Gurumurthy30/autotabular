import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import BASE_DIR

DB_PATH = BASE_DIR / "app_metadata.db"


class ArtifactIndex:
    """Manages artifact records in a local SQLite table."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifact_index (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    run_id TEXT,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    version TEXT,
                    parent_id TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def register_artifact(
        self,
        project_id: str,
        artifact_type: str,
        path: str,
        run_id: str | None = None,
        version: str | None = None,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Registers a new artifact reference in the database."""
        artifact_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO artifact_index (id, project_id, run_id, artifact_type, path, version, parent_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, project_id, run_id, artifact_type, path, version, parent_id, created_at),
            )
            conn.commit()

        return {
            "id": artifact_id,
            "project_id": project_id,
            "run_id": run_id,
            "artifact_type": artifact_type,
            "path": path,
            "version": version,
            "parent_id": parent_id,
            "created_at": created_at,
        }

    def list_artifacts(self, project_id: str, artifact_type: str | None = None) -> list[dict[str, Any]]:
        """Lists registered artifacts for a given project."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if artifact_type:
                cursor = conn.execute(
                    "SELECT * FROM artifact_index WHERE project_id = ? AND artifact_type = ? ORDER BY created_at DESC",
                    (project_id, artifact_type),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM artifact_index WHERE project_id = ? ORDER BY created_at DESC",
                    (project_id,),
                )
            return [dict(row) for row in cursor.fetchall()]
