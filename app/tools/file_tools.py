from pathlib import Path

from app.config import PROJECTS_DIR


class FileTools:
    """Scoped file operations with path traversal protection."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_root = (PROJECTS_DIR / project_id).resolve()
        self.project_root.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """Ensures that the resolved path stays inside the project directory."""
        target = (self.project_root / relative_path).resolve()
        try:
            target.relative_to(self.project_root)
        except ValueError:
            raise PermissionError(f"Access denied: path '{relative_path}' traverses outside project workspace.")
        return target

    def read_file(self, relative_path: str) -> str:
        """Reads text from a file within the project workspace."""
        path = self._resolve_safe_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        return path.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> str:
        """Writes text to a file within the project workspace, creating parent directories."""
        path = self._resolve_safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def list_dir(self, relative_path: str = "") -> list[str]:
        """Lists files and directories relative to the target subdirectory."""
        path = self._resolve_safe_path(relative_path)
        if not path.exists():
            return []
        return [p.name for p in path.iterdir()]
