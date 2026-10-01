from app.tools.artifact_index import ArtifactIndex
from app.tools.dataset_tools import DatasetTools
from app.tools.execution_manager import ExecutionManager
from app.tools.file_tools import FileTools
from app.tools.mlflow_tools import MLflowTools


class AgentTools:
    """Container holding specifically allowed tools for an individual agent."""

    def __init__(
        self,
        file_tools: FileTools | None = None,
        dataset_tools: DatasetTools | None = None,
        execution_manager: ExecutionManager | None = None,
        mlflow_tools: MLflowTools | None = None,
        artifact_index: ArtifactIndex | None = None,
    ):
        self.files = file_tools
        self.dataset = dataset_tools
        self.execution = execution_manager
        self.mlflow = mlflow_tools
        self.artifacts = artifact_index


class ToolRegistry:
    """Central registry factory providing role-scoped tool bundles."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._file_tools = FileTools(project_id)
        self._dataset_tools = DatasetTools(project_id)
        self._execution_manager = ExecutionManager(project_id)
        self._mlflow_tools = MLflowTools(project_id)
        self._artifact_index = ArtifactIndex()

    def get_tools_for_role(self, role: str) -> AgentTools:
        """Returns only the tools authorized for the requested agent role as per PROJECT_SPEC.md §9."""
        if role == "coder":
            return AgentTools(
                file_tools=self._file_tools,
                execution_manager=self._execution_manager,
            )
        elif role == "profile" or role == "eda" or role == "feature_engineering":
            return AgentTools(
                file_tools=self._file_tools,
                dataset_tools=self._dataset_tools,
                artifact_index=self._artifact_index,
            )
        elif role in ("model", "judge", "evaluator"):
            return AgentTools(
                file_tools=self._file_tools,
                mlflow_tools=self._mlflow_tools,
                artifact_index=self._artifact_index,
            )
        elif role == "report" or role == "supervisor":
            return AgentTools(
                file_tools=self._file_tools,
                artifact_index=self._artifact_index,
            )
        else:
            return AgentTools(
                file_tools=self._file_tools,
                artifact_index=self._artifact_index,
            )
