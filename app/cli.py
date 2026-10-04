import uuid
from datetime import datetime
from pathlib import Path

import click

from app.core.model_router import ModelRouter
from app.core.state import ProjectState
from app.graph.build_graph import build_ml_graph
from app.tools.dataset_tools import DatasetTools


@click.group()
def cli():
    """Agentic Tabular ML Engineering Platform CLI."""


@cli.command()
@click.option("--project-name", "-p", default="demo", help="Unique project name / ID")
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="Path to input tabular dataset (CSV)")
@click.option("--target", "-t", required=True, help="Target column name")
@click.option("--metric", "-m", default=None, help="Target metric (e.g. f1, roc_auc, rmse, r2)")
@click.option("--description", default="", help="Project description or user goal")
def run(project_name: str, dataset: str, target: str, metric: str | None, description: str):
    """Executes the autonomous tabular ML workflow end-to-end."""
    print("=" * 70)
    print(f"Starting Tabular ML Pipeline for project: '{project_name}'")
    print(f"Dataset: {dataset}")
    print(f"Target Column: {target}")
    print(f"Target Metric: {metric or 'Auto-selected'}")
    print("=" * 70)

    # 1. Register dataset into project structure
    dataset_tools = DatasetTools(project_name)
    version = "dataset_v1"
    dataset_dest = dataset_tools.register_dataset_from_file(dataset, version=version)
    print(f"Registered dataset to version '{version}' at: {dataset_dest}")

    # Resolve target column case-insensitively against dataset headers
    target = dataset_tools.resolve_target_column(version, target)

    # 2. Setup initial state
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    initial_state: ProjectState = {
        "project_id": project_name,
        "run_id": run_id,
        "user_goal": description or f"Train tabular model to predict '{target}'",
        "target_column": target,
        "target_metric": metric,
        "description": description,
        "constraints": {},
        "dataset_id": Path(dataset).stem,
        "dataset_version": version,
        "task_type": None,
        "profile_summary": {},
        "eda_findings": {},
        "feature_summary": {},
        "model_summary": {},
        "current_stage": "start",
        "iteration": 1,
        "best_experiment_id": None,
        "best_metric_value": None,
        "supervisor_memory": {},
        "artifacts": [],
        "next_action": "profile",
        "status": "RUNNING",
    }

    # 3. Build graph & execute
    router = ModelRouter()
    graph = build_ml_graph(project_name, router=router)

    print("\n--- Invoking LangGraph Pipeline ---", flush=True)
    start_time = datetime.now()
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print(f"Pipeline Completed in {elapsed:.1f}s with status: {final_state.get('status')}")
    print(f"Best Experiment Run: {final_state.get('best_experiment_id')}")
    print(f"Best Metric Value: {final_state.get('best_metric_value')}")
    print(f"Total Iterations: {final_state.get('iteration')}")
    print(f"Total Artifacts Created: {len(final_state.get('artifacts', []))}")
    print("=" * 70)


if __name__ == "__main__":
    cli()
