"""Shared Knowledge Board persisted per run under <run_dir>/memory/knowledge.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import PROJECTS_DIR
from app.utils.logger import get_logger

_log = get_logger(__name__)


class EDAFinding(BaseModel):
    """Structured insight discovered during exploratory data analysis."""
    id: str
    category: str = "general"
    finding: str
    evidence: Any = ""
    implication: str = ""
    recommendation: str = ""
    priority: str = "medium"  # high | medium | low
    status: Literal["open", "used", "rejected"] = "open"


class FeatureTried(BaseModel):
    """Record of an engineered or transformed feature."""
    id: str
    version: str = "v1"
    description: str = ""
    source_cols: list[str] = Field(default_factory=list)
    reason: str = ""
    cv_effect: float | None = None
    outcome: Literal["kept", "dropped", "untested"] = "untested"


class ModelTried(BaseModel):
    """Record of an evaluated model candidate and its cross-validation metrics."""
    id: str
    family: str
    params_summary: str = ""
    cv_mean: float | None = None
    cv_std: float | None = None
    train_score: float | None = None
    gap: float | None = None
    fit_time: float | None = None
    outcome: str = ""


class Hypothesis(BaseModel):
    """Testable hypothesis proposed by an agent."""
    id: str
    text: str
    owner_agent: str
    status: Literal["open", "testing", "confirmed", "rejected"] = "open"
    evidence: str = ""


class JudgeVerdict(BaseModel):
    """Judge evaluation verdict and targeted improvement advice."""
    step: int
    blame_stage: str
    overall: str
    advice: list[str] = Field(default_factory=list)
    addressed: bool = False


class Decision(BaseModel):
    """Supervisor high-level routing decision."""
    step: int
    action: str
    reason: str


class KnowledgeBoard:
    """Shared knowledge board replacing blind hand-offs with a persistent, structured store."""

    def __init__(
        self,
        run_dir: str | Path | None = None,
        project_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        elif project_id and run_id:
            self.run_dir = PROJECTS_DIR / project_id / "runs" / run_id
        else:
            raise ValueError("KnowledgeBoard requires either run_dir or (project_id, run_id)")

        self.memory_dir = self.run_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.memory_dir / "knowledge.json"

        self.eda_findings: list[EDAFinding] = []
        self.features_tried: list[FeatureTried] = []
        self.models_tried: list[ModelTried] = []
        self.hypotheses: list[Hypothesis] = []
        self.judge_verdicts: list[JudgeVerdict] = []
        self.decisions: list[Decision] = []

        self.load()

    def load(self) -> None:
        """Loads state from knowledge.json if present."""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.eda_findings = [EDAFinding(**item) for item in data.get("eda_findings", [])]
            self.features_tried = [FeatureTried(**item) for item in data.get("features_tried", [])]
            self.models_tried = [ModelTried(**item) for item in data.get("models_tried", [])]
            self.hypotheses = [Hypothesis(**item) for item in data.get("hypotheses", [])]
            self.judge_verdicts = [JudgeVerdict(**item) for item in data.get("judge_verdicts", [])]
            self.decisions = [Decision(**item) for item in data.get("decisions", [])]
        except Exception as exc:
            _log.warning("[KNOWLEDGE] Failed to load %s: %s", self.path, exc)

    def save(self) -> None:
        """Saves current state to knowledge.json."""
        data = {
            "eda_findings": [f.model_dump() for f in self.eda_findings],
            "features_tried": [f.model_dump() for f in self.features_tried],
            "models_tried": [m.model_dump() for m in self.models_tried],
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "judge_verdicts": [j.model_dump() for j in self.judge_verdicts],
            "decisions": [d.model_dump() for d in self.decisions],
        }
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.replace(self.path)

    # --- Add Methods ---

    def add_eda_finding(self, finding: EDAFinding | dict[str, Any]) -> None:
        """Adds or updates an EDA finding."""
        obj = finding if isinstance(finding, EDAFinding) else EDAFinding(**finding)
        for i, existing in enumerate(self.eda_findings):
            if existing.id == obj.id:
                self.eda_findings[i] = obj
                self.save()
                return
        self.eda_findings.append(obj)
        self.save()

    def add_feature_tried(self, feature: FeatureTried | dict[str, Any]) -> None:
        """Adds or updates a tried feature."""
        obj = feature if isinstance(feature, FeatureTried) else FeatureTried(**feature)
        for i, existing in enumerate(self.features_tried):
            if existing.id == obj.id:
                self.features_tried[i] = obj
                self.save()
                return
        self.features_tried.append(obj)
        self.save()

    def add_model_tried(self, model: ModelTried | dict[str, Any]) -> None:
        """Adds or updates a tried model."""
        obj = model if isinstance(model, ModelTried) else ModelTried(**model)
        for i, existing in enumerate(self.models_tried):
            if existing.id == obj.id:
                self.models_tried[i] = obj
                self.save()
                return
        self.models_tried.append(obj)
        self.save()

    def add_hypothesis(self, hypothesis: Hypothesis | dict[str, Any]) -> None:
        """Adds or updates a hypothesis."""
        obj = hypothesis if isinstance(hypothesis, Hypothesis) else Hypothesis(**hypothesis)
        for i, existing in enumerate(self.hypotheses):
            if existing.id == obj.id:
                self.hypotheses[i] = obj
                self.save()
                return
        self.hypotheses.append(obj)
        self.save()

    def add_judge_verdict(self, verdict: JudgeVerdict | dict[str, Any]) -> None:
        """Appends a judge verdict."""
        obj = verdict if isinstance(verdict, JudgeVerdict) else JudgeVerdict(**verdict)
        self.judge_verdicts.append(obj)
        self.save()

    def add_decision(self, decision: Decision | dict[str, Any]) -> None:
        """Appends a supervisor decision."""
        obj = decision if isinstance(decision, Decision) else Decision(**decision)
        self.decisions.append(obj)
        self.save()

    # --- Update Status ---

    def update_status(self, section: str, item_id: str | int, new_status: Any) -> bool:
        """Updates the status/outcome/addressed state of an item in the given section."""
        target_list: list[Any]
        if section in ("eda_findings", "eda"):
            target_list = self.eda_findings
        elif section in ("features_tried", "features", "fe"):
            target_list = self.features_tried
        elif section in ("models_tried", "models", "model"):
            target_list = self.models_tried
        elif section in ("hypotheses", "hypothesis"):
            target_list = self.hypotheses
        elif section in ("judge_verdicts", "judge"):
            target_list = self.judge_verdicts
        else:
            return False

        for item in target_list:
            match = False
            if hasattr(item, "id") and str(item.id) == str(item_id):
                match = True
            elif hasattr(item, "step") and str(item.step) == str(item_id):
                match = True

            if match:
                if hasattr(item, "status") and isinstance(new_status, str):
                    setattr(item, "status", new_status)
                elif hasattr(item, "outcome") and isinstance(new_status, str):
                    setattr(item, "outcome", new_status)
                elif hasattr(item, "addressed") and isinstance(new_status, bool):
                    setattr(item, "addressed", new_status)
                self.save()
                return True
        return False

    def mark_judge_advice_addressed(self, target_stage: str) -> int:
        """Marks unaddressed judge advice targeting this stage as addressed."""
        target_norm = target_stage.lower().strip()
        stage_aliases = {
            "fe": {"fe", "features", "feature_engineering"},
            "features": {"fe", "features", "feature_engineering"},
            "feature_engineering": {"fe", "features", "feature_engineering"},
            "model": {"model", "modeling"},
            "modeling": {"model", "modeling"},
            "eda": {"eda"},
        }.get(target_norm, {target_norm})

        count = 0
        for verdict in self.judge_verdicts:
            if not verdict.addressed and verdict.blame_stage.lower() in stage_aliases:
                verdict.addressed = True
                count += 1
        if count > 0:
            self.save()
        return count

    # --- Role-Tailored Digest ---

    def digest(self, role: str, max_chars: int = 4000) -> str:
        """Returns a compact text digest tailored specifically for the querying role."""
        role_norm = role.lower().strip()
        lines: list[str] = []

        if role_norm == "supervisor":
            # 1. ALL open eda findings (id + 1 line)
            open_findings = [f for f in self.eda_findings if f.status == "open"]
            lines.append("=== OPEN EDA FINDINGS ===")
            if open_findings:
                for f in open_findings:
                    lines.append(f"- [{f.id}] ({f.category}, prio={f.priority}): {f.finding}")
            else:
                lines.append("None open.")

            # 2. Hypotheses with status
            lines.append("\n=== HYPOTHESES ===")
            if self.hypotheses:
                for h in self.hypotheses:
                    lines.append(f"- [{h.id}] [{h.status.upper()}] ({h.owner_agent}): {h.text}")
            else:
                lines.append("None recorded.")

            # 3. Last 5 models_tried
            lines.append("\n=== RECENT MODELS TRIED (last 5) ===")
            recent_models = self.models_tried[-5:]
            if recent_models:
                for m in recent_models:
                    mean_str = f"{m.cv_mean:.4f}" if m.cv_mean is not None else "N/A"
                    std_str = f"±{m.cv_std:.4f}" if m.cv_std is not None else ""
                    lines.append(f"- [{m.id}] {m.family}: mean={mean_str}{std_str} outcome={m.outcome} params={m.params_summary[:60]}")
            else:
                lines.append("None tried yet.")

            # 4. Last 5 features_tried
            lines.append("\n=== RECENT FEATURES TRIED (last 5) ===")
            recent_features = self.features_tried[-5:]
            if recent_features:
                for ft in recent_features:
                    lines.append(f"- [{ft.id}] ({ft.version}) {ft.description or ', '.join(ft.source_cols[:3])}: outcome={ft.outcome} effect={ft.cv_effect}")
            else:
                lines.append("None tried yet.")

            # 5. Unaddressed judge advice
            lines.append("\n=== UNADDRESSED JUDGE ADVICE ===")
            unaddressed = [v for v in self.judge_verdicts if not v.addressed]
            if unaddressed:
                for v in unaddressed:
                    for a in v.advice:
                        lines.append(f"- Step {v.step} [blame={v.blame_stage}, overall={v.overall}]: {a}")
            else:
                lines.append("None (all advice addressed or no verdicts yet).")

        elif role_norm == "eda":
            # Existing findings + hypotheses already tested (so it does not repeat)
            lines.append("=== EXISTING EDA FINDINGS (DO NOT DUPLICATE) ===")
            if self.eda_findings:
                for f in self.eda_findings:
                    lines.append(f"- [{f.id}] ({f.category}, status={f.status}): {f.finding}")
            else:
                lines.append("None recorded.")

            lines.append("\n=== HYPOTHESES TESTED ===")
            tested_h = [h for h in self.hypotheses if h.status in ("confirmed", "rejected")]
            if tested_h:
                for h in tested_h:
                    lines.append(f"- [{h.id}] [{h.status.upper()}]: {h.text} (ev: {h.evidence[:60]})")
            else:
                lines.append("None tested yet.")

        elif role_norm in ("fe", "features", "feature_engineering"):
            # Open findings, features_tried, top model feature importance, unaddressed judge advice (fe, eda)
            lines.append("=== OPEN FINDINGS FOR FE ===")
            open_findings = [f for f in self.eda_findings if f.status == "open"]
            if open_findings:
                for f in open_findings:
                    rec = f" -> Rec: {f.recommendation}" if f.recommendation else ""
                    lines.append(f"- [{f.id}] ({f.category}, prio={f.priority}): {f.finding}{rec}")
            else:
                lines.append("None open.")

            lines.append("\n=== ALL FEATURES TRIED ===")
            if self.features_tried:
                for ft in self.features_tried:
                    lines.append(f"- [{ft.id}] ({ft.version}) {ft.description or ', '.join(ft.source_cols[:3])}: reason={ft.reason} outcome={ft.outcome}")
            else:
                lines.append("None tried yet.")

            lines.append("\n=== UNADDRESSED JUDGE ADVICE (FE / EDA) ===")
            fe_unaddressed = [
                v for v in self.judge_verdicts
                if not v.addressed and v.blame_stage.lower() in ("fe", "eda", "features", "feature_engineering")
            ]
            if fe_unaddressed:
                for v in fe_unaddressed:
                    for a in v.advice:
                        lines.append(f"- Step {v.step} [blame={v.blame_stage}]: {a}")
            else:
                lines.append("None.")

        elif role_norm in ("model", "modeling"):
            # Open findings, models_tried (never repeat family+params), features summary, unaddressed judge advice (model)
            lines.append("=== OPEN FINDINGS FOR MODELING ===")
            open_findings = [f for f in self.eda_findings if f.status == "open"]
            if open_findings:
                for f in open_findings:
                    imp = f" -> Implication: {f.implication}" if f.implication else ""
                    lines.append(f"- [{f.id}] ({f.category}): {f.finding}{imp}")
            else:
                lines.append("None open.")

            lines.append("\n=== MODELS TRIED (DO NOT REPEAT FAMILY + PARAMS) ===")
            if self.models_tried:
                for m in self.models_tried:
                    mean_str = f"{m.cv_mean:.4f}" if m.cv_mean is not None else "N/A"
                    std_str = f"±{m.cv_std:.4f}" if m.cv_std is not None else ""
                    gap_str = f" gap={m.gap:.4f}" if m.gap is not None else ""
                    lines.append(f"- [{m.id}] {m.family} | cv={mean_str}{std_str}{gap_str} | outcome={m.outcome} | params={m.params_summary}")
            else:
                lines.append("None tried yet.")

            lines.append("\n=== FEATURES SUMMARY ===")
            kept = [ft for ft in self.features_tried if ft.outcome == "kept"]
            dropped = [ft for ft in self.features_tried if ft.outcome == "dropped"]
            lines.append(f"Total features recorded: {len(self.features_tried)} (Kept: {len(kept)}, Dropped: {len(dropped)})")

            lines.append("\n=== UNADDRESSED JUDGE ADVICE (MODEL) ===")
            model_unaddressed = [
                v for v in self.judge_verdicts
                if not v.addressed and v.blame_stage.lower() in ("model", "modeling")
            ]
            if model_unaddressed:
                for v in model_unaddressed:
                    for a in v.advice:
                        lines.append(f"- Step {v.step}: {a}")
            else:
                lines.append("None.")

        elif role_norm in ("judge", "evaluator"):
            # Everything, compact
            lines.append("=== KNOWLEDGE BOARD SUMMARY (JUDGE) ===")
            lines.append(f"- EDA Findings ({len(self.eda_findings)} total, {len([f for f in self.eda_findings if f.status == 'open'])} open):")
            for f in self.eda_findings[:8]:
                lines.append(f"  * [{f.id}] ({f.category}): {f.finding[:80]} [{f.status}]")
            lines.append(f"- Hypotheses ({len(self.hypotheses)} total):")
            for h in self.hypotheses[:5]:
                lines.append(f"  * [{h.id}] [{h.status}]: {h.text[:80]}")
            lines.append(f"- Features Tried ({len(self.features_tried)} total):")
            for ft in self.features_tried[-8:]:
                lines.append(f"  * [{ft.id}] {ft.description or ', '.join(ft.source_cols[:2])}: outcome={ft.outcome}")
            lines.append(f"- Models Tried ({len(self.models_tried)} total):")
            for m in self.models_tried[-5:]:
                mean_str = f"{m.cv_mean:.4f}" if m.cv_mean is not None else "N/A"
                lines.append(f"  * [{m.id}] {m.family}: cv={mean_str} outcome={m.outcome}")
            lines.append(f"- Judge Verdicts ({len(self.judge_verdicts)} total, {len([v for v in self.judge_verdicts if not v.addressed])} unaddressed)")

        elif role_norm in ("report", "reporting"):
            # Everything, compact for reporting
            lines.append("=== KNOWLEDGE BOARD LINEAGE (REPORT) ===")
            lines.append(f"Decisions Taken ({len(self.decisions)}):")
            for d in self.decisions[-10:]:
                lines.append(f"- Step {d.step}: action={d.action} | reason={d.reason[:80]}")
            lines.append(f"\nConfirmed Findings ({len([f for f in self.eda_findings if f.status != 'rejected'])}):")
            for f in self.eda_findings:
                if f.status != "rejected":
                    lines.append(f"- [{f.id}] {f.finding}")
            lines.append(f"\nModel Trajectory ({len(self.models_tried)} models evaluated):")
            for m in self.models_tried:
                mean_str = f"{m.cv_mean:.4f}" if m.cv_mean is not None else "N/A"
                lines.append(f"- {m.family} ({m.params_summary[:40]}): cv={mean_str} outcome={m.outcome}")
            lines.append(f"\nFinal Features ({len([ft for ft in self.features_tried if ft.outcome == 'kept'])} kept):")
            for ft in [ft for ft in self.features_tried if ft.outcome == "kept"][:15]:
                lines.append(f"- {ft.id}: {ft.description or ', '.join(ft.source_cols)}")

        else:
            lines.append(f"=== KNOWLEDGE BOARD ({role_norm.upper()}) ===")
            lines.append(f"Findings: {len(self.eda_findings)} | Models: {len(self.models_tried)} | Features: {len(self.features_tried)}")

        text = "\n".join(lines)
        if len(text) > max_chars:
            return text[:max_chars - 3] + "..."
        return text
