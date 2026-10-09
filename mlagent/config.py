"""Run configuration.  Precedence:  config.yaml  <  environment  <  CLI flags.

config.yaml is searched in: --config PATH, <data_dir>/mlagent.yaml, ./mlagent.yaml,
~/.config/mlagent/config.yaml.  The resolved config is frozen as <workspace>/config.yaml, and every
agent reads that copy (edit it, then `mlagent resume`, to change a run).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from .state import Base, Budget


class ModelCfg(Base):
    tag: str = "gemma4:31b-cloud"            # cloud-hosted Ollama model
    host: str | None = None            # Ollama host (also OLLAMA_HOST); key via OLLAMA_API_KEY
    temperature: float = 0.2


class ParallelCfg(Base):
    workers: int = 1                   # >1: experiments in a batch run concurrently (LLM calls stay serialized)


class DocsCfg(Base):
    web: bool = True                   # official-docs fallback after local introspection
    web_timeout: int = 15


class HardwareCfg(Base):
    oom_downgrade: bool = True         # retry out-of-memory crashes with a smaller / CPU variant
    max_oom_retries: int = 2


class FinisherCfg(Base):
    methods: list[str] = Field(default_factory=lambda: ["greedy", "voting", "stacking"])
    voting_top_k: int = 5


class KaggleCfg(Base):
    competition: str | None = None
    submit: bool = False               # submit submission.csv and compare the public LB with CV
    wait_seconds: int = 180
    gap_rel: float = 0.02              # flag if |LB - CV| > max(3 * cv_std, gap_rel * |cv|)


class Config(Base):
    model: ModelCfg = Field(default_factory=ModelCfg)
    seed: int = 42
    budget: Budget = Field(default_factory=Budget)
    target_score: float | None = None  # stop early when reached (else parsed from the goal text)
    parallel: ParallelCfg = Field(default_factory=ParallelCfg)
    docs: DocsCfg = Field(default_factory=DocsCfg)
    hardware: HardwareCfg = Field(default_factory=HardwareCfg)
    finisher: FinisherCfg = Field(default_factory=FinisherCfg)
    kaggle: KaggleCfg = Field(default_factory=KaggleCfg)


CLI_MAP = {"model": "model.tag", "workers": "parallel.workers", "max_experiments": "budget.max_experiments",
           "max_minutes": "budget.max_minutes", "seed": "seed", "target_score": "target_score",
           "kaggle": "kaggle.competition", "submit": "kaggle.submit"}

TEMPLATE = """\
# mlagent configuration (precedence: this file < env vars < CLI flags)
model:
  tag: gemma4:31b-cloud    # exact Ollama tag; env MLAGENT_MODEL overrides
  host: null               # env OLLAMA_HOST overrides; API key only via env OLLAMA_API_KEY
  temperature: 0.2
seed: 42
target_score: null         # stop early once the best CV reaches this (else read from the goal text)
budget:
  max_experiments: 15
  max_minutes: 60
  plateau_n: 3             # approved runs without a counted gain
  analyze_every: 3         # K: Analyzer every K runs
  tune_top_n: 3
  tune_trials: 20
  max_consecutive_crashes: 3
  loop_fraction: 0.85      # experiment loop stops at this share of max_minutes
  max_run_minutes: 15
  tune_minutes_per_model: 15
parallel:
  workers: 1               # >1 runs a batch of experiments concurrently
docs:
  web: true                # official docs (allowlisted domains) when local introspection is not enough
  web_timeout: 15
hardware:
  oom_downgrade: true      # CUDA/RAM out-of-memory -> retry smaller, then CPU + lighter model
  max_oom_retries: 2
finisher:
  methods: [greedy, voting, stacking]
  voting_top_k: 5
kaggle:
  competition: null        # e.g. titanic
  submit: false            # needs the kaggle CLI + ~/.kaggle/kaggle.json
  wait_seconds: 180
  gap_rel: 0.02
"""


def _deep_set(d: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def find(explicit: str | Path | None = None, data_dir: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        return p
    for c in ([Path(data_dir) / "mlagent.yaml"] if data_dir else []) + [
            Path.cwd() / "mlagent.yaml", Path.home() / ".config" / "mlagent" / "config.yaml"]:
        if c.exists():
            return c
    return None


def _build(raw: dict, cli: dict | None) -> Config:
    if os.getenv("MLAGENT_MODEL"):
        _deep_set(raw, "model.tag", os.environ["MLAGENT_MODEL"])
    if os.getenv("OLLAMA_HOST"):
        _deep_set(raw, "model.host", os.environ["OLLAMA_HOST"])
    for k, v in (cli or {}).items():
        if v is not None and k in CLI_MAP:
            _deep_set(raw, CLI_MAP[k], v)
    cfg = Config.model_validate(raw)
    cfg.parallel.workers = max(1, cfg.parallel.workers)
    return cfg


def resolve(cli: dict | None = None, data_dir: str | Path | None = None,
            explicit: str | Path | None = None) -> Config:
    f = find(explicit, data_dir)
    raw = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}) if f else {}
    return _build(raw, cli)


def override(cfg: Config, cli: dict | None) -> Config:
    """Re-apply env + CLI on top of an existing (workspace) config — used by resume."""
    return _build(cfg.model_dump(), cli)


def save(root: str | Path, cfg: Config) -> Path:
    p = Path(root) / "config.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
    return p


_cache: dict[tuple[str, float], Config] = {}


def load_workspace(root: str | Path) -> Config:
    p = Path(root) / "config.yaml"
    if not p.exists():
        return Config()
    key = (str(p), p.stat().st_mtime)
    if key not in _cache:
        _cache[key] = Config.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return _cache[key]
