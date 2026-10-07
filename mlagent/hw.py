"""Hardware facts for prompts (so the Coder never asks for a GPU that is not there) and per-worker
device/thread assignment for parallel experiments."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def gpus() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
        rows = [r.split(",") for r in out.stdout.strip().splitlines() if r.strip()] if out.returncode == 0 else []
        return [{"name": r[0].strip(), "total_mb": int(float(r[1])), "free_mb": int(float(r[2]))} for r in rows]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return []


def cores() -> int:
    return os.cpu_count() or 1


@lru_cache(maxsize=1)
def ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def describe() -> str:
    g = gpus()
    gpu = "; ".join(f"{x['name']} {x['total_mb'] // 1024}GB ({x['free_mb'] // 1024}GB free)" for x in g) or "none (CPU only)"
    ram = f", RAM {ram_gb()}GB" if ram_gb() else ""
    return f"{cores()} CPU cores{ram}; GPU: {gpu}"


def device_env(worker: int, workers: int) -> dict[str, str]:
    """Env for worker `worker` of `workers` concurrent scripts: its own GPU (or CPU) and a thread share."""
    env: dict[str, str] = {}
    g = gpus()
    if g:
        env["CUDA_VISIBLE_DEVICES"] = str(worker) if worker < len(g) else ""   # more workers than GPUs -> CPU
    if workers > 1:
        share = str(max(1, cores() // workers))
        env.update(MLAGENT_N_JOBS=share, OMP_NUM_THREADS=share, MKL_NUM_THREADS=share, OPENBLAS_NUM_THREADS=share)
    return env
