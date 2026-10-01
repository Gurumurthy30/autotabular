"""Detects available ML libraries in environment (Phase 3.9)."""

import importlib.util

_DETECTED_LIBS: list[str] | None = None


def get_available_libs() -> list[str]:
    """Detects optional high-performance tabular libraries installed in python environment."""
    global _DETECTED_LIBS
    if _DETECTED_LIBS is not None:
        return _DETECTED_LIBS

    libs = ["sklearn"]
    for mod_name in ("lightgbm", "xgboost", "catboost", "imblearn"):
        if importlib.util.find_spec(mod_name) is not None:
            libs.append(mod_name)

    _DETECTED_LIBS = libs
    return libs
