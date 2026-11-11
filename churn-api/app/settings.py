"""Configuration from environment variables (twelve-factor style)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _keys(raw: str) -> tuple[str, ...]:
    return tuple(k.strip() for k in raw.split(",") if k.strip())


@dataclass(frozen=True)
class Settings:
    model_dir: Path | None = None            # CHURN_MODEL_DIR, else artifacts/CURRENT
    artifacts_root: Path = Path("artifacts")  # CHURN_ARTIFACTS_ROOT
    api_keys: tuple[str, ...] = field(default=(), repr=False)   # CHURN_API_KEYS
    log_level: str = "INFO"                  # LOG_LEVEL
    max_batch_size: int = 1000               # CHURN_MAX_BATCH
    drift_min_rows: int = 50                 # CHURN_DRIFT_MIN_ROWS
    drift_max_rows: int = 20000              # CHURN_DRIFT_MAX_ROWS
    drift_psi_threshold: float = 0.2         # CHURN_DRIFT_PSI_THRESHOLD

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        return cls(
            model_dir=Path(e["CHURN_MODEL_DIR"]) if e.get("CHURN_MODEL_DIR") else None,
            artifacts_root=Path(e.get("CHURN_ARTIFACTS_ROOT", "artifacts")),
            api_keys=_keys(e.get("CHURN_API_KEYS", "")),
            log_level=e.get("LOG_LEVEL", "INFO").upper(),
            max_batch_size=int(e.get("CHURN_MAX_BATCH", 1000)),
            drift_min_rows=int(e.get("CHURN_DRIFT_MIN_ROWS", 50)),
            drift_max_rows=int(e.get("CHURN_DRIFT_MAX_ROWS", 20000)),
            drift_psi_threshold=float(e.get("CHURN_DRIFT_PSI_THRESHOLD", 0.2)),
        )
