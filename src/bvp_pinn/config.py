"""Load the locked physical case and the reference settings.

Code reads these files and never writes them. The physical values are the
supervisor's locked values (docs/parameters.md); the reference settings are
our numerical choices (configs/reference.json).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE_PATH = ROOT / "configs" / "physical-case-pending.json"
REFERENCE_SETTINGS_PATH = ROOT / "configs" / "reference.json"

PARAMETER_NAMES = (
    "Delta", "B", "Da", "kappa", "delta_star", "theta_w", "lambda_m", "beta_m",
    "epsilon", "b", "Pr", "Le", "N1", "Bi", "Rd", "Nr", "Nt", "Nb", "Lb", "Pe",
    "ell", "fw",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(path: Path = CASE_PATH) -> tuple[dict[str, float], str]:
    """Return the 22 locked coefficients and the SHA-256 of the case file."""
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if record.get("locked") is not True:
        raise ValueError(f"{path} is not marked locked")
    values = record["physical_parameters"]
    missing = [name for name in PARAMETER_NAMES if values.get(name) is None]
    extra = sorted(set(values) - set(PARAMETER_NAMES))
    if missing or extra:
        raise ValueError(f"case file mismatch: missing {missing}, unexpected {extra}")
    return {name: float(values[name]) for name in PARAMETER_NAMES}, hashlib.sha256(raw).hexdigest()


def load_reference_settings(path: Path = REFERENCE_SETTINGS_PATH) -> dict:
    return json.loads(Path(path).read_text())
