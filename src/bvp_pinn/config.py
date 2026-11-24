"""Load the locked physical case and the reference settings.

The problem is fixed (AGENTS.md section 3): Eqs. 14 to 18, boundary conditions B1 to B12 and the
22 supervisor values never change, in any run, tool or use case. This module guards the values:

  - the case file must have exactly the SHA-256 it had when the values were locked, the same hash
    every reference folder records, so an edited file stops every load;
  - load_case returns a read-only mapping, so no code can change a value in place;
  - check_locked raises unless a value set is exactly the locked one; the reference study, every
    PINN training run and the error estimate call it before they start.

Code reads these files and never writes them. The reference settings are our numerical choices
(configs/reference.json), kept apart from the supervisor values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
CASE_PATH = ROOT / "configs" / "physical-case-pending.json"
REFERENCE_SETTINGS_PATH = ROOT / "configs" / "reference.json"

# SHA-256 of configs/physical-case-pending.json since the supervisor values were locked on
# 2026-11-16. It equals case_sha256 in reference/locked-branch*/meta.json.
LOCKED_CASE_SHA256 = "70d131f66d018a1a6baf3a5d8a7a64ef88d4825132c6f1ed81ae1472583b4ec8"

PARAMETER_NAMES = (
    "Delta", "B", "Da", "kappa", "delta_star", "theta_w", "lambda_m", "beta_m",
    "epsilon", "b", "Pr", "Le", "N1", "Bi", "Rd", "Nr", "Nt", "Nb", "Lb", "Pe",
    "ell", "fw",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(path: Path = CASE_PATH):
    """Return the 22 locked coefficients (read-only) and the SHA-256 of the case file.

    Refuses any file other than the locked one. Only the "physical_parameters" block is read;
    nothing else in the file (such as the retired debug case) is ever used.
    """
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != LOCKED_CASE_SHA256:
        raise ValueError(f"{path} is not the locked case file (SHA-256 {digest[:12]}..., expected "
                         f"{LOCKED_CASE_SHA256[:12]}...). The supervisor values are fixed; see AGENTS.md section 3.")
    record = json.loads(raw)
    if record.get("locked") is not True:
        raise ValueError(f"{path} is not marked locked")
    values = record["physical_parameters"]
    missing = [name for name in PARAMETER_NAMES if values.get(name) is None]
    extra = sorted(set(values) - set(PARAMETER_NAMES))
    if missing or extra:
        raise ValueError(f"case file mismatch: missing {missing}, unexpected {extra}")
    return MappingProxyType({name: float(values[name]) for name in PARAMETER_NAMES}), digest


def check_locked(p):
    """Raise unless p holds exactly the 22 locked supervisor values, no more and no fewer."""
    locked = load_case()[0]
    if set(p) != set(locked):
        raise ValueError(f"parameter names differ from the locked case: {sorted(set(p) ^ set(locked))}")
    changed = {k: (p[k], locked[k]) for k in locked if float(p[k]) != locked[k]}
    if changed:
        raise ValueError(f"these values differ from the locked supervisor values (given, locked): {changed}. "
                         "The problem is fixed; see AGENTS.md section 3.")


def load_reference_settings(path: Path = REFERENCE_SETTINGS_PATH) -> dict:
    return json.loads(Path(path).read_text())
