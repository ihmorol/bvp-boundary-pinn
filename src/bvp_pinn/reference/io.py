"""Write and load reference data (docs/SPEC_reference_solutions.md section 6).

PINN evaluation code calls load_reference after model selection. Training and
selection code must not import this module (gate G5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import CASE_PATH, file_sha256

FILES = ("reference.npz", "meta.json")


def write_reference(folder: Path, arrays: dict, meta: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "reference.npz", **arrays)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    (folder / "SHA256SUMS").write_text("".join(f"{file_sha256(folder / n)}  {n}\n" for n in FILES))


def load_reference(folder: Path, case_path: Path = CASE_PATH):
    """Return (arrays, meta). Refuses data whose hashes or physical case do not match."""
    folder = Path(folder)
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        if file_sha256(folder / name) != digest:
            raise ValueError(f"hash mismatch for {folder / name}")
    meta = json.loads((folder / "meta.json").read_text())
    if meta["case_sha256"] != file_sha256(case_path):
        raise ValueError("reference was built from a different physical case file")
    with np.load(folder / "reference.npz") as data:
        arrays = {k: data[k] for k in data.files}
    return arrays, meta
