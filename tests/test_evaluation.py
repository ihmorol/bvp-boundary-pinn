"""Evaluation and campaign bookkeeping: the target reference, resumed campaigns and the tables."""
import json
import shutil

import numpy as np
import pytest
import torch

from bvp_pinn.config import load_case
from bvp_pinn.evaluation import evaluate, plots
from bvp_pinn.reference import io

torch.set_default_dtype(torch.float64)


def test_reference_folders_follow_the_reference_settings():
    """The folder names come from L_reference and the final tolerance, as reference.run names them."""
    assert evaluate.REFERENCES["A"].name == "locked-branchA-L20-tol1e-08"
    assert evaluate.REFERENCES["B"].name == "locked-branchB-L20-tol1e-08"


def _fake_reference(folder, **checks):
    _, sha = load_case()
    eta = np.linspace(0.0, 20.0, 5)
    io.write_reference(folder, {"eta": eta, "z": np.ones((12, 5)), "H": np.ones((5, 5))},
                       {"case_sha256": sha, "L": 20.0, "checks": checks})


def test_evaluation_refuses_a_target_that_failed_its_checks(tmp_path, monkeypatch):
    good, bad = tmp_path / "good", tmp_path / "bad"
    _fake_reference(good, G3_fixed_L_pass=True, far_field_valid=True)
    _fake_reference(bad, G3_fixed_L_pass=True, far_field_valid=False)  # branch B-like: no decay
    monkeypatch.setattr(evaluate, "REFERENCES", {"A": good, "B": tmp_path / "missing"})
    assert set(evaluate.load_references()) == {"A"}  # B may be missing; it only names the nearest branch
    monkeypatch.setattr(evaluate, "REFERENCES", {"A": bad})
    with pytest.raises(ValueError, match="not a usable target"):
        evaluate.load_references()


def test_saved_target_reference_passes_the_guard():
    refs = evaluate.load_references()
    assert refs["A"]["eta"][-1] == 20.0 and "B" in refs


def test_paired_seed_table_counts_wins_and_failed_runs(tmp_path):
    """Rule G6 needs, per pair of arms, how many paired seeds one beats the other. A failed run
    (non-finite error) never wins, and a finite run beats it."""
    rows = []
    for seed, (a, b) in enumerate([(1e-4, 2e-4), (1e-4, 2e-4), (3e-4, 2e-4), (np.nan, 2e-4), (1e-4, np.nan)]):
        for arm, v in (("B0", a), ("B1", b)):
            rows.append({"arm": arm, "seed": str(seed), "accepted": "True", "wall_rel_max": v,
                         "max_field_rel_l2": v, "after_adam_wall_rel_max": 10 * v,
                         "after_adam_max_field_rel_l2": 10 * v})
    plots.accuracy_table(rows, tmp_path)
    text = (tmp_path / "accuracy.md").read_text()
    # B0 wins seeds 0, 1 and 4 (B1 failed there); B1 wins seed 2 and seed 3 (B0 failed there)
    assert "| B0 |  | 3 of 5 |" in text and "| B1 | 2 of 5 |  |" in text


def test_resumed_campaign_judges_a_run_that_lost_its_verdict(tmp_path, monkeypatch):
    """A session that stops after training but before the acceptance rule leaves meta.json and
    model.pt without selection.json. The next campaign start judges it and does not retrain it."""
    if shutil.which("g++") is None:
        pytest.skip("needs g++ to build the engine")
    from bvp_pinn.pinn import campaign
    from bvp_pinn.pinn.run import run
    folder = run("B0", 0, dry_run=True, runs_root=tmp_path, assess=False)
    torch.set_default_device("cpu")
    assert (folder / "meta.json").exists() and not (folder / "selection.json").exists()
    monkeypatch.setattr(campaign, "planned", lambda *a, **k: [("B0", 0, folder, True)])
    monkeypatch.setattr(campaign, "run", lambda *a, **k: pytest.fail("a finished run was trained again"))
    campaign.main(["--arms", "B0", "--seeds", "0", "--dry-run"])
    verdict = json.loads((folder / "selection.json").read_text())
    assert verdict["accepted"] is False  # 40 Adam steps cannot pass the rule
    assert not list(folder.glob("*.tmp"))  # atomic writes leave no temporary files
