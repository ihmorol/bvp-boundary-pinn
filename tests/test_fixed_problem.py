"""The problem is fixed: Eqs. 14 to 18, boundary conditions B1 to B12 and the 22 supervisor values.

AGENTS.md section 3: none of the three changes in any run, tool or use case; only the way the
conditions are imposed and the problem is solved may change. These tests check the guards that
hold every results path to that problem: the reference study, every PINN run, the acceptance
rule and the C++ engine. They use a fresh solve of the locked case, not saved reference data.
"""
import json
import shutil

import numpy as np
import pytest
import torch

from bvp_pinn import config
from bvp_pinn.config import check_locked, load_case, load_reference_settings
from bvp_pinn.pinn import constraints, losses, model, runs, train
from bvp_pinn.reference import solver as sv

torch.set_default_dtype(torch.float64)
P = load_case()[0]


@pytest.fixture(scope="module")
def solution():
    """solve_bvp on the locked case at L = 20, from the smooth guess (branch A), tol 1e-7."""
    r = sv.solve_case(P, 20.0, 1e-7, load_reference_settings(), label="fixed-problem test",
                      guess={"rate": 1.0, "theta0": 0.3})
    assert r.ok
    return r


def test_case_file_is_pinned(tmp_path):
    """The case file must hash to the value locked on 2026-11-16; one changed value is refused."""
    _, sha = load_case()
    assert sha == config.LOCKED_CASE_SHA256
    record = json.loads(config.CASE_PATH.read_text())
    record["physical_parameters"]["kappa"] = 0.3
    edited = tmp_path / "case.json"
    edited.write_text(json.dumps(record, indent=2))
    with pytest.raises(ValueError, match="not the locked case"):
        load_case(edited)


def test_values_are_read_only_and_checked():
    p = load_case()[0]
    with pytest.raises(TypeError):
        p["kappa"] = 1.0
    check_locked(p)
    check_locked(dict(p))
    for bad in (dict(p, kappa=1.0), dict(p, Da=0.2000001), {k: v for k, v in p.items() if k != "Pr"},
                dict(p, M=5.0)):
        with pytest.raises(ValueError):
            check_locked(bad)


def test_training_refuses_other_values():
    cfg = {"structure": "single", "windows": [0.0, 20.0], "L": 20.0, "width": 6, "depth": 2,
           "input_map": "log", "map_scale": 0.1, "bc_mode": "exact", "tfc_decay": 0.5, "weight_alpha": 0.9,
           "adam_steps": 1, "lr": 1e-3, "lr_end": 1e-4, "weight_every": 1000, "log_every": 1}
    nets = model.build(cfg, P)
    eta = torch.linspace(0.0, 19.9, 20).reshape(-1, 1)
    with pytest.raises(ValueError, match="locked"):
        train.train(nets, dict(P, kappa=1.0), cfg, eta, eta, lambda row: None)


def test_run_settings_cannot_name_a_value_or_change_the_problem():
    for key, value in (("kappa", 1.0), ("Pr", 7.0), ("theta_w", 1.5)):
        with pytest.raises(ValueError, match="locked supervisor values"):
            runs.resolve("B0", **{key: value})
    with pytest.raises(ValueError, match="unknown settings"):
        runs.resolve("B0", far_form="unit")  # the far-condition form is not a training setting
    with pytest.raises(ValueError, match="reference L"):
        runs.resolve("B0", L=30.0, windows=[0.0, 30.0])
    assert runs.resolve("B0")["L"] == load_reference_settings()["L_reference"]


@pytest.mark.parametrize("mode", ["soft", "linear", "exact"])
@pytest.mark.parametrize("structure", ["single", "windows"])
def test_every_mode_imposes_all_twelve_conditions(mode, structure):
    """Each of B1 to B12 is either built in (exact for any weights) or a term of the loss."""
    cfg = {"structure": structure, "windows": [0.0, 0.5, 2.0, 20.0], "L": 20.0, "width": 6, "depth": 2,
           "input_map": "log", "map_scale": 0.1, "bc_mode": mode, "tfc_decay": 0.5}
    torch.manual_seed(8)
    nets = model.build(cfg, P)
    bc = losses.boundary_errors(nets, P, 20.0).detach()
    soft = constraints.SOFT_BCS[mode]
    exact = [i for i in range(12) if i not in soft]
    assert sorted(exact + soft) == list(range(12))
    assert not exact or bc[exact].abs().max() < 1e-12
    terms = train.loss_parts(nets, P, cfg, train.full_problem(cfg, torch.linspace(0.0, 19.9, 30).reshape(-1, 1)))[0]
    assert len(terms) == 1 + len(soft) + (structure == "windows")
    for j, i in enumerate(soft):
        torch.testing.assert_close(terms[1 + j].detach(), bc[i] ** 2)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++ to build the engine")
def test_engine_solves_the_same_problem(solution):
    """The C++ engine has its own transcription of Eqs. 14 to 18 and B1 to B12. From the solve_bvp
    solution its Newton step must be at its own discretization level, and it must see a known change."""
    from bvp_pinn.pinn.estimate import WALL, engine_step
    eta = sv.wall_mesh(20.0, 3201, 2e-3)
    z = solution.sol(eta)
    step, _ = engine_step(eta, z)
    assert max(abs(step[i, 0]) / abs(z[i, 0]) for i in WALL.values()) < 1e-3
    bump = 1e-3 * np.exp(-eta)
    zb = z.copy()
    zb[6] += bump
    zb[7] -= bump
    step_b, _ = engine_step(eta, zb)
    assert abs((step_b[6, 0] - step[6, 0]) / 1e-3 - 1) < 0.05


def test_reference_takes_f_bounded_at_infinity(solution):
    """Reference audit D1: f' = C/(eta + kappa) solves the homogeneous Eq. 14, so the far conditions
    alone allow a family of solutions with (eta + kappa) f' -> C. The locked-case solution is the
    member with C = 0 (f bounded): there q f' decays exponentially instead of levelling off at C."""
    eta = np.linspace(10.0, 20.0, 201)
    qf = (eta + P["kappa"]) * solution.sol(eta)[1]
    at = lambda x: abs(qf[np.argmin(abs(eta - x))])
    assert np.max(np.abs(qf)) < 1e-3
    assert at(15.0) < 0.1 * at(10.0) and at(18.0) < 0.1 * at(15.0)


def test_reference_settings_keep_the_fixed_far_conditions():
    s = load_reference_settings()
    assert s["far_field"] == "asymptotic" and s.get("far_form", "local") == "local"
    assert not any("continuation" in k for k in s)  # no solve through other parameter values
