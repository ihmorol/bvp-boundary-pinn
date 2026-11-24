"""Gate G1 for the PINN: derivatives, equations on tensors, loss terms, and no reference leakage."""
from pathlib import Path

import numpy as np
import pytest
import torch

from bvp_pinn.config import load_case
from bvp_pinn.pinn import constraints, losses, model, physics
from bvp_pinn.reference import equations as eq

torch.set_default_dtype(torch.float64)
P = load_case()[0]


def known(eta):
    """Five functions with easy derivatives, standing in for a network."""
    return torch.cat([torch.sin(eta), torch.exp(-eta), eta**3, torch.cos(2 * eta), 1 / (1 + eta)], 1)


def test_derivatives_up_to_order_four():
    eta = torch.linspace(0.1, 3, 50).reshape(-1, 1).requires_grad_(True)
    z, H = model.states(known, eta)
    e = eta.detach()
    expected_z = [torch.sin(e), torch.cos(e), -torch.sin(e), -torch.cos(e),
                  torch.exp(-e), -torch.exp(-e), e**3, 3 * e**2,
                  torch.cos(2 * e), -2 * torch.sin(2 * e), 1 / (1 + e), -1 / (1 + e) ** 2]
    expected_H = [torch.sin(e), torch.exp(-e), 6 * e, -4 * torch.cos(2 * e), 2 / (1 + e) ** 3]
    for got, want in zip(z + H, expected_z + expected_H):
        torch.testing.assert_close(got.detach(), want, rtol=1e-12, atol=1e-12)


def test_torch_equations_match_numpy():
    rng = np.random.default_rng(0)
    e = rng.uniform(0, 8, 40)
    z = rng.uniform(-1, 1, (12, 40))
    z[6] = rng.uniform(0, 1, 40)
    H = rng.uniform(-3, 3, (5, 40))
    col = lambda a: torch.tensor(a).reshape(-1, 1)
    got = physics.residuals(col(e), [col(r) for r in z], [col(r) for r in H], P).numpy().T
    coef = np.stack([np.full(40, 1 + P["Delta"]), np.full(40, 1 + P["Delta"] / 2), eq.thermal_coefficient(z, P),
                     np.full(40, 1 / (P["Le"] * P["Pr"])), np.full(40, 1 / P["Lb"])])
    np.testing.assert_allclose(got * coef, eq.residuals(e, z, H, P), rtol=1e-12, atol=1e-12)
    za, zb = rng.uniform(-1, 1, 12), rng.uniform(-1, 1, 12)
    got = physics.boundary([torch.tensor(v) for v in za], [torch.tensor(v) for v in zb], P, 20.0)
    np.testing.assert_allclose(got.numpy(), eq.boundary(za, zb, P, L=20.0), rtol=1e-12, atol=1e-14)


def test_gradient_norm_weights_follow_wang_et_al_2023():
    w = torch.tensor([1.0, -2.0], requires_grad=True)
    res, term = (3 * w[0]) ** 2, (w[1] - 1) ** 2  # gradient norms 18 and 6
    weights = losses.GradNormWeights(2, alpha=0.9)
    weights.update([w], [res, term])
    hat = torch.tensor([24 / 18, 24 / 6])
    torch.testing.assert_close(weights.w, 0.9 + 0.1 * hat)


@pytest.mark.parametrize("structure", ["single", "windows"])
@pytest.mark.parametrize("mode", ["linear", "exact"])
def test_built_in_conditions_hold_for_random_weights(mode, structure):
    """The constraint layer must satisfy its conditions for any weights (docs/pinn/BC_ENFORCEMENT_REVIEW.md)."""
    cfg = {"structure": structure, "windows": [0.0, 0.5, 2.0, 20.0], "L": 20.0, "width": 16, "depth": 3,
           "input_map": "linear", "map_scale": 0.1, "bc_mode": mode, "tfc_decay": 0.5}
    torch.manual_seed(1)
    nets = model.build(cfg, P)
    bc = losses.boundary_errors(nets, P, 20.0).detach().abs()
    soft = constraints.SOFT_BCS[mode]
    exact = [i for i in range(12) if i not in soft]
    assert bc[exact].max() < 1e-12, bc
    assert not soft or bc[soft].min() > 1e-6  # the soft ones are not satisfied by accident


def test_self_replay_is_zero_at_start_and_targets_get_no_gradient():
    net = torch.nn.ModuleList([model.MLP(8, 2, 0.0, 20.0)])
    eta = torch.linspace(0, 5, 20).reshape(-1, 1)
    frozen = net[0](eta).detach()
    loss = losses.self_replay_mse(net, eta, frozen)
    assert loss.item() == 0.0
    loss.backward()
    assert not frozen.requires_grad


def test_interface_is_zero_for_identical_window_networks():
    cfg = {"structure": "windows", "windows": [0.0, 1.0, 2.0], "L": 2.0, "width": 8, "depth": 2,
           "input_map": "linear", "map_scale": 0.1, "bc_mode": "soft"}
    nets = model.build(cfg, P)
    nets[1].load_state_dict(nets[0].state_dict())
    nets[1].lo, nets[1].hi = nets[0].lo, nets[0].hi  # same function on both sides
    assert losses.interface_mse(nets).item() == pytest.approx(0.0, abs=1e-30)


def test_training_code_never_reads_the_reference():
    """Gate G5: no file of the training and selection package reads reference data or imports evaluation."""
    import ast
    import re
    folder = Path(model.__file__).parent
    files = list(folder.glob("*.py"))
    assert len(files) >= 15
    for path in files:
        text = path.read_text()
        # load_reference_settings reads configs/reference.json (our numerical settings), not reference data
        assert not re.search(r"load_reference(?!_settings)", text) and "locked-branch" not in text, path.name
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom):
                names = [f"{node.module or ''}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                assert "evaluation" not in name and "reference.io" not in name, (path.name, name)


@pytest.mark.parametrize("mode", ["soft", "linear", "exact"])
@pytest.mark.parametrize("input_map", ["linear", "log"])
def test_forward_jets_equal_nested_autograd(mode, input_map):
    """jets.states must give the same z, H and the same weight gradients as states_autograd."""
    cfg = {"structure": "windows", "windows": [0.0, 0.5, 20.0], "L": 20.0, "width": 16, "depth": 3,
           "input_map": input_map, "map_scale": 0.1, "bc_mode": mode, "tfc_decay": 0.5}
    torch.manual_seed(2)
    for net in model.build(cfg, P):
        eta = (net.lo + (net.hi - net.lo) * torch.rand(30, 1)).requires_grad_(True)
        a = torch.cat(sum(model.states_autograd(net, eta), []), 1)
        j = torch.cat(sum(model.jets.states(net, eta), []), 1)
        torch.testing.assert_close(j, a, rtol=1e-11, atol=1e-11)
        ga = torch.autograd.grad(a.pow(2).sum(), list(net.parameters()))
        gj = torch.autograd.grad(j.pow(2).sum(), list(net.parameters()))
        for x, y in zip(gj, ga):
            torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)


def test_start_profile_matches_the_reference_smooth_guess():
    """start.SmoothStart is the analytic guess of reference/solver.py, not reference data."""
    from bvp_pinn.pinn.start import SmoothStart
    from bvp_pinn.reference.solver import smooth_guess
    eta = np.linspace(0, 20, 41)
    want = smooth_guess(eta, P, 1.0, 0.3)
    jets = SmoothStart(P, 1.0, 0.3).jets(torch.tensor(eta).reshape(-1, 1), 2)
    got = {0: jets[0][:, 0], 1: jets[1][:, 0], 2: jets[2][:, 0], 4: jets[0][:, 1], 5: jets[1][:, 1],
           6: jets[0][:, 2], 7: jets[1][:, 2], 8: jets[0][:, 3], 9: jets[1][:, 3], 10: jets[0][:, 4], 11: jets[1][:, 4]}
    for i, g in got.items():
        np.testing.assert_allclose(g.numpy(), want[i], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("mode", ["soft", "exact"])
def test_start_profile_with_zero_last_layer(mode):
    """With zero_last the raw network is exactly the start profile; jets still match autograd."""
    cfg = {"structure": "single", "windows": [0.0, 20.0], "L": 20.0, "width": 16, "depth": 3,
           "input_map": "log", "map_scale": 0.1, "bc_mode": mode, "tfc_decay": 0.5,
           "start": "smooth", "zero_last": True}
    torch.manual_seed(3)
    net = model.build(cfg, P)[0]
    eta = (20 * torch.rand(30, 1)).requires_grad_(True)
    inner = net.net if mode == "exact" else net
    torch.testing.assert_close(inner(eta.detach()), inner.start(eta.detach()), rtol=0, atol=0)
    a = torch.cat(sum(model.states_autograd(net, eta), []), 1)
    j = torch.cat(sum(model.jets.states(net, eta), []), 1)
    torch.testing.assert_close(j, a, rtol=1e-11, atol=1e-11)
    if mode == "exact":
        assert losses.boundary_errors(model.build(cfg, P), P, 20.0).detach().abs().max() < 1e-12


def test_residual_scale_is_fixed_and_keeps_the_zero_set():
    eta = torch.tensor([[0.0], [1.0], [10.0]])
    assert losses.residual_scale(eta, P, "none") is None
    s = losses.residual_scale(eta, P, "log")
    h = eta + P["b"]
    torch.testing.assert_close(s, torch.cat([h**4, h**2, h**2, h**2, h**2], 1))
    assert (s > 0).all()
    with pytest.raises(ValueError):
        losses.residual_scale(eta, P, "from_reference")


def test_weight_cap_bounds_the_weights():
    w = torch.tensor([1.0, -2.0], requires_grad=True)
    weights = losses.GradNormWeights(2, alpha=0.0, cap=2.0)
    weights.update([w], [(3 * w[0]) ** 2, 1e-6 * (w[1] - 1) ** 2])
    assert weights.w.max().item() == 2.0


def test_full_bfgs_solves_a_small_least_squares_problem():
    """bfgs.minimize on an ill-conditioned linear least-squares fit reaches the exact answer."""
    from bvp_pinn.pinn import bfgs
    torch.manual_seed(4)
    lin = torch.nn.Linear(6, 1)
    X = torch.randn(40, 6) * torch.logspace(0, 2, 6)
    y = X @ torch.arange(1.0, 7.0).reshape(-1, 1) + 0.5
    info = bfgs.minimize(lin, lambda: (lin(X) - y).pow(2).mean(), 200)
    torch.testing.assert_close(lin.weight.detach().reshape(-1), torch.arange(1.0, 7.0), rtol=1e-8, atol=1e-8)
    assert info["bfgs_iterations"] < 200


def test_levenberg_marquardt_lowers_the_exact_mode_loss():
    """lm.minimize on a tiny exact-mode network: its loss is r.r, and accepted steps only lower it."""
    from bvp_pinn.pinn import lm
    cfg = {"structure": "single", "windows": [0.0, 20.0], "L": 20.0, "width": 6, "depth": 2,
           "input_map": "log", "map_scale": 0.1, "bc_mode": "exact", "tfc_decay": 0.5,
           "start": "smooth", "zero_last": False, "residual_scale": "log"}
    torch.manual_seed(5)
    nets = model.build(cfg, P)
    eta = torch.linspace(0.0, 19.9, 40).reshape(-1, 1)
    before = losses.equation_mse(nets, [(0, eta)], P, "log").sum().item()
    rows = []
    info = lm.minimize(nets, P, cfg, {"groups": [(0, eta)]}, torch.ones(1), 4, log=rows.append, chunk=40)
    after = losses.equation_mse(nets, [(0, eta)], P, "log").sum().item()
    assert info["lm_accepted_steps"] >= 1 and after < before
    assert info["lm_final_loss"] == pytest.approx(after, rel=1e-10)
    assert all(b["loss"] <= a["loss"] for a, b in zip(rows, rows[1:]))


def test_balanced_scale_follows_the_coefficients():
    """balanced: q^4/(1 + c14 q^2) for R14, q^2/(1 + c15 q^2) for R15, q^2 for R16 to R18."""
    eta = torch.tensor([[0.0], [1.0], [19.0]])
    s = losses.residual_scale(eta, P, "balanced")
    q2 = (eta + P["kappa"]) ** 2
    c14 = (1 / P["Da"]) / (1 + P["Delta"])
    c15 = 2 * P["Delta"] * P["B"] / (1 + P["Delta"] / 2)
    torch.testing.assert_close(s[:, 0:1], q2**2 / (1 + c14 * q2))
    torch.testing.assert_close(s[:, 1:2], q2 / (1 + c15 * q2))
    torch.testing.assert_close(s[:, 2:], q2.expand(-1, 3))
    assert (s > 0).all() and s[-1, 0] < (eta[-1, 0] + P["b"]) ** 4  # far R14 weight is capped


def test_tail_fit_finds_a_constant_tail():
    """selfcheck.tail_fit returns C1 = 0 for theta = C q^-m and C1 = c for theta = c + C q^-m."""
    from bvp_pinn.pinn import selfcheck
    m_, c = 0.24, 0.05

    class Tail(torch.nn.Module):
        def __init__(self, const):
            super().__init__()
            self.const = const
            self.lo, self.hi = 0.0, 20.0

        def forward(self, eta):
            th = self.const + 0.2 * (eta + P["kappa"]) ** -m_
            return torch.cat([torch.zeros_like(eta), torch.zeros_like(eta), th, th, torch.zeros_like(eta)], 1)

    for const in (0.0, c):
        out = selfcheck.tail_fit(torch.nn.ModuleList([Tail(const)]), 20.0, P, m_, m_)
        assert out["tail_C1_theta"] == pytest.approx(const, abs=1e-9)


def tiny(structure, mode, **extra):
    """A small config of the given structure and boundary mode, for fast tests."""
    cfg = {"structure": structure, "windows": [0.0, 0.5, 2.0, 20.0], "L": 20.0, "width": 6, "depth": 2,
           "input_map": "log", "map_scale": 0.1, "bc_mode": mode, "tfc_decay": 0.5, "start": "smooth",
           "zero_last": False, "residual_scale": "log", "replay": "physics", "replay_weight": 1.0}
    return {**cfg, **extra}


@pytest.mark.parametrize("structure,mode", [("windows", "exact"), ("windows", "soft"), ("single", "linear")])
def test_levenberg_marquardt_works_on_the_training_objective(structure, mode):
    """lm.minimize starts from exactly the frozen-weight training loss of train.weighted, for every arm
    shape and boundary mode (B1 window networks with the interface term, soft boundary terms), and lowers it."""
    from bvp_pinn.pinn import lm, train
    cfg = tiny(structure, mode)
    torch.manual_seed(6)
    nets = model.build(cfg, P)
    eta = torch.linspace(0.0, 19.9, 60).reshape(-1, 1)
    stage = train.full_problem(cfg, eta)
    terms = train.loss_parts(nets, P, cfg, stage)[0]
    w = 0.5 + torch.rand(len(terms))  # frozen weights, not all 1
    loss = train.weighted(terms, torch.zeros(()), w, cfg).item()
    info0 = lm.minimize(nets, P, cfg, stage, w, 0)
    assert info0["lm_final_loss"] == pytest.approx(loss, rel=1e-10)
    info = lm.minimize(nets, P, cfg, stage, w, 3, chunk=30)
    after = train.weighted(train.loss_parts(nets, P, cfg, stage)[0], torch.zeros(()), w, cfg).item()
    assert info["lm_accepted_steps"] >= 1 and after < loss
    assert info["lm_final_loss"] == pytest.approx(after, rel=1e-9)


def test_config_checks_and_one_refine_chain_for_every_arm():
    from bvp_pinn.pinn import runs
    chains = {runs.resolve(arm)["refine"] for arm in ("B0", "B1", "B2", "B3")}
    assert chains == {"lbfgs:log+lm:balanced"}  # every arm ends on the same objective and optimizer
    with pytest.raises(ValueError):
        runs.resolve("B1", L=30.0)  # the last window edge would no longer be L
    with pytest.raises(ValueError):
        runs.resolve("B0", refine="lbfgs+newton")
    with pytest.raises(ValueError):
        runs.resolve("B0", refine="lm:plain")
    assert runs.parse_set(["adam_steps=10", "lr=1e-3", "refine=lm"]) == {"adam_steps": 10, "lr": 1e-3, "refine": "lm"}


def test_acceptance_rule():
    from bvp_pinn.pinn.estimate import judge
    rule = {"score_max": 1e-3, "tail_share_max": 0.05, "boundary_max": 1e-8}
    good = {"result": "finished", "self_check": {"decaying_far_field": True, "m": 0.24, "mp": 0.2,
                                                "tail_share": 0.001, "max_boundary_abs": 1e-14}}
    assert judge(good, {"score": 4e-5}, rule)["accepted"] is True
    assert judge(good, {"score": 2e-3}, rule)["accepted"] is False
    assert judge(good, {"score": None, "error": "no g++"}, rule)["accepted"] is None
    flat_tail = {**good, "self_check": {**good["self_check"], "tail_share": 0.6}}
    assert judge(flat_tail, {"score": 1e-5}, rule)["accepted"] is False
    no_decay = {**good, "self_check": {**good["self_check"], "decaying_far_field": False, "m": -0.2}}
    out = judge(no_decay, {"score": None, "error": "x"}, rule)
    assert out["accepted"] is False and len(out["reasons"]) == 2
    assert judge({"result": "non-finite loss in Adam stage 0"}, {"score": 1e-5}, rule)["accepted"] is False


def test_every_arm_trains_end_to_end(tmp_path):
    """The whole pipeline of each arm with tiny settings: stages, Adam, the full refine chain
    (L-BFGS, then Levenberg-Marquardt, also for B1's window networks) and a complete run folder."""
    import json
    from bvp_pinn.pinn.run import run
    small = {"adam_steps": 8, "lbfgs_iters": 2, "lm_iters": 1, "n_points": 48, "n_monitor": 24, "log_every": 4,
             "width": 6, "depth": 2}
    for arm in ("B0", "B1", "B2", "B3"):
        folder = run(arm, 0, runs_root=tmp_path, assess=False, **small)
        meta = json.loads((folder / "meta.json").read_text())
        assert meta["result"] == "finished", arm
        assert [s["refine"] for s in meta["refine_stages"]] == ["lbfgs", "lm"], arm
        assert meta["refine_stages"][1]["lm_iterations"] == 1, arm
        for name in ("config.json", "history.csv", "model.pt", "points.npz", "stage0.pt"):
            assert (folder / name).exists(), (arm, name)
    torch.set_default_device("cpu")
    # The acceptance rule on a finished run, from a fresh process state (default dtype float32).
    from bvp_pinn.pinn.estimate import assess_run
    torch.set_default_dtype(torch.float32)
    verdict = assess_run(folder)
    assert torch.get_default_dtype() == torch.float64
    est = json.loads((folder / "error_estimate.json").read_text())
    assert est["score"] is not None and verdict["accepted"] is not None  # the estimate ran
    assert json.loads((folder / "selection.json").read_text()) == json.loads(json.dumps(verdict))
