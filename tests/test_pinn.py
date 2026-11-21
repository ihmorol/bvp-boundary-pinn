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
    """The constraint layer must satisfy its conditions for any weights (docs/BC_ENFORCEMENT_REVIEW.md)."""
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
    """Gate G5: only evaluate.py may load reference data."""
    folder = Path(model.__file__).parent
    for path in folder.glob("*.py"):
        if path.name in ("evaluate.py", "plots.py"):
            continue
        text = path.read_text()
        assert "load_reference" not in text and "reference import io" not in text, path.name
