"""SPEC V10 (Runge-Kutta shooting against solve_bvp), V12 (slope check) and the reference data loader.

Every solve is the locked case (AGENTS.md section 3).
"""
import json

import numpy as np
import pytest

from bvp_pinn.config import load_case
from bvp_pinn.reference import io, verify


def test_loader_round_trip_and_tamper(tmp_path):
    _, sha = load_case()
    arrays = {"eta": np.linspace(0, 1, 5), "z": np.ones((12, 5))}
    io.write_reference(tmp_path, arrays, {"case_sha256": sha})
    got, meta = io.load_reference(tmp_path)
    np.testing.assert_array_equal(got["z"], arrays["z"])
    io.write_reference(tmp_path, arrays, {"case_sha256": "0" * 64})
    with pytest.raises(ValueError, match="different physical case"):
        io.load_reference(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"case_sha256": sha}))
    with pytest.raises(ValueError, match="hash mismatch"):
        io.load_reference(tmp_path)


def test_multiple_shooting_agrees_with_solve_bvp():
    """The second solver, Runge-Kutta multiple shooting from the smooth guess, reaches the same
    locked-case solution as solve_bvp at L = 20 (both branch A, both with the reference's far conditions)."""
    from bvp_pinn.config import load_reference_settings
    from bvp_pinn.reference import equations as eq
    from bvp_pinn.reference import solver as sv
    from bvp_pinn.reference.shooting import multiple_shooting

    p, s = load_case()[0], load_reference_settings()
    L, guess = 20.0, {"rate": 1.0, "theta0": 0.3}
    colloc = sv.solve_case(p, L, 1e-8, s, label="test", guess=guess)
    nodes = sv.wall_mesh(L, s["rk"]["segments"] + 1, s["rk"]["first_segment"])
    sol, _, mismatch = multiple_shooting(lambda x, y: eq.rhs(x, y, p),
                                         lambda a, b: eq.boundary(a, b, p, L=sv.far_L(L, s)),
                                         nodes, sv.smooth_guess(nodes[:-1], p, guess["rate"], guess["theta0"]),
                                         rtol=1e-10)
    assert colloc.ok and mismatch < 1e-9
    assert sv.rel_diff(colloc.wall, sv.wall_values(lambda x: sol(x)[:, 0], L)) < 1e-7


def test_slope_check_on_known_functions():
    eta = np.linspace(0.0, 5.0, 2001)
    e = np.exp(-eta)
    z = np.array([1 - e, e, -e, e, e, -e, e, -e, e, -e, e, -e])
    H = np.array([-e, e, e, e, e])
    assert max(verify.slope_check(eta, z, H).values()) < 1e-9

