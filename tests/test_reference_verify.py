"""SPEC V3 (manufactured solution), V11 (benchmarks), V12 (slope check) and the reference data loader."""
import json

import numpy as np
import pytest

from bvp_pinn.config import load_case
from bvp_pinn.reference import io, verify
from bvp_pinn.reference.verify import mms_order


def test_manufactured_solution_order():
    out = mms_order(load_case()[0], 5.0, [161, 321, 641])
    assert out["mesh_kept_fixed"]
    assert out["observed_orders"][-1] >= 3.6


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


def test_multiple_shooting_recovers_manufactured_solution():
    from bvp_pinn.reference import equations as eq
    from bvp_pinn.reference.shooting import multiple_shooting
    from bvp_pinn.reference.verify import manufactured

    p, L = load_case()[0], 5.0

    def fun(x, y):
        zm, dzm = manufactured(np.array([x]))
        return eq.rhs(x, y, p) + dzm[:, 0] - eq.rhs(x, zm[:, 0], p)

    end = manufactured(np.array([0.0, L]))[0]
    shift = eq.boundary(end[:, 0], end[:, 1], p)
    nodes = np.linspace(0.0, L, 11)
    sol, _, mismatch = multiple_shooting(lambda x, y: fun(x, y), lambda a, b: eq.boundary(a, b, p) - shift,
                                         nodes, manufactured(nodes[:-1])[0] + 0.02)
    grid = np.linspace(0.0, L, 301)
    assert mismatch < 1e-10
    assert np.max(np.abs(sol(grid) - manufactured(grid)[0])) < 1e-8


def test_benchmarks_published_and_exact():
    """SPEC V11: Abbas et al. (2013) at kappa = 5 (L = 10) and the exact flat-sheet Nusselt number."""
    from bvp_pinn.config import load_reference_settings
    from bvp_pinn.reference import benchmark as bm
    from bvp_pinn.reference import solver as sv

    s = load_reference_settings()
    base = dict(load_case()[0], **bm.OFF)
    r = sv.solve_case(dict(base, kappa=5.0, Pr=1.0), 10.0, 1e-9, s, label="t", guess=bm.GUESS)
    assert abs(-r.wall["f2_0"] + 1 / 5 - bm.ABBAS[5]) <= 1e-5
    r = sv.solve_case(dict(base, kappa=bm.FLAT_KAPPA, Pr=3.0), 20.0, 1e-9, s, label="t", guess=bm.GUESS)
    assert abs(-r.wall["th1_0"] / r.wall["theta_0"] / bm.nusselt_exact(3.0) - 1) <= 1e-5


def test_slope_check_on_known_functions():
    eta = np.linspace(0.0, 5.0, 2001)
    e = np.exp(-eta)
    z = np.array([1 - e, e, -e, e, e, -e, e, -e, e, -e, e, -e])
    H = np.array([-e, e, e, e, e])
    assert max(verify.slope_check(eta, z, H).values()) < 1e-9
