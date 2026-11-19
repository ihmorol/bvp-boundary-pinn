"""Build and verify the classical reference for the locked case.

Run from the repository root:
    uv run python -m bvp_pinn.reference.run

Writes reference/study.json, reference/study.md and one folder per solution
branch, reference/<case_id>/, holding reference.npz, meta.json, SHA256SUMS and
figures/. Steps follow docs/SPEC_reference_solutions.md section 5.
"""
from __future__ import annotations

import datetime
import json
import platform
import subprocess

import numpy as np
import scipy

from ..config import ROOT, load_case, load_reference_settings
from . import equations as eq
from . import benchmark, io, plots, shooting, verify
from . import solver as sv


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def scan_branches(p, L, tol, s):
    """Solve from every scan guess; group converged solutions by wall values."""
    branches, failures = [], []
    for rate in s["branch_scan"]["rates"]:
        for theta0 in s["branch_scan"]["theta0"]:
            guess = {"rate": rate, "theta0": theta0}
            r = sv.solve_case(p, L, tol, s, label=f"r{rate}-t{theta0}", guess=guess)
            if not r.ok:
                failures.append(r.summary())
                continue
            for b in branches:
                # Distinct branches differ at order one; 1e-2 groups repeats found at tol 1e-5.
                if sv.rel_diff(b["result"].wall, r.wall) <= 1e-2:
                    b["guesses"].append(guess)
                    break
            else:
                branches.append({"result": r, "guesses": [guess]})
    branches.sort(key=lambda b: b["result"].wall["theta_0"])
    for name, b in zip("ABCDEFGH", branches):
        b["name"] = name
    return branches, failures


def ladder_tol(p, start, s):
    out, cur = [], start
    for tol in s["tol_values"]:
        cur = sv.solve_case(p, start.L, tol, s, label=f"tol{tol:g}", x=cur.x, y=cur.y)
        out.append(cur)
        if not cur.ok:
            break
    return out


def ladder_L(p, start, s):
    out, cur, tol = [], start, s["tol_values"][-1]
    for L in s["L_values"]:
        x, y = (cur.x, cur.y) if L == cur.L else sv.extend(cur, L, s)
        cur = sv.solve_case(p, L, tol, s, label=f"L{L}", x=x, y=y)
        out.append(cur)
        if not cur.ok:
            break
    return out


def export(p, case_sha, s, branch, final, study_branch, out_root, extra):
    eta = sv.wall_mesh(final.L, s["eval_points"], s["eval_first_step"])
    z = final.sol(eta)
    H = eq.highest_derivatives(eta, z, p)
    min_theta_coeff = float(np.min(eq.thermal_coefficient(z, p)))
    case_id = f"locked-branch{branch['name']}-L{final.L:g}-tol{final.tol:.0e}"
    meta = {
        "case_id": case_id, "branch": branch["name"], "case_sha256": case_sha,
        "physical_parameters": p, "solver": "scipy.integrate.solve_bvp", "scipy_version": scipy.__version__,
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "L": final.L, "tol": final.tol, "bc_tol": final.tol, "max_nodes": s["max_nodes"],
        "initial_nodes": s["initial_nodes"], "final_nodes": int(final.x.size), "status": final.status,
        "message": final.message, "max_rms_residual": final.max_rms,
        "boundary_errors": dict(zip(eq.BC_NAMES, map(float, final.bc))),
        "wall_values": final.wall, "min_thermal_coefficient": min_theta_coeff,
        "eval_points": int(eta.size), "eval_first_step": s["eval_first_step"],
        "state_order": list(eq.STATE_NAMES), "highest_order": list(eq.HIGH_NAMES),
        "scan_guesses_reaching_branch": branch["guesses"], "checks": study_branch["checks"],
        "git": extra["git"], "created": extra["created"],
        "use": "Testing and plotting only. Never a PINN training label (AGENTS.md section 3).",
    }
    arrays = {"eta": eta, "z": z, "H": H, "mesh": final.x, "mesh_states": final.y, "rms_residuals": final.rms}
    # Not written yet: meta["checks"] is the caller's dict, finished after this returns.
    return out_root / case_id, arrays, meta, min_theta_coeff


def main():
    out_root = ROOT / "reference"

    p, case_sha = load_case()
    s = load_reference_settings()
    bar, L0, tol0 = s["agreement_rel"], s["L_start"], s["tol_values"][0]
    extra = {"git": git_state(), "created": datetime.datetime.now().isoformat(timespec="seconds")}
    study = {"case_sha256": case_sha, "settings": s, **extra}

    mms = verify.mms_order(p, s["mms"]["L"], s["mms"]["meshes"])
    mms["pass"] = bool(mms["mesh_kept_fixed"] and mms["observed_orders"][-1] >= s["mms"]["min_order"])
    study["V3_mms"] = mms
    print(f"V3 manufactured solution: orders {np.round(mms['observed_orders'], 2)}, pass={mms['pass']}")

    branches, failures = scan_branches(p, L0, tol0, s)
    study["branch_scan"] = {"L": L0, "tol": tol0, "failed": failures,
                            "branches": [{"name": b["name"], "count": len(b["guesses"]), "wall": b["result"].wall}
                                         for b in branches]}
    print(f"Branch scan at L={L0}, tol={tol0:g}: {[ (b['name'], len(b['guesses'])) for b in branches]}, "
          f"{len(failures)} failed")
    if not branches:
        raise SystemExit("No guess converged. Continuation path in configs/reference.json is the next step; "
                         "it is not run automatically so the attempt can be reviewed first.")

    Lref = float(s["L_reference"])
    exported, profiles = [], {}
    for b in branches:
        name = b["name"]
        tl = ladder_tol(p, b["result"], s)
        ll = ladder_L(p, tl[-1], s) if tl[-1].ok else []
        # Same L ladder with the old zero-at-L far conditions, kept only for comparison figures.
        dl = ladder_L(p, tl[-1], {**s, "far_field": "dirichlet"}) if tl[-1].ok else []
        for mode, ladder in (("asymptotic", ll), ("dirichlet", dl)):
            for r in (r for r in ladder if r.ok):
                e = sv.wall_mesh(r.L, 1500, s["first_step"])
                profiles[f"{name}_{mode}_L{r.L:g}_eta"], profiles[f"{name}_{mode}_L{r.L:g}_z"] = e, r.sol(e)
        sb = {"tol_ladder": [r.summary() for r in tl], "L_ladder": [r.summary() for r in ll],
              "dirichlet_L_ladder": [r.summary() for r in dl], "checks": {}}
        study[f"branch_{name}"] = sb
        c = sb["checks"]
        final = next((r for r in ll if r.L == Lref and r.ok), None)
        if final is None:
            print(f"Branch {name}: no converged solution at L={Lref:g}; nothing exported")
            continue
        ok_ll = len(ll) == len(s["L_values"]) and all(r.ok for r in ll)
        c["V4_tol_two_tightest"] = bool(all(r.ok for r in tl) and sv.rel_diff(tl[-1].wall, tl[-2].wall) <= bar)
        nxt = ll[ll.index(final) + 1] if ok_ll and final is not ll[-1] else None
        c["V5_core_change_to_next_L"] = float(sv.rel_diff(final.wall, nxt.wall, sv.CORE_KEYS)) if nxt else None
        c["V5_core_change_to_largest_L"] = float(sv.rel_diff(final.wall, ll[-1].wall, sv.CORE_KEYS)) if ok_ll else None
        c["V5_chi_slope_change_two_largest_L"] = (float(sv.rel_diff(ll[-2].wall, ll[-1].wall, ("ch1_0",)))
                                                  if ok_ll else None)
        c["V5_core_L_independent"] = bool(nxt and c["V5_core_change_to_next_L"] <= bar)
        if nxt:
            c["V5_change_to_next_L_by_key"] = {k: float(sv.rel_diff(final.wall, nxt.wall, (k,))) for k in sv.WALL_KEYS}
        # theta decays like q^-m at infinity only if m > 0 (equations.far_conditions)
        A_L = 1 + (p["theta_w"] - 1) * final.sol(final.L)[6]
        c["far_decay_exponent_m"] = float(p["Pr"] * p["kappa"] * final.wall["f_L"] / (1 + p["Nr"] * A_L**3))
        c["far_field_valid"] = c["far_decay_exponent_m"] > 0
        c["V6_scan_guesses_in_branch"] = len(b["guesses"])
        c["V7_max_abs_bc"] = float(np.max(np.abs(final.bc)))
        c["V7_pass"] = bool(c["V7_max_abs_bc"] <= final.tol)
        v8 = verify.core_then_chi(p, final.L, final.tol, s, b["guesses"][0], final)
        c["V8_core_then_chi"] = v8
        c["V8_pass"] = bool(v8["core_status"] == 0 and v8["chi_status"] == 0 and v8["max_wall_rel_diff"] <= bar)
        folder, arrays, meta, min_coeff = export(p, case_sha, s, b, final, sb, out_root, extra)
        rk_sol, rk = shooting.rk_reference(p, final.L, s, b["guesses"][0])
        arrays["z_rk"] = rk_sol(arrays["eta"])
        per_state = np.max(np.abs(arrays["z_rk"] - arrays["z"]), axis=1) / np.max(np.abs(arrays["z"]), axis=1)
        rk["max_rel_diff_by_state"] = dict(zip(eq.STATE_NAMES, map(float, per_state)))
        rk["max_wall_rel_diff"] = float(sv.rel_diff(final.wall, sv.wall_values(lambda x: rk_sol(x)[:, 0], final.L)))
        c["V10_rk_multiple_shooting"] = rk
        c["V10_pass"] = bool(per_state.max() <= bar and rk["max_wall_rel_diff"] <= bar)
        c["V9_min_thermal_coefficient"] = min_coeff
        c["V12_slope_check"] = verify.slope_check(arrays["eta"], arrays["z"], arrays["H"])
        c["V12_pass"] = bool(max(c["V12_slope_check"].values()) <= final.tol)
        c["G3_fixed_L_pass"] = bool(mms["pass"] and c["V4_tol_two_tightest"] and c["V7_pass"] and c["V8_pass"]
                                    and c["V10_pass"] and c["V12_pass"]
                                    and min_coeff > 0)
        c["G3_far_field_pass"] = c["V5_core_L_independent"]
        io.write_reference(folder, arrays, meta)
        sb["final"] = final.summary()
        sb["folder"] = str(folder.relative_to(ROOT))
        exported.append((name, folder, final))
        print(f"Branch {name}: L={final.L:g}, nodes={final.x.size}, fixed-L G3={c['G3_fixed_L_pass']}, "
              f"core L-independent={c['V5_core_L_independent']}, m={c['far_decay_exponent_m']:.4f}, wrote {folder.name}")

    out_root.mkdir(parents=True, exist_ok=True)
    np.savez(out_root / "l_study.npz", **profiles)
    bench = benchmark.main()
    study["V11_benchmarks"] = {
        "abbas_max_rel_dev_L10": max(abs(r["L10"] / r["published"] - 1) for r in bench["abbas"]),
        "flat_max_rel_dev": max(abs(r["code"] / r["exact"] - 1) for r in bench["flat"]),
        "slip_darcy_rel_dev": abs(bench["slip_darcy"]["code_f2_0"] / bench["slip_darcy"]["exact_f2_0"] - 1),
    }
    # Pass: the published table to one unit of its 5th decimal; exact solutions to 1e-5 (O(1/kappa) at 1e6).
    v = study["V11_benchmarks"]
    study["V11_pass"] = bool(all(abs(r["L10"] - r["published"]) <= 1e-5 for r in bench["abbas"])
                             and v["flat_max_rel_dev"] <= 1e-5 and v["slip_darcy_rel_dev"] <= 1e-5)
    (out_root / "study.json").write_text(json.dumps(study, indent=2, default=float) + "\n")
    (out_root / "study.md").write_text(plots.study_markdown(study))
    if exported:
        plots.make_all(out_root, study, exported)
    print(f"Wrote {out_root / 'study.json'} and {out_root / 'study.md'}")


if __name__ == "__main__":
    main()
