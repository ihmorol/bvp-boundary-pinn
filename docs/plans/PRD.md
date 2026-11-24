# PRD: reference solutions and PINN for the five-field boundary value problem

Date: 2026-11-19. Status: planning document, nothing implemented. Authoritative sources: [equations.md](../problem/equations.md) for the equations and [parameters.md](../problem/parameters.md) for the values. If this file and those two ever differ, they win.

No spec or PRD skill is installed in this environment (see [SKILLS_AND_TOOLS.md](../team/SKILLS_AND_TOOLS.md)), so this PRD follows a plain structure.

## 1. Problem statement

### 1.1 What we solve

We solve a system of five coupled nonlinear ordinary differential equations (Eqs. 14 to 18) with twelve boundary conditions (Eq. 19). The source is the draft by M.J. Uddin and M.I. Hasan (`sources/project-draft/project3Nov_2026.docx`), given to us by the supervisor.

The independent variable is eta. The unknowns are five functions of eta: f, g, theta, phi, chi.

The supervisor's instruction (2026-11-19, recorded in [questions-for-supervisor.md](../problem/questions-for-supervisor.md)) defines the task:

1. Use the given values exactly as given.
2. Compute a reference solution with a classical numerical method.
3. Train a PINN whose only input is eta.
4. Test the PINN against the reference.

The physical meaning of each symbol is not needed.

### 1.2 Domain

eta runs over [0, L]. eta = 0 is the wall. L is a finite numerical endpoint that stands for the far field. L is not given by the supervisor. The reference study sets L = 20, with asymptotic far conditions for theta and phi ([SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md) section 8). The received value L = 0.5 is lambda_m, not this endpoint.

### 1.3 Short names

```text
q = eta + kappa      h = eta + b      c = theta_w - 1      A = 1 + c*theta
f1..f4 = first to fourth eta-derivatives of f; g1, g2; th1, th2; ph1, ph2; ch1, ch2 likewise
```

### 1.4 Equations (residual form; each target is zero)

```text
R14 = (1+Delta)*(f4 + 2*f3/q - f2/q**2 + f1/q**3)
    + kappa/q*(f*f3 - f1*f2)
    + kappa/q**2*(f*f2 - f1**2)
    - kappa/q**3*f*f1
    - 0.5*delta_star*((eta*f3 + 3*f2) + (0.5*eta*f2 + f1)/q)
    - (f2 + f1/q)/Da
    - Delta*(g2 + g1/q)
    - 2*beta_m/h**4*((1/q - 4/h)*theta + th1)

R15 = (1+Delta/2)*(g2 + g1/q)
    - Delta*B*(f2 + f1/q + 2*g)
    - kappa/q*(f1*g - f*g1)
    - delta_star/2*(eta*g1 + 3*g)

R16 = (th2 + th1/q)/Pr
    + Nr/Pr*(A**3*th1/q + 3*c*A**2*th1**2 + A**3*th2)
    + kappa/q*f*th1
    + Nt/Pr*th1**2
    + Nb/Pr*th1*ph1
    + 2*beta_m*lambda_m*(theta-epsilon)/(Pr*h**3)*(kappa*f/q*(1-2/h**2) - 2*f1/h)
    - delta_star*eta/2*th1

R17 = (ph2 + ph1/q)/(Le*Pr)
    + Nt/(Le*Pr*Nb)*(th2 + th1/q)
    + kappa/q*f*ph1
    - delta_star*eta/2*ph1

R18 = (ch2 + ch1/q)/Lb
    + kappa/q*f*phi
    + Pe/Pr*(chi*ph2 + ch1*ph1)
    - delta_star*eta/2*ph1
```

R18 is kept exactly as printed and confirmed by the supervisor, including the f*phi and ph1 terms.

### 1.5 Boundary conditions (suffix 0 means eta = 0, suffix L means eta = L)

```text
B1  = f1_0 - 1 - N1*(f2_0 - f1_0/kappa)
B2  = f_0 - fw/(Pr*Le)*ph1_0
B3  = g_0 + ell*f2_0
B4  = -th1_0 - Bi*(1-theta_0) - Rd/(theta_w-1)*(theta_w**4 - (1+(theta_w-1)*theta_0)**4)
B5  = ph1_0 + Nt/Nb*th1_0
B6  = chi_0 - 1
B7  = f1_L      B8  = f2_L      B9  = g_L
B10 = theta_L   B11 = phi_L     B12 = chi_L
```

B4 also has a stable algebraic form, given in [equations.md](../problem/equations.md). Use it in code after testing that both forms agree.

### 1.6 Parameter values (locked 2026-11-16, from [parameters.md](../problem/parameters.md))

| Symbol | Value | Symbol | Value | Symbol | Value |
|---|---:|---|---:|---|---:|
| Delta | 0.5 | B | 0.5 | Da | 0.2 (from M = 5, M = 1/Da) |
| kappa | 0.2 | delta_star | 0 | theta_w | 0.2 |
| lambda_m | 0.5 | beta_m | 0.2 | epsilon | 0.1 |
| b | 0.2 | Pr | 8.8 | Le | 1 |
| N1 | 0.1 | Bi | 1 | Rd | 1 |
| Nr | 0.5 | Nt | 0.3 | Nb | 0.1 |
| Lb | 1 | Pe | 2 | ell | 0.3 |
| fw | 0.2 | | | | |

Received but unused, because no term in Eqs. 14 to 19 needs them: H = 1, Prm = 0.1.

These values are never tuned or changed by code. delta_star = 0 removes every unsteady term. At the wall, 1/q = 5, 1/q^3 = 125 and 2*beta_m/h^4 = 250, so the wall region has large coefficients (arithmetic, not a computed result).

## 2. Research goal

Find out how well a PINN with eta as its only input solves this BVP, measured against the classical reference. Compare the two training structures the supervisor asked for:

- **Basic structure:** one network on the whole domain (B0), and separate networks per eta window trained together (B1).
- **Knowledge retention:** one network trained window by window from the wall outward, keeping what it learned on earlier windows (B2, B3).

Every arm imposes all twelve boundary conditions at every training step. Arms, rules and gates are in [PINN_PLAN.md](../pinn/PINN_PLAN.md).

This repository does only the BVP work. The same two structures are tested on the Lorenz-1960 problem in a separate repository, and both results feed one journal paper. That alignment lives in [../journal/README.md](../journal/README.md). Nothing there is a task for this workspace.

## 3. Scope

In scope:

1. The classical reference solution at the locked case, with convergence evidence.
2. PINN arms B0 to B3 (defined in [PINN_PLAN.md](../pinn/PINN_PLAN.md) section 2), 5 seeds each.
3. Error evaluation against the reference.
4. A cost comparison with the classical solver.
5. Figures and tables FB1 to FB5, TB1, TB2, FX1, TX1.

Non-goals:

- **No physical interpretation.** No physical reporting quantities and no physics discussion, per the supervisor's instruction.
- **No parameter sweeps.** No case other than the locked one is reported as a result.
- **No changes to the model.** No change to Eqs. 14 to 19 to match other papers.
- **No reference labels in PINN training.** The reference is for testing only ([AGENTS.md](../../AGENTS.md)).
- **No claim that the PINN beats the classical solver.** Unless measured, the cost comparison only reports.

## 4. Deliverables

| ID | Deliverable | Defined in |
|---|---|---|
| D1 | Verified reference data for the locked case, with metadata and hash | [SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md) |
| D2 | Reference convergence table (L study, tolerance and mesh study, guesses) | SPEC section 5 |
| D3 | Tested residual and boundary code (NumPy and PyTorch versions) | [TODO.md](TODO.md) phases R1 and P1 |
| D4 | PINN runs B0 to B3, 5 seeds each, in the run format | [PINN_PLAN.md](../pinn/PINN_PLAN.md) section 3 |
| D5 | Figures and tables listed in section 3 | PINN_PLAN.md section 6 |
| D6 | One-page results memo answering the BVP questions | PINN_PLAN.md section 7 |

## 5. Success metrics

| Metric | Success means |
|---|---|
| Reference gate G3 | Wall values plateau across L and tolerance; manufactured-solution order check passes; all initial guesses reach one solution, or every branch is reported |
| Code gate G1 | Derivative tests up to order 4 and residual tests against an independent SymPy evaluation pass |
| PINN completion | All 20 runs (4 arms x 5 seeds) finished and reported, failures included |
| PINN accuracy | Reported, not pre-set: relative L2 and max error per field and derivative, residual per equation, error per boundary condition. The supervisor set no accuracy target |
| Claims | "Arm X beats arm Y" only if it holds in all 5 paired seeds (gate G6) |
| No leakage | Reference not read during training or model selection (gate G5) |

## 6. Assumptions and open points

- **Checked 2026-11-19:** the locked case has at least two smooth solutions (branches A and B); solve_bvp finds both without continuation ([SPEC](../reference-solver/SPEC_reference_solutions.md) section 8).
- **Checked 2026-11-19:** theta and phi decay only as powers of (eta+kappa), and chi has no decaying solution as Eq. 18 is printed. The reference uses L = 20 with asymptotic far conditions for theta and phi; chi is defined by chi(L) = 0. The PINN uses the same conditions.
- **Checked 2026-11-19:** only branch A decays at infinity; branch B is flagged.
- **Open:** supervisor questions 7 to 9 (target branch, far-field treatment, chi at infinity).
- **Open:** author order with Uddin and Hasan. It does not block implementation.
