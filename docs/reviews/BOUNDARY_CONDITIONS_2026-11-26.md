# Boundary conditions: linear or not, hard or soft, and the far end

Date: 2026-11-26. Record of a side discussion. Nothing in the code, configs or equations was changed by it.

## 0. Decisions

- **The source is the truth.** Eqs. 14 to 19, B1 to B12 and the 22 locked values are fixed and correct as printed. That includes Eq. 8, Eq. 11 and Eq. 18. We do not question them with the supervisor (user's decision, 2026-11-26).
- **L stays 20.** Runs at other L are planned for later ([L_STUDY.md](../plans/L_STUDY.md)).
- **All 12 conditions stay exact (hard).** Section 3 explains why.

## 1. The 12 conditions

Source: Eq. 19 in `sources/project-draft/project3Nov_2026.docx` (image49), [equations.md](../problem/equations.md), and `boundary_terms` in `src/bvp_pinn/reference/equations.py`. In the formulas below, c = theta_w - 1 and q_L = L + kappa.

At the wall, eta = 0:

| BC | As printed |
|---|---|
| B1 | f'(0) = 1 + N1 (f''(0) - f'(0)/kappa) |
| B2 | f(0) = fw/(Pr Le) phi'(0) |
| B3 | g(0) = -ell f''(0) |
| B4 | -theta'(0) = Bi (1 - theta(0)) + Rd/c [theta_w^4 - (1 + c theta(0))^4] |
| B5 | phi'(0) + (Nt/Nb) theta'(0) = 0 |
| B6 | chi(0) = 1 |

As eta -> infinity: f' -> 0 (B7), f'' -> 0 (B8), g -> 0 (B9), theta -> 0 (B10), phi -> 0 (B11), chi -> 0 (B12).

The code applies them at eta = L = 20:

- B7, B8, B9 and B12 as zero values at L.
- B10 and B11 in asymptotic form (Lentini and Keller 1980): theta'(L) + m theta(L)/q_L = 0, and phi'(L) + (mp phi(L) - (Nt/Nb) m theta(L))/q_L = 0. Here m = Pr kappa f(L)/(1 + Nr A^3), A = 1 + c theta(L), and mp = Le Pr kappa f(L).

The asymptotic form is our way of applying "-> 0 at infinity" at a finite end. It is our choice, not the source's wording.

## 2. Linear or nonlinear

A condition is **linear (affine)** when it is a fixed weighted sum of values and derivatives that equals a known number. Products of unknowns, powers of unknowns, and unknowns inside a denominator make it nonlinear. Linear TFC works because adding k times a correction shape moves the condition by exactly k. Nonlinear terms break that: for example, (0.1 + 0.1)^4 = 0.0016, while 0.1^4 + 0.1^4 = 0.0002.

| BC | Type | Reason |
|---|---|---|
| B1 | Linear | Fixed multiples of f'(0) and f''(0) |
| B2, B3, B5 | Linear, coupled | Two functions each, but only fixed multiples of them |
| B4 | **Nonlinear, as printed** | (1 + c theta(0))^4: its theta(0)^4 coefficient is Rd c^3 = -0.512 |
| B6, B7, B8, B9, B12 | Linear | One value or derivative equals a number |
| B10 | Linear as printed; **nonlinear as applied** | m theta(L) contains f(L) theta(L), and theta(L) sits inside A^3 in a denominator |
| B11 | Linear as printed; **nonlinear as applied** | mp phi(L) contains f(L) phi(L) |

**Coupled is not the same as nonlinear.** B2, B3 and B5 involve two functions each but are linear. The code makes them exact by building the fields in a fixed order with no loop:

1. theta(0) comes from the network.
2. B4 gives theta'(0).
3. B5 gives phi'(0).
4. B2 gives f(0).
5. The built f gives f''(0), and B3 gives g(0).

Example: f''(0) = -4.2033, so g(0) = 1.2610, which equals the reference value.

## 3. Hard or soft

**Exact nonlinear conditions.** Plain linear TFC cannot make B4, B10 and B11 exact. Our layer can, because each of them is linear in one slope. It keeps the value as the network gives it, and the correction shapes leave that value alone. Then it sets the slope from the value.

- B4: theta(0) = 0.2502 gives theta'(0) = -1.2594.
- B10: f(L) = 0.1932 and theta(L) = 0.0683 give m = 0.2390, so theta'(L) = -8.08e-4.

Both match the reference to the last digit. Checked on a random, untrained network:

| Mode | Boundary errors |
|---|---|
| Exact | All 12 at most 4.4e-16 |
| Hybrid ("linear") | B4 0.86, B10 7.9e-3, B11 2.9e-2; the other 9 exact |
| Soft | All 12 open |

Other ways to handle nonlinear conditions:

- **Iterative TFC:** a Newton solve inside every forward pass. Not needed here, because the conditions form no loop.
- **Rewriting B10 and B11 as linear zeros at L:** costs several percent at the wall (section 4).
- **Soft penalty for the three:** the hybrid mode, which is the usual practice in the TFC literature (Johnston 2021).

**Limits of exact enforcement:**

- It is exact at L = 20, not at infinity (section 4).
- Exact B10 and B11 do not pick the decaying solution. Runs E1 and E2 met all 12 conditions with m < 0, so the acceptance rule's m > 0 check is needed.
- The output contains a divisor, 1 + Nr A^3, which is zero when theta(L) is about 2.82.

**The three strategies compared** (all three exist in the code through `bc_mode`):

| | All hard | All soft | Hybrid |
|---|---|---|---|
| Boundary error | About 1e-16 | Depends on the weights | 9 exact, 3 depend on the weights |
| Our runs | B0 reaches reference A (2 seeds and the check run) | 2026-11-21 pilot: cold wall, theta(0) -0.031, B4 error 2.39 | 2026-11-21 pilot: theta(0) 0.0004, B4 error 2.30 |
| Weights | None (one loss term) | 12; reached 7.1e10 | 3; reached 2.2e6 and 2.8e9 |

**Recommendation:** keep all hard. The biggest reason this could be wrong: soft and hybrid were only tested with the old pipeline (linear input map, short budgets, no residual scale, no Levenberg-Marquardt). A hybrid B0 run with the current pipeline, about 1.5 hours, would settle it. It has not been run.

**Soft mode works in our code:**

- Adam loss: w0 times the equation term, plus the sum of w_i B_i^2 over the soft conditions. The weights follow the gradient-norm rule, capped at 1000.
- Levenberg-Marquardt: each soft condition is an extra residual row sqrt(w_i) B_i (`lm._GlobalRows`). These add 3 or 12 dense Jacobian rows, which go into the step through the Woodbury identity.

Obstacles for soft and hybrid:

- The weights blow up near zero, so a cap value has to be chosen.
- The acceptance rule requires boundary errors at most 1e-8. Soft runs would need a new bar, declared before the runs.
- B4 sits where the residual terms are about 1e3.
- A soft B10 alone holds the constant part of the theta tail (risk of an E8-type drift; inferred, not tested).

## 4. The far end

The source puts the far conditions at infinity. Every solver stops at a finite L, so some form has to stand in for the limit. At L = 20, theta is still 0.068 and phi about 0.18. Setting them to zero there bends the whole curve. In the saved L study, zero conditions at L = 20 give theta(0) 0.241 (against 0.250) and phi(0) -0.395 (against -0.120).

Known ways to handle the far end:

| Way | Good | Bad |
|---|---|---|
| Zero at a finite L | Literal wording | Large wall errors unless L is huge |
| Zero at many L, then extrapolate | Literal wording, 0.5 s per L | The PINN cannot train at very large L |
| Asymptotic conditions (current) | About 2e-4 from the limit at L = 20 for most values; phi(0) 1.8e-3 | Nonlinear; not the printed wording |
| Better asymptotic conditions | Smaller error | More algebra |
| Map infinity to a point | Literal, at infinity | theta behaves like (1 - xi)^0.24, with an infinite slope at the mapped end |
| Decay built into the network | Nothing to impose | The decay rate is unknown |

chi: with Eq. 18 as given, chi'(0) keeps changing with L (-2.01 at L = 20, -1.08 at L = 160). So chi is defined by chi(L) = 0 at L = 20.

**Planned evidence, not yet run.** A table of wall values with zero conditions at L = 20 to 10 240, next to the asymptotic form and the extrapolated limit, using the classical solver only. It would show whether the literal reading and our form head to the same values.

## 5. Reading the source draft

The draft gives Eqs. 1 to 8, the change of variables (Eq. 11), and the final Eqs. 14 to 19. It does not show the derivation steps. It cites references [16, 17, 44] for the conditions and says nothing on how to apply "-> infinity" at a finite end.

Two notes were made while reading. They are kept for the record only and are **not** raised with the supervisor (user's decision; the source is taken as correct):

- Eq. 11 (phi = C/C_inf) together with Eq. 8 (C -> C_inf) would suggest phi -> 1. Eq. 19 says phi -> 0, and we use Eq. 19.
- Eq. 7 has convection terms for n. The printed Eq. 18 has kappa/q f phi and no f chi' term. We use Eq. 18 as printed.

## 6. Open items

- Hybrid-mode B0 check with the current pipeline (optional, about 1.5 hours).
- Far-end comparison table (zero at L against the asymptotic form), classical solver only.
- Supervisor questions 7 to 10 are still open (branch, far-field treatment, chi, f bounded).
