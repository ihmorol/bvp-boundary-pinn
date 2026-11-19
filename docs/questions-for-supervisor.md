# Questions for the supervisor

The user's 2026-11-16 message reporting direct supervisor values answers the numerical-value questions for all 22 physical coefficients. The values and mappings are locked in [parameters.md](parameters.md#supervisor-confirmation-2026-11-16). The questions below concern definitions and reporting, not missing numbers or permission to replace supplied values automatically.

Copy these into a message. Write dated answers into docs/parameters.md so the record stays in one place. Conditional calculations below are not supervisor-confirmed replacements for the locked values.

1. What is **DeltaT**, and is **T_infinity the same as Tc**? If DeltaT=Tf-T_infinity and T_infinity=Tc imply epsilon=-1/(theta_w-1), the supplied theta_w=0.2 gives epsilon=1.25, not the supplied 0.1. Which temperature definitions apply to this case?
2. Is **Lb** defined with DB or with Dn? If it uses DB and Lb=Pr*Le, the supplied Pr=8.8 and Le=1 give Lb=8.8, not the supplied 1. Which definition applies?
3. Must the **recorded page 7 definitions** hold together for this case, or are the supplied coefficients independent inputs? With delta_star=0, those relations give B=1, not the supplied 0.5.
4. Is **fw** a signed flow parameter, or strictly Cw/(1-Cw)? What sign convention applies? If that formula applies, the supplied fw=0.2 gives Cw=1/6. The earlier negative-concentration example used the superseded proposal fw=-0.5 and does not apply to the current value.
5. What do the received **H=1** and **Prm=0.1** mean? Both are recorded as confirmed values, but neither has an active physical coefficient in Eqs. 14 to 19. We have not added equation terms or assumed a magnetic Prandtl mapping. This interpretation question does not leave the 22 coefficients numerically incomplete.
6. Which **wall quantities** should we report, and what **accuracy** is good enough?

## Decisions received 2026-11-19

Source: the user's message of 2026-11-19 reporting the supervisor's position. These settle method questions, not parameter values.

- Both structures are tried on both problems: the basic windowed structure from the Lorenz work and one network with knowledge retention.
- The BVP network input is eta only. Physical parameters are fixed values, never network inputs. No physical parameter sweep.
- The problem comes from M.J. Uddin and M.I. Hasan through the supervisor, and we work on it with them. Author order is still to be agreed.

## Supervisor answer to questions 1 to 6, 2026-11-19

Source: the user's message of 2026-11-19 reporting the supervisor's response. The physical meaning of each parameter and variable is not needed. The task is numerical: use the given values in Eqs. 14 to 19 exactly as written, compute a reference solution with a classical numerical method, train the PINN on eta, and test the PINN against that reference.

- Questions 1 to 5 are closed. The supplied values are used as given. The conditional calculations above stay on record as notes only; they do not change any value. H and Prm stay unused because no term in Eqs. 14 to 19 needs them.
- Question 6 is closed as our choice: we report the error of all five fields and their eta derivatives against the reference, including the derivative values at eta=0 as plain numerical checks. No physical reporting quantity or accuracy target was set by the supervisor.

## New questions from the reference study, 2026-11-19 (updated)

Details: [SPEC_reference_solutions.md](SPEC_reference_solutions.md) section 8 and [reference/study.md](../reference/study.md).

7. **Target branch.** Two solutions satisfy the conditions on a finite domain. Branch A has theta(0) = 0.2502, f''(0) = -4.2033, and theta decays like (eta+kappa)^-0.24. Branch B has f(infinity) < 0, so its theta cannot decay to zero at infinity; it is not a solution of the eta -> infinity problem. We use branch A. Do you agree?
8. **Far field.** theta and phi decay only as powers of (eta+kappa), so setting them to zero at a finite eta (as eta_inf = 7 to 12 in the papers we read) shifts the wall values by several percent. We impose asymptotic conditions at eta = 20 instead (Lentini and Keller, 1980). The wall values then agree with eta = 160 to about 1e-4, phi(0) to about 1e-3. Is this acceptable?
9. **chi at infinity.** As printed, Eq. 18 has no chi convection term (it has kappa/(eta+kappa)*f*phi). With it, chi cannot decay to zero at infinity: chi'(0) keeps changing as the far boundary moves (-2.01 at eta = 20, -1.08 at eta = 160). We keep Eq. 18 as given and define chi by chi = 0 at eta = 20. Is Eq. 18 as intended, and is this treatment acceptable?
