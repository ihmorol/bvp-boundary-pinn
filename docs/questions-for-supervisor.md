# Questions for the supervisor

The user's 2026-11-16 message reporting direct supervisor values answers the numerical-value questions for all 22 physical coefficients. The values and mappings are locked in [parameters.md](parameters.md#supervisor-confirmation-2026-11-16). The questions below concern definitions and reporting, not missing numbers or permission to replace supplied values automatically.

Copy these into a message. Write dated answers into docs/parameters.md so the record stays in one place. Conditional calculations below are not supervisor-confirmed replacements for the locked values.

1. What is **DeltaT**, and is **T_infinity the same as Tc**? If DeltaT=Tf-T_infinity and T_infinity=Tc imply epsilon=-1/(theta_w-1), the supplied theta_w=0.2 gives epsilon=1.25, not the supplied 0.1. Which temperature definitions apply to this case?
2. Is **Lb** defined with DB or with Dn? If it uses DB and Lb=Pr*Le, the supplied Pr=8.8 and Le=1 give Lb=8.8, not the supplied 1. Which definition applies?
3. Must the **recorded page 7 definitions** hold together for this case, or are the supplied coefficients independent inputs? With delta_star=0, those relations give B=1, not the supplied 0.5.
4. Is **fw** a signed flow parameter, or strictly Cw/(1-Cw)? What sign convention applies? If that formula applies, the supplied fw=0.2 gives Cw=1/6. The earlier negative-concentration example used the superseded proposal fw=-0.5 and does not apply to the current value.
5. What do the received **H=1** and **Prm=0.1** mean? Both are recorded as confirmed values, but neither has an active physical coefficient in Eqs. 14 to 19. We have not added equation terms or assumed a magnetic Prandtl mapping. This interpretation question does not leave the 22 coefficients numerically incomplete.
6. Which **wall quantities** should we report, and what **accuracy** is good enough?
