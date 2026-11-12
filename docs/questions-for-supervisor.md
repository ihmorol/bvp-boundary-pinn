# Questions for the supervisor

Copy these into a message. Write the answers, with dates, into docs/parameters.md so the record stays in one place.

1. What are the values or ranges for **Nr, Nt, Nb, Lb and Pe**? All five are needed in Eqs. 16 to 18.
2. Is **lambda_m** a fixed value or a range? The line on the sheet is not clearly readable.
3. Is **epsilon** fixed at 1, or is it a range like the other lines?
4. What is **DeltaT**, and is **T_infinity the same as Tc**? If DeltaT = Tf - T_infinity, then epsilon = -1/(theta_w - 1).
5. Is **Lb** defined with DB or with Dn? If it uses DB, then Lb = Pr*Le = 6.8 for our chosen Le = 1.
6. Should each parameter be picked inside its own range, or must the page 7 relations hold together? Example: with delta_star = 0, those relations give **B = 1**, not 0.1.
7. Is **fw** a signed flow parameter, or strictly Cw/(1-Cw)? In that formula, fw = -0.5 gives a negative Cw.
8. Which wall quantities should we report, and what accuracy is good enough?
