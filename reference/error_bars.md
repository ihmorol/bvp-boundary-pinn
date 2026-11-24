# Reference error bars (branch A)

Built by `tools/reference_error_bars.py` from `engine/out/engine_convergence.csv` and `reference/l_study.npz`.
Grid: three finest box-scheme levels, r = 2. L: v(L) = v_inf + c L^-p fitted to L = 40, 80, 160.

| Value | Finest grid | Observed order | GCI (fine) | L = 20 | v_inf (L fit) | L = 20 minus v_inf | Order in L |
|---|---:|---:|---:|---:|---:|---:|---:|
| f2_0 | -4.203299261 | 2 | 4.93e-07 | -4.2032976 | -4.2040606 | 0.000763 | 0.362 |
| theta_0 | 0.2501853767 | 2 | 4.36e-07 | 0.25018529 | 0.25026504 | -7.98e-05 | 0.36 |
| th1_0 | -1.259435078 | 2 | 2.64e-07 | -1.2594353 | -1.2591924 | -0.000243 | 0.36 |
| phi_0 | -0.1201549539 | 2 | 6.95e-08 | -0.12015496 | -0.11992905 | -0.000226 | 0.367 |
| ph1_0 | 3.778305233 | 2 | 2.64e-07 | 3.778306 | 3.7775771 | 0.000729 | 0.36 |
| g1_0 | -2.871303171 | 2 | 3.79e-07 | -2.8713023 | -2.8717832 | 0.000481 | 0.362 |
| ch1_0 | -2.008734512 | 2 | 1.01e-07 | -2.0087347 | n/a | n/a | -0.496 |
| f_L | 0.193173606 | 2 | 8.46e-07 | 0.19317374 | 0.19310849 | 6.53e-05 | 0.388 |

chi'(0) has no limit in L: Eq. 18 as given has no decaying chi solution (SPEC section 8.2), so its fit is not a limit.
