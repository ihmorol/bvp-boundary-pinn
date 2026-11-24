# Research notes: references we will use

Date: 2026-11-19.

How the references were checked:

- **Crossref:** DOI references were checked against Crossref metadata (title, authors, venue, volume, pages) on 2026-11-19.
- **Local PDF:** the reference has a local copy in `sources/literature/` with our own `analysis.md`.
- **arXiv:** the arXiv page answered HTTP 200 on 2026-11-19.

Broader reading is in [reading-list.md](reading-list.md) and [literature-synthesis.md](literature-synthesis.md).

## 1. The BVP model

| Reference | Check | Why it matters |
|---|---|---|
| M.J. Uddin, M.I. Hasan. "Transient bio-nano-micropolar fluid slip flow convectively heated radiated curve surface with magnetic dipole in porous medium." Unpublished draft, 14 May 2026. `sources/project-draft/project3Nov_2026.docx` | Local file | Source of Eqs. 14 to 19. The equations are used exactly as given ([PRD.md](../plans/PRD.md)) |
| P. Kumar, B. Nagaraja, F. Almeida, A.R. AjayKumar, Q. Al-Mdallal, F. Jarad. "Magnetic dipole effects on unsteady flow of Casson-Williamson nanofluid propelled by stretching slippery curved melting sheet with buoyancy force." *Scientific Reports* 13, 12770 (2023). https://doi.org/10.1038/s41598-023-39354-5 | Crossref | The draft cites it for its similarity transformation and wall quantities. Its full text (Europe PMC PMC10406858) uses RKF45 shooting and gives no eta_inf value; background only |
| M. Naveed, Z. Abbas, M. Sajid. "Hydromagnetic flow over an unsteady curved stretching surface." *Engineering Science and Technology, an International Journal* 19, 841-845 (2016). https://doi.org/10.1016/j.jestch.2015.11.009 | Crossref | The draft compares its code to this paper. A possible extra code check, only with supervisor approval, since it uses other parameter values |

## 2. Numerical methods for the reference

| Reference | Check | Why it matters |
|---|---|---|
| J. Kierzenka, L.F. Shampine. "A BVP solver based on residual control and the Maltab [sic] PSE." *ACM Transactions on Mathematical Software* 27, 299-316 (2001). https://doi.org/10.1145/502800.502801 | Crossref (the published title and the SciPy manual both read "Maltab") | The algorithm behind `solve_bvp`, per the SciPy manual |
| P. Virtanen et al. "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature Methods* 17, 261-272 (2020). https://doi.org/10.1038/s41592-019-0686-2 | Crossref | Citation for SciPy and `solve_bvp` |
| SciPy `solve_bvp` documentation, SciPy v1.18.0 manual. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html | Page read 2026-11-19 | Source of the solver facts in [SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md) section 3 |
| M. Lentini, H.B. Keller. "Boundary value problems on semi-infinite intervals and their numerical solution." *SIAM Journal on Numerical Analysis* 17, 577-604 (1980). https://doi.org/10.1137/0717049 | Crossref | Asymptotic boundary conditions for semi-infinite BVPs. The method used for theta and phi at eta = L (SPEC section 8) |
| E. Hairer, S.P. Norsett, G. Wanner. *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Springer Series in Computational Mathematics (1993). https://doi.org/10.1007/978-3-540-78862-1 | Crossref (title, series, year; authors from the book record) | Standard source for the Dormand-Prince Runge-Kutta methods behind SciPy's DOP853, used in the second reference (SPEC V10) |
| SciPy `solve_ivp` documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html | Link answered HTTP 200 on 2026-11-19 | The Runge-Kutta integrator used for multiple shooting |
| U.M. Ascher, R.M.M. Mattheij, R.D. Russell. *Numerical Solution of Boundary Value Problems for Ordinary Differential Equations.* SIAM (1995). https://doi.org/10.1137/1.9781611971231 | Crossref | Standard text on collocation, conditioning and continuation for two-point BVPs |
| Z. Abbas, M. Naveed, M. Sajid. "Heat transfer analysis for stretching flow over a curved surface with magnetic field." *Journal of Engineering Thermophysics* 22, 337-345 (2013). https://doi.org/10.1134/S1810232813040061 | Crossref; paper not read. Its table values are taken from the reprint below | Published skin friction for the curved stretching sheet, the curvature-only limit of Eq. 14 (SPEC V11) |
| A. Shafiq, S. Nadeem, N. Muhammad. "Boundary layer flow over a curved surface imbedded in porous medium." *Communications in Theoretical Physics* 71, 344 (2019). https://doi.org/10.1088/0253-6102/71/3/344 | Crossref; article page read 2026-11-19 | Its Table 1 reprints the Abbas et al. values (kappa = 5 to 1000) that V11 compares against. It states bvp4c with tolerance 1e-7 and does not state eta_inf |
| L.J. Crane. "Flow past a stretching plate." *Zeitschrift fur angewandte Mathematik und Physik* 21, 645-647 (1970). https://doi.org/10.1007/BF01587695 | Crossref | Exact flat-sheet solution f = 1 - exp(-eta), the kappa -> infinity limit used in V11 |
| P.J. Roache. "Code verification by the method of manufactured solutions." *Journal of Fluids Engineering* 124, 4-10 (2002; Crossref lists 2001 as first online). https://doi.org/10.1115/1.1436090 | Crossref | Basis for the manufactured-solution test (SPEC V3) |
| A. Pantokratoras. "A common error made in investigation of boundary layer flows." *Applied Mathematical Modelling* 33, 413-422 (2009). https://doi.org/10.1016/j.apm.2007.11.009 | Crossref | Too short a domain distorts wall values in boundary-layer problems. Why the L study is required (SPEC V5) |

## 3. PINNs for boundary value problems

| Reference | Check | Why it matters |
|---|---|---|
| M. Raissi, P. Perdikaris, G.E. Karniadakis. "Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics* 378, 686-707 (2019). https://doi.org/10.1016/j.jcp.2018.10.045 | Crossref; local PDF of arXiv 1711.10561 | The PINN method we use |
| I.E. Lagaris, A. Likas, D.I. Fotiadis. "Artificial neural networks for solving ordinary and partial differential equations." *IEEE Transactions on Neural Networks* 9, 987-1000 (1998). https://doi.org/10.1109/72.712178 | Crossref | Trial functions that satisfy conditions exactly; basis for any hard-constraint variant |
| S. Wang, Y. Teng, P. Perdikaris. "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." *SIAM Journal on Scientific Computing* 43, A3055-A3081 (2021). https://doi.org/10.1137/20M1318043 | Crossref; local PDF | Unbalanced loss terms, which twelve boundary terms and large wall coefficients make likely |
| S. Berrone, C. Canuto, M. Pintore, N. Sukumar. "Enforcing Dirichlet boundary conditions in physics-informed neural networks and variational physics-informed neural networks." *Heliyon* 9, e18820 (2023). https://doi.org/10.1016/j.heliyon.2023.e18820 | Crossref; local PDF | Exact versus penalty enforcement of value conditions; penalty weights matter |
| M.F. Endalew, X.J. Zhang. "Hard-constrained physics-informed neural network with adaptive regional residual balancing for the generalized Falkner-Skan problem." arXiv:2609.23493 (2026), preprint. https://arxiv.org/abs/2609.23493 | arXiv; local PDF | Closest recent PINN study of a semi-infinite boundary-layer similarity BVP |
| H.R. Soltani Motlagh et al. "Physics-informed neural networks for dissipative micropolar nanofluid flow with microrotation dynamics and zero nanoparticle mass flux." *Modelling* 7, 145 (2026). https://doi.org/10.3390/modelling7040145 | Local PDF and our record; DOI not checked through Crossref | A PINN on a micropolar nanofluid BVP of the same family. Our analysis notes internal inconsistencies, so read with care |
| R. Mattey, S. Ghosh. "A novel sequential method to train physics informed neural networks for Allen Cahn and Cahn Hilliard equations." *Computer Methods in Applied Mechanics and Engineering* 390, 114474 (2022). https://doi.org/10.1016/j.cma.2021.114474 | Crossref; arXiv 2106.07606 read | The knowledge-retention method (one network, backward-compatibility loss) behind our retention arms |
| N. McGreivy, A. Hakim. "Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations." *Nature Machine Intelligence* 6, 1256-1269 (2024). https://doi.org/10.1038/s42256-024-00897-5 | Crossref | Rules for fair comparison with classical solvers |

## 4. Figure design

| Reference | Check | Why it matters |
|---|---|---|
| N.P. Rougier, M. Droettboom, P.E. Bourne. "Ten simple rules for better figures." *PLoS Computational Biology* 10(9), e1003833 (2014). https://doi.org/10.1371/journal.pcbi.1003833 | Article page read 2026-11-19 | Rules used for the reference figures: know the message, adapt to the medium, captions, no defaults, colour with purpose, no misleading scales, no chartjunk |
| *Fundamentals of Data Visualization*, chapter 20, "Redundant coding". https://clauswilke.com/dataviz/redundant-coding.html | Page read 2026-11-19 (the page itself does not show the author) | Encode series by colour and a second cue, and prefer direct labels or legends that do not cover data |
| T. Kassis, V. Agarwal, Y. He, D. Patel, A.M. Brueckner. "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents." arXiv:2609.00065 (2026). https://doi.org/10.48550/arXiv.2609.00065 | arXiv page read 2026-11-19 (v2, no journal version) | The scientific-visualization skill used to design and review the figures; its Okabe-Ito palette values are used in `plots.py` |

Paul Tol's colour-scheme notes (personal.sron.nl) could not be reached on 2026-11-19 and are not used.
