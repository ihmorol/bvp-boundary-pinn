# Coupled high-order BVP papers using PINNs

Verified and saved: **6 November 2026**. This literature folder is separate from the Lorenz project.

## What this list establishes

Target: a nonlinear similarity-coordinate boundary-value problem with five physical fields and derivative-order pattern **(4, 2, 2, 2, 2)**, known physical parameters, and wall/far-field conditions.

The four references below have verified equation-residual and boundary-constraint neural formulations. They establish useful precedents, **not an exact match or a guarantee that a PINN will outperform a conventional BVP solver**. No exact five-field (4,2,2,2,2) match was verified in this search. The strongest complementary references are the micropolar nanofluid paper for physical coupling and boundary conditions, and the squeezing-flow paper for nonlinear fourth-order momentum plus transport.

Here “known parameters” includes parameters varied between cases or supplied as network inputs; it does not mean those coefficients are estimated as inverse unknowns.

| Read | Paper | Verified original system | Similarity and limitation | Saved material |
|---|---|---|---|---|
| 1 | Soltani Motlagh et al. (2026) | Four physical fields, **(3,2,2,2)**; nine BCs | Micropolar flow, heat, nanoparticles, blowing and zero nanoparticle flux; no microorganism field, no fourth-order equation | Full journal article |
| 2 | Daraz & Ahmad (2026) | Three physical fields, **(4,2,2)**; eight BCs | Nonlinear fourth-order fluid momentum plus heat/concentration; planar finite squeezing channel, not curved micropolar flow | Full journal article |
| 3 | Sharma et al. (2021) | Two independent physical stress fields, **(4,4)**; eight BCs | Direct coupled fourth-order ODE residuals; linear structural equations, not nonlinear nanofluid physics | Publisher implementation supplement only; main article not downloaded |
| 4 | Kapoor et al. (2024; preprint 2023) | Two beam displacements, each fourth order in space | Verified coupled fourth-order PINN; time-dependent PDE initial-boundary problem, not eta-only ODE BVP | Full arXiv manuscript |

## 1. Closest verified micropolar/transport example

**Hamid Reza Soltani Motlagh; A. M. Amer; Nourhan I. Ghoneim; Ahmed M. Megahed; Amr M. Abdallah; Seyed Behbood Issa-Zadeh.**

**Physics-Informed Neural Networks for Dissipative Micropolar Nanofluid Flow with Microrotation Dynamics and Zero Nanoparticle Mass Flux.** *Modelling* **7**(4), 145 (2026). Published 22 July 2026.

- [DOI / publisher](https://doi.org/10.3390/modelling7040145)
- [Saved full article](01_micropolar_nanofluid_2026/micropolar_nanofluid_PINN_2026.pdf)
- [Public PDF](https://mdpi-res.com/d_attachment/modelling/modelling-07-00145/article_deploy/modelling-07-00145.pdf)

**Verified match.** Equations **(11)–(14)** contain a third-order flow equation and second-order microrotation, temperature and concentration equations. Four physical fields; total differential order nine. Equations **(15)–(16)** impose **five wall plus four far-field constraints**, including Stefan blowing, wall microrotation and zero nanoparticle flux. The two phi-like symbols distinguish microrotation and concentration; do not copy their notation ambiguously into new code.

**Known parameters.** Eleven listed groups include Prandtl, magnetic, Lewis, Eckert, porous resistance, Stefan blowing, Brownian motion, thermophoresis, reaction, vortex-viscosity and heat-source parameters; a known wall microrotation coefficient also appears. The parametric PINN takes the similarity coordinate and seven prescribed parameters as inputs; this is not inverse parameter identification.

**Genuine PINN evidence.** Residuals **(22)–(25)** and losses **(28)–(30)** explicitly use differential-equation and BC errors, with JAX differentiation and no labeled solution data.

**Validation scope.** Table 1 compares wall shear against Kumar's finite-element values for four vortex-viscosity values. This is not evidence that all four complete profiles were independently validated against a conventional solver.

**Read first:** equations (11)–(16), then residual/loss equations (22)–(30). Useful for coupled boundary-loss assembly. It has no microorganism field and does not address fourth-order momentum.

## 2. Closest verified nonlinear fourth-order fluid example

**Umar Daraz; Khaleel Ahmad.**

**A Heuristic Physics-Informed Neural Network for Unsteady Magnetohydrodynamic Squeezing Flow through a Darcy-Forchheimer Porous Channel with Brownian Motion, Thermophoresis, Thermal Radiation, and Heat Generation.** *International Scientific Spectrum* (2026). Published online 26 September 2026.

- [DOI](https://doi.org/10.66972/iscis202617)
- [Saved full article](02_squeezing_nanofluid_2026/squeezing_nanofluid_PINN_2026.pdf)
- [Public PDF](https://iscis-journal.org/index.php/iscis/article/download/17/29)

**Verified match.** Equations **(16)–(17)** use three functions, flow f, temperature theta and nanoparticle concentration phi, with orders **(4,2,2)**. The fourth derivative follows pressure elimination. Momentum drives the mutually coupled temperature/concentration equations; coupling is hierarchical, not bidirectional across every field.

**Known parameters.** The stated baseline includes S, Ha, Kp, Fr, Pr, Rd, Nb, Nt, Sc and Hs; nanoparticle loadings and material-property ratios are also specified. Parameters change between forward-solution cases rather than being inferred.

**BCs.** Equation **(18)** gives eight constraints on eta in [0,1]: at zero, f=0, f''=0, theta'=0, phi'=0; at one, f=1, f'=0, theta=1, phi=1. These are centreline/wall conditions, not semi-infinite far-field conditions.

**Genuine PINN evidence.** Section **7**, particularly **(39)–(42)**, represents each physical field directly with smooth sigmoid networks and forms the squared momentum, thermal and concentration residuals plus eight BC residuals. Derivatives through fourth order are evaluated analytically. Optimization uses PSO–NNA and least-squares polishing; it is not a demonstration that this optimizer is preferable to Adam/L-BFGS for another model.

**Reference.** Section **6** converts the system to eight first-order states for residual-controlled Lobatto collocation, described as mathematically equivalent to bvp4c. Table 3 and Figures 16/18 compare predictions with reference solutions. This first-order conversion is for the classical reference, not the stated high-order PINN formulation.

**Important separation:** Section **8** also uses a **different, supervised Levenberg–Marquardt surrogate trained on reference data**. Its regression plots are not proof of the Section 7 PINN's physics-only training. The retrieved PDF retains “Volume 00, Issue 00”; metadata and equations were verified, but journal quality and reported accuracy were not independently audited. Treat it as a very recent technical precedent rather than a sole validation authority.

## 3. Direct coupled fourth-order BVP with inspectable implementation

**Shivam Sharma; Rajneesh Awasthi; Yedlabala Sudhir Sastry; Pattabhi Ramaiah Budarapu.**

**Physics-informed neural networks for estimating stress transfer mechanics in single lap joints.** *Journal of Zhejiang University–SCIENCE A* **22**, 621–631 (2021). Published 27 August 2021.

- [DOI / main article](https://doi.org/10.1631/jzus.A2000403)
- [Saved publisher implementation supplement](03_lap_joint_coupled_fourth_order_2021/lap_joint_PINN_2021_supplement_code.pdf)
- [Public supplement PDF](https://static-content.springer.com/esm/art%3A10.1631%2Fjzus.A2000403/MediaObjects/11582_2021_371_MOESM1_ESM.pdf)

**Verified match.** Two independent axial-stress functions for the upper and lower tablets, each governed by a fourth-order equation. Supplement **Listing S2** uses one coordinate input, two physical outputs and TensorFlow derivatives u_xxxx and v_xxxx. Each residual couples both fields and their second/fourth derivatives with fixed numerical coefficients. This is a **linear coupled ODE BVP**, not a single fourth-order equation split into helper fields.

**BCs and loss.** **Listing S5** adds two equation MSEs to four endpoint-value errors and four endpoint-first-derivative errors. The first-derivative targets are zero. Numerical endpoint values are passed into the code and are not specified in the inspected snippets.

**Validation.** The publisher abstract reports comparison with MAPLE solutions. The main article was not accessible as full text; full physical parameter tables, main-text equation numbers and validation details were not verified. The public supplement nevertheless directly establishes high-order derivative and BC-loss implementation. It contains code snippets, not a checked standalone executable repository; nothing was executed.

## 4. Coupled fourth-order PDE evidence from beam systems

**Taniya Kapoor; Hongrui Wang; Alfredo Núñez; Rolf Dollevoet.**

**Physics-Informed Neural Networks for Solving Forward and Inverse Problems in Complex Beam Systems.** *IEEE Transactions on Neural Networks and Learning Systems* **35**(5), 5981–5995 (2024). Associated arXiv manuscript: **2303.01055**, inspected v2 dated 29 August 2023.

- [Journal DOI](https://doi.org/10.1109/TNNLS.2023.3310585)
- [arXiv record](https://arxiv.org/abs/2303.01055)
- [Saved full manuscript](04_coupled_beams_PINN/complex_beam_systems_PINN_arxiv_2303.01055.pdf)

**Verified match.** Section **IV-C, equation (15)** solves two coupled Euler–Bernoulli displacement fields with spatial fourth derivatives and time second derivatives. The interaction is a Winkler foundation. The fields are independent physical beam displacements, not derivative helpers.

**Forward benchmark.** Nondimensional coefficients are fixed to one, so this example does not match the target's large parameter set. Both beams have zero displacement and second spatial derivative at the two ends, plus initial displacement/velocity conditions. Equation **(7)**, accompanying text and Figure 4 establish physics plus initial/boundary losses without interior solution data for forward training.

**Validation.** The double-beam example is compared against constructed analytical solutions. It is an initial-boundary PDE problem, not a pure ODE BVP. Do not use the paper's Timoshenko examples as evidence of fourth-order equations.

## Suggested use

1. Use paper 1 to understand a multi-field fluid PINN with mixed wall and far-field conditions.
2. Use paper 2 to inspect nonlinear fourth-order momentum plus thermal/concentration residuals; distinguish its PINN from its supervised surrogate.
3. Use paper 3 for a compact, explicit implementation of two physical fourth-order outputs and eight BCs.
4. Use paper 4 as additional high-order coupled-system evidence, not as a like-for-like fluid benchmark.

For the new problem, these sources motivate implementation choices and tests. They do not remove the need to clarify the supplied equations, obtain a converged BVP reference, or check accuracy for each field and wall quantity. No claimed error, speedup, uniqueness or applicability to the exact five-field system was independently established here.

## Saved-file integrity and access

`manifest.json` records source URLs, file types, sizes and SHA-256 hashes. There are **three full-text PDFs** (two journal articles and one arXiv manuscript) and **one publisher code supplement**. Each downloaded document occupies its own directory. No downloaded code was executed, no original user document was changed, and nothing was saved in the Lorenz workspace.
