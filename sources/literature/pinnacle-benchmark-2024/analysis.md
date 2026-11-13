# Analysis: PINNacle benchmark, 2024

**TL;DR.** 22 PDE tasks across about a dozen method columns, three seeds each. **Gives us:** evaluation practice (NaN logged as non-convergence, 10 percent threshold, max-error metric), boundary weight 100 helps elliptic and hurts steady NS, NTK reweighting strongest, L-BFGS NaN on 7 of 22 tasks yet best where it converged. **Avoid:** RAR as a default (little gain); the suite has no ODE BVP. **Relevance:** medium.

## Record
- Title (as printed): "PINNacle: A C~~omprehensive~~ Benchmark of Physics-Informed Neural Networks for Solving PDEs". The struck word is part of the quoted title, not my wording.
- Authors: Zhongkai Hao, Jiachen Yao, Chang Su, Hang Su, Ziao Wang, Fanzhi Lu, Zeyu Xia, Yichi Zhang, Songming Liu, Lu Lu, Jun Zhu (p.1).
- Venue: PDF header says "Preprint, under review", arXiv:2306.08827v2, dated 5 Oct 2023 (p.1). Our role note assigns it to the NeurIPS 2024 Datasets and Benchmarks track.
- Pages read: all 50 pages by text extraction. Pages 7, 8, 24, 31 re-checked as rendered PDF for table layout. Appendix figures (pp. 25-30, 43-50) skimmed: they are reference-solution and prediction galleries. References (pp. 10-13) read. Nothing was unreadable.

## Problem
- 22 cases built from over 20 PDEs (Sec 3.2, p.4): Burgers 1d and 2d; Poisson in 4 cases; Heat in 4; Navier-Stokes in 3; Wave in 3; chaotic Gray-Scott and Kuramoto-Sivashinsky; 5d Poisson and 5d heat; two inverse problems PInv and HInv (Sec 3.2, p.4; Appendix B, pp. 14-20).
- Four named challenge classes: complex geometry, multi-scale, nonlinearity, high dimension (Table 1, p.5). Long time span shows up as a recurring fifth difficulty (Sec 4.1, pp. 7-8).
- Closest to steady BVPs: Poisson2d-C (Laplace, Dirichlet, pp. 14-15), Poisson2d-CG (Helmholtz k=8, Dirichlet, p.15), Poisson3d-CG (two-domain coefficients, Neumann, pp. 15-16), Poisson2d-MS (piecewise coefficient, Robin BC, p.16), and steady NS2d-C and NS2d-CG at Re=100 with no-slip walls (p.17). None is an ODE BVP and there is no similarity or boundary-layer problem in the suite.
- Stiffness analog: Poisson2d-MS has strongly varying coefficients and defeats every variant (Table 3, p.7). NS2d-C is boundary-condition dominated (p.17).

## Method
- Variants (Sec 3.3.1, pp. 5-6; Table 2, p.6): vanilla PINN; PINN-w with boundary weight 100 (App C.3, p.22); reweighting by gradient norms (PINN-LRA) and by neural tangent kernel (PINN-NTK); residual-based resampling (RAR); MultiAdam optimizer; gPINN and hp-VPINN loss forms; LAAF, GAAF and FBPINN architectures. An L-BFGS arm appears in Table 3 (p.7).
- Setup: 5 layer MLP, width 100; Adam lr 1e-3, betas (0.9, 0.999); 20,000 epochs; 8192 collocation and 2048 boundary points in 2D, 32768 and 8192 in 3D (App C.1-C.2, p.22).
- Metrics: L2RE and L1RE (Eq. 4, p.6); MSE, max error mERR, Fourier error in low, mid and high bands (Eq. 5, p.7); error vs time for time problems (p.7); runtime (Table 12, p.38).
- Protocol: every experiment run 3 times, mean and std reported (Sec 4.1, p.7; Tables 5-8, p.31 ff.). Success threshold L2RE 10% (p.7). "NaN" marks non-convergence and "-" marks not applicable (Table 3 caption, p.7). References come from COMSOL FEM and Chebfun spectral solves and are used only for scoring (Sec 3.2, p.5).

## Key results (author-reported)
- Vanilla PINN solves 10 of 22 tasks at the 10% threshold (Sec 4.1, p.7).
- PINN-w cuts Poisson2d-C from 6.94E-1 to 3.49E-2 and Poisson2d-CG from 6.36E-1 to 6.08E-2, but worsens NS2d-C from 4.70E-2 to 1.45E-1 and Heat2d-MS from 6.21E-2 to 2.42E-1 (Table 3, p.7).
- PINN-NTK is best on Poisson2d-C (1.23E-2), Poisson2d-CG (1.43E-2), Wave1d-C (9.79E-2), and wins 3 of 6 parametric tasks (Table 3, p.7; Sec 4.2, p.8).
- PINN-LRA diverges on NS2d-C (NaN, Table 3, p.7; Fig 25 caption, p.48). RAR gives little gain, e.g. Poisson2d-C 6.99E-1 against 6.94E-1 for vanilla (Table 3, p.7).
- Multi-scale steady Poisson2d-MS fails for all variants, best 5.90E-1 with MultiAdam; all methods also fail on Poisson3d-CG and the long-time cases (Table 3, p.7; Sec 4.1, p.8).
- FBPINN is best on chaotic GS (7.99E-2) but worst on NS2d-CG (8.27E+0); LAAF is best on Heat2d-CG (2.39E-2) (Table 3, p.7; Sec 4.1, p.8).
- vPINN helps inverse problems but shows "no clear improvement" on forward cases (Sec 4.1, p.8).
- L-BFGS is NaN on 7 of 22 tasks in Table 3 (p.7: Poisson2d-C, NS2d-CG, Wave1d-C, GS, KS, PInv, HInv) yet is best on PNd (4.67E-4) and HNd (1.19E-4); MultiAdam converges more often but with worse accuracy where L-BFGS works (Table 24, p.42).
- gPINN hurts Poisson2d-C at every tested weight, 0.67 to 0.70 L2RE (Table 18, p.41; Sec E.2, p.40).
- Larger batch and more epochs help to a plateau near 8192 points and 20k-80k epochs (Sec 4.3, p.9; Tables 14-15, pp. 39-40). Adam betas (0.99, 0.99) are best on average (Table 16, p.40). lr 1e-3, 1e-4 or step decay is stable, 1e-2 spikes (Sec 4.3, p.9; Table 13, p.39).
- Domain scale changes results a lot and MultiAdam is the most stable across scales (Sec E.2, p.41; Table 23, p.42).
- No wall-specific metric appears anywhere; max error is their only worst-case metric (Eq. 5, p.7) and the NS wall regions get no dedicated analysis (Sec 4.1, p.8).
- Our observation, not theirs: Table 3 lists vPINN on HInv as 1.19E-2 while Table 5 lists 4.56E-1 for the same cell (p.7 vs p.31), and Table 3 and Table 24 disagree on L-BFGS for NS2d-C (p.7 vs p.42). Treat single-cell claims with care.

## Adopt for our pipeline
- Report mean and std over at least 3 seeds per variant, and log non-converged runs as NaN instead of dropping them (Sec 4.1, p.7; Table 3 caption, p.7). This fits our Adam then L-BFGS plan, where L-BFGS may fail.
- Use a fixed success threshold like their 10% L2RE, plus a max-error metric over the eta grid for worst-case near-wall error (Eq. 5, p.7). They note metric rankings differ across metrics (App E.1, pp. 23-24), so we score wall-band and global error separately.
- Sweep boundary weights around our fixed scales instead of assuming bigger is safer. Their 100x BC weight helped steady elliptic cases and hurt the steady NS case (Table 3, p.7; App C.3, p.22).
- Add PINN-NTK as one comparison arm for balancing our 12 boundary penalties and 5 residual terms. It was their strongest automatic balancer and the best under parameter changes (Sec 4.2, p.8; Sec 5, pp. 9-10).
- Keep L-BFGS but with a fallback to the Adam checkpoint when it diverges, mirroring their NaN handling (Table 24, p.42).
- Run Adam at lr 1e-3 or 1e-4, try betas (0.99, 0.99), and consider step decay (Table 13, p.39; Table 16, p.40).
- Ablate collocation density; more points helped to saturation and our 1D eta domain makes dense sampling cheap (Sec 4.3, p.9; Table 14, p.39).
- Test two or three far-field lengths L. PINN error depends on domain scale even when the equation is only rescaled (Sec E.2, p.41; Table 23, p.42).
- Keep the classical reference separate from training labels, as they keep FEM and spectral references for scoring only (Sec 3.2, p.5).

## Do not import
- The PDE suite itself: time-dependent or multidimensional PDEs, no ODE BVP, no near-wall inverse-power coefficients (Sec 3.2, p.4).
- RAR resampling as a default: near-zero gain across their suite (Table 3, p.7).
- gPINN gradient terms: hurt one Poisson case at all weights, and our 4th-order equation would push derivative order higher still (Table 18, p.41; Sec E.2, p.40).
- vPINN weak form as the main formulation: no forward-accuracy gain in their results, and a weak form of a 4th-order ODE adds integration weights we do not need (Sec 4.1, p.8).
- FBPINN decomposition as a default: gains sat in 2D geometry and chaotic cases, results swing with overlap ratio (5.81E-1 to 4.19E-2 on Poisson2d-C, Table 22, p.42), and our domain is 1D.
- MultiAdam as a default: the paper reports it "does not significantly outperform other methods" overall; its wins sit in high dimension and domain-scale stability (Sec 4.1, p.8; Table 23, p.42).
- Fourier error as a headline metric: built for spectral content of fields over space-time; our target is wall accuracy in one coordinate (Eq. 5, p.7).
- Their 5x100 MLP as a final choice: tuned for 2D PDEs, not a 5-output system with a 4th-order equation (App C.1, p.22).

## Quotes worth keeping
- "vanilla PINN only solves 10 out of 22 tasks" (p.7).
- "directly optimizing an average of the PDE losses and initial/boundary condition losses leads to critical issues such as loss imbalance, suboptimal convergence, and limited expressiveness" (p.7).
- "PINN-NTK is a strong method for balancing loss weights" (pp. 9-10, the sentence crosses the page break).
- "L-BFGS does not converge in many cases as it is unstable while MultiAdam has a better convergence property. However, L-BFGS achieves better accuracy on some of the problems like high dimensional PDEs." (p.41).
- "We suggest that adding higher-order PDE residuals might harm the training process in some situations." (p.40).

## Relevance
- Medium: the paper gives us evaluation practice, reweighting and optimizer evidence, and failure reporting habits that transfer, but every problem in it is a PDE, so no result carries over directly to our five-field ODE BVP.
