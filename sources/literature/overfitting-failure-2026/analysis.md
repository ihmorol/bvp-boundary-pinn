# Analysis: PINN failure as overfitting, 2026

**TL;DR.** PINN failures are residual overfitting: the residual dips at collocation points and stays large between them. **Gives us:** monitor-set divergence as the failure detector, fresh-set-only final metrics, an optional gradient penalty (a fifth derivative for f), and the L-BFGS stop rule. **Avoid:** equal loss weights; the regularization weight is never reported. **Relevance:** high.

## Record

- Paper: "PINNs Failure Modes are Overfitting" (running title, p.1).
- Authors: Nigel T. Andersen (Hokkaido University) and Takashi Matsubara (Hokkaido University and RIKEN AIP) (p.1).
- Status: preprint, marked "Preprint." on p.1. arXiv:2605.30910v1, 29 May 2026 (p.1).
- Pages read: all 15. Text extracted with pdftotext; pages 2, 4 and 5 were also inspected as images for Figures 1 to 3 and Eq. 5. Table 1 came from the text layer and was complete.
- Role for us: R42, support for fresh-point evaluation and a residual-gradient regularizer.

## Problem

- Claim: PINN failure modes are overfitting. The residual loss is minimized at the collocation points but stays large between them, so training converges to the wrong solution at low loss (Abstract p.1; Sec 4 p.5).
- Evidence 1, residual plots: for the convection equation at FP32, the residual along a time slice dips to about 1e-11 to 1e-15 exactly at the collocation point, while between points it sits many orders higher, up to 1e5 on the color scale. With the regularizer the residual is smooth, around 1e-4 to 1e-6, with no dip at the points (Fig 3(c),(e) p.5).
- Evidence 2, loss curves: train loss falls to about 1e-1 while the test loss on a large fresh point set stays high (Fig 1(d) p.2). The paper calls this train/test divergence the hallmark of a failure mode and reports seeing it in all their tests (Sec 4.1.1 p.5).
- Evidence 3, precision alone does not fix it: FP64 still fails the wave equation (Fig 2 p.4), and cutting the wave grid from 10201 to 144 points brings failure back even at FP64, while regularization restores success on the small grid (Sec 4.1.2 p.6).
- Their reading: dense grids work because they over-determine the problem, so the network cannot fit the residual only at the points (Sec 1 p.2).
- Separate diagnosis: Allen-Cahn is not a failure mode in their runs. Train and test loss move together, so the trouble is stiff optimization, fixed by FP64, with a late-stage test-loss rise that regularization controls (Sec 4.2 p.6; Fig 4 p.6).
- Cross-reference warning: the convection text cites Figure 1, which is captioned wave, and the wave text cites Figure 3, which is captioned convection FP32. The panel art matches the captions: Figure 1(a) shows the wave-shaped solution, Figure 3(a) shows the striped convection solution. We read the captions as correct and the citations as a preprint slip (p.2, p.5-6).

## Method

- Architecture: vanilla MLP, 4 hidden layers of 128 tanh neurons, Glorot normal init, inputs scaled to [-1,1] (Sec 3 p.3; App A p.13). Implementation in JAX with Flax (App A p.13).
- Optimizer: L-BFGS only, memory set to 100. Stop when the relative train-loss change over 100 iterations drops below sqrt(machine epsilon), 3.453e-4 at FP32 and 1.490e-8 at FP64, plus an iteration cap (App A p.13).
- Two point sets: a small training grid, and a much larger test set that is never trained on, 10000 random domain points plus 10000 per boundary and initial condition (Sec 3 p.3; App A p.13). Train loss on the small set, test loss on the fresh set; divergence flags overfitting (App A p.13).
- double PINN, Eq. 5 (p.4): adds (lambda_r/2) times the squared gradient of each loss with respect to the network inputs, so the domain residual gets a penalty on grad F, each boundary loss a penalty on grad of the boundary residual, each initial loss likewise. For our 1D case the norm reduces to one squared scalar per term.
- Difference from plain double backpropagation: the penalty covers every PINN loss term, not just one data loss, and flattened residuals stop the network from lowering the loss at the points at the cost of the space between them (Sec 3.1 p.4).
- Baselines: L2 and L1 weight penalties on convection beta=50 at FP32 (Sec 5.1 p.8-9); PINNsFormer, PINNMamba and vanilla FP64 PINN on the full benchmark (Sec 4.3 p.7).
- Benchmarks: convection, reaction, wave, Allen-Cahn, all 1D time-dependent PDEs with closed-form solutions except Allen-Cahn, which uses a spectral reference (App B p.13-14). Training grids: 400, 256, 144, 4096 domain points, 100 to 200 points per condition (Sec 4.3 p.7).
- Metrics: final loss, relative MAE, relative RMSE, and total collocation count N (Sec 4.3 p.7; Table 1 p.8).
- Cost: the paper states the input gradients theoretically double the compute per step, offset by faster convergence and fewer points (Sec 4.3 p.8; App C p.15).
- Gap: no numeric value for lambda_r appears anywhere in the paper.

## Key results (author-reported)

- Convection beta=50 at FP32: unregularized train loss about 1e-1 with test loss stuck high (Sec 4.1.1 p.5). Double PINN succeeds in a little over 5000 L-BFGS iterations; L2 also succeeds but slower with higher test loss; L1 avoids failure yet keeps large error (Fig 6 p.9; Sec 5.1 p.8-9).
- Wave at FP64: fails on a 12x12 = 144 grid, succeeds on the same grid once regularized (Sec 4.1.2 p.6).
- Table 1 (p.8), double PINN at FP64 versus three baselines trained on 10201 to 10504 points; their N is 800, 456, 444, 4396 for convection, reaction, wave, Allen-Cahn:
  - Convection: loss 4.0e-11, rMAE 4.9e-6, rRMSE 5.4e-6. Best baseline rMAE 5.9e-3, from PINN_FP64.
  - Reaction: loss 3.5e-12, rMAE 1.1e-5, rRMSE 2.9e-5. Best baseline rMAE 9.2e-3, from PINNMamba.
  - Wave: loss 3.5e-7, rMAE 3.8e-4, rRMSE 3.8e-4. Best baseline rMAE 8.0e-3, from PINN_FP64.
  - Allen-Cahn: loss 4.0e-5, rMAE 9.6e-5, rRMSE 3.6e-4. Best baseline rMAE 1.6e-2, from PINN_FP64.
- The abstract's "up to 23x fewer collocation points" (p.1) matches Table 1 ratios: wave 444 versus 10504, reaction 456 versus 10403. Allen-Cahn is only about 2.4x because its sharp interfaces need points (Sec 4.3 p.7).
- Table 1 footnote: loss values are not directly comparable across methods because of the penalty term (p.8).
- All numbers above are the paper's own, run on Titan RTX for FP32 and A100 for FP64 (Sec 3 p.3). We have not reproduced them.

## Adopt for our pipeline

1. Monitor on fresh points during training: log the residual penalty on our monitor set next to train loss. A falling train loss with a flat or rising monitor loss is their failure signature (App A p.13; Sec 4.1.1 p.5). The residual alone is enough, no analytic solution needed, which fits our problem where the classical solution is check-only.
2. Keep the fresh final evaluation set untouched by training: their headline rMAE and rRMSE come from a large fresh set (App A p.13; Table 1 p.8). Our final numbers should come only from our fresh set.
3. Test a residual-gradient penalty as a side run: for each residual R_i(eta) add (lambda_r/2)(dR_i/deta)^2 at train points. Our f equation is order 4, so the penalty pulls order-5 derivatives through autograd; g, theta, phi, chi need order 3. Expect roughly double residual cost per step, per their own cost note (Eq. 5 p.4; Sec 4.3 p.8; App C p.15).
4. Try plain L2 weight decay first: it fixed the same failure at FP32 with no extra autograd, only more iterations (Fig 6 p.9). Cheap to test inside our Adam then L-BFGS loop.
5. Grid density over precision: FP64 failed on sparse grids (Sec 4.1.2 p.6). Keep the eta grid dense enough that monitor points sit between train points, and watch between-point residual there.
6. Copy the stop rule for the L-BFGS phase: relative train-loss change over a 100-iteration window versus sqrt(machine epsilon) (App A p.13). Also stop on monitor loss, since they saw test loss rise late after the solution was learned (Sec 4.2 p.6).

## Do not import

1. The benchmark equations and their metrics as-is: they are 1D time-dependent PDEs with known solutions, not our five-field ODE BVP in eta (Sec 4.3 p.7; App B p.13-14).
2. Equal loss weights, lambda_F = lambda_B = lambda_I = 1 (Sec 3.1 p.4): our design uses weighted soft penalties across five fields and 12 BCs; this paper offers no evidence for all-ones in our setting.
3. A specific lambda_r: the value is never reported, so any value we pick is our own tuning choice, not theirs (Eq. 5 p.4; Sec 5.1 p.8).
4. Boundary-loss gradient penalties by default (Eq. 5 p.4): their boundary residuals are curves over t, ours are point conditions at eta=0 and eta=L, so the input gradient of a point condition is a different object and needs its own decision.
5. The 23x claim as a target for us: it is measured against transformer and state-space baselines on their PDE suite (Table 1 p.8), not against ODE BVP solvers.
6. Dropping Adam: they run L-BFGS from the start (Sec 3 p.3). That is their choice on smooth PDE losses; keep our Adam warm start.
7. Their verdict that Allen-Cahn is not a failure mode: it is equation-specific (Sec 4.2 p.6). Keep per-field diagnostics for our own equations.

## Quotes worth keeping

1. "the loss is minimized on the collocation points, but not elsewhere" (Abstract, p.1).
2. "we consider this divergence the hallmark of a PINNs failure mode" (Sec 4.1.1, p.5).
3. "By flattening the residuals, double PINN penalizes the network for reducing the losses at the collocation points at the expense of the surrounding region." (Sec 3.1, p.4)
4. "Unregularized PINN show a clear preference for reducing the residual at the collocation point, while double PINN remains smooth and shows no such preference." (Fig 3 caption, p.5)
5. "more collocation points makes it harder for the network to memorize the data" (Sec 1, p.2).

## Relevance

High. The mechanism, residual overfitting at collocation points, applies to any collocation-trained PINN including our 1D BVP, and the paper directly validates our monitor set and fresh final evaluation while offering a testable gradient regularizer.
