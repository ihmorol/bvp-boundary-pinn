"""The PINN for Eqs. 14 to 19 (docs/pinn/PINN_PLAN.md). Training and selection only: nothing in
this package reads reference data (gate G5). Evaluation is in bvp_pinn.evaluation.

model.py        networks and their derivatives
jets.py         all derivatives in one forward pass (Taylor mode)
start.py        optional fixed start profile under the network
physics.py      the equations and boundary conditions on tensors (the shared reference functions)
points.py       training and monitor points
constraints.py  boundary conditions built into the output (TFC)
losses.py       loss terms, the fixed residual scale and the gradient-norm weights
train.py        the four arms: stages, Adam, refine stages (L-BFGS, BFGS, Levenberg-Marquardt)
bfgs.py         full BFGS with a dense inverse Hessian
lm.py           Levenberg-Marquardt on the training objective, one block per network
selfcheck.py    checks of a trained network that need no reference
estimate.py     Newton-step error estimate (C++ engine) and the acceptance rule
runs.py         configs, checks, run folders
run.py          command line: train one arm and seed
campaign.py     command line: every arm and seed of the campaign, resumable
"""
