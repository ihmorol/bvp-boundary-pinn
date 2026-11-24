"""The PINN for Eqs. 14 to 19 (docs/pinn/PINN_PLAN.md).

model.py        networks and their derivatives
jets.py         all derivatives in one forward pass (Taylor mode)
start.py        optional fixed start profile under the network
physics.py      the equations and boundary conditions on tensors
points.py       training and monitor points
constraints.py  boundary conditions built into the output (TFC)
losses.py       loss terms, the fixed residual scale and the gradient-norm weights
train.py        the four arms: stages, Adam, refine stage (L-BFGS, BFGS or Levenberg-Marquardt)
bfgs.py         full BFGS with a dense inverse Hessian
lm.py           Levenberg-Marquardt for the exact-mode sum of squares
selfcheck.py    checks of a trained network that need no reference
runs.py         configs and run folders
run.py          command line: train one arm and seed
evaluate.py     compare finished runs with the reference, build results/summary.csv
plots.py        result figures
"""
