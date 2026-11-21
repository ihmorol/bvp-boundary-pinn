"""The PINN for Eqs. 14 to 19 (docs/PINN_PLAN.md).

model.py     networks and their derivatives
physics.py   the equations and boundary conditions on tensors
points.py    training and monitor points
constraints.py  boundary conditions built into the output (TFC)
losses.py    loss terms and the gradient-norm weights
train.py     the four arms: stages, Adam, L-BFGS
runs.py      configs and run folders
run.py       command line: train one arm and seed
evaluate.py  compare finished runs with the reference, build results/summary.csv
plots.py     result figures
"""
