"""Training and monitor points on [0, L], denser near the wall."""
from __future__ import annotations

import numpy as np
import torch


def wall_points(n, L, stretch, rng):
    """n points in [0, L]: one random point in each of n equal cells of s in [0, 1], then
    eta = L (exp(stretch s) - 1) / (exp(stretch) - 1). Larger stretch puts more points near the wall."""
    s = (np.arange(n) + rng.random(n)) / n
    return np.sort(L * np.expm1(stretch * s) / np.expm1(stretch))


def make_points(cfg, seed):
    """Fixed training points and a separate set of monitor points, both from one seed.

    The monitor points never enter a gradient; they only catch overfitting between training points.
    """
    rng = np.random.default_rng(seed)
    train = wall_points(cfg["n_points"], cfg["L"], cfg["stretch"], rng)
    monitor = wall_points(cfg["n_monitor"], cfg["L"], cfg["stretch"], rng)
    return column(train), column(monitor)


def column(x):
    return torch.as_tensor(np.asarray(x, dtype=float).reshape(-1, 1), dtype=torch.get_default_dtype())


def in_range(eta, lo, hi):
    """The points with lo <= eta < hi. Training points never sit exactly on L."""
    e = eta.reshape(-1)
    return eta[(e >= lo) & (e < hi)]
