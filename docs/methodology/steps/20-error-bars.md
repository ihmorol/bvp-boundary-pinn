# Step 20 · Error bars

[Index](README.md) · page 2, the classical reference · group 7, cross-checks · [← step 19](19-branch-search.md) · [step 21 →](21-l-study.md)

**In one line:** how accurate the reference is: the mesh error, and how far L = 20 is from infinity.

## What it does
1. **Mesh error (GCI):** from the three finest engine meshes, the observed order p and an extrapolated value give an error bar. About 10⁻⁷ relative.
2. **L extrapolation:** fit v(L) = v∞ + c L^(−p) to the values at L = 40, 80, 160, and compare v(20) with v∞.
3. **Two far forms:** solve L = 20, 40, …, 10 240 twice, with A taken at θ(L) (the reference's form) and with A = 1. The two approach the limit from opposite sides, so they bracket it.

## The equations
```text
p   = log(e21/e32)/log 2,   e21 = f1 − f2, e32 = f2 − f3  (coarse, middle, fine)
GCI = 1.25 abs(e32/f3)/(2^p − 1)
v(L) = v∞ + c L^(−p)
```

## In the code
- [tools/reference_error_bars.py](../../../tools/reference_error_bars.py): `three_level` ([line 38](../../../tools/reference_error_bars.py#L38)), `grid_index`, `l_extrapolation`, `far_ladder` ([line 71](../../../tools/reference_error_bars.py#L71)). Writes `reference/error_bars.json` and `.md`.

## Ready?
Yes.
