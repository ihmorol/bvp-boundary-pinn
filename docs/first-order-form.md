# From Eqs. 14 to 19 to the first-order form

Date: 2026-11-19. Purpose: explain, step by step, how the reference solver turns Eqs. 14 to 19 into the form `M*H = -r0` and then into a first-order system. Authoritative equations: [equations.md](equations.md). Residual forms used here: [PRD.md](PRD.md) section 1.4. Code: `src/bvp_pinn/reference/equations.py` (`residuals`, `highest_matrix`, `highest_derivatives`, `rhs`, `boundary`).

## 1. Is this step needed?

Yes, for the reference solution. No, for the PINN.

- **Reference solver.** `scipy.integrate.solve_bvp` only solves systems written as `dz/deta = F(eta, z)`, with first derivatives only. MATLAB's bvp4c and bvp5c have the same rule. Our equations have fourth derivatives (f) and second derivatives (the others), so they must be rewritten. Rewriting a higher-order ODE as a first-order system is the standard textbook way to use these solvers. This document is that rewrite, nothing more.
- **PINN.** The PINN does not use any of this. It takes autograd derivatives of its outputs, puts them into Eqs. 14 to 18 exactly as written, and squares the residuals in the loss. M and r0 never appear in training.
- **Accuracy.** The rewrite is exact algebra. It does not approximate anything, so it adds no error. The only care needed is that we never divide by a number close to zero (section 7).

So the step stays. It is the usual way, not an extra method.

## 2. The plan in three moves

1. Write the unknowns as a list of twelve values `z`, so each one only needs a first derivative.
2. Find the five derivatives that are not in `z`. Call them `H`. Each equation gives one of them.
3. Return `dz/deta`, which is `z` shifted by one derivative, with `H` filled in.

## 3. Names

```text
q = eta + kappa      h = eta + b      c = theta_w - 1      A = 1 + c*theta
f1, f2, f3, f4 = first to fourth eta-derivatives of f
g1, g2         = first and second derivatives of g
th1, th2       = the same for theta
ph1, ph2       = the same for phi
ch1, ch2       = the same for chi
```

With the locked values, `c = 0.2 - 1 = -0.8`, so `A = 1 - 0.8*theta`.

## 4. Step 1: the state z

f appears up to its fourth derivative, so we keep f and its first three derivatives. The other four fields appear up to their second derivative, so we keep each field and its first derivative.

```text
z = [f, f1, f2, f3, g, g1, theta, th1, phi, ph1, chi, ch1]     (12 values)
```

Now take the derivative of each entry:

| Entry of z | Its derivative | Already in z? |
|---|---|---|
| f | f1 | yes |
| f1 | f2 | yes |
| f2 | f3 | yes |
| f3 | **f4** | no |
| g | g1 | yes |
| g1 | **g2** | no |
| theta | th1 | yes |
| th1 | **th2** | no |
| phi | ph1 | yes |
| ph1 | **ph2** | no |
| chi | ch1 | yes |
| ch1 | **ch2** | no |

Seven derivatives are just copies from z. Five are missing:

```text
H = [f4, g2, th2, ph2, ch2]     (the highest derivatives)
```

The five equations (Eqs. 14 to 18) have exactly the job of giving these five values.

## 5. Step 2: split each equation into an H part and the rest

Look at where H appears in each equation. In every case it is only multiplied by a coefficient. It is never squared, never inside a fraction's denominator, never inside another function. That means each residual can be written as

```text
R = (coefficient times H) + (everything else)
```

We call the coefficients **M** (a 5 by 5 table) and the "everything else" **r0** (five numbers).

**How to get r0 in practice:** evaluate the residual formula with `f4 = g2 = th2 = ph2 = ch2 = 0`. Whatever is left is r0. That is exactly what the code does.

**About delta_star.** The delta_star terms contain only eta, f1, f2, f3, g, g1, th1 and ph1. None of these is in H, so in general these terms belong in r0. With delta_star = 0 they are zero, so they are left out below.

### 5.1 Eq. 14 (gives f4)

**Start:**

```text
R14 = (1+Delta)*(f4 + 2*f3/q - f2/q**2 + f1/q**3)
    + kappa/q*(f*f3 - f1*f2)
    + kappa/q**2*(f*f2 - f1**2)
    - kappa/q**3*f*f1
    - (f2 + f1/q)/Da
    - Delta*(g2 + g1/q)
    - 2*beta_m/h**4*((1/q - 4/h)*theta + th1)
```

**Find H.** f4 is inside the first bracket. g2 is inside the `Delta*(...)` bracket.

**Open those two brackets:**

```text
(1+Delta)*(f4 + 2*f3/q - f2/q**2 + f1/q**3)
    = (1+Delta)*f4  +  (1+Delta)*(2*f3/q - f2/q**2 + f1/q**3)

-Delta*(g2 + g1/q)
    = -Delta*g2  -  Delta*g1/q
```

**H part:** `(1+Delta)*f4 - Delta*g2`

**Row 1 of M** (columns f4, g2, th2, ph2, ch2):

```text
[1+Delta, -Delta, 0, 0, 0]
```

**Rest (r0_14):**

```text
r0_14 = (1+Delta)*(2*f3/q - f2/q**2 + f1/q**3)
      + kappa/q*(f*f3 - f1*f2)
      + kappa/q**2*(f*f2 - f1**2)
      - kappa/q**3*f*f1
      - (f2 + f1/q)/Da
      - Delta*g1/q
      - 2*beta_m/h**4*((1/q - 4/h)*theta + th1)
```

**Check.** Put f4 = g2 = 0 into R14. You get exactly r0_14.

### 5.2 Eq. 15 (gives g2)

**Start:**

```text
R15 = (1+Delta/2)*(g2 + g1/q)
    - Delta*B*(f2 + f1/q + 2*g)
    - kappa/q*(f1*g - f*g1)
```

**Find H.** Only g2, inside the first bracket.

**Open it:**

```text
(1+Delta/2)*(g2 + g1/q) = (1+Delta/2)*g2 + (1+Delta/2)*g1/q
```

**H part:** `(1+Delta/2)*g2`

**Row 2 of M:**

```text
[0, 1+Delta/2, 0, 0, 0]
```

**Rest (r0_15):**

```text
r0_15 = (1+Delta/2)*g1/q
      - Delta*B*(f2 + f1/q + 2*g)
      - kappa/q*(f1*g - f*g1)
```

### 5.3 Eq. 16 (gives th2)

**Start:**

```text
R16 = (th2 + th1/q)/Pr
    + Nr/Pr*(A**3*th1/q + 3*c*A**2*th1**2 + A**3*th2)
    + kappa/q*f*th1
    + Nt/Pr*th1**2
    + Nb/Pr*th1*ph1
    + 2*beta_m*lambda_m*(theta-epsilon)/(Pr*h**3)*(kappa*f/q*(1-2/h**2) - 2*f1/h)
```

**Find H.** th2 appears in two places:

```text
first bracket:   th2/Pr
second bracket:  Nr/Pr*A**3*th2
```

**Add the two coefficients:**

```text
1/Pr + Nr*A**3/Pr = (1 + Nr*A**3)/Pr
```

**H part:** `(1 + Nr*A**3)/Pr * th2`

Note that A depends on theta, so this coefficient changes along eta. It is the only diagonal entry of M that is not a fixed number.

**Row 3 of M:**

```text
[0, 0, (1+Nr*A**3)/Pr, 0, 0]
```

**Rest (r0_16):**

```text
r0_16 = th1/(q*Pr)
      + Nr/Pr*(A**3*th1/q + 3*c*A**2*th1**2)
      + kappa/q*f*th1
      + Nt/Pr*th1**2
      + Nb/Pr*th1*ph1
      + 2*beta_m*lambda_m*(theta-epsilon)/(Pr*h**3)*(kappa*f/q*(1-2/h**2) - 2*f1/h)
```

### 5.4 Eq. 17 (gives ph2)

**Start:**

```text
R17 = (ph2 + ph1/q)/(Le*Pr)
    + Nt/(Le*Pr*Nb)*(th2 + th1/q)
    + kappa/q*f*ph1
```

**Find H.** ph2 in the first bracket, and th2 in the second bracket. So this equation mixes two highest derivatives.

**Open both brackets:**

```text
(ph2 + ph1/q)/(Le*Pr)          = ph2/(Le*Pr) + ph1/(q*Le*Pr)
Nt/(Le*Pr*Nb)*(th2 + th1/q)    = Nt/(Le*Pr*Nb)*th2 + Nt/(Le*Pr*Nb)*th1/q
```

**H part:** `Nt/(Le*Pr*Nb)*th2 + 1/(Le*Pr)*ph2`

**Row 4 of M:**

```text
[0, 0, Nt/(Le*Pr*Nb), 1/(Le*Pr), 0]
```

**Rest (r0_17):**

```text
r0_17 = ph1/(q*Le*Pr)
      + Nt/(Le*Pr*Nb)*th1/q
      + kappa/q*f*ph1
```

### 5.5 Eq. 18 (gives ch2)

**Start:**

```text
R18 = (ch2 + ch1/q)/Lb
    + kappa/q*f*phi
    + Pe/Pr*(chi*ph2 + ch1*ph1)
```

**Find H.** ch2 in the first bracket, and ph2 in the last bracket (multiplied by chi).

**Open both brackets:**

```text
(ch2 + ch1/q)/Lb           = ch2/Lb + ch1/(q*Lb)
Pe/Pr*(chi*ph2 + ch1*ph1)  = Pe*chi/Pr*ph2 + Pe/Pr*ch1*ph1
```

**H part:** `Pe*chi/Pr*ph2 + 1/Lb*ch2`

The coefficient of ph2 contains chi, so it changes along eta. That is fine: chi is in z, so at any point it is a known number.

**Row 5 of M:**

```text
[0, 0, 0, Pe*chi/Pr, 1/Lb]
```

**Rest (r0_18):**

```text
r0_18 = ch1/(q*Lb)
      + kappa/q*f*phi
      + Pe/Pr*ch1*ph1
```

## 6. Step 3: put the five rows together and solve

Each equation must equal zero:

```text
R = M*H + r0 = 0      so      M*H = -r0
```

Written out:

```text
| 1+Delta   -Delta      0               0           0    |   | f4  |   | -r0_14 |
| 0         1+Delta/2   0               0           0    |   | g2  |   | -r0_15 |
| 0         0           (1+Nr*A**3)/Pr  0           0    | * | th2 | = | -r0_16 |
| 0         0           Nt/(Le*Pr*Nb)   1/(Le*Pr)   0    |   | ph2 |   | -r0_17 |
| 0         0           0               Pe*chi/Pr   1/Lb |   | ch2 |   | -r0_18 |
```

### 6.1 Solving by hand

M is almost triangular, so the unknowns come out one at a time. Each line below is just "move the other terms to the right, then divide by the coefficient".

```text
From row 2:   (1+Delta/2)*g2 = -r0_15
              g2  = -r0_15 / (1+Delta/2)

From row 1:   (1+Delta)*f4 - Delta*g2 = -r0_14
              f4  = (-r0_14 + Delta*g2) / (1+Delta)

From row 3:   (1+Nr*A**3)/Pr*th2 = -r0_16
              th2 = -r0_16 * Pr / (1+Nr*A**3)

From row 4:   Nt/(Le*Pr*Nb)*th2 + ph2/(Le*Pr) = -r0_17
              ph2 = (-r0_17 - Nt/(Le*Pr*Nb)*th2) * (Le*Pr)

From row 5:   Pe*chi/Pr*ph2 + ch2/Lb = -r0_18
              ch2 = (-r0_18 - Pe*chi/Pr*ph2) * Lb
```

Order matters only a little: g2 before f4, th2 before ph2, ph2 before ch2.

### 6.2 What the code does

The code builds M and r0 and calls `np.linalg.solve(M, -r0)`. This gives the same five numbers as the hand solution. We use it because it is one line instead of five and leaves less room for a sign mistake. We do not compute the inverse of M, because solving directly is cheaper and slightly more accurate. Either way the answer is the same to rounding error.

### 6.3 The first-order system

Now every derivative is known:

```text
d/deta [f,  f1, f2, f3, g,  g1, theta, th1, phi, ph1, chi, ch1]
     = [f1, f2, f3, f4, g1, g2, th1,   th2, ph1, ph2, ch1, ch2]
```

The plain entries are copies from z. The five H entries come from section 6.1. This is the function `rhs` that solve_bvp calls at every mesh point.

## 7. When could this go wrong?

Every line in section 6.1 divides by a diagonal entry of M. If one of them were close to zero, a tiny error in r0 would become a huge error in H.

| Diagonal entry | Locked value | Can it get small? |
|---|---|---|
| 1+Delta | 1.5 | No, it is fixed |
| 1+Delta/2 | 1.25 | No, it is fixed |
| (1+Nr*A**3)/Pr | depends on theta | Yes, if A**3 nears -2 (A near -1.26, theta near 2.8) |
| 1/(Le*Pr) | 0.114 | No, it is fixed |
| 1/Lb | 1 | No, it is fixed |

Only the third one needs watching, so the solver logs its smallest value along the solution. On 2026-11-19 it was 0.144 for branch A and 0.117 for branch B ([SPEC](SPEC_reference_solutions.md) section 8, check V9). That is far from zero.

**Why not just check the determinant?** For this matrix the determinant is the product of the five diagonal entries. "Not zero" only means you can divide. A value like 1e-12 is not zero but still blows errors up. So we look at how small the diagonal gets, not only whether it is exactly zero.

## 8. Eq. 19: the boundary conditions

The boundary conditions need no rearranging. They use only f, f1, f2, g, theta, th1, phi, ph1 and chi, and all of these are already in z. No H value appears in any of them. So the boundary function reads z at the wall (`za`, eta = 0) and at the far end (`zb`, eta = L), and returns twelve numbers that must be zero:

```text
At eta = 0 (from za):
B1  = f1_0 - 1 - N1*(f2_0 - f1_0/kappa)
B2  = f_0 - fw/(Pr*Le)*ph1_0
B3  = g_0 + ell*f2_0
B4  = -th1_0 - Bi*(1-theta_0) - Rd/(theta_w-1)*(theta_w**4 - (1+(theta_w-1)*theta_0)**4)
B5  = ph1_0 + Nt/Nb*th1_0
B6  = chi_0 - 1

At eta = L (from zb):
B7  = f1_L     B8  = f2_L     B9  = g_L
B10 = theta_L  B11 = phi_L    B12 = chi_L
```

Count: twelve conditions for twelve first-order states. That is the number solve_bvp needs.

B4 is used in its stable algebraic form from [equations.md](equations.md), after a test showed both forms agree.

## 9. How we know the algebra is right

| Test | What it checks | File |
|---|---|---|
| `R(H) - R(0) = M*H` at random states | The split in section 5 matches the original equations exactly | `tests/test_equations_algebra.py` |
| NumPy residuals against SymPy | The residual code matches an independent transcription | `tests/test_equations_sympy.py` |
| Manufactured solution, observed order 3.38 to 3.83 | The whole first-order system and solver work together | SPEC check V3 |
| All twelve boundary errors below 1e-15 at the solution | The boundary function and solve are consistent | SPEC check V7 |

All of these pass as of 2026-11-19.

## 10. Summary

- solve_bvp needs first derivatives only, so we list twelve states and ask the equations for the five missing highest derivatives.
- Each equation is linear in those five values, so we split it into `M*H + r0` and solve `M*H = -r0`.
- M is almost triangular, so the solve is five simple divisions in a fixed order.
- The rewrite is exact and adds no error. Only the entry `(1+Nr*A**3)/Pr` needs watching, and it stays at 0.12 or above.
- The PINN does not use this. It trains on Eqs. 14 to 18 directly. This rewrite only serves the reference that the PINN is tested against.
