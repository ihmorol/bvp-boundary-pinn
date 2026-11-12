# Equations and boundary calculation sheet

Working target: the supervisor-confirmed Eqs. 14–19 in [the source draft](../sources/project-draft/project3Nov_2026.docx). We are solving the given numerical specification. We are not correcting its physical model by copying a related paper.

There are five physical outputs: f, g, theta, phi, chi. Primes mean derivatives with respect to eta, not NN weights. The NN weights are named `w` in the strategy so they are not confused with the physical output theta.

## Short names used below

```text
q = eta + kappa
h = eta + b
c = theta_w - 1
A = 1 + c*theta

f1 = df/deta; f2 = d2f/deta2; f3 = d3f/deta3; f4 = d4f/deta4
g1 = dg/deta; g2 = d2g/deta2
th1 = dtheta/deta; th2 = d2theta/deta2
ph1 = dphi/deta; ph2 = d2phi/deta2
ch1 = dchi/deta; ch2 = d2chi/deta2
```

**Read:** f one is f prime; f two is f double prime; f four is the fourth derivative of f. A is one plus theta-w minus one times theta. These names shorten calculations; they are not new physical parameters.

## Corrected source transcription of Eq. 14

The November 10 Lorenz-repo method file is not authoritative for this implementation. Its Eq. 14 transcription incorrectly placed part of the unsteady term outside its bracket and later introduced a guessed reading of the convective bracket. The enlarged original page was inspected again on November 12.

The printed first convective bracket is **f*f3 - f1*f2**, not f4-f1*f3. The printed `-0.5*delta_star` multiplies the whole bracket containing both `(eta*f3+3*f2)` and `(0.5*eta*f2+f1)/q`. The new transcription below follows that grouping. This is a copying correction, not a change to the supervisor's model.

```text
R14 = (1+Delta)*(f4 + 2*f3/q - f2/q**2 + f1/q**3)
    + kappa/q*(f*f3 - f1*f2)
    + kappa/q**2*(f*f2 - f1**2)
    - kappa/q**3*f*f1
    - 0.5*delta_star*((eta*f3 + 3*f2)
                      + (0.5*eta*f2 + f1)/q)
    - (f2 + f1/q)/Da
    - Delta*(g2 + g1/q)
    - 2*beta_m/h**4*((1/q - 4/h)*theta + th1)
```

**Read:** assemble each signed term from the predictions and derivatives at the same eta. R fourteen is the resulting left-side error. Its target is zero. Keep parentheses: changing them changes the equation.

## Eq. 15

```text
R15 = (1+Delta/2)*(g2 + g1/q)
    - Delta*B*(f2 + f1/q + 2*g)
    - kappa/q*(f1*g - f*g1)
    - delta_star/2*(eta*g1 + 3*g)
```

**Read:** one plus Delta over two times the g-derivative bracket; minus the Delta-times-B bracket; minus the kappa-over-q product bracket; minus the delta-star-over-two bracket. The result should be zero.

## Eq. 16

```text
R16 = (th2 + th1/q)/Pr
    + Nr/Pr*(A**3*th1/q + 3*c*A**2*th1**2 + A**3*th2)
    + kappa/q*f*th1
    + Nt/Pr*th1**2
    + Nb/Pr*th1*ph1
    + 2*beta_m*lambda_m*(theta-epsilon)/(Pr*h**3)
      *(kappa*f/q*(1-2/h**2) - 2*f1/h)
    - delta_star*eta/2*th1
```

**Read:** th one and th two are theta's first and second eta derivatives. Squares and cubes here are ordinary powers, not derivative orders. This error also uses f, f prime and phi prime. All coefficients stay fixed during the run.

## Eq. 17

```text
R17 = (ph2 + ph1/q)/(Le*Pr)
    + Nt/(Le*Pr*Nb)*(th2 + th1/q)
    + kappa/q*f*ph1
    - delta_star*eta/2*ph1
```

**Read:** phi double prime and phi prime form the first bracket. Theta double prime and theta prime form the second. The flow term uses f times phi prime. Divide by the full products shown in the denominators.

## Eq. 18, retained as confirmed

```text
R18 = (ch2 + ch1/q)/Lb
    + kappa/q*f*phi
    + Pe/Pr*(chi*ph2 + ch1*ph1)
    - delta_star*eta/2*ph1
```

**Read:** chi double prime plus chi prime over q, divided by Lb; plus kappa over q times f times phi; plus Pe over Pr times the two-product bracket; minus delta-star times eta over two times phi prime. The printed f-times-phi and final phi-prime terms are intentional in this working transcription because the supervisor confirmed the source as written. Do not replace them with chi terms silently.

## Boundary errors from Eq. 19

Suffix `0` means eta=0. Suffix `L` means eta=L, approximating the far-away boundary.

```text
B1 = f1_0 - 1 - N1*(f2_0 - f1_0/kappa)
B2 = f_0 - fw/(Pr*Le)*ph1_0
B3 = g_0 + ell*f2_0
B4 = -th1_0 - Bi*(1-theta_0)
     - Rd/(theta_w-1)*(theta_w**4
                       - (1+(theta_w-1)*theta_0)**4)
B5 = ph1_0 + Nt/Nb*th1_0
B6 = chi_0 - 1
B7 = f1_L
B8 = f2_L
B9 = g_L
B10 = theta_L
B11 = phi_L
B12 = chi_L
```

**Read:** each B is a condition moved to the left, so its target is zero. B3 is g at zero plus ell times f double prime at zero. B5 is phi prime at zero plus Nt over Nb times theta prime at zero. B11 is phi at L; its target is the printed zero, not a reinterpreted concentration value.

The original zero-flux relation is a cancellation, not two separate zero slopes. Do not set theta(0)=1, phi(0)=1, f(0)=0, f(L)=0 or f3(L)=0 as extra conditions.

For stable evaluation of the radiation term, when c is nonzero:

```text
Aw = 1 + c*theta_0
Qw = (1-theta_0)*(theta_w+Aw)*(theta_w**2+Aw**2)
B4 = -th1_0 - Bi*(1-theta_0) - Rd*Qw
```

This is algebraically equal to the original quartic difference divided by c. Test both forms before selecting the stable form. A removable algebraic limit at c=0 does not prove that temperature normalization is physically defined there.

## Dependency map for implementation

| Equation | Fields needed | Highest derivative |
|---|---|---|
| R14 | f, g, theta | f4, g2 |
| R15 | f, g | f2, g2 |
| R16 | f, theta, phi | th2, ph1 |
| R17 | f, theta, phi | th2, ph2 |
| R18 | f, phi, chi | ph2, ch2 |

The first four equations and their boundary conditions do not contain chi. For the **classical reference**, the ten-state core can be checked first and chi can then be solved using the core curves, because the printed R18 is linear in chi for fixed core fields. This is an optional cross-check. The baseline PINN still predicts all five fields and trains their original equations together.

## Highest-derivative system for the reference

Use twelve first-order states:

```text
z = [f, f1, f2, f3, g, g1, theta, th1, phi, ph1, chi, ch1]
H = [f4, g2, th2, ph2, ch2]
```

Collect the original equations as `R = M*H + r0`. Here r0 means evaluate the original residual formulas with H=0 at the given state. The matrix is:

```text
M = [
 [1+Delta, -Delta,          0,                         0,         0],
 [0,       1+Delta/2,       0,                         0,         0],
 [0,       0,              (1+Nr*A**3)/Pr,            0,         0],
 [0,       0,              Nt/(Le*Pr*Nb),             1/(Le*Pr), 0],
 [0,       0,              0,                         Pe*chi/Pr, 1/Lb]
]
```

Solve `M*H=-r0`; do not form an explicit matrix inverse. This triangular derivative structure follows from the checked transcription above. Its diagonal coefficients must be nonzero for the regular first-order reference. Monitor suitably scaled conditioning, especially `(1+Nr*A**3)/Pr` along the computed state. A nonzero determinant alone is not a useful numerical accuracy certificate.

## Mathematical integrity tests

1. Test analytic derivatives through fourth order with a smooth exponential and polynomial examples, including exact constant/zero derivatives handled deliberately.
2. Verify the two R14 bracket groupings against the source; never import the previous wrong file as code.
3. Check `R(H)-R(0)=M*H` at random finite artificial states. This confirms the algebraic rearrangement, not physical validity.
4. Test a quadratic zero-flux wall cancellation and B2/B5's consequence `f0=-fw*Nt/(Pr*Le*Nb)*th1_0` when all denominators are nonzero.
5. Verify all residual outputs have shape `(N,1)` before stacking into `(N,5)`.
6. Compare a separately coded NumPy reference evaluator with the Torch evaluator at identical artificial states. Shared helper code is useful, but an independent transcription test is needed to catch shared errors.
