# Substituted reading sheet for Eqs. 14 to 19

This sheet restates [Eqs. 14 to 19](equations.md) with the [locked 2026-11-16 values](parameters.md#supervisor-confirmation-2026-11-16) substituted. It is a reading aid, not a new specification. The authoritative calculation sheet stays [equations.md](equations.md). The locked values stay in [parameters.md](parameters.md) and [the case config](../../configs/physical-case-pending.json).

Every number below is arithmetic on locked values. No solver, reference, training or evaluation ran. The five functions f, g, theta, phi, chi are still unknown.

## Replacements used

```text
q = eta + 0.2          (kappa = 0.2)
A = 1 - 0.8*theta      (theta_w = 0.2, so c = theta_w - 1 = -0.8)
```

With b = 0.2 as well, h = eta + b equals q = eta + 0.2. This merge holds only while kappa and b keep their locked values. [equations.md](equations.md) keeps q and h separate; this sheet merges them for reading.

Other substituted coefficients:

| Expression | Value | From |
|---|---:|---|
| 1+Delta | 1.5 | Delta=0.5 |
| 1+Delta/2 | 1.25 | Delta=0.5 |
| Delta*B | 0.25 | Delta=0.5, B=0.5 |
| 1/Da | 5 | Da=0.2, derived from M=5 and M=1/Da |
| kappa | 0.2 | k=0.2 |
| 2*beta_m | 0.4 | bm=0.2 |
| 2*beta_m*lambda_m | 0.2 | bm=0.2, L=lambda_m=0.5 |
| Pr*kappa | 1.76 | Pr=8.8, kappa=0.2 |
| Nt/Nb | 3 | Nt=0.3, Nb=0.1 |
| Pe/Pr | 5/22 | Pe=2, Pr=8.8 |
| fw/(Pr*Le) | 1/44 | fw=0.2, Pr=8.8, Le=1 |
| ell | 0.3 | l=0.3 |
| Bi, Rd | 1, 1 | direct |

delta_star = 0 (received dt=0). Every term it multiplies is gone, so all five equations below are steady forms.

Eqs. 16 and 17 are multiplied by Pr=8.8 below. Multiplying an equation by a positive constant does not change where it equals zero.

## Eq. 14 with numbers

```text
0 = 1.5 * (f'''' + 2*f'''/q - f''/q^2 + f'/q^3)

  + (0.2/q)   * (f*f''' - f'*f'')
  + (0.2/q^2) * (f*f'' - (f')^2)
  - (0.2/q^3) * f*f'

  - 5   * (f'' + f'/q)            (1/Da = 5)
  - 0.5 * (g'' + g'/q)            (Delta = 0.5)

  - (0.4/q^4) * (theta' - 3*theta/q)
```

The last line comes from 2*beta_m/h^4*((1/q - 4/h)*theta + th1). With h=q, the bracket is (1/q - 4/q)*theta + theta' = theta' - 3*theta/q.

At the wall eta=0, q=0.2, so 1/q=5, 1/q^2=25, 1/q^3=125 and 1/q^4=625. These are coefficient sizes, not solution values; [parameters.md](parameters.md#coefficient-sizes-for-the-locked-target) records them.

## Eq. 15 with numbers

```text
0 = 1.25 * (g'' + g'/q)
  - 0.25 * (f'' + f'/q + 2*g)
  - (0.2/q) * (f'*g - f*g')
```

## Eq. 16 with numbers (multiplied by 8.8)

```text
0 = (1 + 0.5*A^3) * (theta'' + theta'/q)

  + (0.3 - 1.2*A^2) * (theta')^2
  + 0.1 * theta' * phi'
  + (1.76/q) * f * theta'

  + [0.2*(theta - 0.1)/q^4]
      * [0.2*f*(1 - 2/q^2) - 2*f']
```

Reading notes:

- The theta'' coefficient is 1 + 0.5*A^3 with A = 1 - 0.8*theta. It still depends on the unknown theta, so the equation stays nonlinear. Its size must be checked along any computed solution; the locked values alone do not prove regularity.
- 0.3 is Nt. 1.2 is 3*c*Nr = 3*(-0.8)*0.5. 1.76 is Pr*kappa.
- The last line is 2*beta_m*lambda_m*(theta-epsilon)/(h^3)*(kappa*f/q*(1-2/h^2) - 2*f1/h) with h=q and 2*beta_m*lambda_m = 0.2. Both inner pieces carry one factor of 1/q, which moves the bracket from 1/q^3 to 1/q^4.

## Eq. 17 with numbers (multiplied by 8.8)

```text
0 = phi'' + phi'/q
  + 3 * (theta'' + theta'/q)
  + (1.76/q) * f * phi'
```

The number 3 is Nt/Nb. The theta bracket is the same theta''+theta'/q bracket as in Eq. 16, here without the A^3 weight.

## Eq. 18 with numbers

```text
0 = chi'' + chi'/q
  + (0.2/q) * f * phi
  + (5/22) * (chi*phi'' + chi'*phi')
```

Lb=1, so the chi bracket keeps no denominator, and Pe/Pr = 2/8.8 = 5/22. The flow term stays f*phi, as printed and confirmed; [equations.md](equations.md) says not to replace it, and this sheet does not.

## Eq. 19 with numbers

At the wall, eta=0:

```text
B1: 1.5*f'(0) = 1 + 0.1*f''(0)
B2: f(0) = phi'(0)/44
B3: g(0) = -0.3*f''(0)
B4: -theta'(0) = 0.998 - theta(0) + 1.25*[1 - 0.8*theta(0)]^4
B5: phi'(0) = -3*theta'(0)
B6: chi(0) = 1
```

Derivations worth recording:

- B1: f'(0) - 1 - 0.1*(f''(0) - 5*f'(0)) = 0, where 5 is 1/kappa. Collecting the f'(0) terms gives 1.5*f'(0) = 1 + 0.1*f''(0).
- B2: fw/(Pr*Le) = 0.2/8.8 = 1/44.
- B4: with Bi=Rd=1 and theta_w-1=-0.8, the radiation part is 1.25*(0.0016 - A_0^4) = 0.002 - 1.25*A_0^4, where A_0 = 1 - 0.8*theta(0). Moving terms gives the printed form; 0.998 = 1 - 0.002.
- B5: Nt/Nb = 3.

Combining B2 with B5 gives f(0) = -3*theta'(0)/44. This is a consequence of the printed conditions, not an extra one.

The stable wall-radiation form in [equations.md](equations.md) becomes:

```text
Aw = 1 - 0.8*theta(0)
Qw = (1 - theta(0)) * (0.2 + Aw) * (0.04 + Aw^2)
B4 = -theta'(0) - (1 - theta(0)) - Qw
```

This is algebraically equal to the quartic form above because Rd=1. Test both forms before selecting the stable one for software.

Far away, at the numerical endpoint eta=L used for infinity:

```text
f'(L) = 0,  f''(L) = 0,  g(L) = 0,  theta(L) = 0,  phi(L) = 0,  chi(L) = 0
```

L itself is not set. The received L=0.5 is lambda_m, not this endpoint. Endpoint trials in [training-proposal.json](../../configs/training-proposal.json) remain proposals.

## What this sheet does not do

- It does not replace [equations.md](equations.md), change [the locked case](../../configs/physical-case-pending.json) or touch [training-proposal.json](../../configs/training-proposal.json).
- It does not solve anything. f, g, theta, phi, chi remain unknown. No software test, reference calculation, training run or evaluation exists.
- It does not settle the open definitions (epsilon, Lb, B, fw, H, Prm) listed in [questions-for-supervisor.md](questions-for-supervisor.md). If a definition answer changes a locked value, this sheet must be recomputed from the new record.

## How this sheet was checked

Each line was re-derived by hand from [equations.md](equations.md) and the locked values in [parameters.md](parameters.md), then compared term by term with the original expression during the 2026-11-16 session. An earlier chat draft printed 0.9984 for the B4 constant; the correct value is 0.998, and the chat was corrected. The repository check tools validate records, links and the writing rule. They do not verify this arithmetic or any physical claim.
