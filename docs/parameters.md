# Parameter record

Source: [supervisor's scanned page](../sources/parameter-sheet/parameter-values.pdf). Recorded on 2026-11-12.

**These are mostly conditions, not fixed numerical values.** The source condition and our chosen value are separate. The requested first case uses small rounded values close to allowed boundaries, without setting a denominator to zero. Our choices are proposals, not numbers directly supplied by the supervisor.

The user chose to wait for the supervisor for missing values. We have not filled missing values from papers. The first physical case is therefore **not runnable yet**.

## Source condition and first-case choice

| Symbol in equations | Reading of the sheet | Proposed first value | Basis and limits |
|---|---|---:|---|
| Delta | Nonnegative or positive; inequality handwriting slightly unclear | 0.1 | Positive choice satisfies either reading and keeps microrotation coupling active |
| B | Nonnegative or positive | 0.1 | Positive choice satisfies either reading |
| Da | Positive | 0.1 | Never use zero in the printed 1/Da expression |
| kappa | Handwritten K, positive | 0.1 | Mapped to the curvature symbol in Eqs. 14–19; denominator eta+kappa stays nonzero |
| delta_star | 0, 1, 2, ... | 0 | Preserve the listed discrete values; use 1 and 2 as later cases |
| theta_w | Positive | 0.1 | Requested small rounded choice; do not assume this is a physically validated temperature regime |
| lambda_m | Symbol readable; bound unclear | pending | Do not turn an incomplete inequality into a number |
| beta_m | Nonnegative or positive | 0.1 | Keep the named beta_m distinct from the unclear separate M line |
| epsilon | Appears to read epsilon=1 followed by >=0 | pending | Fixed 1 versus a range is unclear; follow-up question was not answered |
| b | Constant, nonnegative or positive | 0.1 | Inverse powers of eta+b prohibit b=0 for the regular baseline |
| Pr | Fixed 6.8 | 6.8 | **Do not round a supplied fixed value to 7 or 6.5** |
| Le | 1, 2, 3, 4 | 1 | First listed value; later test 2, 3 and 4 separately |
| N1 | Nonnegative or positive | 0.1 | Positive value, rather than switching slip off at zero |
| Bi | Nonnegative or positive | 0.1 | Positive value, so wall heating term remains present |
| Rd | Appears twice; at least one line is positive | 0.1 | Positive value satisfies the visible conditions |
| ell | 0 <= ell <= 1 | 0.1 | Near lower boundary, but not zero |
| fw | -1 <= fw <= 1 | -0.5 | Rounded nonzero value towards lower endpoint; do not infer suction/blowing sign without the stated convention |
| Nr | No usable uncrossed value | pending | Await supervisor |
| Nt | Not given | pending | Await supervisor |
| Nb | No usable uncrossed value | pending | Await supervisor; current equations divide by Nb |
| Lb | Not given | pending | Await supervisor |
| Pe | Not given | pending | Await supervisor |

## Clarifications the user already gave

1. The two P-like handwritten lines refer to **magnetic Prandtl number Pm**. Pm is not in the specified equations. It is not Pe, Lb or the supplied Pr=6.8.
2. The red/crossed-out M line is **unresolved**. Do not replace beta_m with M or add a new force term.
3. Missing physical values must **wait for the supervisor**. Literature numbers are not defaults.
4. Work lives in this new local repository, not the Lorenz repository.
5. The marked Eqs. 14–19 are the confirmed working equations. Differences from related papers are not permission to alter them.

A crossed-out line near the radiation entries is not a reliable active instruction. Keep its appearance in the source PDF rather than pretending it supplies Nr or Nb.

## Numerical warning for the requested near-boundary choice

At eta=0, with kappa=b=Da=0.1 and beta_m=0.1:

- 1/(eta+kappa)=10.
- 1/(eta+kappa)^3=1000.
- 1/Da=10.
- 2*beta_m/(eta+b)^4=2000.
- The magnetic bracket in Eq. 14 contains (1/kappa-4/b)=-30 at the wall.

These are coefficient sizes, not predicted solution values. They explain why a rounded allowed number can still make the numerical problem difficult. They do not prove the case has no solution.

At theta_w=0.1, the quantity `A=1+(theta_w-1)*theta` becomes `1-0.9*theta`. The effective coefficient of theta'' in Eq. 16 is `(1+Nr*A^3)/Pr`. Nr is missing, and theta is unknown. The strategy must check this coefficient along the computed solution; it must not assume regularity before solving.

## Two cases, kept separate

**Target:** the requested small rounded choices above. This is the research case we will report if it passes verification.

**Optional debug starting case:** kappa=1, b=1, Da=1, theta_w=1.5, with the same remaining supplied coefficients. Its inverse-power terms are less extreme. It can help test the implementation or support parameter continuation towards the target. It is also blocked until the missing values arrive and is not presented as the supervisor's fixed baseline.

Do not choose theta_w=1 merely because it is a rounded number: the printed boundary condition divides by theta_w-1. A stable algebraic limit can be studied later, but it does not automatically define the full physical normalization.

## Machine-readable record

[physical-case-pending.json](../configs/physical-case-pending.json) contains every physical field, chosen values, null missing values, exclusions and debug overrides. Null means unknown, never zero.

A complete parameter table needs 22 coefficients for Eqs. 14–19. Seven remain pending: lambda_m, epsilon, Nr, Nt, Nb, Lb and Pe. A future validator must reject this pending file as a simulation input until all seven are supplied and the denominator/regularity checks pass.
