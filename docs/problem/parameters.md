# Parameter record

## Supervisor confirmation: 2026-11-16

Source: the user's message received on 2026-11-16, reporting values directly from the supervisor. This message is the source of the active numbers, not the old scanned sheet or papers.

All 24 received values are locked. After the mappings below, all 22 physical coefficients needed by Eqs. 14 to 19 have numerical values. H and Prm are confirmed received values kept outside that coefficient set. Numerical completeness is not software validation or proof of convergence.

### Exact received values

```text
%PARAMETERS
b=0.2; bm = 0.2; B = 0.5; dt=0; Dt=0.5;
H=1; k=0.2; L =0.5; M=5; tw = 0.2; ep = 0.1;

Pe=2;Pr=8.8;Prm = 0.1;
Nb=0.1; Nr = 0.5; Nt=0.3;
Lb=1; Le=1;

% BD parameters
Bi=1; l = 0.3; fw = 0.2; Rd = 1; N1 = 0.1;

bm := beta_m, dt := delta*, Dt = Delta,  L := lambda_m; tw = theta_w, M:= 1/Da
```

### Locked physical coefficients and mappings

The receipt explicitly states `bm=beta_m`, `dt=delta_star` (`delta*`), `Dt=Delta`, `L=lambda_m`, `tw=theta_w` and `M=1/Da`. The user also approved the notation aliases `k=kappa`, `l=ell` and `ep=epsilon`. These three aliases are not mappings explicitly stated in the supervisor receipt.

Every received value in this table comes from the message dated above. **Da=0.2 is derived as 1/M=1/5, not directly supplied.**

| Physical field | Locked value | Received value | Mapping or basis |
|---|---:|---|---|
| Delta | 0.5 | Dt=0.5 | Explicit mapping: Dt=Delta |
| B | 0.5 | B=0.5 | Direct, same name |
| Da | 0.2 | M=5 | Derived from the explicit relation M=1/Da |
| kappa | 0.2 | k=0.2 | User-approved notation alias: k=kappa |
| delta_star | 0 | dt=0 | Explicit mapping: dt=delta_star, also written delta* |
| theta_w | 0.2 | tw=0.2 | Explicit mapping: tw=theta_w |
| lambda_m | 0.5 | L=0.5 | Explicit mapping: L=lambda_m, not the far endpoint |
| beta_m | 0.2 | bm=0.2 | Explicit mapping: bm=beta_m |
| epsilon | 0.1 | ep=0.1 | User-approved notation alias: ep=epsilon |
| b | 0.2 | b=0.2 | Direct, same name |
| Pr | 8.8 | Pr=8.8 | Direct; supersedes the historical sheet reading 6.8 |
| Le | 1 | Le=1 | Direct, same name |
| N1 | 0.1 | N1=0.1 | Direct, same name |
| Bi | 1 | boundary group Bi=1 | Direct boundary coefficient in Eq. 19 |
| Rd | 1 | Rd=1 | Direct, same name |
| Nr | 0.5 | Nr=0.5 | Direct, same name |
| Nt | 0.3 | Nt=0.3 | Direct, same name |
| Nb | 0.1 | Nb=0.1 | Direct, same name |
| Lb | 1 | Lb=1 | Direct, same name |
| Pe | 2 | Pe=2 | Direct, same name |
| ell | 0.3 | l=0.3 | User-approved notation alias: l=ell |
| fw | 0.2 | fw=0.2 | Direct value; meaning and sign convention remain open |

### Confirmed values outside the physical coefficient set

| Received symbol | Locked received value | Treatment |
|---|---:|---|
| H | 1 | No active physical coefficient in Eqs. 14 to 19; interpretation remains open |
| Prm | 0.1 | No active physical coefficient in Eqs. 14 to 19; do not assume a magnetic Prandtl mapping |

Both values come from the same 2026-11-16 receipt. Keep them separately; do not invent equation terms for them. Asking what they mean does not make the 22 physical coefficients numerically incomplete.

In [the reference calculation sheet](equations.md#highest-derivative-system-for-the-reference), `M` is a local highest-derivative matrix and `H` is a local derivative vector. Those meanings are separate from the received scalars M=5 and H=1. The received L=0.5 means lambda_m=0.5, not the numerical far endpoint L used in the strategy.

## Lock policy

No automatic changes or tuning of the locked target are allowed. Any change needs an explicit user or supervisor update, with its date and source recorded. Do not replace a supplied number to enforce a conditional definition. The earlier proposals and Pr=6.8 are superseded, not fallback defaults.

Eqs. 14 to 19 stay unchanged. Numerical and training settings are our choices, kept apart from these values: [configs/reference.json](../../configs/reference.json) for the reference and [configs/pinn.json](../../configs/pinn.json) for the PINN, each with dated notes. They are not physical coefficients.

## Coefficient sizes for the locked target

The following arithmetic uses the locked values above, not a computed solution. With `q=eta+kappa`, `h=eta+b`, kappa=b=Da=0.2 and beta_m=0.2, the wall eta=0 gives:

- 1/q=5.
- 1/q^3=125.
- 1/Da=5.
- 2*beta_m/h^4=250.
- The magnetic bracket in Eq. 14 contains (1/kappa-4/b)=-15 at the wall.

These are coefficient sizes, not predicted solution values. They explain why an allowed number can still make the numerical problem difficult. They do not prove that the case has no solution.

With theta_w=0.2, `A=1+(theta_w-1)*theta` becomes `1-0.8*theta`. The effective coefficient of theta'' in Eq. 16 is `(1+0.5*A^3)/8.8`, using the received Nr=0.5 and Pr=8.8. It must stay away from zero along the solution; the check is below.

## One case only

**Target:** the locked 22-field case above. Every reference solve and every PINN run uses it.

**Retired debug case (2026-11-23):** the case file still holds an `auxiliary_debug_case` (kappa=1, b=1, Da=1, theta_w=1.5), proposed on 2026-11-12 as a numerical stepping stone. It is not used. AGENTS.md section 3 says we never solve or train through other parameter values; only the way the fixed problem is imposed and solved may change. The block stays in the file because the file is locked and its hash is part of every reference folder. The code reads only the `physical_parameters` block, refuses the file if its hash changes, and refuses any value set other than the locked one (`config.load_case`, `config.check_locked`).

Effective coefficient check: along the computed branch A, theta stays in [0.068, 0.250], so `(1+0.5*A^3)/8.8` stays at or above 0.1427, well away from zero ([reference audit](../reviews/REFERENCE_AUDIT_2026-11-23.md), check 2).

## Open definitions, not missing numbers

These conditional checks use the recorded definitions and the locked values. They are not supervisor-confirmed derived values and do not authorize changes to the target.

- If DeltaT=Tf-T_infinity and T_infinity=Tc imply `epsilon=-1/(theta_w-1)`, then theta_w=0.2 gives epsilon=1.25, not the supplied 0.1. The temperature definitions remain open.
- If Lb uses DB and must satisfy `Lb=Pr*Le`, then Pr=8.8 and Le=1 give Lb=8.8, not the supplied 1. The DB versus Dn definition remains open.
- If the recorded page 7 definitions apply together, delta_star=0 gives B=1, not the supplied 0.5. Ask whether those relations govern this case; do not change B automatically.
- If `fw=Cw/(1-Cw)`, the supplied fw=0.2 gives Cw=1/6. The old negative-concentration example used the historical proposal fw=-0.5 and no longer applies. The meaning and sign convention of fw still need an answer.
- The meanings of H=1 and Prm=0.1 remain open. Neither is an active physical coefficient in the confirmed equations.
- The required wall quantities and acceptable accuracy have not been specified.

Keep the questions in [questions-for-supervisor.md](questions-for-supervisor.md). On 2026-11-19 the supervisor answered that the physical meaning of the values is not needed: the supplied values are used exactly as given, so the definition questions are closed and change nothing in this record.

## Historical sheet readings and choices: 2026-11-12

Historical source: [supervisor's scanned page](../../sources/parameter-sheet/parameter-values.pdf). Recorded on 2026-11-12. The following readings and choices are preserved for history only. They are not active defaults. The 2026-11-16 receipt supersedes them, including the old Pr=6.8 reading.

Most sheet entries were conditions, not fixed numerical values. The requested first case then used small rounded proposals near allowed boundaries, without making a denominator zero. The user chose to wait for the supervisor rather than fill the remaining values from papers. The word "pending" in this historical table describes that earlier state only.

| Symbol in equations | Historical reading of the sheet | Historical proposed value | Historical basis and limits |
|---|---|---:|---|
| Delta | Nonnegative or positive; inequality handwriting slightly unclear | 0.1 | Positive choice satisfied either reading and kept microrotation coupling active |
| B | Nonnegative or positive | 0.1 | Positive choice satisfied either reading |
| Da | Positive | 0.1 | Zero was excluded by the printed 1/Da expression |
| kappa | Handwritten K, positive | 0.1 | K was mapped to the curvature symbol; eta+kappa stayed nonzero |
| delta_star | 0, 1, 2, ... | 0 | First listed value; 1 and 2 were proposed as later cases, not approved target changes |
| theta_w | Positive | 0.1 | Requested small rounded proposal, not a validated temperature regime |
| lambda_m | Symbol readable; bound unclear | pending | An incomplete inequality was not turned into a number |
| beta_m | Nonnegative or positive | 0.1 | Kept beta_m distinct from the unclear separate M sheet line |
| epsilon | Appeared to read epsilon=1 followed by >=0 | pending | Fixed 1 versus a range was unclear; the follow-up question was unanswered then |
| b | Constant, nonnegative or positive | 0.1 | Inverse powers of eta+b excluded b=0 for the regular baseline |
| Pr | Fixed 6.8 | 6.8 | Recorded then as fixed, not rounded; now superseded by the receipt's Pr=8.8 |
| Le | 1, 2, 3, 4 | 1 | First listed value; 2, 3 and 4 were proposed as separate later cases |
| N1 | Nonnegative or positive | 0.1 | Positive choice kept slip active |
| Bi | Nonnegative or positive | 0.1 | Positive choice kept the wall heating term active |
| Rd | Appeared twice; at least one line was positive | 0.1 | Positive choice satisfied the visible conditions |
| ell | 0 <= ell <= 1 | 0.1 | Near the lower boundary, but not zero |
| fw | -1 <= fw <= 1 | -0.5 | Rounded nonzero proposal towards the lower endpoint; no suction/blowing sign was inferred |
| Nr | No usable uncrossed value | pending | Awaited supervisor at that time |
| Nt | Not given | pending | Awaited supervisor at that time |
| Nb | No usable uncrossed value | pending | Awaited supervisor at that time; the equations divide by Nb |
| Lb | Not given | pending | Awaited supervisor at that time |
| Pe | Not given | pending | Awaited supervisor at that time |

### Historical clarifications recorded on 2026-11-12

1. The user identified the two P-like handwritten lines as magnetic Prandtl number Pm, outside the specified equations. This older clarification does not assign that meaning to the newly received Prm.
2. The red/crossed-out M sheet line was unresolved then. The new receipt explicitly supplies M=1/Da. Neither the old line nor the new scalar replaces beta_m.
3. Missing physical values were to wait for the supervisor, not use literature defaults. The 2026-11-16 receipt now supplies them.
4. Work lives in this new local repository, not the Lorenz repository.
5. The marked Eqs. 14 to 19 are the confirmed working equations. Differences from related papers do not permit changes to them.

The crossed-out line near the radiation entries was not a reliable numerical instruction. Keep its appearance in the source PDF as history; use the new receipt for Nr and Nb.

## Machine-readable record

[physical-case-pending.json](../../configs/physical-case-pending.json) now holds the numerically complete, locked physical case with status `parameters_complete_validation_pending`. Its filename is retained only for link compatibility; it no longer means that physical numbers are missing. The record keeps the receipt, mappings, derived Da, excluded H/Prm and the retired debug block separate. No physical coefficient is null. Its status string `parameters_complete_validation_pending` is part of the locked file and its hash, so it is left as it is; the state of the work is in [strategy.md](../plans/strategy.md) section 2.

The record checker compares the equation coefficients with the dated receipt and its mappings. A SHA-256 checksum guards the received values against unrecorded changes. Any authorized future update must update the dated record and checksum together; it cannot silently replace the locked target.

Where the work stands is in [strategy.md](../plans/strategy.md) section 2 and the [status log](../status-log.md). The record checker checks record consistency; it is not proof of solver correctness or convergence.
