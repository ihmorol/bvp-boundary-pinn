# Fixed-case BVP methodology

**Status: parameters complete; validation pending.** The 2026-11-16 receipt locks all 22 physical coefficients. No solver, software tests, physical reference, training or evaluation exists yet. This is a proposed procedure, not a result; required numerical choices remain pending.

Follow the stages in order. Tasks are small diagram boxes; mathematics uses separate strips. The builder supplies canonical R14 to R18, B1 to B12 and the matrix from [equations.md](equations.md).

## 1. Freeze the specification and case

### 1.1 Read the dated receipt

Read the 2026-11-16 receipt in [parameters.md](parameters.md). Match its 24 received values to the active record: 22 complete physical coefficients and two separate symbols.

### 1.2 Check the notation mappings

Apply receipt mappings and user-approved aliases exactly. Derive Da only from received scalar M. Distinguish received L, M and H from endpoint, matrix and derivative-vector notation.

### 1.3 Lock the active case

Lock [the current case](../configs/physical-case-pending.json), including known delta_star=0 and Pr=8.8. Preserve Eqs. 14 to 19. Physical changes need dated authorization; debug overrides never replace this target.

### 1.4 Check the physical denominators

Check finite coefficients, recorded ranges and denominators on proposed intervals. Check the thermal coefficient on computed curves later. Diagnose invalid expressions without epsilon patches or changed numbers.

### 1.5 Log the open definitions

Carry epsilon, Lb, B and fw questions from [the supervisor questions](questions-for-supervisor.md). Keep H and Prm interpretations separate. Open definitions do not undo numerical completeness.

#### Mathematics

All values below come from the dated receipt and its recorded mappings, except the explicitly derived Da:

$$
\begin{aligned}
&\Delta=0.5,\ B=0.5,\ \kappa=0.2,\ \delta_\star=0,\ \theta_w=0.2,\ \lambda_m=0.5,\\
&\beta_m=0.2,\ \epsilon=0.1,\ b=0.2,\ Pr=8.8,\ Le=1,\ N_1=0.1,\\
&Bi=1,\ Rd=1,\ Nr=0.5,\ Nt=0.3,\ Nb=0.1,\ Lb=1,\ Pe=2,\ \ell=0.3,\ f_w=0.2,\\
&Da=1/M_{\rm received}=1/5=0.2.
\end{aligned}
$$

$$
\begin{aligned}
&bm\mapsto\beta_m,\quad dt\mapsto\delta_\star,\quad Dt\mapsto\Delta,\quad
L_{\rm received}\mapsto\lambda_m,\quad tw\mapsto\theta_w,\\
&k\mapsto\kappa,\quad l\mapsto\ell,\quad ep\mapsto\epsilon.
\end{aligned}
$$

The last three aliases have user approval. Received H=1 and Prm=0.1 enter no confirmed equation. Scalar M=5 is not local matrix M; received L=0.5 is not endpoint L.

$$
q=\eta+\kappa>0,\quad h=\eta+b>0,\quad
Da,Pr,Le,Nb,Lb\ne0,\quad \theta_w-1\ne0.
$$

For the locked positive kappa and b, q and h stay positive on a nonnegative interval. Numerical margins still need declared checks. Conditional definition discrepancies stay in the question record, not in replacement formulas here.

#### Gate and output

The numerical-completeness gate is satisfied by the locked receipt, not by this document's arithmetic alone. Output the case identifier, receipt and equation hashes, mappings and denominator checks. Keep unresolved definitions visible. Software correctness, reference regularity and physical interpretation have not been established.

## 2. Freeze the numerical contract

### 2.1 Declare the domain schedule

Declare ordered endpoints, wall extent, reference meshes, guesses, continuation and uncertainty estimation. [Config proposals](../configs/training-proposal.json) are L=10,15,20, 200 initial nodes and tolerances 10^-5,10^-7. These are uncertified trials.

### 2.2 Fix the network and scales

Record four proposed width-64 tanh layers, five linear outputs and float64. Choose fixed positive equation/BC scales and positive weights. Scales represent characteristic magnitudes, never current residuals.

### 2.3 Specify both optimizers

Declare Adam's schedule and transition rule, plus L-BFGS history, search, budgets and tolerances. Adam betas, optimizer epsilon and geometric schedule below remain methodology proposals, not supervisor/config values.

### 2.4 Declare decisions and thresholds

Predeclare ranking, monitor cadence, final counts, stopping tests and acceptance thresholds. Include fallback scales, branch matching and sensitivity rules. Required unset choices remain pending before physical runs.

### 2.5 Assign seeds and runtime settings

Assign independent initialization and coordinate streams with proposed seeds 0,1,2. Record generators, libraries, device, threads, deterministic policy, revision and local changes. Do not promise cross-hardware bitwise identity.

#### Mathematics

Required scales and weights satisfy

$$
s_i>0,\quad t_j>0,\quad \lambda_i>0,\quad \mu_j>0,
\qquad i=1,\ldots,5,\quad j=1,\ldots,12.
$$

The config proposes weights of one. Its eight null fields remain null: `equation_scales`, `boundary_scales`, `max_iterations`, `max_function_evaluations`, `tolerances`, `equation_residual_thresholds`, `wall_quantity_thresholds`, and `domain_sensitivity_thresholds`. Wall extent, grids, cadence, branch rules, fallbacks and other thresholds also remain pending.

#### Gate and output

Output a frozen run manifest with source notes for every adopted proposal, exact budgets and threshold definitions. Any required unset choice blocks a physical run, not the already satisfied coefficient-completeness gate. Do not substitute library defaults or invented numbers. Later changes require a separate manifest.

## 3. Pass the software tests

### 3.1 Test the schema and transcription

Test missing, nonfinite and invalid coefficient rejection with no hidden defaults. Compare transcriptions against enlarged source pages, especially R14 brackets and R18 terms. Require exactly twelve BCs.

### 3.2 Test derivatives and shapes

Test polynomial/exponential derivatives through fourth order, normalization and independent rows. Keep N-by-1 components before stacking. Handle mathematical zeros deliberately; disconnected graphs are not automatically valid zeros.

### 3.3 Compare independent residual calculations

Compare independent NumPy/Torch residual transcriptions at matching finite synthetic states. Test the matrix identity with eta and coefficients fixed. This checks algebra, not physical validity.

### 3.4 Test boundaries and weight gradients

Test every BC, radiation equivalence and nonzero slopes that cancel. Check finite component gradients through highest derivatives with nondegenerate fixtures. Not every weight gradient must be nonzero.

### 3.5 Test manufactured runs and restoration

Use synthetic forcing and targets for manufactured solutions. Test repeatable fresh closures, checkpoint/RNG restoration and short finite updates. Synthetic results check software, never the physical target.

#### Mathematics

$$
R(\eta,z,H)-R(\eta,z,0)=M(z)H.
$$

Use the canonical matrix. Repeated closures at fixed weights/coordinates must reproduce objective and gradients within declared test tolerances using fresh graphs. Never detach loss terms.

#### Gate and output

All required tests must pass predeclared tolerances before reference computation. Output fixtures, expected values, test versions and failure records. Failed tests stop the physical pipeline. A record-integrity check is not a solver test, and a manufactured solution is not evidence that the target BVP is solved.

## 4. Build and validate the reference

### 4.1 Construct the first-order system

Construct twelve-state z and local derivative vector H. Solve the canonical system without inversion at each state. Check denominator margins, scaled conditioning and the thermal coefficient.

### 4.2 Prepare the ordered guesses

Build smooth guesses with consistent derivative states on frozen meshes. Follow declared guess/continuation order. Keep debug continuation separate, with normalization questions explicit when theta_w crosses one.

### 4.3 Solve each scheduled candidate

Run solve_bvp for declared guesses and endpoints. Save status, mesh and branch labels. Success flags alone do not pass validation; retain failed guesses and distinct branches.

### 4.4 Refine and check independently

Refine mesh and tolerance separately, then extend the domain by the declared rule. Independently check original residuals, signed BCs, profiles, wall derivatives and derivative-state compatibility.

### 4.5 Estimate branch-specific uncertainty

Match branches using predeclared diagnostics. Select the first ordered domain passing all checks. Save uncertainties and unresolved branches. Failure is not proof that no solution exists.

#### Mathematics

$$
z=(f,f',f'',f''',g,g',\theta,\theta',\phi,\phi',\chi,\chi')^T,
\qquad H=(f^{(4)},g'',\theta'',\phi'',\chi'')^T.
$$

$$
r_0=R(\eta,z,0),\qquad M(z)H=-r_0.
$$

$$
z'=(z_2,z_3,z_4,H_1,z_6,H_2,z_8,H_3,z_{10},H_4,z_{12},H_5)^T.
$$

State indices start at one. H_1 here is a derivative component, not received scalar H=1.

$$
A=1+(\theta_w-1)\theta=1-0.8\theta,\qquad
a_T=\frac{1+Nr A^3}{Pr}=\frac{1+0.5A^3}{8.8}.
$$

The substituted coefficients come from the receipt. Check a_T along computed states; the PINN residual form does not remove a singular highest-derivative system.

Do not differentiate a piecewise-cubic f interpolant four times, or obtain derivatives from a right-hand side and reuse that same expression as an independent residual check. A ten-state core followed by linear chi is only an optional cross-check.

#### Gate and output

Require a regular reference, passed refinement checks, matched branches and uncertainty estimates below declared limits. Output all attempts and branch-specific records; estimates are not rigorous bounds. If no candidate passes, stop. Separately authorized exploration carries `UNVERIFIED_REFERENCE`. Reference curves never supply baseline training labels.

## 5. Construct the point sets

### 5.1 Generate fixed training coordinates

Generate stratified coordinates on full and wall intervals. The config proposes 1024 points, split equally. Save realized coordinates, retaining outer coverage. Baseline training does not resample.

### 5.2 Keep endpoint traces separate

Evaluate eta=0 and eta=L traces separately from interiors. Supply derivatives for all twelve BCs. Zero flux is one cancellation, not separate zero slopes or extra labels.

### 5.3 Freeze the monitor grid

Freeze an independent dense monitor, endpoint traces and quadrature weights before training. Check sorting, weights and coverage. This monitor supports ranking; reference-label errors never enter ranking.

### 5.4 Seal the final-test recipe

Freeze dense/random final recipes before fitting, but withhold coordinates from training and selection. Generate and seal arrays only after model lock, adding endpoint traces for quadrature.

### 5.5 Record streams and objective meaning

Record separate init, train.full, train.wall and final.random streams with generator versions/states. Equal empirical weights emphasize wall-biased sampling, not uniform quadrature. This fixed case has no parameter split.

#### Mathematics

For either sampling interval, use endpoint names distinct from physical b:

$$
\eta_j=\alpha+(\beta-\alpha)\frac{j+u_j}{n},\qquad
j=0,\ldots,n-1,\quad 0<u_j<1.
$$

Use [0,L] or [0,L_wall], with 0<L_wall<=L. Reject and redraw endpoint-valued random draws. For a sorted evaluation grid spanning the interval,

$$
0=\xi_0<\cdots<\xi_K=L,
$$

$$
\omega_0=\frac{\xi_1-\xi_0}{2},\qquad
\omega_n=\frac{\xi_{n+1}-\xi_{n-1}}{2}\ (1\le n<K),\qquad
\omega_K=\frac{\xi_K-\xi_{K-1}}{2}.
$$

$$
\|v\|_{2,\omega}=\left(\frac{\sum_n\omega_n v_n^2}{\sum_n\omega_n}\right)^{1/2},
\qquad \|v\|_\infty=\max_n|v_n|.
$$

Sort random grids and remove duplicate coordinates before quadrature. Maxima are sampled values, not continuous supremum bounds.

#### Gate and output

Require correct counts, finite coordinates, separate endpoint traces, declared coverage and positive quadrature weights. Output coordinate arrays, weights, stream states and checksums. Training, monitoring and withheld evaluation must have distinct roles. Frozen final recipes do not permit inspecting final coordinates to improve fitting or choose a checkpoint.

## 6. Train the fixed-case PINN

### 6.1 Initialize one physical-output network

Use the init stream for proposed Xavier-uniform weights, gain one and zero biases. Zero Adam moments. Train only NN weights/biases; physical coefficients remain locked.

### 6.2 Predict fields and eta derivatives

Clear gradients and create fresh coordinate leaves. Predict five fields through coordinate normalization. Compute f derivatives through fourth order and others through second order, retaining NN-weight gradient paths.

### 6.3 Assemble residuals, boundaries and loss

Assemble five canonical residuals and twelve endpoint BCs with fixed positive scales/weights. Include no reference labels, output clipping, dropout, batch normalization or second trainable physics network.

### 6.4 Apply one checked Adam update

Check finite residuals/loss, then backpropagate. Check gradients before the declared NN-only Adam update. Record unmodified gradient norms, learning rate and objective evaluations at each step.

### 6.5 Monitor eligible snapshots

Recheck updated weights and monitor at fixed cadence, including initial/stage endpoints. Save finite diagnostics and optimizer/RNG states. Apply declared transition/budget rules; flat loss alone is not convergence.

#### Mathematics

$$
x=2\eta/L-1,\qquad
u_{\mathbf w}(\eta)=\mathcal N_{\mathbf w}(x)=(f,g,\theta,\phi,\chi)^T,
$$

$$
\frac{d^ku}{d\eta^k}=\left(\frac2L\right)^k\frac{d^k\mathcal N}{dx^k}.
$$

Differentiate through eta normalization once, not once through eta and again with an extra chain factor.

$$
W_{ab}^{(l)}\sim U\!\left[-\sqrt{\frac6{n_{\rm in}+n_{\rm out}}},
\sqrt{\frac6{n_{\rm in}+n_{\rm out}}}\right],\qquad
\mathbf a^{(l)}=0,\quad m_0=v_0=0.
$$

Here a denotes network biases. This initialization is a methodology proposal specifying the config's Xavier choice.

$$
r_i=R_i/s_i,\qquad d_j=B_j/t_j,\qquad \rho_n=1/N,
$$

$$
J_{\rm train}(\mathbf w)=
\sum_{i=1}^{5}\lambda_i\sum_{n=1}^{N}\rho_n r_i(\eta_n)^2
+\sum_{j=1}^{12}\mu_j d_j^2.
$$

Index residuals in R14 to R18 order. Deterministic chunks divide by the total training count and include BCs once. Logged values may be detached; loss terms may not.

The config proposes the rate endpoints and budget. Betas, optimizer epsilon and schedule shape remain methodology proposals:

$$
\beta_1=0.9,\quad\beta_2=0.999,\quad\varepsilon_A=10^{-8},\quad T_A=20000,
$$

$$
\alpha_t=10^{-3}\left(\frac{10^{-4}}{10^{-3}}\right)^{(t-1)/(T_A-1)},
\qquad t=1,\ldots,T_A.
$$

$$
\begin{aligned}
G_t&=\nabla_{\mathbf w}J_{\rm train}(\mathbf w_{t-1}),\\
m_t&=\beta_1m_{t-1}+(1-\beta_1)G_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)(G_t\odot G_t),\\
\widehat m_t&=m_t/(1-\beta_1^t),\qquad
\widehat v_t=v_t/(1-\beta_2^t),\\
\mathbf w_t&=\mathbf w_{t-1}-\alpha_t\widehat m_t/(\sqrt{\widehat v_t}+\varepsilon_A).
\end{aligned}
$$

Operations are componentwise. Optimizer epsilon is not physical epsilon. This proposal uses no weight decay, AMSGrad or gradient clipping.

#### Gate and output

Nonfinite residuals, derivatives, losses, gradients or updated weights stop progress and identify the first failing term. Do not patch physics denominators or replace errors with zeros. Output component histories and eligible snapshots. Budget exhaustion is a stopping status, not convergence or evidence that final accuracy targets were met.

## 7. Refine and select the checkpoint

### 7.1 Freeze the refinement objective

Start fresh L-BFGS from best eligible Adam. Freeze coefficients, domain, coordinates, scales, weights and monitor. Objective changes require a new contract and reset optimizer history.

### 7.2 Build a fresh checked closure

Clear gradients, rebuild graphs/coordinates, recompute all residuals/BCs, check finiteness and backpropagate. Never resample, alter weights or clip closure gradients. Count every repeated objective evaluation.

### 7.3 Compute directions and line searches

Apply declared limited-memory directions and line search to the frozen objective. Count trial evaluations separately from accepted steps. Recompute accepted-state diagnostics; returned losses may describe earlier weights.

### 7.4 Apply stopping and fallback rules

Distinguish convergence tests, budget exhaustion and failure. Restore Adam on refinement failure or worse monitor rank. Keep Adam without rank improvement; retain all failed refinement records.

### 7.5 Rank complete eligible checkpoints

Rank finite snapshots with complete diagnostics and checked gradients using predeclared positive thresholds. Compare full-precision ranks without rounding or undeclared ties. Exclude reference errors and final tests.

#### Mathematics

The refinement direction proposal uses a limited-memory inverse-Hessian approximation K_k, distinct from the reference matrix M:

$$
p_k=-K_k\nabla J_{\rm train}(\mathbf w_k),\qquad
\mathbf w_{k+1}=\mathbf w_k+a_kp_k.
$$

For the config's proposed strong-Wolfe search,

$$
J(\mathbf w_k+a_kp_k)\le J(\mathbf w_k)+c_1a_k\nabla J(\mathbf w_k)^Tp_k,
$$

$$
|\nabla J(\mathbf w_k+a_kp_k)^Tp_k|
\le c_2|\nabla J(\mathbf w_k)^Tp_k|,\qquad 0<c_1<c_2<1.
$$

History 50 is a config proposal. Search constants, curvature policy, budgets and stopping tolerances need an explicit installed-library contract.

Define J_monitor using the same scaled loss with normalized monitor quadrature weights. Every tau below must be positive and predeclared:

$$
Q=\max\left\{
\max_{1\le i\le5}\frac{\|r_i\|_{2,\omega}}{\tau_{R_i}},
\max_{1\le i\le5}\frac{\|r_i\|_\infty}{\tau_{R_i,\max}},
\max_{1\le j\le12}\frac{|d_j|}{\tau_{B_j}}
\right\}.
$$

$$
\operatorname{rank}(c)=(Q_c,J_{{\rm monitor},c},E_c,o_c).
$$

Minimize lexicographically. E counts cumulative full-objective evaluations, including training, repeated closures and monitoring; o is increasing checkpoint ordinal. Exact score ties prefer fewer evaluations, then the earlier checkpoint. Line-search trials are not candidates; candidate cadence is predeclared.

#### Gate and output

Output the selected checkpoint, every eligible rank, stopping reason and Adam fallback decision. No eligible snapshot means failed training. Neither optimizer convergence nor Q<=1 establishes final accuracy. Keep the selected state restorable and record refinement failures even when the Adam fallback remains usable for locked evaluation.

## 8. Lock and test the selected model

### 8.1 Lock the model and open tests

Lock weights and all design decisions before opening withheld coordinates. Save final arrays; evaluate dense/random sets separately. Never rank checkpoints or silently redesign using final results.

### 8.2 Compare fields and velocity

Compare five fields and f-prime to matched-branch reference values. Report absolute errors, weighted RMS, sampled maxima and normalized errors with predeclared near-zero fallbacks and recorded denominator choices.

### 8.3 Measure original residuals and boundaries

Report every R14 to R18 using original/scaled RMS and sampled maximum. Report every signed original/scaled B1 to B12. Check endpoint derivatives separately from interior averages.

### 8.4 Check wall outputs and uncertainty

Record raw wall derivatives and branch-specific uncertainty. Eq. 23 remains a separate pending checked task: confirm requested quantities and enlarged source formulas before defining outputs or thresholds.

### 8.5 Apply every acceptance requirement

Apply the final-test requirements to each set and required output. Record a provisional pass with sensitivity pending, or an explicit failure. Final acceptance is decided after phase 9.

#### Mathematics

For each field or derivative u, use p=2 for weighted RMS and p=infinity for sampled maximum. Positive d and S are required contract choices:

$$
D_{u,p}=\begin{cases}
\|u_*\|_p,&\|u_*\|_p\ge d_{u,p},\\
S_{u,p},&\|u_*\|_p<d_{u,p},
\end{cases}
\qquad
E_{u,p}=\frac{\|u-u_*\|_p}{D_{u,p}}.
$$

Report absolute errors too. Apply the same fallback principle to scalar wall quantities without tiny epsilon additions. Raw wall reporting includes

$$
\mathcal W_0=(f'(0),f''(0),f'''(0),g'(0),\theta'(0),\phi'(0),\chi'(0)).
$$

These are derivatives, not unchecked Eq. 23 transfer formulas. Interpret discrepancies against estimated reference uncertainty; unresolved uncertainty or branch identity prevents verified accuracy.

Phase 8 checks the final sets and outputs; phase 9 supplies sensitivity evidence. Evaluate the complete conjunction only after both. Let F mean all required values are finite, V mean a reliable same-branch reference, and M_metrics contain all declared residual, BC, field, velocity, wall, uncertainty and sensitivity checks:

$$
\mathrm{ACCEPT}=F\land V\land
\bigwedge_{m\in\mathcal M_{\rm metrics}}(m\le\tau_m).
$$

#### Gate and output

Require the final-test thresholds on every declared set before advancing to sensitivity checks. Output full diagnostics and `FINAL_TEST_PASS_SENSITIVITY_PENDING`, or retain the exact failure. Without a matched reference, retain `UNVERIFIED_REFERENCE`. Pending Eq. 23 targets block their claim. Later redesign requires a separate run and fresh withheld coordinates.

## 9. Check sensitivity and method changes

### 9.1 Check domains and reference meshes

Vary L, reference mesh and tolerance separately. Compare matched branches on the common interval and check wall quantities. Save refinement differences and uncertainties, not just solver status.

### 9.2 Check training point density

Vary declared counts and wall coverage with other factors fixed. Compare residuals, fields and wall derivatives; charge generation/scoring costs. Point-density changes never retune physical coefficients.

### 9.3 Check all declared seeds

Run every declared seed and retain failures. Report median, spread and acceptance counts by branch. Distinct branches are separate outcomes, never averaged into one claimed solution.

### 9.4 Compare one optional change

After baseline checks, compare one-factor arms: Adam only, full BFGS, augmented Lagrangian, resampling, enrichment, bounded weights, first-order outputs or exact endpoints. Each needs a separate declared contract.

### 9.5 Freeze costs and new-test policy

Declare budgets, evaluation accounting and comparison grids before studies. Redesign after test exposure needs recorded exposure and fresh withheld coordinates. Include failures and sampling/weight-scoring costs.

#### Mathematics

For two endpoint choices, compare only

$$
I=[0,\min(L_a,L_b)],\qquad
\Delta_{u,p}^{a,b}=\frac{\|u_a-u_b\|_{p,I}}{D_{u,p,I}}
\le\tau^{\rm sensitivity}_{u,p}.
$$

Choose the denominator's comparison curve, near-zero threshold and fallback scale before comparison. Keep scales fixed and include wall differences. Predeclared meshes supply the quadrature on I.

Optional first-order helper outputs need compatibility tests and final original-equation checks using physical-head derivatives. Adaptive or augmented objectives freeze before L-BFGS; any change resets history. Full BFGS and augmented-Lagrangian updates remain separate method proposals. [Literature context](literature-synthesis.md) is not evidence that they work here.

#### Gate and output

Sensitivity passes when every required difference meets its fixed threshold with matched branches. Combine this result with phase 8 to decide final ACCEPT. Output settings, common intervals, costs and seed outcomes. Unresolved branches block like-for-like claims. Optional arms cannot rescue a failed baseline by hidden retuning; each has its own contract.

## 10. Report and archive every outcome

### 10.1 Plot each field and error

Plot separate f,g,theta,phi,chi and f-prime panels: PINN/reference curves, absolute errors, wall detail and reference uncertainty. Captions identify the recorded case and tested interval, without wider claims.

### 10.2 Tabulate every mathematical check

Tabulate original/scaled RMS/max per R, signed errors per BC and raw wall derivatives. Include normalized errors, uncertainty, thresholds and decisions, plus L, mesh and point-density sensitivity tables.

### 10.3 Report seed statistics and failures

Report attempted, finite, accepted and failed seeds by branch. Give medians and declared spread, stating statistic membership. Never zero-fill failures or report only the best seed.

### 10.4 Report time and peak memory

Separate reference, Adam, refinement and evaluation time, including failures. Report objective/closure counts, accepted updates, peak memory, device and precision. Compare costs under declared budgets, not optimizer names.

### 10.5 Archive the complete run record

Archive receipt/code/config hashes, manifests, coordinates, weights, RNG/optimizer states, references and diagnostics. Preserve pending choices, failures, stop reasons, test exposure and decisions with all later reruns.

#### Gate and output

Every reported number needs a source or saved computation. Output accepted-under-declared-tests, failed, blocked, budget-exhausted or unverified-reference status with limits and pending outputs. This methodology claims no solved target, uniqueness, temporal stability or cross-hardware bitwise identity. Rendering this diagram is not a physical run.

## Sources and build notes

- [Parameters](parameters.md): locked receipt, aliases, derived Da and open definitions.
- [Strategy](strategy.md): ordered gates, baseline and numerical proposals.
- [Equations](equations.md): unchanged canonical residuals, boundary errors and matrix.
- [Supervisor questions](questions-for-supervisor.md): definitions and requested output requirements.
- [Literature synthesis](literature-synthesis.md): optional comparisons; author-reported results remain author-reported.

The diagram follows the supplied design theme: pale task boxes, numbered circles, green dashed groups and tabs, yellow decision diamonds, dark arrows and serif text. Fifty tasks stay short; full formulas have separate strips. The canvas grows with the content. SVG preserves vector shapes; HTML and this Markdown are the editable text sources. Document checks do not test a solver.

Build the HTML with `uv run --no-project --with latex2mathml --with markdown python -I tools/build_methodology.py`. Render it with the PDF skill's single-page exporter at width 4200px; its CommonJS source needs a `.cjs` copy where the package uses ES modules. Run `uv run --no-project --with pymupdf --with pypdf python -I tools/check_methodology.py --source-pdf <rendered-pdf>` to check canonical content, add PDF metadata, export SVG and render section previews. The SVG has outlined text for portable mathematics. No physical parameters are changed by these tools.
