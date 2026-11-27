# Journal strategy: from the ICEEICT Lorenz paper to one journal paper

Date: 2026-11-19. Status: advisory report, no experiment run for it. Owner: the FYDP team. For discussion with the supervisor.

Tags used throughout:

- **[V]** checked in the linked source. Some sources were read through search-result excerpts because the publisher site blocked direct reading. Those are marked "excerpt" and must be re-checked on the live page before you rely on them.
- **[J]** my judgment as a reviewer and author in this field. Not a sourced fact.
- **[A]** an assumption, because the material was not available.

Materials read: the submitted conference paper (`writings/iceeict2027-final/main.tex`, `main.pdf`, `ablation-table.tex`, `review/` in ihmorol/lorenz1960-pinn), the supervisor's draft (`sources/project-draft/project3Nov_2026.docx`), and in this repo [equations.md](../problem/equations.md), [parameters.md](../problem/parameters.md), [strategy.md](../archive/strategy-2026-11-16.md), [methodology.md](../methodology/methodology.md), [lorenz-to-bvp-methodology.md](../archive/lorenz-to-bvp-methodology-2026-11-16.md), [questions-for-supervisor.md](../problem/questions-for-supervisor.md) and the Lorenz code under `prior-work/`.

---

## 1. Executive summary

**Update, 2026-11-19:** the team proposed one network with knowledge retention as the method. [single-network-retention.md](single-network-retention.md) finds this realistic and takes it as the core method. With that change, the error estimator below becomes optional and Neurocomputing becomes the primary journal. Everything else in this report still applies.

The plan as it stands does not add up to one paper. The conference paper's contributions (time windows, endpoint handoff, causal weights, hard initial condition) are exactly the parts our BVP plan says must not be carried over. The BVP equations also come from an unpublished manuscript by other authors (M.J. Uddin, AIUB, and M.I. Hasan, BUET). So today we have two unrelated projects, and one of them has an open ownership question.

The rescue is to stop presenting the BVP as "the next application" and present both problems as test cases for one question. The conference paper's own strongest finding can carry that question. Its full-interval network had a small residual (MSE 2.80e-4) but a trajectory error of RMSE 2.237. My recommended framing is therefore: **when does a small PINN residual mean a small solution error, and how can we tell without a reference solution?**

Concretely, the journal paper should add four things:

1. A reference-free error estimator for PINN solutions of nonlinear ODE systems, built from the linearized error equation. It applies to the initial value problem (Lorenz) and the two-point boundary value problem (micropolar flow) alike.
2. One training framework for both problems, in which only the sequencing step is problem-specific: time windows with warm start for the IVP, parameter continuation with warm start for the BVP.
3. Matched-budget ablations with at least five seeds.
4. An honest comparison with classical solvers at equal accuracy and equal time.

This is new, substantive content well above 30%. My estimate is about 75 to 80% new by words and over 90% new by figures. The primary target is the Journal of Scientific Computing. The backups are Neurocomputing and Mathematics and Computers in Simulation.

Two things must happen before any coding:

- The supervisor must settle who owns the BVP model and how authorship works with Uddin and Hasan.
- The team must wait for the ICEEICT decision, due 20 November 2026.

---

## 2. Target journal shortlist

### 2.1 Comparison table

Metrics come from aggregators unless marked otherwise. Check them in Clarivate JCR and SCImago before you quote them. I could not verify review times on publisher pages, because ScienceDirect blocked direct reading. Each journal's "Journal insights" page lists them, so read it before choosing.

| Journal | Scope fit for the recommended framing | Stated policy on extended conference papers | Typical expectations [J] | Standing and competitiveness |
|---|---|---|---|---|
| **Journal of Scientific Computing** (Springer) | Good. Numerical methods with careful verification; publishes PINN analysis and error-estimation papers [J] | Journal page not read. Springer's general computer-science guidance says an extension needs at least 30% new material and must cite the original [V, excerpt: [Springer LNCS guidelines](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)] | Clear numerical-analysis content (a proposition about the estimator), classical baselines, convergence studies. Accepts small problems when the method insight is real | Q1; JIF 3.3 for 2024 per [Wikipedia](https://en.wikipedia.org/wiki/Journal_of_Scientific_Computing) [V, aggregator]. Competitive, but the field is reachable for careful work [J] |
| **Neurocomputing** (Elsevier) | Good. Neural-network training methods with experiments | Journal-wide rule not found. One special-issue call asks for at least 40% new material plus a letter describing the differences [V, excerpt: [special issue page](https://www.sciencedirect.com/special-issue/105M4TS8C78)] | Clear method novelty, ablations, comparisons with several PINN variants. Less theory needed | Q1 [V, aggregator: [myhuiban](https://www.myhuiban.com/journal/234)]. Large volume, mid-high acceptance difficulty [J] |
| **Mathematics and Computers in Simulation** (Elsevier, IMACS) | Good. Simulation methods, ODE/BVP numerics, PINNs appear | Not found; fall back on Elsevier's general test (below) | Correct numerics, verification, comparison with standard solvers | JCR 2024 Q1 in three categories per [Peeref](https://peeref.com/journals/5739) [V, aggregator; other aggregators say SJR Q2] |
| **Engineering Applications of Artificial Intelligence** (Elsevier) | Fair. Wants an engineering application; the heat-transfer BVP gives one. It already published a PINN study of bioconvection nanofluid flow over a stretching surface (2024) [V, excerpt: [Altmetric record](https://altmetric.com/details/138202784/citations)] | Not found | Real engineering use, validation, comparison with other AI methods. "Just another PINN application" risk is high here [J] | Q1, high volume [J, verify] |
| **Neural Computing and Applications** (Springer) | Fair | Guest-editor guidance expects roughly 30% new content; one later topical call asks for 50% [V, excerpt: [NCA guidance](https://link.springer.com/journal/521/updates/17231220)] | Method plus experiments; lower theory bar | Q1/Q2 [J, verify] |
| **Machine Learning: Science and Technology** (IOP, open access) | Good for an ML-for-science reliability paper | At least 30% new material, different title and abstract, cite conference paper with DOI in the introduction, cover letter listing the additions [V: [IOP policy](https://publishingsupport.iopscience.iop.org/questions/expanding-a-cs-proceedings-article-for-journal-publication/)] | Physical-science insight, open data and code | Fully open access; APC listed as £1900 / $2680 with reduced rates for some countries [V, excerpt: [IOP page](https://publishingsupport.iopscience.iop.org/?p=3555)]. Bangladesh's waiver group must be checked |
| **Journal of Computational Physics** (Elsevier) | Fair in topic, weak in scale. Two ODE test problems will read as small for JCP [J] | Not found; Elsevier general test applies | New method with clear advantage on hard problems, high-dimensional or PDE tests | Top venue. Stretch target only [J] |
| **SIAM Journal on Scientific Computing** | Fair in topic, high bar | Considered only if significantly revised and extended. Footnote on first page, conference paper uploaded with the submission, cover letter explaining the extension [V: [SISC editorial policy](https://epubs.siam.org/journal/sisc/editorial-policy), [instructions](https://epubs.siam.org/journal/sisc/instructions-for-authors), excerpt] | Strong analysis, careful numerics | Stretch target only [J] |
| **Neural Networks** (Elsevier) | Fair | Instructions say articles must not have been published elsewhere except in abstract form; no extension rule stated [V: [INNS instructions](https://www.inns.org/instructions-to-authors)] | High ML novelty bar | Stretch target; the policy wording is a risk for an extension [J] |
| **Journal of Machine Learning for Modeling and Computing** (Begell House) | Very good topic fit | Not found | Method focus, small community | Indexing unclear: one aggregator says SCImago Q1, but Scopus coverage is not confirmed [V, excerpt: [colab.ws](https://colab.ws/journals/55109)]. Verify before choosing |

Dropped after checking:

- **Chaos, Physica D, Chaos Solitons & Fractals.** Our Lorenz-1960 system is not chaotic. The conference paper itself says it is "non-chaotic and periodic". The model reduces to the free rigid body equations [V: [Bihlo and Popovych](https://arxiv.org/abs/0909.1957)]. A dynamical-systems journal would see a periodic 3-ODE and a heat-transfer BVP as off-scope [J].
- **CMAME.** It is a computational-mechanics journal, and two one-dimensional ODE problems are too small for it [J].
- **Scientific Reports.** It is open access with an APC and accepts sound work, but for a methods paper it gives less signal than the options above. I did not find its extension rule [J].

### 2.2 Ranked recommendation

1. **Primary: Journal of Scientific Computing.** The recommended framing (residual vs error, estimator, classical baselines) is a numerical-analysis story, and this audience rewards it [J]. It also forces us to do the verification a reviewer elsewhere would demand anyway.
2. **Backup 1: Neurocomputing.** Use it if the estimator proposition stays empirical and the paper reads more as a training-method study [J].
3. **Backup 2: Mathematics and Computers in Simulation.** Use it if the reviewers at the first two say the ML part is thin but the numerics are sound [J].

If the supervisor insists on an application paper about the flow physics, switch to EAAI and read section 5.1, framing C, first.

### 2.3 Policies that apply whichever journal we pick

- **Elsevier general test.** Elsevier states that "an author may build on a preliminary conference paper with additional experiments, new data or further discussion" (quote checked 2026-11-19). Conference papers are judged case by case, and in computer science a conference paper often counts as formal publication [V: [Elsevier, what counts as prior publication](https://www.elsevier.com/connect/ive-seen-this-somewhere-before-what-counts-as-prior-publication)]. A thesis or report does not count as prior publication on Elsevier's list [V, same page]. So the FYDP report is not a duplication problem.
- **IEEE side.** The conference paper's copyright goes to IEEE when the IEEE copyright form is signed for Xplore [V: [IEEE copyright transfer](https://conferences.ieeeauthorcenter.ieee.org/get-published/about-transferring-copyright-to-ieee/)]. IEEE accepts a conference-to-journal evolution when the journal paper has substantially more technical content, cites the conference paper and states the difference [V, excerpt: [IEEE post-publication policies](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/post-publication-policies/)]. Reusing small parts of text and figures needs full citation; larger third-party reuse needs IEEE permission via pubs-permissions@ieee.org [V, excerpt, same source]. In practice: redraw every figure and rewrite every paragraph, then no permission is needed [J].
- **ICEEICT 2027 dates:** acceptance notification 20 Nov 2026, camera-ready and registration 20 Dec 2026, conference 28 to 30 Jan 2027 [V: [important dates](https://iceeict.mist.ac.bd/authors-important-dates), [call for papers](https://iceeict.mist.ac.bd/authors/call-for-papers/)]. Our own checklist records that simultaneous submission is not allowed during review (review/requirements-verdict.md, item S13). Do not submit the journal paper before the conference decision. If accepted, submit it after the conference version is final.

---

## 3. Lessons from editors and reviewers

1. **Weak baselines kill ML-for-PDE papers.** McGreivy and Hakim found that 79% (60 of 76) of ML-for-fluid-PDE papers claiming to beat a numerical method compared against a weak baseline. Their two rules: compare at equal accuracy or equal runtime, and compare against an efficient numerical method. They also found widespread reporting bias [V: [arXiv 2407.07218](https://arxiv.org/abs/2407.07218), Nat. Mach. Intell. 2024]. *For us:* the conference paper has no cost comparison with DOP853, and the BVP plan uses solve_bvp only as a check. Both must change (I4).
2. **Interpretation over leaderboard.** The Physical Review Fluids editorial (Brenner and Koumoutsakos, 2021) asks ML papers to contribute insight and interpret results, not just report strong performance [V, excerpt: PRFluids 6, 070001]. *For us:* explain why the residual and the error separate, not only that they do.
3. **The boundary-layer community has known errors.** Pantokratoras reviewed many boundary-layer papers and found profiles cut off by a too-small domain, which corrupts wall shear and heat transfer [V, excerpt: Appl. Math. Model. 2009, doi 10.1016/j.apm.2007.11.009; [example comment](https://arxiv.org/pdf/physics/0702021)]. *For us:* a domain-length study for L is mandatory (I12).
4. **"Just another PINN application" is the default rejection** for a PINN that solves one more similarity ODE. A PINN for bioconvection nanofluid flow over a stretching sheet is already published in EAAI (2024) [V, excerpt]. *For us:* the BVP alone has no novelty as an application [J].
5. **Reference-free error bounds for PINNs on ODEs already exist.** Hillebrecht and Unger derived a rigorous a posteriori bound for ODE PINNs (IJCNN 2022) and extended it to PDEs (IEEE TNNLS 2023) [V: [arXiv 2203.17055](https://arxiv.org/abs/2203.17055), [arXiv 2210.03426](https://arxiv.org/abs/2210.03426)]. Related work: error majorants used for training (Astral, [arXiv 2406.02645](https://arxiv.org/abs/2406.02645)) and bounds for a first-order nonlinear ODE ([arXiv 2411.13848](https://arxiv.org/abs/2411.13848)). I found no paper that does this for a coupled, high-order, nonlinear two-point BVP with soft boundary terms and domain truncation [J, based on these searches only]. *For us:* our novelty must be stated against these papers, not claimed in a vacuum.
6. **Small loss is not small error.** The Acta Numerica PINN survey makes this point, and our own conference Table III shows it (residual MSE 2.80e-4, RMSE 2.237) [V: conference paper; repo note in [literature-synthesis.md](../literature/literature-synthesis.md)]. *For us:* this is our hook.
7. **Report everything, with enough seeds.** Our conference paper used 1 seed for two arms, 3 seeds for ablations, and a p-value of 0.196 with n=3 vs n=5 [V: conference Table II and text]. Reviewers at Q1 journals will ask for more seeds and for every run, failures included [J].
8. **Do not claim the obvious.** Adam followed by L-BFGS is known to beat Adam alone on PINN benchmarks [V: Rathore et al. ICML 2024, cited in our conference paper]. The supervisor's draft lists this hybrid as a contribution. Do not repeat that claim (I18).
9. **Reproducibility is now expected.** IOP requires a data availability statement [V: IOP page above]. Most Q1 computational journals expect code and data links [J]. We already have config hashing and saved checkpoints; release them with a DOI.

---

## 4. Inconsistency audit

Severity: **critical** means the paper cannot be accepted as one paper without fixing it, **major** means a reviewer will likely ask for a major revision, and **minor** is a cleanup item. "Fix" points to the section of this report where the fix lives.

| ID | Issue | Where it appears | Severity | Why a reviewer cares | Fix |
|---|---|---|---|---|---|
| I1 | No research question links an IVP in time to a BVP in space | Conference paper vs [strategy.md](../archive/strategy-2026-11-16.md) section 1 | Critical | Two problems with no shared question read as two papers stapled together | One framing, section 5.2. Both problems test the same estimator and framework, section 6 |
| I2 | The conference method does not transfer. Windows, handoff, causal weights and hard IC are all listed as "do not carry over" | Conference contributions vs `prior-work/README.md`, [lorenz-to-bvp-methodology.md](../archive/lorenz-to-bvp-methodology-2026-11-16.md) | Critical | A journal "extension" whose new half drops the old method is not an extension | Split the method into a shared layer and a sequencing layer; BVP sequencing is parameter continuation with warm start, section 6.3 |
| I3 | Eqs. 14 to 19 come from an unpublished manuscript by Uddin and Hasan. Ownership, permission and authorship are unclear, and their own PINN paper may overlap ours | `sources/project-draft/project3Nov_2026.docx` header and section 5 | Critical | Publishing another group's unpublished model, or two papers with the same results, is a publication-ethics problem | Supervisor decision first, section 10 Q1. Add a published benchmark BVP so the methods paper does not depend on the draft, section 6.1 |
| I4 | No classical-solver baseline and no cost comparison. PINNs are far slower than DOP853 or solve_bvp for one forward 1D solve | Conference results; [strategy.md](../archive/strategy-2026-11-16.md) section 13 | Critical | Weak-baseline rule (lesson 1) | Classical baselines at equal accuracy and equal time; do not claim speed, section 6.6 |
| I5 | Conference comparisons are descriptive: the full-interval run differs in parameters (27x), sampling, optimizer budget; 1 seed for two arms; 3 seeds for ablations; p=0.196 | Conference Tables II and III, text | Major | Confounded comparisons cannot support "windows help" | Matched-budget re-runs, at least 5 seeds per arm, 10 for headline arms, section 6.8 |
| I6 | One integrable, periodic orbit on [0, 13.26] from one initial state. "Long horizon" is one loop | Conference abstract and section II | Major | Thin test case for a journal; the conference paper itself promised other initial states and coefficients | Several initial states and 1 to 5 orbits; report error growth with horizon, section 6.7 |
| I7 | Constraint enforcement differs: hard IC vs twelve soft BC terms, never compared | Conference eq. 3 vs [strategy.md](../archive/strategy-2026-11-16.md) section 5 | Major | Reviewer asks whether results depend on hard vs soft | Make enforcement an ablation axis on both problems, section 6.4 |
| I8 | Network size differs: 27 x 11,283 = 304,641 parameters vs one 12,933-parameter network | Conference Table I vs [lorenz-to-bvp-methodology.md](../archive/lorenz-to-bvp-methodology-2026-11-16.md) | Major | Accuracy gains may just be capacity | Report parameter counts; include matched-budget arms, section 6.2 |
| I9 | Loss definitions differ: per-point mean over three components with causal weights vs a scaled sum of 5 equation and 12 boundary terms | Conference eqs. 6 to 8 vs [strategy.md](../archive/strategy-2026-11-16.md) section 5 | Minor | Notation looks inconsistent across the two halves | One loss template with declared scales, section 6.4 |
| I10 | Reference handling differs. Lorenz code logs reference error during training and returns terminal weights; BVP plan ranks checkpoints independently and holds out final evaluation | [lorenz-to-bvp-methodology.md](../archive/lorenz-to-bvp-methodology-2026-11-16.md); [strategy.md](../archive/strategy-2026-11-16.md) section 12 | Major | Any use of the reference for selection is leakage | One protocol: selection by residual and estimator only, reference opened once at the end, section 6.8 |
| I11 | Metrics differ, and the BVP accuracy target and wall quantities are undefined | Conference eq. 9; [questions-for-supervisor.md](../problem/questions-for-supervisor.md) Q6 | Major | No common yardstick across the two problems | Common metric set plus estimator effectivity, section 6.7 |
| I12 | Truncated far field L and stiff wall coefficients (1/q^3 = 125, 2*beta_m/h^4 = 250 at eta=0) not yet checked | [parameters.md](../problem/parameters.md), [strategy.md](../archive/strategy-2026-11-16.md) section 2 | Major | Known boundary-layer error (lesson 3); stiffness is the likely PINN failure | Domain-length study, wall clustering, continuation, section 6.5 |
| I13 | BVP definitions conflict: draft title says "transient" but delta_star=0 removes unsteady terms; draft table and conclusions use M as a magnetic parameter, the receipt says M=1/Da; epsilon, Lb, B, fw, H, Prm questions open | Draft docx; [parameters.md](../problem/parameters.md); [questions-for-supervisor.md](../problem/questions-for-supervisor.md) | Major | An application reviewer will spot it; a methods reviewer will ask what was solved | State the case as a fixed test problem with every value sourced; settle the definitions with the supervisor, section 10 Q5 |
| I14 | Existence and uniqueness of the BVP solution at the locked case are not checked. Stretching-surface problems can have more than one solution [J] | No reference run yet ([README](../../README.md)) | Major | A PINN may converge to a different branch than solve_bvp | Multiple initial guesses and continuation in the reference gate, section 6.5 |
| I15 | Notation clashes: kappa (wavenumber vs curvature), epsilon (causal weight vs Curie temperature), theta (network weights vs temperature), L (loss, lambda_m, far endpoint), M, H, k | Conference paper vs [equations.md](../problem/equations.md) | Minor | Confusing; looks careless | One notation table for the journal paper, section 8 |
| I16 | Optimizer schedules differ, and the BVP proposal has eight unset settings | Conference Table I vs [configs/training-proposal.json](../archive/training-proposal-2026-11-16.json) | Minor | Unclear protocol | One declared schedule family, set by pilot runs, section 6.4 |
| I17 | The FYDP context file describes a data-driven ANN sweep (MSE on RK4 targets), while the conference paper is a PINN | `prior-work/fydp-context/CONTEXT.md` vs conference paper | Minor | Not a publication conflict, but the FYDP report and the journal paper must tell the same story | Update the FYDP report chapters to the PINN story; do not cite the ANN sweep as our method, section 7 phase 4 |
| I18 | The supervisor's draft presents Adam plus L-BFGS as a developed contribution, and uses MATLAB | Draft section 1 highlights (iii); section 5 | Minor | Claiming a known technique as new invites rejection | Present Adam then L-BFGS as standard practice with citation, section 6.4 |
| I19 | The conference paper lists causal weighting as a contribution but its own results do not show a consistent accuracy gain | Conference contributions vs section V-C | Minor | Overclaim | Keep causal weighting as an ablation, not a headline, section 6.8 |
| I20 | Procedure: conference under review until 20 Nov 2026; IEEE copyright at camera-ready; no simultaneous submission | Section 2.3 | Major | A desk reject or ethics query if mishandled | Compliance checklist, section 9; timeline, section 7 |
| I21 | Fourth-order autograd derivatives in float64 are untested; the Lorenz work only needed first order | [strategy.md](../archive/strategy-2026-11-16.md) section 4 | Minor | Wrong derivatives give wrong physics silently | Manufactured-solution derivative tests before training, section 6.8 |
| I22 | The brief we were given calls the Lorenz system chaotic. The paper correctly says non-chaotic and periodic | User brief vs conference section II | Minor | If "chaotic" slips into the journal paper, reviewers will catch it | Use "integrable, periodic" and cite the rigid-body reduction, section 8 |

---

## 5. Unified narrative

### 5.1 Candidate framings

**Framing A: one sequential training framework for ODE PINNs.** Time windows with warm start for IVPs and parameter continuation with warm start for BVPs, presented as one idea: "solve an easier problem first, then move along a path".

- Novelty: low to moderate. Windowing, causal sweeping and curriculum training are all published (Penwarden et al., Krishnapriyan et al., Wang et al.). The unification is mostly packaging [J].
- Feasibility: high. Most code exists.
- Risk: reviewers say "incremental".
- Best fit: Neurocomputing, NCA.

**Framing B (recommended): residual is not error.** Reference-free error estimation for PINN solutions of nonlinear ODE IVPs and BVPs, with the sequential framework as the training method under test. The PINN's equation residual and boundary or initial residual drive the linearized error equation around the PINN solution. Solving this linear problem once gives an error estimate. We test it where the true error is known, then use it for model selection and to explain failures.

- Novelty: moderate to good. ODE IVP bounds exist (Hillebrecht and Unger), but I found nothing on coupled high-order BVPs with soft boundary residuals and domain truncation. The idea of the linearized defect equation is classical in numerical ODEs (defect correction), so the contribution is the transfer to PINNs, the treatment of soft boundary terms, and the evidence [J].
- Feasibility: good for an undergraduate team. It needs Jacobians (autograd or SymPy), and SciPy solve_ivp and solve_bvp for linear problems. The conference checkpoints let us test it on Lorenz within weeks.
- Risk: medium. If the estimator is poor at large errors, we report that honestly. That is still a result, because it tells users when not to trust it.
- Best fit: Journal of Scientific Computing, MATCOM, MLST.

**Framing C: verification-first PINN for boundary-layer similarity BVPs.** An application paper on the micropolar flow, with the Lorenz work as background.

- Novelty: low as ML. As physics, it depends on the model by Uddin and Hasan, which is not ours to publish alone.
- Feasibility: high.
- Risk: high. The Lorenz link is cosmetic, so the "extension" claim fails; ownership conflict (I3); "another PINN application" (lesson 4).
- Best fit: EAAI or a heat-transfer journal, and only as a paper with Uddin and Hasan.

### 5.2 Chosen framing and defense

I recommend Framing B with A's training framework inside it.

1. It is the only framing in which the Lorenz paper and the BVP answer the same question. The Lorenz full-interval run already shows the problem. The BVP's stiff wall and truncated domain are the second, different way the residual can mislead.
2. It turns our honesty rules into the contribution. We already separate the reference from training and refuse to call small loss success. Framing B makes that discipline the scientific claim.
3. It survives the classical-solver objection. We do not claim PINNs beat solve_bvp. We claim a way to know whether a PINN answer can be trusted, which matters wherever PINNs are used, including inverse and parametric settings where a classical reference is not available [J].
4. It reduces the I3 risk. The methods paper needs a hard BVP, not this specific physics. A published benchmark from the same family can carry the paper if permission for Eqs. 14 to 19 is not granted.

Working title: *Residual is not error: reference-free error estimation and sequential training for physics-informed neural networks on nonlinear ODE initial and boundary value problems.*

### 5.3 Contribution bullets (as they would appear in the introduction)

- We derive a reference-free error estimate for PINN solutions of nonlinear ODE systems by solving the linearized error equation forced by the PINN's interior and initial or boundary residuals. We show the estimate matches the true error up to second-order terms, and we measure its effectivity index (estimated error over true error) on an initial value problem and a coupled five-field, high-order boundary value problem.
- We present one training framework for both problem classes, in which only the sequencing step differs: time windows with warm start for the IVP, and parameter continuation with warm start for the BVP. With matched parameter and optimizer budgets and at least five seeds per arm, we isolate the effect of each component, including hard vs soft constraint enforcement.
- We compare every PINN result with classical solvers (DOP853; collocation BVP solvers) at equal accuracy and equal wall time, and with three PINN variants, and we report all runs, including failures.
- We show where the residual and the error separate: error growth over several orbits in the IVP, and near-wall stiffness and domain truncation in the BVP. We show the estimator detects these cases without a reference solution. Code, configurations, checkpoints and data are released.

Each bullet is testable. The second-order claim is a short proposition. The effectivity range is a measured result. If the estimator fails in some regime, the fourth bullet reports where.

---

## 6. Revised methodology plan

One framework covers both problems. Each fix notes the audit ID it closes.

### 6.1 Problem formulation

Write both problems in one form: find u on an interval Omega with N(u) = 0 inside and C(u) = 0 on the constraint set, where s is the coordinate. For Lorenz, s = t, u = (x, y, z), N is first order and C is the initial state. For the BVP, s = eta, u = (f, g, theta, phi, chi), N is Eqs. 14 to 18 (orders 4, 2, 2, 2, 2) and C is the twelve conditions of Eq. 19 at eta = 0 and eta = L. (I1, I15)

Problem set:

- **P1 Lorenz-1960** with the conference coefficients a = (-0.1, 1.6, -0.75). Use the conference initial state plus at least 2 further initial states, and horizons of 1 and 3 orbits (5 orbits and more states are optional, see 7.2.1). Describe it as integrable and periodic, with two conserved quadratic quantities. (I6, I22)
- **P2 micropolar BVP** at the locked case in [parameters.md](../problem/parameters.md), only if the supervisor confirms we may publish it (I3). Keep the open definition questions on record as stated assumptions. (I13)
- **P3 published benchmark BVP** from the same family, for example the unsteady curved stretching surface of Naveed, Abbas and Sajid (2016). The supervisor's draft already uses it for validation. This gives literature values to check against and keeps the paper alive if P2 is withheld. (I3) [J]

### 6.2 PINN design

- Pointwise MLP with tanh, float64, Xavier initialization, zero biases, and seeds recorded. Input normalized to [-1, 1] inside the model.
- P1: the conference design (separate 1-60-60-60-60-3 networks per window) plus a single network and a shared network. P2 and P3: one 1-64-64-64-64-5 network.
- Every comparison reports trainable parameters. Add one matched-budget arm per problem: a single P1 network with about 304k parameters, and a narrower window network whose 27-window total is close to 12.9k. (I8)
- Wall handling for P2: cluster collocation points near eta = 0, as already proposed in [strategy.md](../archive/strategy-2026-11-16.md) section 7. (I12)

### 6.3 Sequencing layer: the only problem-specific part

- P1: 27 time windows (also 9 and 54 as an ablation), endpoint handoff, warm start of weights.
- P2: parameter continuation on (kappa, b, Da) through 1, 0.5, 0.2 with warm start, already proposed in [strategy.md](../archive/strategy-2026-11-16.md) section 2. Each stage reuses the previous weights. Continuation is to a BVP what windowing is to an IVP: a path from an easy problem to the target. (I2)
- Ablation on both: sequencing on vs off at matched total budget.

### 6.4 Loss and constraint enforcement

One loss template for both problems: J = sum_i lambda_i * mean(r_i/s_i)^2 + sum_j mu_j * (b_j/t_j)^2, with declared fixed scales s_i, t_j and positive weights. P1's conference loss is the special case of three equal components with hard IC. Causal weights are an optional per-point multiplier, used only on P1. (I9)

Constraint enforcement is an ablation axis on both problems (I7):

- P1: hard IC (trial function) vs soft IC penalty.
- P2 and P3: all-soft (baseline) vs partial-hard. The simple Dirichlet values (chi(0) = 1; g, theta, phi, chi, f' at L) are built into the trial function, and the Robin-type conditions (B1, B4) stay soft.

Optimizer: Adam with step decay, then L-BFGS with strong Wolfe line search, cited as standard practice, not as our contribution (I18). Fix the budgets from pilot runs and write them into the configs, replacing the eight nulls (I16). The repo literature notes suggest adding a full-memory BFGS arm; keep it optional.

### 6.5 Reference solutions and BVP gates

- P1: DOP853 at rtol 1e-10 and atol 1e-12, as in the conference paper, for every initial state and horizon.
- P2 and P3: solve_bvp on the twelve-state first-order system, as planned in [strategy.md](../archive/strategy-2026-11-16.md) section 6. Add a second solver as a cross-check: MATLAB bvp5c, which the supervisor's collaborators use, or a Chebyshev collocation. (I4)
- **Gate before any PINN training (I12, I14):** solutions for L in {10, 15, 20, 30}, with wall values converged to a stated tolerance. Use at least five different initial guesses to look for second solutions. Continuation from the debug case to the locked case must not jump branches.

### 6.6 Baselines

- **Classical (I4):** DOP853 (P1) and solve_bvp (P2, P3). Two comparisons: wall time to reach the PINN's achieved error, and error achieved in the PINN's wall time. Same machine, single thread, timings repeated. Say plainly that classical solvers win on one forward solve, if they do.
- **PINN variants:**
  - vanilla single network (both problems)
  - causal-weighted single network (P1)
  - gradient-based loss weights following Wang et al. 2021 (P2, P3)
  - hard vs soft constraints (both)
- Three variants per problem is enough. More arms cost compute without adding to the story [J].

### 6.7 Metrics

Common to both problems (I11):

- relative L2 error and max absolute error per component, on a dense grid not used in training
- residual mean square on the same grid, and each boundary or initial residual separately
- estimator effectivity index = ||e_est|| / ||e_true||, reported per run and per component
- cost: wall time, Adam steps, L-BFGS function calls, parameter count, hardware

Problem-specific:

- P1: drift of the two invariants, and error against time across 1 to 5 orbits (I6)
- P2 and P3: wall quantities f''(0), g'(0), theta'(0), phi'(0), chi'(0) with their reference values. The supervisor confirmed on 2026-11-19 that no physical reporting is needed, so these are plain numerical checks against the reference.

### 6.8 Error estimator, ablations and validation protocol

**Estimator.** Let u_hat be the trained PINN, r = N[u_hat] its interior residual and c = C[u_hat] its constraint residual. The true error e = u_hat - u satisfies N'[u_hat] e = r - Q(e) and C'[u_hat] e = c - Q_C(e), where Q and Q_C are of second order in e for smooth N and C.

Dropping Q gives a linear problem for e_est: a linear IVP for P1 (solve_ivp), and a linear two-point BVP for P2 and P3 (solve_bvp). The proposition to prove: if N'[u_hat] has a bounded inverse, then ||e - e_est|| <= K ||e||^2 for small e [J: standard argument, to be written with the supervisor].

For P2, add a separate truncation term from the L study, because the estimator only sees the error inside [0, L].

**Ablations (≥5 seeds each; 10 seeds for the main arm if time allows). Section 7.2.1 marks which levels are must-have and which are optional:**

| Problem | Axis | Levels |
|---|---|---|
| P1 | Windows | 1, 9, 27, 54 |
| P1 | Warm start | on, off |
| P1 | Causal weights | on, off (I19) |
| P1 | IC enforcement | hard, soft |
| P1 | Budget | matched parameters, matched steps |
| P1 | Horizon | 1, 3, 5 orbits |
| P1 | Initial state | 4 to 5 states |
| P2, P3 | Constraints | all-soft, partial-hard |
| P2, P3 | Loss scaling | declared scales, unit scales |
| P2, P3 | Continuation | on, off |
| P2, P3 | Wall clustering | on, off |
| P2, P3 | Optimizer | Adam only, Adam then L-BFGS |
| P2, P3 | Far field | L in {10, 15, 20} |

**Validation protocol (I5, I10, I21):**

1. Software tests first: derivative checks up to fourth order against manufactured smooth functions, residual code against hand-computed values, and the estimator on a linear problem where it is exact.
2. Freeze configs and seeds before the full runs. Log git hash and config hash with every result.
3. Select checkpoints by training residual and estimator only. The reference is used once, after selection, for the reported errors.
4. Report medians and interquartile ranges. Report all runs, including diverged ones. Use paired seeds. Do not give p-values for n = 3; with n ≥ 5, prefer effect sizes with bootstrap intervals [J].
5. Keep the rule from [AGENTS.md](../../AGENTS.md): no claim without a run behind it.

---

## 7. Revised implementation plan

### 7.1 Facts and assumptions

- Team: five authors on the FYDP context (Ikramul Hasan Moral, Md. Abu Bakar, Samiur Rahman Omlan, Fariha Islam, Md. Touhidul Islam) [V: `prior-work/fydp-context/CONTEXT.md`].
- Hardware: this machine is a 4-thread Intel i5-6200U with about 7 GB RAM and no GPU [V: checked 2026-11-19]. The Lorenz single-network run takes under six minutes on a laptop CPU [V: Lorenz README].
- FYDP-III is already running and ends in mid-February 2027. The team may extend the submission by a few weeks, ideally a month, to mid-March [V: team message, 2026-11-19].
- Assumptions:
  - **[A]** Each member has a similar laptop.
  - **[A]** About 10 hours per person per week, less around exams and the conference.
  - **[A]** The team may scatter after FYDP-III, so the journal submission should happen before or at the FYDP end, not months later.
- Small float64 networks on 1D problems do not gain much from a GPU [J]. CPU is fine, but the window-PINN cost per run must be measured in a pilot before the grid is fixed.

### 7.2 Phases

The window is about 13 weeks to mid-February, or about 17 weeks with a one-month extension. That is tight for the full plan, so the work is split into must-have and optional tiers (7.2.1). The rule: the FYDP-III report and the journal paper use the same runs and the same results tables. No result is produced twice.

Two tracks, same work order:

- **Base track:** the FYDP-III report is due mid-February with the must-have results. The journal paper is submitted by mid-March.
- **Extension track (preferred):** the FYDP-III report and the journal submission both land in mid-March, and the extra month buys the optional tier and a proper supervisor review.

| Phase | Dates | Output | Risks and mitigation |
|---|---|---|---|
| 0. Decisions | 20 to 30 Nov 2026 | ICEEICT result handled. Supervisor answers section 10, above all Q1 (ownership) and Q7 (extension). Framing frozen. Shared code package set up from the Lorenz and BVP code | Supervisor slow to answer Q1: start P3 (published benchmark) at once, so no week is lost. Supervisor rejects the reframing: fall back to Framing A with the same experiments |
| 1. References and estimator prototype | 1 to 21 Dec 2026 | P1 references for 3 initial states and 1 and 3 orbits. BVP solve_bvp gate (L study, multiple guesses, continuation). Estimator run on the saved conference checkpoints, including the RMSE 2.237 run: the first real result. Camera-ready by 20 Dec if accepted, owned by one person | solve_bvp fails at the locked case: continue from the debug case and report the branch; ask the supervisor before changing any value |
| 2. BVP PINN baseline and pilots | 22 Dec 2026 to 10 Jan 2027 | Derivative and residual tests pass. BVP baseline PINN trained. Estimator for BVPs working. Timed pilot of every must-have arm; grid sized to the measured cost | PINN does not converge on the BVP: try continuation and partial-hard constraints first; if it still fails, the failure plus the estimator detecting it is a reportable result |
| 3. Must-have campaign | 11 Jan to 7 Feb 2027 | All must-have runs with 5 seeds. Weekly results review against the audit table. Conference presentation 28 to 30 Jan (one or two members; the rest keep runs going) | Compute overrun: run in parallel on all five laptops, use free cloud CPU as overflow; cut optional levels, never seeds |
| 4a. Writing (base track) | 8 to 15 Feb 2027 | FYDP-III report from the must-have results, aligned to the PINN story (I17). Paper sections 1 to 5 were drafted during Phases 2 and 3 | Report and paper drift apart: one shared results folder and one set of figures for both |
| 4b. Optional tier (extension track) | 8 Feb to 1 Mar 2027 | Optional runs from 7.2.1, then the full journal draft and reproducibility package. Two members act as hostile reviewers using section 4 | Optional runs late: drop them; the paper stands on the must-have tier |
| 5. Submission | 2 to 15 Mar 2027, buffer to 31 Mar | Supervisor review, similarity check, cover letter, submission to the primary journal | Supervisor review takes longer: send each section as it finishes, starting in January |

Writing does not wait for results. Sections 2 (related work), 3 (problems) and 4 (method) of the paper are drafted in Phases 1 and 2 by R5, with the estimator derivation by R4.

#### 7.2.1 Must-have and optional experiments

| Tier | Problem | Runs |
|---|---|---|
| Must-have | P1 | Estimator on the conference checkpoints. Matched-budget re-runs of window PINN, single network, no warm start and soft IC. 3 initial states, 1 and 3 orbits, 5 seeds |
| Must-have | BVP (P2, or P3 if Q1 blocks P2) | Reference gate with L study. Baseline PINN; continuation on and off; all-soft vs partial-hard constraints; 5 seeds. Estimator on every run |
| Must-have | Both | Classical comparison at equal accuracy and equal time |
| Optional | P1 | 9 and 54 windows; 5 orbits; causal weights on and off; 10 seeds for the main arm |
| Optional | BVP | Second BVP (both P2 and P3); gradient-based loss weights; wall clustering on and off; Adam-only arm; MATLAB bvp5c cross-check |

The must-have tier alone still carries all four contribution bullets and keeps the new content well above 30%: the estimator, the BVP class and the classical comparison are all in it [J]. If the optional tier is skipped, report fewer levels honestly rather than single-seed runs.

### 7.3 Task split (roles; the team assigns names)

| Role | Main work | Also reviews |
|---|---|---|
| R1 Lead and integration | Shared package, experiment runner, configs, merge reviews | Everything before it is merged |
| R2 Classical numerics | P1 to P3 references, L and continuation studies, classical baselines and timing | R4's estimator |
| R3 BVP PINN | P2 and P3 PINN arms, loss scales, constraint ablations | R2's references |
| R4 Lorenz and estimator | P1 re-runs at matched budgets, horizons and initial states; estimator for both problems | R3's training code |
| R5 Writing and reproducibility | Literature positioning, paper draft, figures, data package, compliance checklist | Every claim against its run record |

### 7.4 Tools and practices

- Python with PyTorch (float64 autograd), NumPy, SciPy (solve_ivp DOP853, solve_bvp), SymPy for Jacobians, Matplotlib. Optional MATLAB bvp5c cross-check.
- One repository and one package for P1 to P3, reusing the parts listed in `prior-work/README.md`. The Lorenz repo stays frozen as the conference record.
- Tracking: keep the present JSON config and checksum approach. Each run writes a run ID, git hash, config hash, seed, hardware, wall time and all metrics to one CSV. A heavier tracker such as MLflow is optional and not needed [J].
- Release: tag the commit used for the paper. Archive code, configs, checkpoints and result tables on Zenodo with a DOI, and state the license.
- Keep the gate order from [strategy.md](../archive/strategy-2026-11-16.md): parameters, software tests, reference, training, evaluation.

---

## 8. Paper blueprint

| Section | Content | Figures and tables |
|---|---|---|
| 1 Introduction | The residual-vs-error problem; why it matters where no reference exists; prior bounds (Hillebrecht and Unger, Astral); contributions; disclosure sentence citing the ICEEICT paper with DOI | Fig. 1: one panel showing small residual with large error (Lorenz full interval) next to a BVP case |
| 2 Related work | PINN failure modes, causal and windowed training, constraint enforcement, a posteriori estimates, classical defect correction, ML baseline practice (McGreivy and Hakim) | None |
| 3 Problems | Unified form; P1 (integrable, periodic, conserved quantities); P2 and P3 with every value sourced; notation table (I15) | Table 1: notation. Table 2: problem data |
| 4 Method | Shared PINN layer; loss template; constraint options; sequencing layer (windows, continuation); estimator and its proposition | Fig. 2: framework diagram with the shared and problem-specific parts. Algorithm 1 |
| 5 Experimental protocol | Software tests, references and gates, seeds, selection without reference, metrics, hardware, baselines | Table 3: all arms and budgets |
| 6 Results: initial value problem | Matched-budget ablations; horizon and initial-state study; estimator effectivity; invariants | Fig. 3: error vs time over orbits. Fig. 4: ablation box plots. Fig. 5: estimated vs true error |
| 7 Results: boundary value problem | L study; continuation; constraint and scaling ablations; wall quantities; estimator effectivity | Fig. 6: profiles vs reference. Fig. 7: near-wall error. Fig. 8: estimated vs true error. Table 4: wall quantities |
| 8 Comparison with classical solvers | Equal accuracy and equal time | Fig. 9: error vs wall time for all methods. Table 5: cost |
| 9 Discussion and limitations | Where the estimator fails; what PINNs do not win; when to use what | None |
| 10 Conclusion | Answers to the contributions, no new claims | None |
| Appendices | Proof of the proposition; full configs; all-run tables | Table A1: every run |

Expected length: about 10,000 words, 9 figures, 5 main tables.

### 8.1 New content estimate (task D)

| Part | Words (est.) | New share | Basis |
|---|---|---|---|
| Introduction and related work | 1,800 | 85% | New question and positioning |
| Problems | 900 | 60% | Lorenz equations restated with citation; BVP new |
| Method: framework | 1,500 | 40% | Window, trial and causal equations summarized with citation; continuation and loss template new |
| Method: estimator | 1,200 | 100% | Entirely new |
| Protocol | 1,000 | 70% | New gates, seeds, selection rule |
| Results IVP | 1,500 | 80% | All runs redone at matched budgets, new horizons and states; conference numbers only cited |
| Results BVP | 1,800 | 100% | Entirely new |
| Classical comparison | 700 | 100% | Entirely new |
| Discussion | 600 | 100% | Entirely new |
| **Total** | **~11,000** | **~80%** | **[J] estimate** |

A strict count that treats every Lorenz result as reused still leaves the BVP, the estimator and the classical comparison: about 3,700 words, or roughly 35% before counting any new Lorenz work. So the 30% bar holds even under the most pessimistic counting [J]. Over 90% of figures will be new, because the conference figures are not reused.

**What we can reuse:** the Lorenz equations and coefficients, the trial-function and window definitions as short cited restatements, the DOP853 settings, and conference numbers cited as prior results.

**What we must rewrite or redo:**

- every sentence of prose
- the title and abstract, which must differ from the conference paper
- all figures, regenerated in the new style
- all tables
- every result we report as ours in the journal, re-run at matched budgets

---

## 9. Conference-to-journal compliance checklist

- [ ] Wait for the ICEEICT decision (20 Nov 2026). Do not submit the journal paper while the conference paper is under review (requirements item S13).
- [ ] If accepted: sign the IEEE copyright form at camera-ready. Keep a copy, and record the final DOI when Xplore publishes it.
- [ ] If rejected: the extension rules no longer apply, but do not submit the conference text elsewhere and the journal paper at the same time. Fold the useful parts into the journal paper and say so to the supervisor.
- [ ] Journal paper: different title and different abstract (IOP requires this explicitly; it is good practice everywhere).
- [ ] Cite the conference paper in the introduction with its DOI. Add a first-page footnote: "A preliminary version of the Lorenz-1960 windowed PINN appeared in [ICEEICT 2027]." SISC requires the footnote; it costs nothing elsewhere.
- [ ] Upload the conference PDF as a non-review file if the journal asks (SISC does).
- [ ] No reused text beyond short cited equations. Redraw every figure. Run a similarity check against the conference paper before submission.
- [ ] Settle authorship for P2 with the supervisor, Uddin and Hasan in writing before submission (I3). Settle author order for the journal paper early.
- [ ] Data and code statement with the Zenodo DOI.
- [ ] AI-use disclosure following the target journal's policy. Keep it consistent with the conference acknowledgment.
- [ ] Cover letter paragraph (template):

  > This manuscript extends our conference paper "[title]", [venue, year, DOI]. The conference paper reported a windowed PINN for one periodic orbit of the Lorenz-1960 system. The present manuscript adds (1) a reference-free error estimate for PINN solutions of nonlinear ODE systems, with a proof of second-order accuracy, (2) a boundary value problem class with five coupled fields and twelve boundary conditions, (3) matched-budget ablations over at least five seeds on several initial states and horizons, and (4) comparisons with classical solvers at equal accuracy and equal time. All Lorenz results are recomputed; no text or figure is reused. We estimate that over 75% of the manuscript is new material. The conference paper is attached for the editor's reference.

---

## 10. Open questions for the supervisor

1. **Ownership (blocking).** Eqs. 14 to 19 come from the draft by M.J. Uddin and M.I. Hasan. May we publish PINN results on this model in our own journal paper? If so, are they co-authors, and how do we avoid overlap with their own PINN paper on the same equations? If not, may we use a published benchmark BVP (P3) instead?
2. **Paper type.** Do you accept a methods paper on PINN reliability (Framing B), with the micropolar case as one test problem, rather than an application paper on the flow physics?
3. **Target journal.** Do you agree with the Journal of Scientific Computing as primary, with Neurocomputing and MATCOM as backups? Is there a budget for open-access fees if we choose MLST?
4. **Authorship and corresponding author** for the journal paper.
5. **Model definitions.**
   - Settle the open items in [questions-for-supervisor.md](../problem/questions-for-supervisor.md): epsilon, Lb, B, fw, H and Prm.
   - The draft's title says "transient", but delta_star = 0. Which is intended?
   - Is M the magnetic parameter or 1/Da?
6. **Accuracy and outputs.** Which wall quantities, and what accuracy counts as good enough?
7. **Timeline.** FYDP-III ends in mid-February 2027. Will you approve the one-month extension to mid-March, so that the FYDP-III report and the journal submission can land together? Is the journal submission a FYDP deliverable?
8. **Theory support.** Can you or a colleague check the estimator proposition before submission?
9. **Conference reviews.** If ICEEICT returns reviews, which points should the journal paper answer?

---

## 11. Coverage check

| Requested item | Where |
|---|---|
| A. Journal landscape, policies, expectations, review time, competitiveness | 2.1 to 2.3; review times not verified, stated in 2.1 |
| A. Editor and reviewer lessons | 3 |
| B. Inconsistency audit with severity, impact, fix | 4 |
| C. Two to three framings, assessment, choice, bullets | 5 |
| D. New-content plan, proportion, reuse and rewrite | 8.1 |
| Deliverables 1 to 10 | Sections 1 to 10 |

Every audit item has a matching fix in the plan:

| Audit ID | Fix location |
|---|---|
| I1 | 5.2, 6.1 |
| I2 | 6.3 |
| I3 | 6.1 (P3), 9, 10 Q1 |
| I4 | 6.5, 6.6 |
| I5 | 6.8 |
| I6 | 6.1, 6.7, 6.8 |
| I7 | 6.4 |
| I8 | 6.2 |
| I9 | 6.4 |
| I10 | 6.8 |
| I11 | 6.7 |
| I12 | 6.2, 6.5 |
| I13 | 6.1, 10 Q5 |
| I14 | 6.5 |
| I15 | 6.1, 8 (Table 1) |
| I16 | 6.4 |
| I17 | 7.2 Phase 4a |
| I18 | 6.4 |
| I19 | 6.8 |
| I20 | 2.3, 7.2 Phase 0, 9 |
| I21 | 6.8 |
| I22 | 6.1 |

Limits of this report:

- Several journal policies come from search excerpts, because ScienceDirect blocked direct reading.
- Metrics come from aggregators.
- The novelty claim for the BVP estimator rests on a limited search and needs a proper literature check in Phase 1.
- No experiment was run for this report.
