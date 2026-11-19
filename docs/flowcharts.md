# Flowcharts

The browser-readable [strategy guide](../strategy-guide.html) renders these flows as labeled steps and branches. Mermaid source below is editable documentation, not code that trains a network.

## Deterministic methodology overview

These stage numbers match [the detailed methodology](methodology.md). The [HTML diagram](../bvp-methodology.html) and [PDF diagram](../bvp-methodology.pdf) show five small tasks per stage, with separate mathematical strips. The builder checks these stage labels against the methodology source. The locked case has all 22 physical coefficients; software tests and numerical choices remain pending. A failed reference never becomes a verified result through a later residual check.

```mermaid
flowchart TD
  S1[1. Freeze the specification and case] -->|Valid case| S2[2. Freeze the numerical contract]
  S1 -->|Missing or invalid| BLOCK[BLOCKED: save the missing input or exact reason]
  S2 -->|Complete contract| S3[3. Pass the software tests]
  S2 -->|Unset setting| BLOCK
  S3 -->|Pass| S4[4. Build and validate the reference]
  S3 -->|Fail| FIX[FAILED_TESTS: repair and repeat stage 3]
  S4 -->|Reliable reference| S5[5. Construct the point sets]
  S4 -->|Fail or branch unresolved| REF[UNVERIFIED_REFERENCE: diagnose; no accuracy claim]
  S5 --> S6[6. Train the fixed-case PINN]
  S6 -->|Finite Adam checkpoint| S7[7. Refine and select the checkpoint]
  S6 -->|Nonfinite value or graph error| FAIL[FAILED_TRAINING: save the first bad term]
  S7 -->|Frozen selected checkpoint| S8[8. Lock and test the selected model]
  S7 -->|Refinement fails| ADAM[Keep the eligible Adam checkpoint and failure reason]
  ADAM --> S8
  S8 -->|All declared checks pass| S9[9. Check sensitivity and method changes]
  S8 -->|Any check fails| TEST[FAILED_ACCEPTANCE: retain the exposed final test]
  S9 --> S10[10. Report and archive every outcome]
  BLOCK --> S10
  FIX --> S10
  REF --> S10
  FAIL --> S10
  TEST --> S10
```

Separately approved exploration without a reference may use stages 5 to 7. It retains the unverified-reference status and cannot pass the reference-based acceptance rule. A redesign after final testing is a new logged run, never a hidden retry of the same test.

## Lorenz-style BVP computational graph

This diagram follows the live Lorenz PINN's network, derivative, physics and optimizer structure, adapted to this two-ended BVP. See [the comparison](lorenz-to-bvp-methodology.md). It uses one shared network, not a separate trainable physics network. The complete method remains in [methodology.md](methodology.md).

```mermaid
flowchart TD
  I[Fixed interior eta coordinates: N by 1] --> X[Normalize x = 2 eta / L - 1]
  X --> NN[One shared MLP: 1 to 64 to 64 to 64 to 64 to 5]
  NN --> U[Physical fields: f g theta phi chi]
  U --> D[Componentwise eta derivatives: f through order 4; others through order 2]
  U --> R[Original R14 to R18]
  D --> R
  P[Locked physical coefficients] --> R
  E[Endpoint coordinates: 0 and L] --> EN[Evaluate the same NN and endpoint derivatives]
  EN --> B[All twelve original boundary errors]
  P --> B
  R --> JR[Scaled weighted equation loss]
  B --> JB[Scaled weighted boundary loss]
  JR --> J[One total loss]
  JB --> J
  J --> FIN{Finite values and connected gradients?}
  FIN -->|No| FAIL[Save failure; stop invalid update]
  FIN -->|Yes| G[Backpropagate to NN weights and biases]
  G --> OPT[Adam then deterministic fixed-objective L-BFGS]
  OPT -.->|Update NN only| NN
  OPT --> MON[Independent monitoring and eligible checkpoints]
  MPTS[Fixed monitoring coordinates] --> MON
  MON --> LOCK[Select and lock checkpoint; retain Adam fallback]
  LOCK --> TEST[Open withheld coordinates; evaluate fields derivatives R and B]
  REF[Checked twelve-state solve_bvp reference] -.->|Comparison only| TEST
  TEST --> SEN[Domain sample and seed sensitivity]
  SEN --> REPORT[Errors uncertainty costs status and all failures]
```

Reference values and final-test errors feed neither the training loss nor checkpoint ranking. Coordinate differentiation keeps a graph for the later loss-to-weight backward pass. The baseline has no Lorenz initial-value lift, time windows, endpoint handoff or causal weighting.

## Architecture and gradient flow

```mermaid
flowchart TD
  X[Eta coordinates] --> N[One smooth five-output NN]
  N --> U[f g theta phi chi predictions]
  U --> D[Componentwise eta derivatives: f1 to f4; other first and second derivatives]
  U --> E[Five original equation residuals R14 to R18]
  D --> E
  P[Fixed complete physical parameter case] --> E
  P --> B[All twelve original boundary residuals B1 to B12]
  N --> B
  D --> B
  E --> JE[Scaled weighted equation means]
  B --> JB[Scaled weighted boundary squares]
  JE --> J[One scalar loss]
  JB --> J
  J --> G[Backpropagation through equations and derivatives]
  G --> O[Optimizer changes NN weights only]
  O --> N
```

Physical parameters feed the equation and boundary arithmetic only. They are not in the optimizer's parameter list. The network, equations and boundary checks share one differentiable computation.

## Full run with blocking and failure paths

```mermaid
flowchart TD
  S[Preserve source and record approved equations] --> P{All physical coefficients supplied?}
  P -->|No| WAIT[Save pending case and wait for supervisor]
  P -->|Yes| V{Ranges, denominators and numerical settings valid?}
  V -->|No| STOP[Stop and report the exact invalid input]
  V -->|Yes| T{Analytic, equation, BC and graph tests pass?}
  T -->|No| FIX[Fix software then repeat tests]
  FIX --> T
  T -->|Yes| REF[Build and refine classical reference; vary L, mesh, guesses, tolerance]
  REF --> RQ{Reliable same-case reference available?}
  RQ -->|No| RD[Diagnose reference; retain unverified-reference status]
  RQ -->|Yes| N[Initialize direct five-output NN for one seed]
  RD --> N
  N --> A[Adam: equations and boundaries together; finite checks; monitor and checkpoint]
  A --> AQ{Finite loss and gradients?}
  AQ -->|No| FAIL[Save failure; diagnose first bad term]
  AQ -->|Yes| SW{Declared switch or budget rule met?}
  SW -->|No| A
  SW -->|Yes| F[Freeze points, physical parameters, scales and loss weights]
  F --> L[L-BFGS with deterministic recomputing closure]
  L --> EV[Select checkpoint; fresh-point original residuals, BCs, fields and derivatives]
  EV --> ACC{Declared accuracy checks met?}
  ACC -->|No| BAD[Record failed validation; controlled diagnosis and new run]
  ACC -->|Yes| SEN[Repeat domain, sample density and seed checks]
  SEN --> SC{Sensitivity acceptable?}
  SC -->|No| BAD
  SC -->|Yes| OUT[Verified tested-case record and measured comparisons]
  BAD --> DONE[Save all outcomes including failures; no hidden retuning of final test]
  FAIL --> DONE
  OUT --> DONE
```

## One Adam update

```mermaid
flowchart TD
  Z[Clear old NN gradients] --> I[Fresh interior coordinate leaves]
  I --> N[NN fields and component derivatives]
  N --> R[Original R14 to R18]
  Z --> B[Fresh endpoint coordinates 0 and L]
  B --> BN[Same NN and required endpoint derivatives]
  BN --> BR[B1 to B12]
  R --> LOSS[Fixed scales and positive weights; one combined loss]
  BR --> LOSS
  LOSS --> FIN{All values finite?}
  FIN -->|No| STOP[Stop update and diagnose]
  FIN -->|Yes| BP[Backward through residual calculation]
  BP --> FG{Gradients finite and connected?}
  FG -->|No| STOP
  FG -->|Yes| STEP[Update NN weights]
  STEP --> MON[Independent monitoring and checkpoint at declared intervals]
```

## Reference check is not training data

```mermaid
flowchart LR
  SPEC[Same approved equations and parameter case] --> REF[Twelve-state classical BVP]
  SPEC --> PINN[Physics-only PINN training]
  REF --> COMP[Independent comparison]
  PINN --> COMP
  COMP --> OUT[Errors, wall quantities, domain sensitivity and status]
```

There is no arrow from reference values to the training loss. A separate supervised or reference-pretrained variant would need a different diagram and an explicit report of data use.
