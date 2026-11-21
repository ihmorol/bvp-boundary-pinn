# Flowcharts

The browser-readable [strategy guide](../strategy-guide.html) renders these flows as labeled steps and branches. Mermaid source below is editable documentation, not code that trains a network.

## Methodology overview

Same numbers as the [methodology diagram](../bvp-methodology.pdf) and [its guide](methodology.md).

```mermaid
flowchart TD
  S1[1. Locked case] --> S2[2. Wall coefficient sizes and loss scales]
  S2 --> S3[3. Software tests]
  S3 --> T{All tests pass?}
  T -->|No: fix the code| S3
  T -->|Yes| S4[4. First-order system with solve_bvp]
  S4 --> S5[5. Convergence checks: tolerance, mesh, L, guesses]
  S5 --> C{Converged?}
  C -->|Yes| REF[Reference curves, check only]
  C -->|No| UNV[Unverified reference: no accuracy claim]
  S5 -->|then train| S6[6. Point sets, half near the wall]
  S6 --> S7[7. Network and loss]
  S7 --> S8[8. Adam training]
  S8 --> S9[9. L-BFGS refinement]
  S9 --> F{All values finite?}
  F -->|No| STOP[Stop and save the first bad term]
  F -->|Yes| S10[10. Pick the checkpoint on monitor points]
  S10 -.->|next seed| S6
  S10 -->|after 3 seeds| S11[11. Final test vs reference]
  REF --> S11
  S11 --> A{Targets met?}
  A -->|Yes| S12[12. Sensitivity, then variants B1 to B6]
  S12 --> S13[13. Report]
  A -->|No: reported as failed| S13
```

## Lorenz-style BVP computational graph

Text version of the [architecture diagram](../bvp-pinn-architecture.pdf). It follows the Lorenz PINN's loop, changed for this two-ended BVP. See [the comparison](lorenz-to-bvp-methodology.md). One shared network, no separate trainable physics network.

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
