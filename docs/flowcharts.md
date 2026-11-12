# Flowcharts

The browser-readable [strategy guide](../strategy-guide.html) renders these flows as labeled steps and branches. Mermaid source below is editable documentation, not code that trains a network.

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
