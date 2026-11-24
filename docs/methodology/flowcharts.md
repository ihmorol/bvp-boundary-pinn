# Flowcharts

Editable Mermaid text versions of the method, updated 2026-11-24 to the code in `src/bvp_pinn/`. The browser-readable [strategy guide](strategy-guide.html) renders them; rebuild it with `python3 -I tools/build_guide.py`. The one-page diagrams are in [methodology.md](methodology.md).

## Methodology overview

Same numbers as the [methodology diagram](figures/bvp-methodology.pdf).

```mermaid
flowchart TD
  S1[1. Locked case] --> S2[2. Wall coefficient sizes, fixed residual scales]
  S2 --> S3[3. Software tests: uv run pytest]
  S3 --> T{All tests pass?}
  T -->|No: fix the code| S3
  T -->|Yes| S4[4. First-order system with solve_bvp]
  S4 --> S5[5. Checks: tolerance, three solvers, L to 10 240, branch search]
  S5 --> C{Converged?}
  C -->|Yes| REF[Reference curves, check only]
  C -->|No| UNV[Unverified reference: no accuracy claim]
  S5 -->|then train| S6[6. Points stretched to the wall, monitor set apart]
  S6 --> S7[7. Network; B1 to B12 built in; loss R14 to R18]
  S7 --> S8[8. Adam, smooth start profile, log scale]
  S8 --> S9[9. Refine: L-BFGS, then Levenberg-Marquardt]
  S9 --> F{All values finite?}
  F -->|No| STOP[Stop and save the first bad term]
  F -->|Yes| S10[10. Accept or reject: error estimate and tail check]
  S10 -.->|next run| S6
  S10 -->|after 20 runs| S11[11. Test against the reference]
  REF --> S11
  S11 --> A{Same in all 5 seeds?}
  A -->|Yes| S12[12. Compare arms B1 to B3 against B0]
  S12 --> S13[13. Report]
  A -->|No: no consistent effect| S13
```

## One training step

Text version of the [architecture diagram](figures/bvp-pinn-architecture.pdf).

```mermaid
flowchart TD
  I[Training points eta, N by 1] --> X[Log input map]
  X --> NN[tanh network 40 x 4, five outputs]
  ST[Smooth start profile, fixed] --> NN
  NN --> BL[Boundary layer: B1 to B12 exact for any weights]
  P[Locked coefficients] --> BL
  BL --> J[Taylor jets: f to order 4, g theta phi chi to order 2]
  J --> R[R14 to R18, original equations]
  P --> R
  R --> S[Fixed scale s of eta, from eta and the locked values]
  S --> L[One scalar loss: mean of scaled squares]
  L --> FIN{Finite?}
  FIN -->|No| FAIL[Stop and save the failure]
  FIN -->|Yes| G[Backward pass to the network weights]
  G --> OPT[Adam, then L-BFGS and Levenberg-Marquardt with weights frozen]
  OPT -.->|new weights| NN
```

The locked coefficients are never in the optimizer's parameter list. In exact mode the boundary conditions hold by construction, so the loss holds only the equations.

## The four arms

```mermaid
flowchart LR
  B0[B0: one network on 0 to L] --> END[Same final objective and refine chain]
  B1[B1: four window networks with interface continuity] --> END
  B2[B2: one network, windows added from the wall, physics replay] --> END
  B3[B3: as B2, self-replay of frozen outputs] --> END
  END --> ACC[Acceptance rule, then evaluation]
```

Window edges 0, 0.5, 2, 6, 20 are pilot values until pilot P3 freezes them ([PINN_PLAN.md](../pinn/PINN_PLAN.md) section 2).

## After training: acceptance, then evaluation

```mermaid
flowchart TD
  RUN[Finished run folder] --> SC[Self check: decay powers, tail share, boundary errors]
  RUN --> EST[C++ engine: one Newton step from the PINN state]
  SC --> RULE{Acceptance rule in configs/pinn.json}
  EST --> RULE
  RULE -->|accepted or rejected, both kept| EVAL[bvp_pinn.evaluation: metrics, summary, failures]
  REFD[Reference A and B] -.->|read only here| EVAL
  EVAL --> FIG[Figures FB1 to FB5, FX1; tables TB2, TX1]
```

There is no arrow from the reference to training or to acceptance. `src/bvp_pinn/pinn/` never imports the reference data or the evaluation package, and a test checks it.
