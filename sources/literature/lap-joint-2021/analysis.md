# Analysis: Lap joint PINN code supplement, 2021

**TL;DR.** Code supplement only: two coupled fourth-order stress fields, chained derivatives, plain unweighted loss. **Gives us:** the chained autodiff pattern for f'''' and the one-term-per-BC loss assembly. **Avoid:** float32; the listings omit optimizer, sizes and accuracy. **Relevance:** medium.

## Record
- Type: publisher implementation supplement, code listings plus weight and bias histograms. It is not the main article.
- Authors: Shivam Sharma, Rajneesh Awasthi, Yedlabala Sudhir Sastry, Pattabhi Ramaiah Budarapu (p. 1).
- Year: 2021. Journal: not named in the supplement. DOI printed on p. 1: https://doi.org/10.1631/jzus.A2000403.
- Pages read: all 5 PDF pages. PDF p. 1 is the cover, p. 2 holds the abstract and Listings S1 to S3, p. 3 holds Listings S4 and S5, pp. 4 and 5 hold Figs. S1 and S2. Printed footers 2 to 4 sit on PDF pages 3 to 5.
- All listings are legible in both the text layer and the page images. Nothing was guessed.

## Problem
- Two axial stress fields u(x) and v(x) for a single lap joint, one coordinate input, two network outputs (Listing S2, p. 2).
- The equations appear only as code constants (Listing S2, p. 2). Each residual is fourth order in its own field and cross coupled: f_u = 5.4968e-9 u_xxxx + 1.3410e-9 v_xxxx + 1.4805e-6 u_xx + 1.4400e-6 v_xx + 0.00038407 u + 0.00038400 v - 0.00038400, and f_v mirrors it with 2.0119e-9 v_xxxx and 5.7390e-7 v_xx. No derivation or parameter names are given.
- BCs: eight terms in the loss (Listing S5, p. 3). Four endpoint values for u and v at both ends, plus four endpoint first derivatives whose targets are implicitly zero.
- Domain [lb, ub]. The numeric lb, ub and N_f never appear in the supplement.

## Method (from listings)
- Network: one input, tf.tanh hidden layers in a loop, linear output layer (Listing S1, p. 2). Depth and width are not stated. Figs. S1a-d (p. 4) track four weight matrices, weights through weights_3, which under Listing S1 means three hidden layers plus output. The captions say "hidden layers 1, 2, 3 and 4", so the depth is ambiguous and widths are never given.
- Input scaling: `A = 1.5*(X - self.lb)/(self.ub - self.lb)`. This maps [lb, ub] to [0, 1.5], not [-1, 1] (Listing S1, p. 2).
- Derivatives: chained tf.gradients, one order at a time, up to u_xxxx and v_xxxx (Listing S2, p. 2).
- Loss: plain sum of ten tf.reduce_mean(tf.square(...)) terms: four endpoint value errors, four endpoint slope errors, two equation residual MSEs. No weights or scaling on any term (Listing S5, p. 3).
- Sampling: Latin hypercube `X_f = lb + (ub-lb)*lhs(1, N_f)`, then sorted, reshaped to a column, saved to X_f.txt (Listing S3, p. 2).
- Init: Xavier stddev sqrt(2/(in_dim+out_dim)) with tf.truncated_normal, dtype tf.float32 (Listing S4, p. 3).
- Optimizer, learning rate and iteration count: absent from every listing. The iteration axes in Figs. S1e-h and S2e-h run to 200 (pp. 4-5). No batching appears anywhere, so the code reads as full batch.

## Key facts (from listings)
- One coordinate input, two outputs: `u = uv[:,0:1]`, `v = uv[:,1:2]` after one shared network call (Listing S2, p. 2).
- Fourth derivatives are computed and enter both residuals (Listing S2, p. 2, lines 14 and 18).
- Each residual contains the other field's fourth derivative, second derivative and value, so the coupling is real, not cosmetic (Listing S2, p. 2).
- net_f_uv also returns u_xx and v_xx, but the shown loss never uses them. They presumably feed stress post-processing in the main article (Listing S2, p. 2).
- Slope BCs are zero targets by omission: the loss squares `u0_x_pred`, `v0_x_pred`, `u_x_ub_pred`, `v_x_ub_pred` with no target subtracted (Listing S5, p. 3).
- Endpoint value targets live in stored tensors u0_tf and v0_tf, two entries each, indexed [0:1,0] and [1:2,0]. Their numeric values are not printed in the supplement (Listing S5, p. 3).
- Precision is float32 (Listing S4, p. 3).
- Histograms show weights staying within about plus or minus 0.10 to 0.15 across training (Figs. S1 and S2, pp. 4-5). Diagnostic only. No accuracy, error or timing numbers appear anywhere in the supplement.

## Adopt for our pipeline
- Chained autodiff for our f fourth derivative: build f, f', f'', f''', f'''' as sequential gradient calls, the Listing S2 pattern (p. 2). This is the direct precedent for our order-4 flow equation.
- One net with column-sliced outputs scales to our five outputs f, g, theta, phi, chi (Listing S2, p. 2).
- Tanh hidden layers with a linear output layer match our float64 tanh MLP plan (Listing S1, p. 2).
- Loss assembly as one term per condition: eight single-point BC terms appended to two residual MSEs (Listing S5, p. 3). Our 12 BCs extend the same way, one penalty tensor per condition at eta=0 or eta=L, with the coupled radiation condition as its own term.
- Run a weight-1 unweighted sum once as a baseline before our soft weights, to measure what the weights buy (Listing S5, p. 3).
- Save the collocation sample to a file for reproducibility, as they save X_f.txt (Listing S3, p. 2).

## Do not import
- float32 (Listing S4, p. 3): our near-wall terms grow like inverse powers of (eta+kappa) and a fourth derivative amplifies round-off, so we keep float64.
- Unweighted loss as the final setup (Listing S5, p. 3): our flux and nonlinear coupled radiation BCs sit at different scales, so weight 1 is a control only.
- Input scaling to [0, 1.5] (Listing S1, p. 2): nonstandard. tanh is centered at 0, so we keep [-1, 1] and may test their variant once.
- Sorting the LHS sample (Listing S3, p. 2): pointless for full batch and harmful for minibatches. We keep unsorted draws or a fixed grid.
- Their coefficients, 5.4968e-9 and the rest (Listing S2, p. 2): lap joint stiffness values with no bearing on our fluid constants.

## Quotes worth keeping
- p. 2, Listing S2: `u_xxxx = tf.gradients(u_xxx, x, name='u_xxxx')[0]`
- p. 2, Listing S1: `A = 1.5*(X - self.lb)/(self.ub - self.lb)`
- p. 3, Listing S5: `tf.reduce_mean(tf.square(self.u0_x_pred))`
- p. 2, Listing S3: `X_f = lb + (ub-lb)*lhs(1, N_f)`
- p. 2, abstract: "snippets of the python script indicating the key implementation steps and the histograms denoting the weights and biases during the network training"

## Relevance
Medium. The chained fourth derivative and the one-term-per-BC loss assembly map straight onto our f equation and 12 BCs, but the supplement omits optimizer, sizes, iteration counts and any accuracy data, so it guides structure, not tuning.
