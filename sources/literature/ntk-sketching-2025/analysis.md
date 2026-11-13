# Analysis: NTK weight convergence and sketching, 2025

**TL;DR.** Proves convergence for NTK weights under plain gradient descent and computes them cheaply by sketching. **Gives us:** affordable NTK weights: factor 2.5 cost, 17 block traces with no matrix storage, moving average alpha 1e-4 to 1e-3. **Avoid:** the theorem does not cover Adam or L-BFGS; the step bound shrinks with stiff residuals. **Relevance:** high (only if NTK weights are pursued).

## Record
- Title: Convergence and Sketching-Based Efficient Computation of Neural Tangent Kernel Weights in Physics-Based Loss (p. 1).
- Authors: Max Hirsch (UC Berkeley), Federico Pichi (SISSA) (p. 1).
- Year: 2025. Preprint, not peer reviewed. arXiv:2511.15530v1, math.NA, stamped 19 Nov 2025, title page dated 20 Nov 2025 (p. 1).
- Pages read: all 29 pages, from text extraction. No page was unreadable. Figure curves were taken from captions and surrounding text, not inspected pixel by pixel.
- Code: the authors give github.com/maxhirsch/Efficient-NTK, built on the PINA library (p. 1, p. 16).

## Problem
- Two questions (p. 2): does training with adaptive NTK-based loss weights converge at all, and can the weight updates be made cheap?
- Exact NTK entries cost grows with parameter count; the paper cites P around 10^7 for modern networks (p. 2). This is the cost problem for us too, since a fourth-order residual makes the Jacobian work heavy.
- The paper proves convergence, in a time-averaged sense, of plain gradient descent with NTK weights refreshed at every step (Section 3, p. 5-9). It then gives a randomized estimator so weights can be refreshed often at low cost (Section 4, p. 10-16).

## Method
- Setting: a residual vector R(θ) stacks PDE and boundary residuals at n collocation points; loss L = (1/2) R^T Λ R with symmetric positive definite weight matrix Λ (Section 2.1, p. 3-4).
- Weight scheme (Eq. 5, p. 5): for each loss block j, λ_j(θt) = Tr(K(θt)) / Tr(K_jj(θt)), where K = ∇R^T ∇R is the PINN NTK and K_jj is its diagonal block for that group. The balanced quantity is the per-block trace against the total trace. The scheme comes from Wang, Yu, Perdikaris (ref [37], p. 2).
- Motivation (p. 5): the singular values of K set residual decay rates. If Λ = Tr(K) K^-1 held exactly, every residual would decay at the common rate e^(-Tr(K) t). The trace ratios are a rough diagonal approximation of that scaled inverse.
- Convergence assumptions (Assumption 1, p. 6): convex open parameter domain; Λ(θt) and K(θt) symmetric positive definite with eigenvalues bounded above and below (kmin, kmax, ℓmin, ℓmax); the weighted gradient field G(θ; z) = (1/2) ||R(θ)||^2_{Λ(z)} has uniformly L-Lipschitz gradient; the unweighted loss has L-Lipschitz gradient. No loss convexity and no PDE linearity is assumed (Remark 2, p. 6). The lower bound kmin > 0 needs at least as many parameters as residual points, n ≤ p (p. 6).
- Optimizer covered: plain fixed-step gradient descent, θt+1 = θt − η∇R(θt)Λ(θt)R(θt) (Eq. 4, p. 5). Not Adam, not L-BFGS.
- Sketching estimator (Algorithm 2, p. 11): sample g ∼ N(0, I_n); take one throwaway predictor step θ̃ = θt + Δt ∇R(θt) g, set up through Λpred = diag(−g) diag(R)^−1; then (R(θ̃) − R(θt))/Δt ≈ K(θt) g by Taylor expansion. One sample gives the symmetrized outer product K̂ = ((R(θ̃)−R)/Δt) g^T. The trace estimate is the Hutchinson-style scalar g^T (R(θ̃)−R(θt))/Δt.
- Error (Theorem 3, p. 11; Theorem 4, p. 13): the K g estimate is unbiased up to mean error C n Δt and mean squared error C^2 n(n+2) Δt^2. With N averaged samples, the trace error squared is a Δt^2 discretization term plus ||K||^2_F / N; the full-matrix version adds a factor 2n(n+2). Rademacher vectors also work (Remark 5, p. 12); clipping negative entries lowers the error (Remark 6, p. 12-13).
- Cost (p. 10, p. 16): one extra forward plus backward for the predictor step, one extra forward for R(θ̃). The paper counts two extra forward passes and one extra backward pass per iteration, a factor 2.5 on training time, with no significant extra memory.
- Practical scheme (Algorithm 3, p. 16): average N samples once at initialization, then one sample per step folded into a moving average K̄t = (1−α) K̄t−1 + α K̂t, α ∈ (0,1]; weights from Eq. 5 with K̄ in place of K. As η → 0 the moving average tracks the true NTK better (p. 16).
- Benchmarks (Section 5, p. 16-24): a 1D steady second-order BVP uxx = −16π^2 sin(4πx), u(0)=u(1)=0, for the convergence rates; a small quadratic regression for the estimator error; the 1D wave equation; and a nonlinear 2D liquid crystal Q-tensor system with two network outputs (p. 3, p. 22-24).

## Key results (author-reported)
- Theorem 1 (p. 6): time-averaged squared residuals ≤ (||R(θ0)||^2 − ||R(θT)||^2)/(Tη) for η ≤ 2 kmin ℓmin / (L kmax ℓmax^2), so the average goes to zero.
- Theorem 2 (p. 8) and Corollary 1 (p. 9): time-averaged squared gradient norms of the weighted and unweighted losses go to zero under the trace-ratio weights of Eq. 5, with an extra kmax/kmin factor.
- Corollary 2 (p. 9): with periodic updates, convergence holds if an update is skipped whenever the accumulated λmax(ΔΛ)||R||^2 budget would exceed h(T), for any h = o(T).
- Theorem 5 (p. 15): a cheaper trace-only estimate from R(θt + g), g ∼ N(0, ε^2 I), has error bounds of order C(n+1)ε and C(n+1)^2(ε+ε^2), but gives no off-diagonal NTK entries.
- Numerics, 1D BVP (Section 5.1, p. 17-18): 4 collocation points, one hidden layer of 100 neurons, exact NTK, weights every step. Loss reached machine precision near epoch 200. The predicted O(T^-1) average decay appeared after about epoch 100, once the NTK eigenvalues stopped growing.
- Numerics, estimator (Section 5.2.1, p. 18-20): error decayed at the predicted O(N^-1) rate (Fig. 6, p. 20). A close full-NTK picture needed about 2×10^4 samples; 2000 samples were already qualitatively close (p. 19).
- Numerics, wave equation (Section 5.2.2, p. 21-23): 300 interior, 300 initial-derivative, 100 initial, 100 per boundary points, resampled each step; 3 hidden layers of 500 neurons; α = 10^-3. Estimated weights tracked exact NTK weights closely, with the largest weight under-approximated and the smallest over-approximated (p. 22); the trained PINN matched the exact solution (p. 23).
- Numerics, Q-tensor (Section 5.2.3, p. 22-24): adaptive weights changed little in solution quality versus a plain PINN, both close to a finite element reference. The claimed gain is feasibility of NTK weights at low cost, not accuracy.

## Adopt for our pipeline
- Scope check for our five-field system first. K = ∇R^T ∇R is symmetric positive semidefinite for any residual stack, coupled or not (Section 2.2, p. 4), and the theory allows any SPD Λ with bounded eigenvalues (Assumption 1.2, p. 6). So 17 loss components fit the frame. What can break: the kmin > 0 lower bound, which needs n ≤ p and residual gradients that are not strongly linearly correlated (p. 6), and the step-size bound, which shrinks as the spread of our 17 weights grows.
- Use the sketch estimator for weight denominators. For each of our 17 components j, estimate Tr(K_jj) by g_j^T (R_j(θ̃) − R_j(θt))/Δt, with g split into the same blocks as R (Algorithm 2, Eq. 12, p. 10-11). No n×n matrix is needed for weights, only 17 scalar traces, so memory stays at a few residual-sized vectors.
- Cost per weight refresh: one extra backward (the predictor) plus two extra forwards. The paper's factor 2.5 applies if run every step (p. 16). Refreshing every k steps during our Adam phase amortizes this; L-BFGS iterations are not covered by any theory here.
- Moving average with small α, paper values 10^-4 and 10^-3 (p. 20, p. 22), instead of exact per-step weights, plus an N-sample warm start at initialization (Algorithm 3, p. 16).
- Update schedule: our fixed-weight baseline is the spaced-update case of Corollary 2 (p. 9). Sparse refreshes are consistent with its h = o(T) budget rule.
- Smooth activation: Assumption 2.3 needs a smooth residual map, which the paper notes higher-order operators require anyway (p. 11). Tanh, as in their tests, fits our fourth-order equation.
- The finite-difference product K g ≈ (R(θ + Δt∇R g) − R(θ))/Δt works for any g (Remark 5, p. 12), so cheap column or diagonal probes of K are available for diagnostics.
- The estimator needs every residual entry nonzero, so diag(R)^-1 exists (Assumption 2.1, p. 11). Near-zero entries in our residuals are the practical failure mode; check before enabling.

## Do not import
- The convergence guarantee itself. It covers plain fixed-step GD under bounded-eigenvalue assumptions (Theorems 1-2, p. 6-8). Adam and L-BFGS are outside it, so our pipeline gets no theorem from this paper.
- The step-size bound η ≤ 2 kmin ℓmin / (L kmax ℓmax^2). It depends on the inverse condition numbers of K and Λ and may need to be very small, as the authors state (Remark 3, p. 7). A fourth-order residual makes kmax/kmin large.
- The two or three block split of Eq. 5 and Eq. 12 as-is (p. 5, p. 10). A 17-component split is a generalization the paper never tests.
- Full NTK reconstruction. It needs thousands of samples (p. 19). We only need block traces.
- Expectations of accuracy gains. On their hardest test the weights changed little in final quality (p. 24). Their evidence is about cost and stable weight trajectories, not accuracy.
- Resampling of collocation points as a required pairing. Their wave-equation weights estimate an expected NTK under resampling (p. 21). Our boundary points are fixed; treat interior resampling as a separate test.
- The word unbiased without the caveat: the weights are quotients of random traces, so estimated weights are biased even though the traces are unbiased up to O(Δt) (p. 22).
- The O(T^-1) rate as a prediction for our loss. Their rate test used 4 collocation points (p. 17) and the rate only appeared after the eigenvalues stabilized (p. 18).

## Quotes worth keeping
- "we prove that under appropriate conditions, gradient descent enhanced with adaptive NTK-based weights is convergent in a suitable sense." (p. 1, abstract)
- "the weighting parameters in Equation (5) can be thought of as a rough approximation to a scaled inverse of K" (p. 5)
- "there must be more parameters in the neural network than the amount of data used to train the model." (p. 6)
- "costs just two extra forward passes and one extra backward pass in the neural network optimization per iteration, effectively only multiplying the training time by a factor of 2.5." (p. 16)
- "even though the use of the adaptive loss weights makes little difference in the quality of solution for this example, we emphasize that the Algorithm 3 can offer enough improvement in computational efficiency to make such an adaptive NTK-based loss weighting feasible." (p. 24)

## Relevance
High: it supplies the piece our plan flags as missing, a cheap sketch-based estimator for NTK block traces that makes frequent NTK weight updates affordable for an expensive residual, with a first convergence result for the scheme as supporting theory.
