# Representation-Space Trust Region: Full Mathematical Specification

This document defines the complete mathematics for implementing a **representation-space trust region** (RST) for policy optimization. It includes:

1. The representation-space constraint
2. Gradient/Hessian thresholding
3. Trust-region update rule
4. Connection to the full theoretical bounding chain

The goal is to replace or augment classical KL-based trust regions (TRPO/PPO) with constraints directly operating on a learned causal representation \(Z(s)\).

---

# **1. Representation-Space Constraint**

Let
- \( Z_\theta(s) \in \mathbb{R}^d \) be the representation produced by an encoder parameterized by \(\theta\),
- \(\pi_\theta(a|s)\) be the corresponding policy.

We define the **representation-space distance** between the current parameters \(\theta\) and candidate parameters \(\theta'\):

\[
D_Z(\theta', \theta)
= \mathbb{E}_{s \sim d_{\pi_\theta}} \, \| Z_{\theta'}(s) - Z_{\theta}(s) \|^2.
\]

This is a quadratic penalty analogous to KL, but acting on **latent causal geometry**.

### **Local approximation**
For small steps:

\[
Z_{\theta'}(s) \approx Z_{\theta}(s) + J_Z(s)(\theta' - \theta),
\]
where
\( J_Z(s) = \frac{\partial Z_\theta(s)}{\partial \theta} \) is the representation Jacobian.

Thus,
\[
D_Z(\theta', \theta)
\approx (\theta' - \theta)^T \underbrace{\mathbb{E}[J_Z(s)^T J_Z(s)]}_{F_Z} (\theta' - \theta),
\]

where **\(F_Z\)** is the **representation-space Fisher metric**, analogous to TRPO’s KL-Fisher.

---

# **2. Gradient and Hessian Thresholding**

We take a standard surrogate objective:

\[
L(\theta') =
\mathbb{E}_{s,a \sim \pi_\theta} [
A_\theta(s,a) \, \frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}],
\]

and approximate it locally with first/second order terms:

\[
L(\theta') \approx L(\theta) + g^T(\theta' - \theta) + \frac{1}{2} (\theta' - \theta)^T H (\theta' - \theta).
\]

Where
- \(g = \nabla_\theta L(\theta)\),
- \(H\) is an optional Hessian or Gauss–Newton approximation.

To ensure numerical stability, we apply **gradient clipping**:

\[
g \leftarrow g \, \frac{\min(1, c_g / \|g\|)}{}
\]

and **Hessian spectral clipping**:

\[
H \leftarrow U \, \mathrm{diag}(\min(\lambda_i, c_H)) \, U^T.
\]

where \(U \Lambda U^T\) is the eigendecomposition of \(H\).

---

# **3. Representation-Space Trust Region Update Rule**

We solve the constrained optimization:

\[
\max_{\theta'} \; g^T(\theta' - \theta) + \frac{1}{2} (\theta' - \theta)^T H (\theta' - \theta)
\quad \text{s.t.} \quad
(\theta' - \theta)^T F_Z (\theta' - \theta) \le \delta_Z.
\]

This is a direct analogue of TRPO’s KL trust region.

### **Closed-form solution (natural-gradient style)**
When \(H\) is ignored (first-order step),

\[
\theta' = \theta +
\sqrt{\frac{\delta_Z}{g^T F_Z^{-1} g}} \, F_Z^{-1} g.
\]

### **Second-order constrained solution**
Solve the linear system:

\[
(H - \lambda F_Z) (\theta' - \theta) = g,
\]

with Lagrange multiplier \(\lambda > 0\) chosen to satisfy the trust-region boundary:

\[
(\theta' - \theta)^T F_Z (\theta' - \theta) = \delta_Z.
\]

This is exactly the TRPO constrained conjugate-gradient procedure, but substituting **the Z-Fisher matrix** \(F_Z\) for the KL-Fisher.

---

# **4. Relationship to the Full Bounding Chain**

Our theory establishes the chain:

\[
(J^* - J)
\le c_1 \|Z^* - Z\|
\le c_2 \|\nabla V(Z^*) - \nabla V(Z)\|
\le c_3 KL(\pi^* \| \pi).
\]

The representation-space trust region ensures:

\[
\|Z_{\theta'} - Z_{\theta}\|^2
\le \delta_Z,
\]
which, under local strong convexity of \(V(z)\), implies:

\[
\|\nabla V(Z_{\theta'}) - \nabla V(Z_{\theta})\|
\le C \sqrt{\delta_Z}.
\]

And by the gradient-to-policy bound:

\[
\| \nabla V_{\pi_{\theta'}} - \nabla V_{\pi_\theta} \|
\le c_3 KL(\pi_{\theta'}\|\pi_\theta).
\]

Thus choosing

\[
\delta_Z = O(KL)
\]

ensures that the model stays within the theoretical **upper bound on causal representation error**.

### **Interpretation**

- **KL trust region ensures policies stay causally stable.**
- **Representation trust region enforces this stability directly in the latent space.**
- This can replace or augment KL constraints.

---

# **Summary of Key Equations**

### Representation distance:
\[
D_Z(\theta',\theta) = \mathbb{E} \|Z_{\theta'}(s) - Z_\theta(s)\|^2.
\]

### Local quadratic approximation:
\[
D_Z \approx (\theta'-\theta)^T F_Z (\theta'-\theta).
\]

### Trust region constraint:
\[
(\theta'-\theta)^T F_Z (\theta'-\theta) \le \delta_Z.
\]

### Update rule:
\[
\theta' = \theta +
\sqrt{\frac{\delta_Z}{g^T F_Z^{-1} g}} F_Z^{-1} g.
\]

Or second-order constrained solve:
\[
(H - \lambda F_Z)(\theta'-\theta) = g.
\]

### Theoretical guarantee chain:
\[
(J^* - J) \le c_1 \|Z^* - Z\|
\le c_2 \|\nabla V(Z^*) - \nabla V(Z)\|
\le c_3 KL.
\]

This document contains everything needed for engineering implementation.

