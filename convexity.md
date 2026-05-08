Parameterization to enforce non-negativity (practical)

Instead of hard-clipping you should parametrize non-negative weights to allow unconstrained optimization:

Let unconstrained parameter 
U~k∈Rm×d
U
~
k
	​

∈R
m×d
. Define 
Uk=softplus⁡(U~k)
U
k
	​

=softplus(
U
~
k
	​

) or 
Uk=exp⁡(U~k)
U
k
	​

=exp(
U
~
k
	​

) (softplus is numerically stable).

Same for 
Wk,w
W
k
	​

,w: use 
Wk=softplus⁡(W~k)
W
k
	​

=softplus(
W
~
k
	​

), 
w=softplus⁡(w~)
w=softplus(
w
~
).

This guarantees elementwise non-negativity while allowing gradients to flow.

3) Making the ICNN 
μ
μ-strongly convex (global / local)

Convexity alone does not give a lower bound on the Hessian eigenvalues. To obtain 
μ
μ-strong convexity you need 
∇2V(z)⪰μI
∇
2
V(z)⪰μI (for all 
z
z in the region of interest). Practical, safe ways to enforce strong convexity:

(A) Add an explicit quadratic term

Define

V~(z)=VICNN(z)+μ2∥z∥22.
V
~
(z)=V
ICNN
	​

(z)+
2
μ
	​

∥z∥
2
2
	​

.

The 
μ2∥z∥2
2
μ
	​

∥z∥
2
 term contributes a constant Hessian 
μI
μI, so if 
VICNN
V
ICNN
	​

 is convex, 
V~
V
~
 is 
μ
μ-strongly convex globally.

Pros: trivial, numerically stable, exact strong convexity constant.

Cons: adds a bias term to the value — usually fine (you can compensate by learning a flexible ICNN offset), and you can scale 
μ
μ small so you get local strong convexity without overwhelming learned shape.

(B) Regularize the minimum eigenvalue of the Hessian (harder)

You can add a penalty that pushes the minimum Hessian eigenvalue above 
μ
μ in a neighborhood:

Lhess=λh Ez∼D[max⁡(0,  μ−λmin⁡(∇2V(z)))2].
L
hess
	​

=λ
h
	​

E
z∼D
	​

[max(0,μ−λ
min
	​

(∇
2
V(z)))
2
].

Estimating 
λmin⁡
λ
min
	​

 can be done approximately with the Lanczos method or inverse power iterations, and Hutchinson trace tricks let you approximate trace metrics. This is more complex and expensive.

Recommendation: use (A) unless you have a special reason not to. It’s simple, exact, and cheap.

4) Activation and architecture choices

Use ReLU or softplus for 
σ
σ. ReLU is convex & non-decreasing and cheap; softplus is smooth (useful if you want smooth Hessian estimates).

Depth: a few hidden layers (2–4) often suffice. Deeper nets are expressive but harder to train.

Width: proportional to latent dimension 
d
d. Keep hidden-to-hidden 
Wk
W
k
	​

 shapes consistent.

Skip connections: allowed if they preserve convexity (i.e., sums of convex functions are convex). For instance, you can have 
V(z)=α∥z∥2/2+w⊤hK(z)
V(z)=α∥z∥
2
/2+w
⊤
h
K
	​

(z) safely.

5) Training: losses / regularizers / diagnostics

Loss: train 
V
V as usual (critic loss / TD errors / MSE to targets) with the ICNN output. If adding 
μ2∥z∥2
2
μ
	​

∥z∥
2
, include it in forward pass.

Regularize for stability: gradient clipping, spectral normalization on unconstrained parts if needed.

Diagnostics: routinely sample 
z
z from your data and check

Hessian-vector products via autograd to spot negative curvature,

empirical 
∥∇zV(z)∥
∥∇
z
	​

V(z)∥ statistics,

minimal eigenvalue estimates (Lanczos on Hessian or power method on inverse).

Initialization: initialize 
W~k,U~k
W
~
k
	​

,
U
~
k
	​

 near zero so softplus weights start small and the 
μ
μ-quadratic dominates early; helps training.

6) PyTorch-style sketch (concise)