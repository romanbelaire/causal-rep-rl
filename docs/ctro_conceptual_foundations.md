# Conceptual foundations: aliasing, value geometry, and what the method is doing

> **Stale relative to the live method (2026-08-16).** Sections that motivate MICo and
> \(D_Z\) as part of the *method* are historical. The live recipe is anti-aliased PPO
> (PPO + one scale-corrected PL hinge). See [`METHOD_CHANGELOG.md`](METHOD_CHANGELOG.md)
> and [`next_round_experiments.md`](next_round_experiments.md).

Reference document. Written in plain language with the formal statements marked where
they will be needed. Companion to `dz_trust_region_proofs.md` (the mathematics) and
`next_round_experiments.md` (the runs; supersedes the experiment section of
`dz_experiment_guide.md`). Tier 0 checks against this document are recorded in
`t0_findings.md`.

---

## 1. Collapse and aliasing are two different things, and one causes the other

**Aliasing** is the failure that matters. Two situations that the world treats
differently end up looking the same to the agent. If the agent cannot tell them apart it
cannot act differently in them, so any policy it learns is wrong in at least one of them.

**Collapse** is one specific route to aliasing. If the representation has shrunk to a
handful of effective dimensions, it does not have enough room to keep everything
separated, and aliasing becomes unavoidable.

**The implication runs one way.** Collapse forces aliasing. Aliasing does not require
collapse: a representation can have plenty of dimensions available and still map two
particular situations to nearly the same point.

> Rank is how many distinct colours are on the palette. Aliasing is whether the two
> things you needed to tell apart actually got different colours. A small palette
> guarantees you eventually run out. A large palette guarantees nothing, because you can
> still paint two different things the same colour with most of the palette unused.

### Why this resolves the tension in the narrative

Moalla and Lyle identified collapse, which is a real pathology and a real route to
aliasing. But because collapse is only *one* route, measuring rank catches that route and
misses the others. This predicts exactly the two anomalies observed:

- A successful policy can have low rank, if the few dimensions it retained happen to
  separate the situations that matter.
- A policy with healthy rank can be poor, if it aliases anyway despite having room.

Rank is therefore a proxy for aliasing, and a lossy one in both directions. That is why
it fails as an optimization target, and why participation ratio failed to separate arms
in the five-seed comparison while closest-pair distance separated them on every seed.

**Narrative order:** the literature identified a symptom; we identify the underlying
failure; we show why the symptom is an unreliable signal for it; we constrain the thing
that actually matters.

---

## 2. What the PL condition is

Picture the value estimate as a landscape over representation space, with height equal to
value. Learning is trying to climb it.

**The PL condition says that anywhere you are well below the summit, the ground beneath
you is steep** — the further below the top, the steeper, in a fixed proportion.

$$\|\nabla f\|^2 \;\ge\; 2\mu\, f, \qquad f = (\text{height of summit}) - (\text{height here}).$$

What it forbids is a flat plateau sitting at a bad height. Gradient-based learning
follows the slope, so a flat region partway up is where learning stalls without ever
discovering that a higher region exists.

The constant $\mu$ measures how strongly the condition holds. Large $\mu$ means steep
everywhere below the top and fast convergence; small $\mu$ means nearly flat somewhere
and learning crawls.

The payoff is a guarantee: when PL holds, gradient descent closes the remaining gap at an
exponential rate, even on a landscape that is irregular and not a simple bowl. That is
why PL is the right assumption — weak enough to be plausible of a neural network, strong
enough to guarantee learning works.

---

## 3. What "PL holds" means precisely

PL is a property of **a function together with a geometry**, never of a representation by
itself. Making the claim precise requires naming three things.

**Which function's landscape.** Two candidates, and they are not the same:

| | What it says | Status |
|---|---|---|
| PL on the *learned critic* $V_\psi$ over $Z$ | The network's current landscape has no plateaus | This is what $\mu_{PL}$ measures and what $D_Z$ protects |
| PL on the *optimal value* $V^*$ over $Z$ | Given this encoding, the true target is learnable | This is the property that makes a representation good |

The second is what you want; the first is what you can measure. A complete argument needs
a claim that the measurable one tracks the meaningful one. **We do not currently have
that claim, and it should be recorded as an open gap.**

**Over which region.** The reachable set rather than all of representation space. The
fixed reference batch approximates it.

**With respect to which distance.** Straight-line (Euclidean) distance, per §1 of the
proofs document.

---

## 4. How PL relates to causal structure

### 4.1 The negative result

Causal sufficiency — the representation containing everything needed to predict what
interventions do — cannot by itself give PL.

The reason is simple once seen. If a representation contains enough information, then any
relabelling of it that loses nothing still contains enough information. Information is
preserved under relabelling. But relabellings stretch and squash the landscape, and
stretching a landscape can flatten it.

> **Proposition (informal).** Sufficiency is invariant under injective reparameterization.
> The PL constant is not. Therefore no property expressible purely in terms of the
> information content of $Z$ can lower-bound $\mu$.

The formal version uses the $\operatorname{diag}(1,c)$ counterexample from Proposition 1
of the proofs document, which is an injective map degrading $\mu$ to $1/c^2$.

**This is the theorem form of the intuition that a good encoding of the world is not by
itself worth anything.** It is also the argument for why a separate geometric mechanism
must exist at all.

### 4.2 Why MICo does not supply PL — the two bounds face opposite directions

A caveat to §4.1: the negative result applies to bisimulation as an *equivalence
relation*, which is purely informational. MICo assigns *distances*, and distances are
geometry, so MICo is not covered by the proposition and does constrain shape.

So why does MICo alone not help? Because of what it constrains.

Bisimulation-style distances have the property that the difference in value between two
situations is *at most* their bisimulation distance. Castro et al.'s MICo proposition
states this for the **diffuse fixed point** $U^\pi$:
$|V^\pi(x)-V^\pi(y)| \le U^\pi(x,y)$. That is an *upper* limit on steepness **with
respect to $U^\pi$**, not automatically with respect to Euclidean (or angular) distance
in the embedding $Z$.

The catch, confirmed in Tier 0 against the paper (see `t0_findings.md` §T0.2): the
quantity our encoder distances approximate is the **reduced** MICo distance
$\Pi U^\pi = U^\pi - \tfrac12 U^\pi(x,x) - \tfrac12 U^\pi(y,y)$, and the value upper
bound does **not** transfer to $\Pi U^\pi$ in general. So the clean slogan
“MICo upper-bounds how fast value may change with latent distance” is true only if
“distance” means the diffuse potential the loss fits ($U_\omega$ / $U^\pi$), not
$\|Z_i-Z_j\|$.

PL remains a *lower* limit on steepness of $V$ over $Z$.

$$\underbrace{\text{MICo on }U}_{\text{value changes no faster than }U} \qquad\text{versus}\qquad \underbrace{\text{PL on }Z}_{\text{value changes no slower than a set rate over }Z}$$

**The two terms still face opposite directions**, but they do not literally sandwich the
same geometric quantity unless the upper bound is kept on $U$ rather than on embedding
distance. A representation can satisfy MICo’s embedding fit and still be the flat
pathology PL excludes. Condition-number targeting (§6) must not be drafted against
embedding distance until this is resolved.

---

## 5. How you optimize for a property

PL is a property, not a knob. You cannot optimize it directly; you optimize a computable
**surrogate** and hope its optimizers coincide with the property holding. That is what
the $\mu_{PL}$ hinge does, and the question is whether the surrogate is faithful.

### 5.1 The surrogate has a degenerate solution

$$\mu_{PL} = \frac{\|\nabla_Z V\|^2}{2f}$$

There are two ways to raise this quantity: make the critic steeper, or make $f$ smaller.
The first has a cheap and destructive route.

**Contracting the latent inflates $\mu_{PL}$.** If every latent is multiplied by $c < 1$
while the value assigned to each *situation* is unchanged, then the value as a function of
the *latent coordinate* becomes steeper by a factor $1/c$, so $\|\nabla_Z V\|$ grows and
$\mu_{PL}$ grows by $1/c^2$. The values did not improve and the landscape did not become
better conditioned. The ruler shrank.

Contraction is exactly what brings situations closer together. **So a naive PL penalty,
optimized on its own, pushes toward aliasing — the failure the method exists to prevent.**

### 5.2 Which is a third, independent reason the terms are needed together

Two mechanisms block the contraction route, and they block it differently:

- **MICo** pins latent distances to a fixed target metric, so contraction raises the MICo
  loss. This is an absolute guard on scale.
- **$D_Z$** limits how fast distances may change, so rapid contraction is prevented even
  where the target is loose. This is a rate limit.

This is a distinct argument from §4.2. There, the terms bound value steepness from
opposite sides. Here, one term closes a loophole in the other's surrogate. Both arguments
conclude that the terms are non-redundant, by different routes.

### 5.3 A prediction that can be checked for free

If §5.1 is correct, then in a PL-only configuration the latent should visibly contract:
falling pairwise distances, rising $\mu_{PL}$, and return that does not follow. In a
MICo-plus-PL configuration it should not.

**The existing Minigrid two-by-two ablation already contains this comparison.** Tier 0
(T0.1) separated the cells and recomputed endpoint median pairwise distance on a shared
probe buffer (`t0_findings.md`). On the Minigrid VAE stack the prediction did **not**
hold: PL-only raised both $\mu_{PL}$ and return without contracting relative to baseline
or FULL. That is consistent with the VAE prior suppressing scale collapse, and moves the
decisive test to non-VAE stacks (E3 / arm A of E2).

### 5.4 The better fix: make the surrogate scale-invariant

Rather than relying on other terms to guard a leaky surrogate, remove the leak. The
problem is that $\mu_{PL}$ has units of inverse length squared and therefore responds to
the size of the ruler. Multiplying by a characteristic latent length $L$ — the median
pairwise distance on the reference batch — cancels it:

$$\tilde\mu_{PL} \;=\; \frac{\|\nabla_Z V\|^2 \, L^2}{2f}.$$

Under $Z \to cZ$ the numerator falls by $c^2$ and $L^2$ rises by $c^2$, so
$\tilde\mu_{PL}$ is unchanged. Contraction no longer buys anything, and the quantity
being optimized is the one intended: how much the value changes over a characteristic
distance, rather than over an arbitrary unit.

For full dimensionlessness, divide additionally by a value scale such as the range of $V$
over the reference batch. Worth doing, since it also makes $\mu_{PL}$ comparable across
tasks with different reward magnitudes — which the current definition is not, and which
matters as soon as results are pooled across a suite.

### 5.5 Other routes to making PL hold

Penalizing a surrogate is one route. Two others are worth recording:

- **Architectural.** Choose a parameterization in which PL holds by construction. The
  input-convex critic explored in earlier program material is an instance: convexity
  implies PL under mild conditions. The cost is expressiveness.
- **Reshaping the target rather than the geometry.** PL concerns a function over a space.
  Value normalization and reward shaping change the function; encoder training changes the
  space. Only the second has been used so far.

### 5.6 The role of MICo, in brief

MICo has been performing three roles: it blocked latent contraction, it supplied an upper
bound on how fast value may change, and it defined which situations must be kept apart.
The scale-invariant diagnostic of §5.4 makes the first role redundant, and the second role
can be obtained more cheaply from spectral normalization or a gradient penalty on the
value head. The third role is irreplaceable. Without a bisimulation target nothing
specifies which situations are causally distinct, so the instruction not to alias them has
no referent, and $D_Z$ would merely preserve faithfully whatever arbitrary geometry
initialization happened to produce.

---

## 6. The mechanism improvement suggested by all of the above

If one term supplies an upper limit on steepness and the other a lower limit, then the
landscape is sandwiched between two bounds and what governs learning is **the ratio
between them** — a condition number for the value landscape over the representation.

Currently the two bounds are enforced separately, with independent weights $\alpha$ and
$\beta$ and nothing coordinating them. Targeting the ratio directly would replace two
weights tuned against each other with a single interpretable quantity, and would let the
method be stated in one sentence.

This should be checked against the precise definitions before being adopted, since the
bisimulation value bound carries conditions and MICo's diffuse construction complicates
the upper side. It is nonetheless the first available change that would *simplify* the
method rather than add to it.

---

## 7. Open gaps recorded

1. No claim yet that PL on the learned critic tracks PL on the optimal value (§3).
2. Nothing derives that PL holds in the first place; §4.1 explains why no informational
   condition can supply it, which makes this a structural gap rather than an oversight.
3. The value-drift cross-term in the $D_Z$ theorems (§5 of the proofs document).
4. The sandwich claim of §4.2: checked against Castro et al. (T0.2). Prop 4.8 bounds
   $|V^\pi(x)-V^\pi(y)|$ by the diffuse $U^\pi$, not by reduced / embedding distance
   $\Pi U^\pi$. §4.2 rewritten accordingly; §6 remains open until an upper bound is
   stated on a quantity we actually optimize jointly with PL.