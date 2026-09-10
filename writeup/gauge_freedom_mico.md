# Gauge freedom between causal and value representations

**Status:** methodology note (text only). Consolidates the design rationale for keeping MICo in CTRO; not a results paper.

## Terms

- **Latent summary \(Z\)**: an intermediate code of the observation shared by the policy and value function, \(\pi(a\mid Z)\) and \(V(Z)\).
- **Causal structure** (informally): distances / neighborhoods in \(Z\) that reflect how states relate under reward and transition — not just any convenient embedding for predicting \(V\).
- **Value geometry**: how \(V\) changes over \(Z\) (gradients, curvature proxies such as \(\mu_{PL}\)).
- **Gauge freedom**: leftover freedom in the pair \((Z, V)\) after training: you can reparameterize or stretch the coordinates of \(Z\), and fold inverse transforms into the value (and policy) heads, without changing the *behavioral* map from observation to action/value. Different gauges can look fine on returns or on raw \(V\), yet disagree on “what \(Z\) is for.”

## The problem we care about

CTRO needs one shared \(Z\) for three jobs at once:

1. **Policy** acts in \(Z\).
2. **Value** is defined on \(Z\) (and we monitor value-geometry health via \(\mu_{PL}\)).
3. **Causal / metric structure** on \(Z\) (bisimulation-style) — so representation error and trust-region reasoning about \(Z\) mean something beyond a disposable critic feature.

If we only couple \(Z\) to the value path (e.g. PL / non-vanishing \(\mu_{PL}\)), the optimization can still choose a gauge that is convenient for \(V\) but poorly identified as a causal state summary: dimensions can be rescaled, rotated, or mixed with nuisance features so that value looks healthy while the latent’s geometry is not fixed by the *dynamics/reward metric* we intend. That is the **causal–value gauge mismatch**.

Conversely, if we only impose structure on \(Z\) without a value link, the latent can look structured without remaining a good place for the policy / value geometry (the role we separately stress-test with PL and with latent-no-link controls).

So the two losses address different failure modes; neither alone pinches the full story.

## Role of each CTRO piece

CTRO objective (schematic):

\[
L_{\mathrm{CTRO}} = L_{\mathrm{PPO}} + \alpha\, L_{\mathrm{MICo}} + \beta\, L_{\mathrm{PL}}.
\]

### MICo (structure / metric on \(Z\))

**What it does.** MICo is an approximate bisimulation (metric) loss: pairs of states should have latent distances that track reward disagreement plus discounted next-state distances (angular / diffuse form in our implementation). Implementation: `src/losses/mico.py`.

**Why it is in the method (gauge story).** MICo anchors \(Z\) to a *task metric* induced by rewards and transitions, not only to whatever features make \(V\) easy. That reduces the freedom of reparameterizations of \(Z\) that leave \(V\)’s fit intact but scramble “causal” neighborhoods in \(Z\). In short: **MICo pinches the causal gauge of \(Z\)**.

**What it is not.** MICo alone is not our claim for protecting return via value geometry. In the Minigrid 2×2 ablation (`EXPERIMENTS.md`), **MICO_ONLY** barely moves \(\mu_{PL}\) or return; it is the right negative control for “structure without value link.”

### PL (value link)

**What it does.** PL penalizes small value-head PL ratios over \(Z\) (gradient of \(V\) wrt \(Z\) relative to Bellman residual). It keeps value geometry over the shared \(Z\) from collapsing. Implementation: `src/losses/pl_coupling.py`.

**Why it is in the method.** Pinches the **value** side of the pair: non-vanishing \(\mu_{PL}\) on the same \(Z\) the policy uses. Empirical story in Minigrid: PL-on cells raise \(\mu_{PL}\) and return relative to baseline / mico-only.

**What it is not.** PL does not, by itself, force \(Z\) to respect a bisimulation-style metric; a gauge that is great for \(V\) can still be weakly tied to dynamics/reward geometry.

### Together

| Piece | Pinches | Failure mode if missing |
|-------|---------|-------------------------|
| MICo | causal / metric structure of \(Z\) | value can look good while \(Z\) remains gauge-free w.r.t. reward-transition geometry |
| PL | value geometry of \(V\) over \(Z\) | structured \(Z\) without a stable value link (and weak \(\mu_{PL}\)) |
| Shared \(Z\) + policy-on-\(Z\) | same coordinates for \(\pi\) and \(V\) | architecture mismatch (baseline vs CTRO stacks is a separate confound) |

**Design commitment:** keep **both** \(\alpha\) and \(\beta\) for full CTRO. Ablations (mico-only, pl-only, latent-nolink \(\alpha=\beta=0\)) exist to separate roles, not because one loss replaces the other.

## How this relates to “latent no link”

`exp_latent_nolink` uses the same stack (shared \(Z\), policy-on-\(Z\)) with \(\alpha=\beta=0\). That is **architecture without either pin**: not a pure gauge experiment, but the natural “shared \(Z\) alone is not enough” control. Full CTRO re-adds both pinches.

Gauge-vs-performance experiments that compare baseline / nolink / CTRO should still report both return and \(\mu_{PL}\) (and optionally PR), with the interpretation:

- \(\mu_{PL}\) mainly tracks the **value** pin (PL / value geometry).
- MICo is for **identifying \(Z\)** as a metric/causal summary; it will not always show up as a big \(\mu_{PL}\) gap by itself (consistent with MICO_ONLY).

## Implementation pointers

| Item | Location |
|------|----------|
| CTRO agent | `src/agents/ctro.py` |
| MICo loss | `src/losses/mico.py` |
| PL loss | `src/losses/pl_coupling.py` |
| Experiment narrative / ablations | `EXPERIMENTS.md` (Minigrid 2×2; performance negative controls) |
| Minigrid ablation table | `plots/ctro/ablation_table.txt` |

## One-paragraph methods blurb (copy-ready)

We train a shared latent \(Z\) used by both \(\pi\) and \(V\). Two extra losses address **gauge freedom** in that pair: **MICo** constrains distances in \(Z\) to track an approximate bisimulation metric (reward and transition structure), so \(Z\) is not free to reparameterize away from causal neighborhood geometry while the value head absorbs the transform; **PL** constrains the value geometry of \(V\) over the same \(Z\) (non-vanishing PL ratio / \(\mu_{PL}\)). Ablations show that structure alone (MICo only) does not replace the value link for return, while the value link alone does not make the “pinch causal gauge” part of the design. Full CTRO uses both.

## Open documentation gap (closed by this note)

Previously, repo writeups motivated MICo only as “structure” and PL as “value link,” without naming **causal–value gauge freedom** as the reason MICo stays in the methodology. This note is the text-only consolidation of that rationale.
