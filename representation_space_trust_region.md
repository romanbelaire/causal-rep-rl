# Relational latent trust region

This document **supersedes** the older pointwise displacement

\[
D_Z(\theta',\theta)=\mathbb{E}\|Z_{\theta'}(s)-Z_\theta(s)\|^2
\]

and the log-ratio penalty implemented in `src/losses/dz_trust_region.py`
(legacy, off in the live recipes). Those quantities do not certify that
pairwise geometry is preserved, and they have latent null directions that
policy KL also misses.

## Finite-reference relational distortion

Let \(B=\{s_i\}_{i=1}^{n}\) be a **frozen** reference set. Let \(d_{\rm old}\)
and \(d_{\rm new}\) be pairwise distances in the snapshot encoder and the
candidate encoder. With floor \(\tau_d>0\):

\[
\widehat D_{Z,\infty}^{B}
=\max_{i\neq j,\; d_{\rm old}(s_i,s_j)\ge\tau_d}
\left|\frac{d_{\rm new}(s_i,s_j)}{d_{\rm old}(s_i,s_j)}-1\right|.
\]

Pairs with \(d_{\rm old}<\tau_d\) are excluded. Log the excluded fraction.
This is a finite-batch certificate, **never** population \(D_{Z,\infty}\).

## Modes

1. **Observe:** compute and log after representation updates.
2. **Hard gate:** snapshot encoder parameters **and** representation optimizer
   state; apply the candidate step; restore both if
   \(\widehat D_{Z,\infty}^{B}>\epsilon_z\).

PPO clipping / `target_kl` and this metric protect different objects.

## Limitations

- Depends on the choice of \(B\) and \(\tau_d\).
- Does not imply bisimulation or causal-factor recovery.
- Compensated encoder/head maps that leave policy logits fixed can still
  change this metric (that is the point of the KL-null toy).
