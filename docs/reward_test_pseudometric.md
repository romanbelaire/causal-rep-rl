# Reward-test pseudometric (v1)

Permitted claim:

> We estimate an action-conditioned reward-test pseudometric through
> self-supervised action experiments and use a finite, confidence-filtered
> separation constraint to reduce false latent aliases on observed support.

Names that must not be overloaded:

| Symbol | Code / log key | Status |
|---|---|---|
| \(D_U\) | (theory only) | Ideal interventional reward-test pseudometric |
| \(\widehat D_{\mathrm{sig}}\) | `D_hat_sig` | Empirical pseudometric on fixed signatures |
| \(\widehat D_{\mathrm{pair}}\) | `D_hat_pair` | Same-action AC-MICo/Q score; not a metric |
| \(\widehat D^-\) | `D_hat_lower` | Conservative LCB used only for negatives |
| \(\widehat D_{Z,\infty}^B\) | `D_Z_inf_B` | Batch relational distortion |

`RewardTestSignature` and `D_hat_sig` are diagnostic only. They do not enter
the MiniGrid training loss in v1.

`L_sep` trains only on confident negatives (`w_neg > 0`). Undecided pairs are
excluded. A batch covariance eigenvalue floor is a secondary collapse diagnostic,
not a substitute for `L_sep`.

PL remains the AA-PPO landscape condition. Target-landscape PL
(`target_mu_pl_*`) is logged on a frozen encoder/critic and must not be
confused with the live hinge.

Not permitted without new assumptions/evidence:

- `D_hat_pair` is a metric or exact bisimulation metric
- reward-only tests identify a full feedback latent MDP
- PL alone prevents latent aliasing
- finite-reference relational distortion is a population worst-case guarantee
- MiniGrid transfer demonstrates robustness to arbitrary interventions
