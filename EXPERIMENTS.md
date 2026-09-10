# Experiment program (E0–E7)

Every experiment states a claim, a baseline, a budget, metrics, and a
falsification criterion. Historical MiniGrid 2×2 / DMControl CTRO catalogs
remain in `docs/` and `NOTES.md`; they are not the v1 success package.

AC-MICo is a sampled surrogate, not exact bisimulation. v1 uses future
rewards. The target is reachable-support control sufficiency. Batch
\(\widehat D_{Z,\infty}^B\) is not population \(D_{Z,\infty}\). Policy KL has
latent null directions. PL is a critic-landscape property.

| ID | Claim | Baseline | Falsifier | Entry |
|---|---|---|---|---|
| E0 | Optimizer split + new modules off matches vanilla PPO and anti-aliased PPO | Pre-split MiniGrid configs | Regression beyond ordinary seed variation | `exp_e0_vanilla`, `exp_e0_aa_ppo` |
| E1 | Same-action MICo/Q separates action aliases | Random-pair MICo | No gain on action-alias toy | `exp_e1_action_alias` |
| E2 | Replay target ensemble disagreement predicts later TD/pair error | Scalar \(V\) | Disagreement uncorrelated with later error | `exp_e2_ensemble_calibration` |
| E3 | Query \(U\) finds discriminating actions faster | Matched uniform / entropy | No sample-efficiency gain | `exp_e3_query_coverage` |
| E4 | Confidence-filtered positives merge nuisances without false merges | Unfiltered diversity | False merges or no merge | `exp_e4_nuisance` |
| E5 | Relational metric detects KL-null geometry change | Policy KL | Cannot distinguish shear from benign head change | `exp_e5_kl_null` |
| E6 | Full stack improves return **and** action-conditioned probe error | Dual E0 locks | Only exploration stats improve | `exp_e6_ablation` |
| E7 | Optional named visual/context shift | Stated shift only | No benefit on that shift; no broad causal claim | deferred |

Run E0–E5 on MiniGrid/toys before Procgen. Report per-seed results, intervals,
wall-clock, action coverage, and component ablations for any benchmark claim.

GPU MiniGrid/Procgen jobs are not part of the CPU toy landings. Toys must run
on CPU and terminate.

## Submit (Bridges-2)

CPU toys first, then MiniGrid GPU array (3 arms × 3 seeds), then CPU aggregation.
Procgen is not in this pipeline.

```bash
bash src/experiments/jobs/submit_active_ctro.sh
# toys only:
TOYS_ONLY=1 bash src/experiments/jobs/submit_active_ctro.sh
# MiniGrid + agg if toys already passed:
MINIGRID_ONLY=1 bash src/experiments/jobs/submit_active_ctro.sh
```

Results:

- toys: `results/active_ctro/toys/summary.json`
- MiniGrid: `results/active_ctro/minigrid/{exp_e0_vanilla,exp_e0_aa_ppo,exp_active_ctro}/seed_{N}/`
- table: `results/active_ctro/minigrid/tables/e0_e6_summary.txt`

Local CPU toys (no GPU):

```bash
CUDA_VISIBLE_DEVICES= python -m src.experiments.run_active_ctro_toys
```
