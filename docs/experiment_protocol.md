# Experiment protocol

Each of E0–E7 needs: claim, baseline, fixed budget, metrics, falsifier.
Toys run on CPU and exit. MiniGrid/Procgen GPU jobs are recorded in `NOTES.md`
with job IDs.

Do not present policy KL as a latent-geometry certificate, nor PL as a pairwise
anti-aliasing certificate. Do not claim external-shift causal robustness
without a named shift (E7).

Primary metrics: action coverage, same-action pair counts, positive/negative/
undecided fractions, `D_hat_pair` / `D_hat_lower`, ensemble Bellman-target
variance, \(\widehat D_{Z,\infty}^B\) and excluded-pair fraction, PPO clip
fraction / approximate KL / return, optimization-ownership deltas, and
action-conditioned probe error on the toys.

Gates G0–G5 in `docs/reward_test_pseudometric.md` and the Stage 5 matrix must
pass before a new full active MiniGrid run is treated as evidence. Incremental
rows: `python -m src.experiments.exp_matrix_row --row N --smoke` (CPU) or
`bash src/experiments/jobs/submit_active_ctro_matrix.sh` (GPU, post-gate).

## Pipeline

1. CPU toys: `src/experiments/jobs/active_ctro_toys_s.sh` (RM-shared). Now includes T1–T5.
2. Shared-ref diagnostics: `python -m src.experiments.exp_diag_reference_ownership` (CPU).
3. MiniGrid Unlock GPU (legacy three-arm): `src/experiments/jobs/active_ctro_minigrid_s.sh`.
4. Stage 5 matrix GPU (post G0–G5): `src/experiments/jobs/active_ctro_matrix_s.sh`.
5. CPU aggregate: `src/experiments/jobs/active_ctro_agg_s.sh`.

Orchestrator: `bash src/experiments/jobs/submit_active_ctro.sh`.
Record job IDs in `NOTES.md` at submit and again when the array finishes.
