# Phase 0 freeze note — `ltro-phase0-v1`

**Freeze date:** 2026-08-23  
**Git tag:** `ltro-phase0-v1`
**Commit:** resolve with `git rev-parse ltro-phase0-v1^{commit}`
**Config:** [`configs/frozen/ltro_phase0_v1.json`](../configs/frozen/ltro_phase0_v1.json)

This freeze is the timestamped boundary for confirmatory work. It was recorded **before** Phase 1 v2 collapse-gating results were inspected.

## Locked method: LTRO-fixed (not adaptive dual)

Phase 0 compared adaptive dual radii \(\eta\in\{0.05,0.07,0.14\}\) against fixed-\(\lambda\) and off. **No adaptive dual arm passed the P0.3 operational criteria** (controller floor/ceiling band, late \(D_Z/\eta^2\in[0.5,2]\), geometry grads not dominating task grads). Dual-tight had late \(D_Z/\eta^2\approx 0.92\) but kept \(\lambda\) above its floor on nearly every update and showed geometry-gradient dominance. Therefore adaptive dual control did **not** provide stable operational value on cartpole-pixels under the predeclared rule.

The frozen recipe is **LTRO-fixed**:

- CTRO (MICo + PL) with `--dz-enabled`
- fixed \(\lambda=1\)
- `dz_adapt=false` (no dual \(\lambda\) updates)
- \(\eta=0.14\) retained only as documented below

## Critical: how \(\eta\) enters the loss

The live objective term is **not** a hinge. In [`src/agents/ctro.py`](../src/agents/ctro.py):

\[
L \;=\; L_{\mathrm{CTRO}} \;+\; \lambda\,(D_Z-\eta^2).
\]

With frozen \(\lambda=1\), \(-\eta^2\) is a constant and has **no gradient**. Training is exactly

\[
L \;=\; L_{\mathrm{CTRO}} \;+\; D_Z.
\]

**Freeze description for manuscripts and reviewers:** fixed geometry penalty with coefficient \(\lambda=1\); \(\eta=0.14\) is retained only as a **monitoring threshold** for logging \(D_Z/\eta^2\) and for continuity with dual-controller diagnostics. It does not gate or scale the geometry gradient under this freeze.

(If a future variant used \(\lambda\max(0,D_Z-\eta^2)\), \(\eta\) would control activation; that is **not** the frozen code.)

## \(D_Z\) and reference scale

- Definition: reference-scaled log-distance \(D_Z^{\log}\) in [`src/losses/dz_trust_region.py`](../src/losses/dz_trust_region.py), \(\delta=0.01\).
- \(s_{\mathrm{ref}}=\mathrm{median}\|z_i-z_j\|/\sqrt{d}\) computed **once** at reference-buffer initialization and then frozen for the run. Refresh of the reference buffer does **not** recompute \(s_{\mathrm{ref}}\).
- Degenerate \(s_{\mathrm{ref}}<10^{-8}\) at initialization **raises** (fail fast). The optimizer never silently replaces a degenerate scale with a later “healthy” scale.
- Eval/diagnostic path: if a PPO run cannot form \(s_{\mathrm{ref}}\) because latents have already collapsed, metrics log saturated collapse (\(C_{0.01}=1\), etc.) without writing a new scale into the store and without using that scale in a geometry loss (PPO has no \(D_Z\) term).

## Environment lock

- Repo pin: [`requirements.txt`](../requirements.txt)
- Bridges runtime used for Phase 0/1 jobs: `/ocean/projects/cis260223p/rbelaire/envs/causal-rep` with `module load pytorch/…` / CUDA as in the Slurm scripts.

## Companion artifacts in this freeze

- Selection report: [`docs/phase0_calibration_report.md`](phase0_calibration_report.md)
- Radius selection JSON: `results/dmcontrol_pixels/phase0_radius_selection.json` (on-disk; not required in git)
- Phase 1 threshold **rules** (values unset until healthy PPO v2 finishes): [`configs/frozen/phase1_collapse_thresholds.json`](../configs/frozen/phase1_collapse_thresholds.json)
- Gate / fit scripts: `src/experiments/select_dz_radius.py`, `fit_phase1_collapse_thresholds.py`, `analyze_phase1_collapse_gate.py`
- Eval logging fix: eval checkpoints always log even when outside the step logger interval (`performance_runner.py`)
