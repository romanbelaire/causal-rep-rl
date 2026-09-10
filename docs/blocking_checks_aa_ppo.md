# Blocking checks for anti-aliased PPO (B1–B4)

Date: 2026-08-16. Companion to [`next_round_experiments.md`](next_round_experiments.md)
and [`METHOD_CHANGELOG.md`](METHOD_CHANGELOG.md).

---

## B1 — Latent collapse / participation ratio

**Probe.** CPU diagnostic on E2 arm A checkpoint
`results/dmcontrol_pixels/exp_e2_pl_tilde/seed_42/cartpole-swingup`
(scale-corrected PL-only; nearest available trained encoder to the live method).

```bash
PYTHONPATH=. /ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin/python \
  -m src.experiments.diagnose_latent_geometry \
  --run-dir results/dmcontrol_pixels/exp_e2_pl_tilde/seed_42/cartpole-swingup \
  --n 512 --device cpu \
  --output-dir results/blocking_b1_e2_pl_tilde_s42
```

**Result.** `gate_b1_ok=true`. Participation ratio ≈ 7.62 (not near one). Pairwise
distance p05 ≈ 0.536, median ≈ 0.831, ratio p05/median ≈ 0.645. Full eigenvalue
spectrum and histogram written to
`results/blocking_b1_e2_pl_tilde_s42/geometry_report.json` and `geometry.png`.

**Decision.** Not collapsed on this checkpoint. Re-run B1 on the first
`anti_aliased_ppo` state-obs checkpoint before quoting E-A geometry (probe above
uses random pixel observations, not the frozen training ref batch).

---

## B2 — Denominator provenance (E2 four-arm)

**Verdict: reference-level denominator.** E2 job `e2_run3_pixels_cartpole_s.sh`
(43614235) logged `v_ref` / `train_v_ref` and value-gap floors (`f_floor_*`).
It did **not** use the earlier Bellman-residual denominator. Arm A
(`exp_e2_pl_tilde`) is the scale-corrected PL hinge on that denominator.

---

## B3 — Scale-correction provenance

**Verdict: scale-corrected statistics were logged.** E2 metrics include
`mu_pl_tilde_*`, `latent_pair_L`, and `train_pl_L`, with
`train_pl_scale_invariant=1` on PL arms. Endpoint median pairwise distances differ
sharply across arms (example seed 42: PL-tilde \(L\approx 18.5\), full CTRO
\(L\approx 0.85\), spectral \(L\approx 10.8\), MICo \(L\approx 2.5\)). Cross-arm
comparisons of raw \(\mu_{PL}\) or uncorrected distances are not interpretable;
use \(\tilde\mu\) or \(L\)-normalized separation.

---

## B4 — State PPO baseline budgets

Targets: cartpole-swingup > 800, walker-walk > 900, cheetah-run > 400.
Hopper-hop = unsolved (not a separator).

| Task | Checkpoint | Seeds | Final `eval_full_return_mean` | Gate |
|---|---|---|---|---|
| cartpole-swingup | `exp_baseline_v3` | 42, 43, 44 | 849 / 874 / 873 | **pass** |
| walker-walk | `exp_baseline_v3` | — | not run | **open** |
| cheetah-run | `exp_baseline_v3` | — | not run | **open** |

**Decision.** Cartpole state baseline meets the bar. Walker and cheetah must be
retrained under `exp_baseline_v3` (or equivalent matched PPO) and clear their
targets before E-B uses those tasks as separators. Until then, E-B may proceed on
cartpole alone for smoke, but the pre-declared multi-task headline requires all
three gates.
