# Run 1 pilot — pre-declared success criteria

Pilot scale: `dmcontrol_pixels` / cartpole-swingup / seed 42 / 1.5M steps.
Job script: `src/experiments/jobs/dz_run1_pilot_s.sh`.

## Arms

| id | exp | clip | \(D_Z\) | \(\eta\) |
|---|---|---|---|---|
| 0 | `exp_dz_run1_stress` | off | absent | — |
| 1 | `exp_dz_run1_eta0.101` | off | on | 0.101 (calibrated under clip) |
| 2 | `exp_dz_run1_eta0.32` | off | on | 0.32 (\(\eta^2\approx 0.10\)) |
| 3 | `exp_dz_run1_eta0.71` | off | on | 0.71 (\(\eta^2\approx 0.50\), early unconstrained scale) |

Reference distribution: Run 2 fixed arm `exp_dz5_fixed` seeds 42–46, pooled per-update `kl`
from `metrics.csv` (job 43539537).

| statistic | value |
|---|---:|
| median | 0.440 |
| p95 | 2.383 |
| p99 | 5.871 |
| late (epoch>600) median | 0.759 |
| Run 2 fixed last-50 return mean ± std | 163.0 ± 33.8 |
| Run 2 fixed last-50 return IQM | 163.3 |

## Outcomes (declare before launch)

**Explosion (negative).** Either:

1. run-wide median `kl` \(>\) Run2 p99 \(= 5.871\), or
2. fraction of logged updates with `kl` \(> 3\times\) Run2 p99 \(= 17.61\) exceeds 1%.

File as: latent trust region did not replace the policy trust region in this
configuration (or \(\eta\) too tight — check mid/loose arms before generalizing).

**Bounded-loose (positive middle).** Not an explosion, and:

1. run median `kl` \(>\) Run2 median \(= 0.440\), and
2. last-50 mean return \(\ge\) Run2 fixed mean \(-\) 1 std \(= 129.2\), and
3. stress control either explodes by the same KL rule, collapses return
   (last-50 \(< 80\)), or finishes with last-50 return at least 40 below the best
   Run-1 \(\eta\) arm.

Interpretation: \(D_Z\) controls policy movement more loosely than PPO clip but
sufficiently; clipping was doing work (stress fails relative to Run 1).

**Bounded-tight (strong positive).** Not an explosion, and:

1. run median `kl` \(\le\) Run2 p95 \(= 2.383\), and
2. last-50 mean return \(\ge\) Run2 fixed IQM \(= 163.3\), and
3. stress fails as in bounded-loose (3).

**Stress alone succeeds (claim collapses).** Stress is not an explosion and
last-50 return \(\ge 129.2\). Then PPO clipping was unnecessary here; a Run-1
"success" cannot be read as "\(D_Z\) replaces KL."

**\(\eta\)-ambiguity rule.** If only \(\eta=0.101\) explodes while \(\eta\in\{0.32,0.71\}\)
is bounded-loose or tight, do **not** conclude Run 1 fails — conclude the
clip-calibrated \(\eta\) is miscalibrated for the unclipped regime and expand
the mid/loose arm to five seeds.

## Theory expectation

Theorems 1–2 do not predict Run 1 success. Gauge freedom under \(D_Z\) (rotations
free; fixed head ⇒ policy change) cuts against it. A negative is consistent with
theory. A positive motivates a follow-up lemma on bounded policy KL under bounded
latent distortion + Lipschitz head + co-adapting critic — only after the
phenomenon is observed.
