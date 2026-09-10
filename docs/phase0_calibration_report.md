# Phase 0 calibration report

**Date:** 2026-08-23 (selection finalized)  
**Freeze identifier:** `ltro_phase0_v1`  
**Mode:** **LTRO-fixed** (frozen \(\lambda=1\), \(\eta=0.14\), `dz_adapt=false`)  
**Radius sweep job:** **44194678** (dual-tight / dual-medium complete)

## Selection rule (Experiment-guide P0.3)

Operational radius is chosen only if:

1. \(\lambda\) is at its upper bound for at most 10% of updates;
2. \(\lambda\) is above its lower floor for about 25–75% of updates (adaptive arms);
3. median late \(D_Z/\eta^2\) (final 20% of logged updates) is in \([0.5, 2]\);
4. geometry-gradient ownership does not dominate task gradients for most updates (`own_encoder_grad` > \(2\times\)(`own_actor_grad`+`own_value_grad`) on fewer than 50% of updates);
5. metrics are finite and ordinary learning is not obviously prevented.

**Return is never used to select \(\eta\).** If no dual arm passes, prefer fixed penalty (`dz_adapt=false`, \(\lambda=1\)).

Script: `python -m src.experiments.select_dz_radius`.

## Arms scored (2026-08-23)

| Arm | Exp | \(\eta\) | Status | Pass? | Late median \(D_Z/\eta^2\) | \(\lambda\) above floor | \(\lambda\) upper sat | Geom-dom rate |
|---|---|---:|---|---|---:|---:|---:|---:|
| dual-tight | `exp_dzb_dual_tight` | 0.05 | complete (5/5) | no | 0.92 | 0.98 | 0.00 | 0.74 |
| dual-medium | `exp_dzb_dual_medium` | 0.07 | complete (5/5) | no | 0.55 | 0.68 | 0.00 | 0.60 |
| dual-loose | `exp_dzb_fixed` | 0.14 | complete (5/5) | no | 0.14 | 0.29 | 0.00 | 0.72 |
| fixed penalty | `exp_dzb_penalty` | 0.14 | complete | fallback | 0.065 | 1.00 (frozen \(\lambda\)) | 0.00 | 0.87 |
| off | `exp_dzb_off` | — | complete | diagnostic | — | — | — | 0.67 |

### Decision

No adaptive dual arm passed P0.3. Dual-tight had the right late \(D_Z/\eta^2\) band but kept \(\lambda\) above the floor almost always and showed geometry-gradient dominance. Dual-medium was closer on the floor band but missed ratio and/or geom criteria across seeds. Per the guide fallback, Phase 0 locks **fixed penalty** at \(\lambda=1\), `dz_adapt=false`.

**Loss semantics (reviewer check):** the code implements \(\lambda(D_Z-\eta^2)\), **not** \(\lambda\max(0,D_Z-\eta^2)\). With frozen \(\lambda=1\), \(\eta\) does not affect gradients; training is \(L_{\mathrm{CTRO}}+D_Z\). The value \(\eta=0.14\) is retained only as a monitoring threshold for \(D_Z/\eta^2\) logs. See [`docs/phase0_freeze_note.md`](phase0_freeze_note.md).

Artifact: `results/dmcontrol_pixels/phase0_radius_selection.json`  
Frozen config: `configs/frozen/ltro_phase0_v1.json`

## Runtime overhead (P0.4)

**Production wall-clock (Stage B confirm job 44000037, 1.5M steps, V100-32):**

| Arm | Mean elapsed | vs off |
|---|---:|---:|
| off (`exp_dzb_off`) | 9.25 h | — |
| dual-loose (`exp_dzb_fixed`) | 8.87 h | −4.1% |
| fixed penalty (`exp_dzb_penalty`) | 8.47 h | −8.3% |

End-to-end \(D_Z\) wall-clock is within noise of off and under the 30% overhead target.

## Git freeze

Approve an intentional commit of the freeze config + selection artifacts, then tag `ltro-phase0-v1`. Do not tag a dirty unrelated tree.
