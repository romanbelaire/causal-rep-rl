# Phase 1 ALE completion note (preliminary scaffold)

**Status:** Stage A gate done (Phoenix qualifies). Stage B Phoenix reduced matrix incomplete (job **44570971** timed out; 2/18 finals).

**Freeze:** `ltro_phase0_v1` / tag `ltro-phase0-v1` — unchanged; do not retune on ALE return.

**Protocol:** see `docs/moalla_ale_protocol.md` (hash stamped in each ALE `config.json`).

## Stage A

- Job IDs: **44279818** + resumes **44487871** / **44570605** (smoke **44279817**)
- Games: Phoenix, NameThisGame
- Stress dial: epochs 4 vs 16; seeds 42–44; 100M steps
- Gate output: `results/ale/phase1_ale_stage_a_gate.json`
- Result: Phoenix **qualified** (3/3 seeds); NameThisGame **not** (return rose under stress)

## Stage B

- Qualifying games: `ALE/Phoenix-v5` only
- Job script: `src/experiments/jobs/p1_ale_stage_b_matrix_s.sh`; credit-reduced run **44570971** tasks 6–23 (PFO/CTRO/LTRO; PPO from Stage A)
- Methods: PPO (Stage A) / PFO / CTRO / LTRO-fixed × {4,16} × {42,43,44}
- Analysis: `src/experiments/analyze_phase1_ale_stage_b.py` (needs Stage A PPO paths or symlinks; blocked until cells finish)
- Figures: `src/experiments/plot_phase1_ale_stage_b.py`
- Primary interaction:
  \(I=(R_{\mathrm{LTRO}}-R_{\mathrm{PPO}})_{\mathrm{stress}}-(R_{\mathrm{LTRO}}-R_{\mathrm{PPO}})_{\mathrm{standard}}\)
- Label: **preliminary** (3 seeds; stress=16 not 32)
- No representation-protection superiority claim without PFO unit checks + full Stage B

## Figures (to produce after Stage B)

1. Return curves standard vs stress (PPO)
2. Cumulant NMSE over training
3. Geometry diagnostics panel
4. Method comparison under stress
5. Interaction \(I\) bar with seed points

Phase 2 starts only after this note is filled and the freeze tag is unchanged
(bugfix → new version id).
