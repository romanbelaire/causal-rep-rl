# Phase 1 ALE Moalla guide (locked program)

This file stores the revised Phase 1 experimental program used for the ALE
collapse-regime work. Cartpole pixel epoch sweeps remain calibration / ordinary-case
only and are not the primary collapse benchmark.

## Frozen method

- Config: `configs/frozen/ltro_phase0_v1.json`
- Tag: `ltro-phase0-v1`
- Do not retune LTRO on ALE return. Bug fixes require a new freeze id.

## Games

1. `ALE/Phoenix-v5`
2. `ALE/NameThisGame-v5`

## Stage A (PPO only)

| Game | Epochs | Condition | Seeds |
|---|---:|---|---|
| Phoenix | 4 | Standard | 42, 43, 44 |
| Phoenix | 16 | Stress | 42, 43, 44 |
| NameThisGame | 4 | Standard | 42, 43, 44 |
| NameThisGame | 16 | Stress | 42, 43, 44 |

Stress dial is **16 epochs** (deliberate compute choice vs Moalla-style stronger 32).
Escalate to 32 only if Stage A fails to qualify either game.

## Stage B

On each qualifying game: PPO, PFO, CTRO, LTRO-fixed × {4, 16} epochs × seeds 42–44.

## Primary hypothesis

Interaction \(I=(R_{\mathrm{LTRO}}-R_{\mathrm{PPO}})_{\mathrm{stress}}-(R_{\mathrm{LTRO}}-R_{\mathrm{PPO}})_{\mathrm{standard}}\)
should be positive when LTRO helps specifically under representation stress.

## Protocol detail

See `docs/moalla_ale_protocol.md` for transcribed Moalla / CleanRL hyperparameters,
wrappers, architecture, PFO coefficient, and metric definitions.
