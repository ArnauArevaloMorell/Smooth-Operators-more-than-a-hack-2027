# Smooth Operators — Technical document

**More than a Hack 2027, Phase 1: ball possession.** One label per tactical-video frame (`H` / `A` / `D`) from the automatic tracking (ATD) and Smart Tagging.
Code: <https://github.com/ArnauArevaloMorell/Smooth-Operators-more-than-a-hack-2027> (branch `solution`).

## 1. Approach, and why

`spellF1` scores whole possessions by temporal overlap, so we treat possession as **segmentation**, not per-frame classification: a per-frame classifier invents a possession with every flicker.

```
ATD ─► per-frame features ─► per-frame models ─┐
          │                                     ├─► emissions ─► 4-state Viterbi ─► H / A / D
          └─► temporal CNN ensemble (±51 s) ────┘        ▲
Smart Tagging ─► restart priors, weak phase prior ───────┘
```

* **Features** (metres, seconds, per half; team-level, since the ATD has no identities): ball–player distance and velocity agreement per team, control episodes, ball visibility and out-of-play location, players' tempo *relative to the half's median*, team shape (advancement, stretch, movement to goal), contests and abrupt ball-velocity changes (touches).
* **Per-frame models:** a dead-ball MLP, ownership and shape logistic models, and a **switch-hazard** MLP that predicts *where* a change of possession is plausible.
* **Temporal CNN** (the main evidence): 251 features at 5 Hz, eight dilated residual blocks (dilations 1–128, each output sees ±51 s), 64 channels, outputs P(home), P(away), P(dead). **Ensemble of 10 networks**: five plain, five that also learn a training-only second output (which team has a player within 1.5 m of the ball in the *clean* tracking). Trained with team-swap augmentation, extra loss weight at possession changes, and read at test time from both teams' sides. Trained in PyTorch, shipped as weights run with **numpy only**.
* **Decoder:** a 4-state Viterbi (`H`, `A`, dead-after-`H`, dead-after-`A`) over each half. It pays a cost for every possession change (lowered where the hazard model sees a touch), so the number of possessions and each boundary come out of the evidence, which is what IoU matching rewards.

Why the effort went into *ownership*: on our development matches, giving the decoder perfect dead/alive raised spellF1 by ~0.06; giving it perfect team ownership raised it to ~0.93. And context decides ownership: whether a touch changed possession depends on the next seconds. That is why there is a ±51 s network.

## 2. Assets

| asset | used | how / why not |
|---|---|---|
| ATD (players, ball) | **yes** | everything is built on it |
| Smart Tagging, set pieces | **yes** | each window contains a restart: dead before, alive after, restarting side when tagged |
| Smart Tagging, phases | weakly | a small nudge only (weight 0.1): on BAR-ATH they agree with clear ball control 78% of the time |
| homography, tactical/broadcast video, broadcast ATD | no | the ATD is already in pitch coordinates; video ball detection was not validated in the time available |
| public 5-minute ground truth | check only | alignment and label meaning; never used to train, tune or choose versions |

## 3. Training data (none from the competition match)

* **Metrica Sports public sample games 1–3** (github.com/metrica-sports/sample-data): tracking plus human-annotated event data by Metrica, ~280 min. Frame labels are derived mechanically from the events (below).
* **DFL/IDSSE open data, 7 Bundesliga matches** (Bassek et al., *Scientific Data* 12, 195, 2025; CC BY 4.0): 25 Hz tracking with the provider's per-frame `BallStatus` and `BallPossession` flags, the same kind of source as the BAR-ATH ground truth. ~670 min. Possession flicks shorter than 1.5 s between two spells of the same team are merged back.
* **Simulated BAR-ATH noise:** the clean tracking is degraded to look like the ATD. Ball gaps are sampled from the ATD's own gap distribution (median ~11 s, taken from the organisers' ATD-derived `baseline_raw.csv`, no labels), alongside player dropout and track fragmentation, slow-drift position noise, and slowly varying ball offsets. Each match is degraded at light, medium and heavy settings (several draws each, plus an extra-heavy setting) and seen from both teams' sides.

## 4. Possession rules implemented

* A possession is a maximal run of one team with the ball alive; a stoppage (`D`) is never a possession, so `H D H` is two possessions.
* **Dead** from the moment the ball leaves the pitch, a foul is whistled or a goal is scored, until the restart is kicked. Carrying or placing the ball for a restart does not restart play.
* **Ownership changes only on a controlled action** by the other team: a pass, carry, shot, recovery or the start of a set piece. Challenges, aerial duels, deflections and touches without control do not change it; a loose ball keeps its last owner.
* **Restarts** always start a new possession; the restarting side comes from who is on the ball at the kick, the set-piece tag's team, and a learned prior.
* **Missing ball:** the decoder carries the current owner through ball gaps unless the players' movement, the shape models and the network say otherwise. "Ball not seen" alone is never read as a stoppage, because on BAR-ATH the ball is often lost in live play.

## 5. Results (held out: every match scored by models that never saw it)

Official `possession_metrics.py`, simulated BAR-ATH noise. Two networks of the shipped recipe per fold (the submission averages ten).

| development matches | starter `baseline_smooth50` | **ours** spellF1 / macroF1 |
|---|---|---|
| Metrica (3 matches × 6 noise draws) | 0.269 / 0.627 | **0.752 / 0.931** |
| DFL (7 matches × 2 draws; labels made like BAR-ATH's) | 0.216 / 0.523 | **0.658 / 0.921** |
| heavy noise only (closest to BAR-ATH): Metrica / DFL | 0.231 / 0.201 | **0.717 / 0.647** |

On BAR-ATH: 276 possessions (2.6 per minute, mean 12.4 s), 46% dead ball. The public 5 minutes give spellF1 0.675 and macroF1 0.934 (correctness check only).

**Adapting to the real match without labels.** Every run prints a health report (ball visibility, player speed, dead share, possessions per minute and more) and an evidence cross-check (how often each independent source agrees with the others), both next to their simulated range. Twice this exposed a gap between our simulation and the real ATD:
* the ATD is smoothed, so players look slower, and the first model read the whole match as dead;
* single-camera ball offsets are larger than we had simulated.

Both were fixed in the *simulation*, never by fitting to BAR-ATH.

## 6. Limitations and next steps

* **Very short possessions** (under 2 s; 30% of DFL's possessions) are found only about a quarter of the time. A one-touch interception rarely shows enough control to pay for two switches. Most of the remaining loss is here.
* **Ball gaps:** team accuracy inside the ATD's long ball gaps is lower than with the ball seen. The obvious next step is a ball detector on the tactical video (through the homography) to fill the gaps.
* **Domain shift:** the method is tuned on ten other matches with simulated noise. BAR-ATH shows fewer possessions per minute than any simulated case, so the real score may differ from the held-out estimate.

## 7. Reproduction

```bash
git clone https://github.com/ArnauArevaloMorell/Smooth-Operators-more-than-a-hack-2027 && cd Smooth-Operators-more-than-a-hack-2027
git checkout solution
pip install -r requirements-solution.txt        # numpy, pandas (pinned)
# unpack the data bundle into ./data (or set MTAH_DATA_DIR), see DATA.md
python -m solution.make_submission --team Smooth_Operators
#  -> evaluation/results/Smooth_Operators.csv, validated; sha256 e9459c8d54eca83d04f428cfc7052d85d74a8367127f21e8032527b495b78bed
```

The run is deterministic: the same inputs give the same file byte for byte, checked across machines. The models, network weights and constants are committed. The scripts that retrain them from public data only are in `dev/` (`train_models.py`, `train_seqnet.py`, `merge_nets.py`).
