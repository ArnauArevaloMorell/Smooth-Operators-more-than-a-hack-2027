# Smooth Operators — Technical document

**More than a Hack 2027, Phase 1: ball possession.** One label per tactical-video frame, `H` / `A`
/ `D`, from the provided automatic tracking and Smart Tagging.

Reproduce the submission (details in [§7](#7-reproduction)):

```bash
pip install -r requirements-solution.txt
python -m solution.make_submission --team <team_name>
```

---

## 1. The approach in one paragraph

Possession is a *segmentation* problem, and the ranking metric (`spellF1`) scores whole
possessions by temporal IoU, so we never decide frames independently. We compute physically
meaningful per-frame evidence from the tracking — *is a player of either team on the ball*, *is
the ball in play*, *which team's collective shape looks like attacking* — turn it into scores
with three small models trained on **other matches**, and decode the whole half at once with a
four-state Viterbi segmentation (`H`, `A`, dead-after-`H`, dead-after-`A`) that pays a fixed cost
for every possession change. That cost is an L0 penalty on the number of possessions: the
possession count and each boundary's position come out of the evidence, which is exactly what
IoU matching rewards. Every free constant was chosen on Metrica's public sample games with
BAR-ATH-like degradation applied; nothing was fitted or tuned on BAR-ATH.

```
ATD (players by team, ball) ──► features (metres, seconds, per half)
                                  │  control:  ball–player distance × velocity agreement
                                  │  dead:     visibility, out-of-play location, tempo, ball speed
                                  │  shape:    collective movement & stretch, attack-oriented
                                  ▼
                 dead-ball MLP ─► logit      ownership logistic ─► logit
                                  ▼
Smart Tagging ──► restart / phase priors ──► emissions (T × 4) ──► Viterbi ──► H/A/D per frame
```

## 2. Why this, rather than something else

* **The ranking metric decides the architecture.** `spellF1` counts possessions as objects, so a
  per-frame classifier, however accurate, invents possessions with every flicker (the starter's
  four baselines show it). A global segmentation with a per-change cost cannot flicker, and puts
  each change where the evidence turns. We also tried replacing the decoder's team assignment by
  the literal rule "team of the last clear control episode": it agreed with the Viterbi output to
  within 0.003 spellF1, which tells us the decoder is not the bottleneck — the evidence is.
* **Ownership matters far more than dead-ball detection.** An oracle experiment on the sample
  games: giving the decoder perfect dead/alive raised spellF1 by ~0.06; giving it perfect team
  ownership raised it to ~0.93. Effort went where that says.
* **Possession is a collective, ball-centred question,** and the tracking has no identities, so
  every feature is team-level: nearest player *per team*, team spreads, team movement.
* **Small models, few constants.** The development set is three matches, so the models are a
  32-unit MLP and two logistic regressions, and the decoder has ~20 constants, all shared across
  matches. All of it is numpy, shipped as JSON, and deterministic.

## 3. Assets: used and dropped

| asset | used? | how / why not |
|---|---|---|
| tactical ATD (players, ball) | **yes** | everything is built on it |
| Smart Tagging — set pieces | **yes** | each tagged window contains a restart: dead before it, alive after it, restarting side when tagged |
| Smart Tagging — phases | **yes, weakly** | a small nudge toward the team a phase names (in-possession codes) or the other one (defending codes); matters mostly inside ball gaps |
| homography | no | the ATD is already in pitch coordinates; the homography is only needed to go back to pixels, i.e. to use the video |
| tactical / broadcast video | no | a better ball detector is the obvious next step (§6), but not in the time available |
| broadcast ATD | no | not synchronised to the tactical timeline; re-aligning it was judged poor value next to the ownership work |
| public 5-minute ground truth | **only to check correctness** | frames aligned, file well-formed, labels meaning what we think — never to choose between versions (RULES.md) |

## 4. Training and tuning data

**No labels of any kind on BAR-ATH were used**, by hand or otherwise. Everything learned or tuned
comes from:

* **Metrica Sports public sample data**, games 1–3 (<https://github.com/metrica-sports/sample-data>,
  anonymised professional matches with tracking and human-annotated event data, labelled by
  Metrica). ~280 minutes of play, ~1,100 possessions, ~280 dead-ball stretches.
* **Labels derived mechanically from those events** (`dev/sample_games.py`) following
  CHALLENGE.md: dead from the frame the ball leaves the pitch, a foul is whistled or a goal is
  scored, until the next set piece or kick; while alive, ownership changes only on a *controlled*
  action by the other side (pass, carry, shot, recovery, set piece, or the start of a ball-lost) —
  challenges, aerial duels and deflections never change it on their own. Actions with a dead ball
  (e.g. carrying the ball to the restart spot) never restart play.
* **Degradation to look like the BAR-ATH ATD** (`dev/degrade.py`). The sample tracking is clean,
  so we damage it: ball gaps injected into live play with lengths resampled from the empirical gap
  distribution of the BAR-ATH ATD (median ~11 s; taken from the organisers' public
  `baseline_raw.csv`, which is ATD-derived and carries no ground truth), half of them starting on
  fast balls; the ball made visible during some dead stretches (being carried back); player slots
  dropped in bursts; position noise. Three strengths (light / medium / heavy).
  *Calibration check:* the starter's `baseline_smooth50`, run on our degraded sample games, scores
  spellF1 0.25–0.28 and macroF1 0.61–0.63 (heavy/medium) — close to its published BAR-ATH
  full-match score (0.2433 / 0.6724). The proxy is realistic, if anything slightly harsh on the
  frame level.
* **Smart Tagging is simulated** on the sample games (they have none): set-piece windows around a
  random 55% of true restarts, opening 2.5–4.5 s early, team missing on 40% and wrong on 5%, plus
  false positives; phases as jittered (±2 s) sub-intervals of 60% of possessions longer than 3 s,
  with the wrong team 15% of the time.

Evaluation is **leave-one-game-out**: the models scoring game *g* never saw game *g*. Each game
is degraded six times (light / medium / heavy, two independent random draws each), so every
reported number and every tuning decision is an average over **18 cases** — a constant is only
kept if it helps across noise realisations, not on one lucky draw. The shipped models
(`solution/models.json`) are then trained on all of them.

## 5. The method in detail

### 5.1 Features (`solution/features.py`)

All in metres (pitch 105.3 × 68.0 m) and seconds; computed per half, so no window crosses
halftime.

* **Ball cleaning.** Isolated detections more than 4 m from their ±3-frame median are dropped
  (false positives); the track is then averaged over 5 frames before differentiating —
  differentiating raw detections turns a few tens of centimetres of noise into several m/s of
  velocity noise. This alone quadrupled clear-control coverage under heavy noise.
* **Control evidence** for each team: `exp(-d²/2σ²) · exp(-max(0, |v_ball − v_player| − v0)/vs)`
  for the team's nearest player — near the ball *and* moving with it. A pass flying past a
  defender is near but not moving with him.
* **Control episodes and last/next controller.** Frames where one team clearly controls (kernel >
  0.55 for one side, < 0.25 for the other, ≥ 3 frames) are 98–99% correct on the sample games.
  Every frame gets the team of the previous and of the next episode and the time since/until — the
  possession rule itself, measured: the ball belongs to whoever last controlled it, through passes
  in flight and loose balls.
* **Dead-ball cues** at 1, 4 and 10 s, centred and one-sided: ball visibility; last and next seen
  ball position (outside the lines, near a touchline, corner, centre spot, six-yard box); time
  since/until the ball was seen; mean player speed, absolute and relative to the half's median
  (players walk during stoppages); ball speed; whether anyone is on the ball.
* **Shape** (no ball needed), oriented so `+` means "home looks like the attacking side": mean
  player velocity toward home's attacking goal, difference in team stretch (x and y), block
  position, and team line height. Attack direction per half comes from where each team's
  centroid sits on average.

### 5.2 Models (`solution/mlp.py`, `solution/logreg.py`)

* **Dead ball:** 1-hidden-layer MLP (32 tanh units) on the dead-ball cues. Leave-one-game-out
  log-loss 0.19–0.21 vs 0.21–0.25 for logistic regression.
* **Ownership:** logistic regression on control evidence at three radii × six windows
  (past-only, future-only and centred — the asymmetry between past and future is what pins a
  change of possession to a frame), last/next controller, and shape. Held-out frame accuracy
  ~0.90–0.92 on live frames, under degradation.
* **Shape-only:** logistic regression on shape alone (held-out ~0.87 with no ball at all) — kept
  for the decoder to fall back on in ball gaps.
* **Switch hazard:** MLP (32 units) predicting "the ball changes hands within ±5 frames of here"
  from the contrast between the last half-second and the next (control by each side, the last and
  next controlling team), from signs of a contest (both sides within 3 m of the ball) and from
  abrupt changes of ball velocity (a touch). It does not say *who* gets the ball — the ownership
  model does — only *where* a change is plausible.

### 5.3 Decoder (`solution/decode.py`, `solution/pipeline.py`)

Per frame, emissions for the four states: the two team states get `±½ · (w_c·(c_H − c_A) +
w_o·owner_logit + w_s·shape_logit)`, the two dead states get `w_d·(dead_logit + b)`, plus Smart
Tagging priors. Transition costs: `die` for alive→dead, `restart` for dead→alive, plus
`restart_same` when play restarts with the side that had the ball before the stoppage (tuned:
restarts more often go to the other side — throw-ins, goal kicks). The H↔A cost is **per frame**:
`switch · exp(−w_h · clip(hazard_logit + b_h))` — cheap where the hazard model sees a contest or a
touch, dearer where nothing is happening. A constant cost has to compromise between missing the
short possessions and inventing ones mid-possession; a per-frame cost does not. Exact Viterbi per
half; frames outside the halves are `X` (never scored).

**Smart Tagging priors.** For each tagged set piece, the restart instant is where the ball first
moves faster than 6 m/s inside the window (else the window centre); before it the dead states get
a bonus, after it the alive states — and the side the tag names, when it names one. Phases add a
small bonus to the side they imply; their weight is deliberately below the tuned optimum (§6).

### 5.4 The possession rules, as implemented

* **Loose balls and passes in flight** stay with the last team that controlled the ball (no
  evidence for either side ⇒ no reason to pay a switch).
* **A touch is not a change.** The other side must show enough control — near the ball *and*
  moving with it, for long enough — to pay the switch cost. A deflection or a won header with no
  follow-up does not.
* **Contested balls**: control is a *difference* between the teams, so a duel with players of
  both sides at the ball is evidence for neither.
* **Restarts** always start a new possession (the D stretch between them is a stoppage, as the
  metric defines it); the restarting side comes from who is on the ball when it is kicked, the
  set-piece tag's team, and the learned prior that it is usually not the side that had it.
* **Dead ball** is whatever the dead model and the restart structure say, never simply "the ball
  is not visible": on BAR-ATH the ball is seen being carried to restarts, and lost during play.

## 6. Results and limitations

Leave-one-game-out on the degraded sample games (official `possession_metrics.py`):

| | spellF1 | macroF1 |
|---|---|---|
| starter `baseline_smooth50`, same data | 0.277 | 0.631 |
| **this pipeline** | **0.614** | **0.893** |

| degradation (6 cases each) | baseline spellF1 | ours spellF1 | ours macroF1 | possessions predicted / true |
|---|---|---|---|---|
| light | 0.314 | 0.649 | 0.908 | 1451 / 2188 |
| medium | 0.280 | 0.617 | 0.896 | 1391 / 2188 |
| heavy | 0.239 | 0.575 | 0.874 | 1345 / 2188 |

How the score was built up (leave-one-game-out; the first rows on 9 cases, the last on 18):

| step | spellF1 |
|---|---|
| hand-made control kernel + linear dead model + Viterbi, untuned | 0.398 |
| tuned decoder constants | 0.56 |
| dead-ball MLP + ownership model with past/future windows | 0.58 |
| ball-track smoothing before differentiating | 0.585 |
| Smart Tagging restart priors, re-tuned | 0.59 |
| Smart Tagging phase prior at the robust weight (see below) | 0.601 |
| learned switch hazard (per-frame possession-change cost), re-tuned on 18 cases | **0.614** |

**Smart Tagging weights were stress-tested, not just tuned.** The tagging is automatic and its
error rate is unknown, so each prior was re-scored against tags much worse than simulated:

* *Phases:* the tuner wanted a weight of 0.58–0.75. Re-simulating phases as only 70% or 60%
  correct, such a weight *loses* up to 0.012 against no phases at all, while 0.2 gains +0.008 /
  +0.005 / −0.002 across the scenarios — so 0.2 ships.
* *Set pieces:* with the tag's team wrong 35% of the time and 40% false tags added, the tuned
  weights still score 0.608 against 0.603 without set-piece tags — they cannot hurt much, so the
  tuned values ship.

With no feedback before the deadline, a setting that cannot hurt much is worth more than one that
wins the simulation.

**Where it fails, and what we would do next:**

* **Very short possessions** (< 2 s, a quarter of all possessions in the sample games' event data)
  are still mostly missed: a one-touch interception rarely shows enough control to pay for two
  switches, even at the lower cost the hazard model allows there.
  Whether the BAR-ATH ground truth has as many of them is unknown (its public 5 minutes have 2 of
  17), so we did not push the switch cost down to chase them.
* **Ball gaps.** Inside the ATD's long gaps (median 11 s) only team shape speaks, and a change of
  possession there is found ~20% of the time. The single biggest improvement available is a
  better ball: run a ball detector on the tactical video through the homography to fill the gaps.
* **Domain shift.** Everything is tuned on three other matches with simulated noise; the
  simulation is calibrated (§4) but not identical. The possession count the pipeline reports on
  BAR-ATH is the first thing to sanity-check.

## 7. Reproduction

```bash
git clone <this repository> && cd <it>
pip install -r requirements-solution.txt            # numpy, pandas -- pinned
# unpack the gated bundle into ./data (or set MTAH_DATA_DIR), see DATA.md
python -m solution.make_submission --team <team_name>
#   -> evaluation/results/<team_name>.csv, validated with the official checks; prints its sha256
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
python evaluation/run_evaluation.py                 # public 5 minutes: a correctness check only
```

`make_submission` also prints a **health report**: aggregate statistics of the run (ball
visibility, tracked players, share of frames with a clear controller, dead-ball share, possessions
per minute, …) next to the range each took on the 18 sample-game cases, flagging anything outside
it — the first place to look if the real tracking differs from our simulation of it — and the
public-window scores as an alignment check.

Deterministic: same inputs → the same file, byte for byte (checked with and without the parse
cache). The models and constants are committed (`solution/models.json`, `solution/params.json`).
To regenerate *them* from scratch — from public data only:

```bash
bash dev/fetch_sample_data.sh          # Metrica public sample games -> data/sample-data/
python dev/evaluate.py --baseline      # the starter baseline on the degraded sample games
python dev/tune.py --rounds 3 --out solution/params.json
python dev/train_models.py             # -> solution/models.json
python dev/evaluate.py                 # leave-one-game-out scores
python -m solution.test                # loader / decoder / writer self-tests, no data needed
```
