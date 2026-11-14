# Smooth Operators — Technical document

**More than a Hack 2027, Phase 1: ball possession.** One label per tactical-video frame, `H` / `A`
/ `D`, from the provided automatic tracking and Smart Tagging.

Reproduce the submission (details in [§8](#8-reproduction)):

```bash
pip install -r requirements-solution.txt
python -m solution.make_submission --team <team_name>
```

---

## 1. The approach in one paragraph

Possession is a *segmentation* problem, and the ranking metric (`spellF1`) scores whole
possessions by temporal IoU, so we never decide frames independently. We compute physically
meaningful per-frame evidence from the tracking — *is a player of either team on the ball*, *is
the ball in play*, *which team's collective shape looks like attacking* — and give it to a
**temporal convolutional network** that reads about ±25 to ±51 s of the match around each moment and
outputs, at every frame, the probability that home has the ball, away has it, or it is dead.
Those probabilities, together with a few direct cues and the Smart Tagging restarts, are decoded
for the whole half at once by a four-state Viterbi segmentation (`H`, `A`, dead-after-`H`,
dead-after-`A`) that pays a cost for every possession change — an L0 penalty on the number of
possessions, so the count and each boundary come out of the evidence, which is what IoU matching
rewards. Everything learned or tuned comes from **other matches** — Metrica's public sample games
and the DFL/IDSSE open Bundesliga data — degraded to look like BAR-ATH's automatic tracking;
nothing was fitted or tuned on BAR-ATH. The network is trained with PyTorch but shipped as plain
weights and run with numpy, so the submission needs numpy only.

```
ATD (players by team, ball) ──► features (metres, seconds, per half)
                                  │  control:  ball–player distance × velocity agreement, episodes
                                  │  dead:     visibility, out-of-play location, relative tempo
                                  │  shape:    per-team advancement, stretch, movement to goal
                                  │  touches:  contests, abrupt ball-velocity changes
                                  ▼
          temporal CNN (10-network ensemble) ─► P(home), P(away), P(dead) per frame
          + dead-ball MLP, ownership & hazard models (direct cues, per-frame switch cost)
                                  ▼
Smart Tagging ──► restart priors ──► emissions (T × 4) ──► Viterbi ──► H/A/D per frame
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
* **Context is the rest of the answer.** Per-frame models top out around 0.91 team accuracy under
  BAR-ATH-like noise, because whether a touch changed possession depends on what happens over the
  next seconds. A temporal CNN that sees ±25 s cut the held-out team errors by ~40% (0.914 → 0.952)
  and the dead/alive errors by half (0.929 → 0.963), and lifted held-out spellF1 on the
  human-labelled matches by +0.08 to +0.09 (§6).
* **Small models, few constants.** The development set is three matches, so the models are a
  32-unit MLP, logistic regressions and a 64-channel network, and the decoder has ~20 constants,
  all shared across matches. Inference is numpy only and deterministic.

## 3. Assets: used and dropped

| asset | used? | how / why not |
|---|---|---|
| tactical ATD (players, ball) | **yes** | everything is built on it |
| Smart Tagging — set pieces | **yes** | each tagged window contains a restart: dead before it, alive after it, restarting side when tagged |
| Smart Tagging — phases | **yes, weakly** | a small nudge toward the team a phase names (in-possession codes) or the other one (defending codes); matters mostly inside ball gaps |
| homography | no | the ATD is already in pitch coordinates; the homography is only needed to go back to pixels, i.e. to use the video |
| tactical / broadcast video | no | a better ball detector is the obvious next step (§6), but not in the time available |
| broadcast ATD | no | not synchronised to the tactical timeline; re-aligning it was judged poor value next to the ownership work |
| Metrica public sample games 1–3 | **yes, training & tuning** | other matches with tracking and human event data (§4) |
| DFL / IDSSE open data, 7 Bundesliga matches | **yes, training & tuning** | other matches with 25 Hz tracking and the provider's per-frame ball-status and possession flags (§4) |
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
* **DFL / IDSSE open data**: 7 full matches of the 2022/23 Bundesliga and 2. Bundesliga — Bassek,
  M. et al. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer",
  *Scientific Data* 12, 195, doi:10.1038/s41597-025-04505-y; files on figshare,
  doi:10.6084/m9.figshare.28196177, CC BY 4.0 (`dev/fetch_dfl_data.sh`, `dev/dfl_games.py`).
  Full-pitch optical tracking at 25 Hz plus the provider's per-frame `BallStatus` (alive/dead)
  and `BallPossession` (home/guest) flags — the same kind of answer the BAR-ATH ground truth is
  built from. The possession flag changes on every touch, so possessions shorter than 1.5 s that
  sit between two spells of the other team are merged back (a touch without control is not a
  change, CHALLENGE.md); that brings DFL to ~3.8 possessions per minute, the Metrica rate. DFL
  tracks the ball through every stoppage, unlike BAR-ATH, so the degradation hides it for all
  (45%) or the first part (25%) of each stoppage. ~650 minutes of play, labelled by the provider's
  systems.
* **Smart Tagging is simulated** on the sample games (they have none): set-piece windows around a
  random 55% of true restarts, opening 2.5–4.5 s early, team missing on 40% and wrong on 5%, plus
  false positives; phases as jittered (±2 s) sub-intervals of 60% of possessions longer than 3 s,
  with the wrong team 15% of the time.

Evaluation is **leave-one-group-out**: the models scoring a match never saw it, under any noise
draw. Folds are each Metrica game on its own and the DFL matches in two groups. Each Metrica game
is degraded six times (light / medium / heavy, two independent draws each) and each DFL match
twice (medium, heavy — the two settings that bracket BAR-ATH, §7): **32 cases**. Decoder
constants are tuned on the 26 medium and heavy cases. The shipped models are then trained on all
cases (`solution/models.json`, `solution/seqnet.npz`).

**A caveat we measured, not assumed.** Held out, the final pipeline scores ~0.95 spellF1 on the DFL
cases against ~0.67–0.74 on Metrica. DFL's possession flag is most likely generated by the
provider's own rule from the tracking, which a network can learn to reproduce; DFL scores are
therefore too flattering to forecast BAR-ATH, and **the Metrica rows (human-made labels) are the
numbers we quote**. DFL is still valuable *training* data: removing it from the network's training
costs −0.064 / −0.080 / −0.068 spellF1 on the held-out Metrica matches (light / medium / heavy).

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
* **Temporal CNN** (`solution/seqnet.py`, trained by `dev/train_seqnet.py`): the dead-ball,
  ownership and hazard feature designs (251 columns) sampled 5 times a second and standardised; a
  1×1 stem to 64 channels; seven or eight residual blocks `x + W₂·relu(dilated 3-tap conv(x))`
  with dilations 1, 2, …, 64 or 1, 2, …, 128 — each output step sees ~±25 s or ~±51 s; a 1×1 head
  to three logits (home, away,
  dead). Cross-entropy with steps within ±0.4 s of a label change weighted ×3 (spellF1's IoU is
  decided at the boundaries), AdamW, one-cycle schedule, 6,000 or 12,000 steps of 32 random 102-s crops. Halves
  are processed separately. Training data is doubled by **team-swap augmentation** — every case
  also appears with the two teams' roles swapped and H/A labels exchanged, so the network has no
  reason to prefer a side — and widened with an **extra-heavy** noise setting beyond BAR-ATH's on
  every axis and with nine more noise draws (three each of light, medium, heavy) of every DFL match
  and six of every Metrica match. At inference the network also reads the match with the teams swapped and the two
  readings are averaged (**test-time team swap**). Shipped as an ensemble of ten networks, all trained on that
  data: five that see ±51 s, trained for 6,000 steps (seeds 0–4), and five that see ±25 s, trained
  for 12,000 (seeds 5–9); their probabilities are averaged; numpy reproduces PyTorch's outputs to < 1e-3 (checked at
  export). Held out: team accuracy 0.952 (ball gaps 0.910), alive/dead accuracy 0.963 (before
  the swap augmentation).
* **Switch hazard:** MLP (32 units) predicting "the ball changes hands within ±5 frames of here"
  from the contrast between the last half-second and the next (control by each side, the last and
  next controlling team), from signs of a contest (both sides within 3 m of the ball) and from
  abrupt changes of ball velocity (a touch). It does not say *who* gets the ball — the ownership
  model does — only *where* a change is plausible.

### 5.3 Decoder (`solution/decode.py`, `solution/pipeline.py`)

Per frame, emissions for the four states: `w_seq · log P_net(state)` from the temporal CNN
(home, away, and dead for both dead states), plus the direct cues — the two team states get
`±½ · (w_c·(c_H − c_A) + w_o·owner_logit + w_s·shape_logit)`, the two dead states
`w_d·(dead_logit + b)` — plus Smart Tagging priors. With the network present, tuning drives the
shape weight and the set-piece team weight to zero: the network has absorbed them. Transition costs: `die` for alive→dead, `restart` for dead→alive, plus
`restart_same` when play restarts with the side that had the ball before the stoppage (tuned:
restarts more often go to the other side — throw-ins, goal kicks). The H↔A cost is **per frame**:
`switch · exp(−w_h · clip(hazard_logit + b_h))` — cheap where the hazard model sees a contest or a
touch, dearer where nothing is happening. A constant cost has to compromise between missing the
short possessions and inventing ones mid-possession; a per-frame cost does not. Exact Viterbi per
half; frames outside the halves are `X` (never scored).

**Smart Tagging priors.** For each tagged set piece, the restart instant is where the ball first
moves faster than 6 m/s inside the window (else the window centre); before it the dead states get
a bonus, after it the alive states. Phases add a small fixed bonus (0.1) to the side they imply:
with the network present, phase weights between 0 and 0.2 change held-out spellF1 by less than
0.001 whatever the phases' accuracy, so the smallest weight is the safest on a real match.

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

Held out (leave-one-group-out), official `possession_metrics.py`, final simulation (§7). The
**Metrica rows are the ones to read** (human-made labels; DFL scores are too flattering, §4):

| held-out cases | starter `baseline_smooth50` | per-frame models only (update 3) | **final, with temporal CNN** |
|---|---|---|---|
| Metrica light (6) | 0.321 / 0.646 | 0.663 / 0.909 | **0.819** (0.739 for the first network) |
| Metrica medium (6) | 0.255 / 0.628 | 0.630 / 0.895 | **0.792** (0.712) |
| Metrica heavy (6) — closest to BAR-ATH | 0.231 / 0.608 | 0.578 / 0.875 | **0.769** (0.670) |
| DFL medium / heavy (7 + 7) | 0.271 / 0.204 | 0.651 / 0.584 | 0.953 / 0.948 (flattering, §4) |

(spellF1 / macroF1; the final column is spellF1 with the shipped network recipe, held out on
the Metrica folds, four networks averaged -- two of each kind; the shipped file averages ten.) The starter baseline on the *heavy* simulation (0.231 / 0.608) sits next to its
published BAR-ATH full-match score (0.243 / 0.672), so the simulation is a faithful stand-in; the
heavy Metrica row, **~0.77 spellF1**, is our best estimate of the full-match result.

How the score was built (held out; each row on the simulation as it stood at that step):

| step | spellF1 |
|---|---|
| hand-made control kernel + linear dead model + Viterbi, untuned | 0.398 |
| tuned decoder constants | 0.56 |
| dead-ball MLP + ownership model with past/future windows | 0.58 |
| ball-track smoothing before differentiating | 0.585 |
| Smart Tagging restart priors | 0.59 |
| Smart Tagging phase prior at a robust weight | 0.601 |
| learned switch hazard (per-frame possession-change cost) | 0.614 |
| *simulation made harder to match BAR-ATH (§7)* | *0.582* |
| rich ball-free team-shape features, tuned on BAR-ATH-like noise | 0.618 |
| DFL matches added to training | 0.630 |
| temporal CNN, decoder re-tuned around it | 0.810 (Metrica heavy 0.578 → 0.670) |
| team-swap augmentation + extra-heavy noise + test-time team swap | Metrica mean 0.706 → 0.732, heavy 0.670 → 0.698 |
| + boundary-weighted loss | Metrica mean 0.739, heavy 0.702 |
| + three more noise draws per DFL match (light, medium, heavy) | Metrica mean 0.758, heavy 0.717 |
| + three further draws per DFL match and three per Metrica match | Metrica mean 0.767, heavy 0.727 |
| four times longer training (6000 steps instead of 1500) | Metrica mean 0.781, heavy 0.752 (two networks) |
| networks trained for 6000 and for 12000 steps, averaged | Metrica mean 0.784, heavy 0.760 (four networks) |
| + a third round of draws (three more per DFL and per Metrica match) for the 6000-step networks | Metrica mean 0.787, heavy 0.763 (four networks) |
| + the same third round for the 12000-step networks | Metrica mean 0.788, heavy 0.766 (four networks) |
| **the 6000-step networks see ±51 s instead of ±25 s (dilation 128)** | Metrica mean **0.793**, heavy **0.769** (four networks) |

Network variants we tested on the held-out Metrica matches (mean / heavy spellF1): team-swap
augmentation 0.721 / 0.682; extra-heavy noise 0.710 / 0.679; **per-match input normalisation
0.704 / 0.659 — rejected**; swap augmentation with test-time swap 0.731 / 0.692; with extra-heavy
noise as well 0.732 / 0.698 (best on heavy, the regime BAR-ATH is in); plus a boundary-weighted
loss 0.739 / 0.702 with weight 2 and 0.739 / 0.702 with weight 4 — the same gain from two
independent trainings, so it is not seed luck (weight 2 kept); plus three more noise draws of
every DFL match in training, 0.758 / 0.717, and three further draws per DFL match plus three per
Metrica match, 0.767 / 0.727 (training variety was the most reliable lever we found for the
network: three rounds, three consistent gains at every noise level -- the third is below). With that much more data the
network was under-trained at 1500 steps: two independent trainings each at 1500, 3000 and 6000
steps gave 0.767 and 0.762, then 0.769 and 0.772, then 0.777 and 0.781 (each 6000-step network
beats every shorter one), and the two networks averaged 0.769 / 0.733, 0.776 / 0.741 and
0.781 / 0.752. Most of the gain shows with the test-time team swap:
the longer-trained networks are more confident, and the two views of the match disagree in more
useful ways; a longer receptive field (±51 s, dilation 128) gave
0.744 / 0.705 at that stage, within seed noise (it paid later, below), and multi-offset averaging at inference +0.001 — not shipped. Re-scanning the
decoder's network weight and switch cost for the new outputs moved nothing beyond ±0.003.
Doubling again to 12,000 steps did not help a single network (0.773 and 0.778; the two averaged
0.781 / 0.758), but networks of the two lengths make different mistakes: one 6,000-step and one
12,000-step network averaged gave 0.782 and 0.783 against 0.781 for two of the same length, and
all four together **0.784 / 0.760** — shipped as five networks of each length. (Mixing 3,000- and
6,000-step networks gave no such gain, 0.780; the decoder re-scan on the four-network outputs
found +0.001, which leave-one-game-out did not confirm, so the decoder is unchanged.)
A third round of noise draws (three more per DFL and per Metrica match) gave a third consistent
gain: two 6,000-step networks scored 0.782 and 0.782 (against 0.777 and 0.781 without it), and
averaged 0.786 / 0.762 (against 0.781 / 0.752); with the two 12,000-step networks, 0.787 /
0.763. (A decoder re-scan on these outputs
again found no setting that leave-one-game-out preferred, 0.784 against 0.787, so the decoder is unchanged.)
The 12,000-step networks gained from the third round as well: 0.778 and 0.779 alone (against
0.773 and 0.778), 0.785 as a pair (against 0.781), and with the two 6,000-step networks **0.788 /
0.766** — shipped, five networks of each length, all on the same data. (Light noise went from
0.813 to 0.809 while medium and heavy rose; heavy is the regime BAR-ATH is in.) A fourth round of
draws gave nothing: 0.783 and 0.781 alone, 0.785 / 0.760 as a pair against 0.786 / 0.762, so
training variety has run its course for this network. The decoder re-scan on the new outputs
found +0.001 that leave-one-game-out did not confirm (0.785 against 0.788); decoder unchanged.
The longer receptive field, which gave nothing at 1,500 steps, paid once the network was trained
longer on all three rounds of draws: two 6,000-step networks that see ±51 s scored 0.789 and
0.784 alone (against 0.782 and 0.782 for ±25 s), 0.792 / 0.766 as a pair (against 0.786 / 0.762),
and with the two 12,000-step ±25 s networks **0.793 / 0.769** against 0.788 / 0.766 -- better on
15 of the 18 held-out cases, worse on 2, and better on each of the three matches (+0.007, +0.004,
+0.005). Shipped: five ±51 s networks at 6,000 steps with the five 12,000-step networks. Doubling
the view again (±102 s, dilation 256, trained on 205-s windows) was worse alone (0.782 / 0.750)
and added nothing to the mix (better on 8 cases, worse on 9). The decoder re-scan on the new
outputs again lost under leave-one-game-out (0.790 against 0.793); decoder unchanged.

**Decisions made against the simulation's preference, for the real match.**

* *Phases:* without the network the tuner wanted weights of 0.58–0.75; re-simulated with phases
  60–70% correct, such weights lose up to 0.012, so we shipped 0.2, and 0.1 once the network made
  phases irrelevant (< 0.001 either way).
* *Set pieces:* with the tag's team wrong 35% of the time and 40% false tags added, the tuned
  weights still beat no tags — kept.
* *Decoder tuned on Metrica alone* (human labels only) gained +0.008 to +0.017 there but lost 0.02–0.05
  on DFL, by switching possession less often (3.37 → 3.03 possessions/min on Metrica). On BAR-ATH
  the pipeline already reports fewer possessions per minute (2.59) than any simulated case, so a
  setting that switches even less is the wrong direction for the real match — not shipped.

**Where it fails, and what we would do next:**

* **Very short possessions** (< 2 s, a quarter of all possessions in the sample games' event data)
  remain the hardest: a one-touch interception rarely shows enough control to pay for two
  switches, even with the network and the per-frame switch cost.
  Whether the BAR-ATH ground truth has as many of them is unknown (its public 5 minutes have 2 of
  17), so we did not push the switch cost down to chase them.
* **Ball gaps.** Inside the ATD's long gaps (median 11 s) the network reaches 0.91 team accuracy
  against 0.95 with the ball seen. The biggest improvement left is a better ball: a ball detector
  on the tactical video, through the homography, to fill the gaps; and, as some teams do, the
  referee's whistle from the video's audio for stoppages — neither could be validated on the
  development matches (no video, no audio), so neither was attempted blind.
* **Domain shift.** Everything is tuned on three other matches with simulated noise; the
  simulation is calibrated (§4) but not identical. The possession count the pipeline reports on
  BAR-ATH is the first thing to sanity-check.

## 7. Adapting to the real match without labels

The method was built entirely on other matches, so the real risk was not the method but the gap
between our simulated noise and BAR-ATH's actual tracking. `make_submission` therefore prints a
**health report**: aggregate statistics of the run next to their range on the simulation. Five
rounds on the real data, using only those statistics — never a label:

1. **Player tempo.** First run: 95% of the match called dead. The report showed why — median
   player speed 1.47 m/s on BAR-ATH against 2.3–2.5 in simulation. The ATD is smoothed; our
   simulated per-frame jitter had inflated every speed, and the dead-ball model had learned "everyone
   slower than ~2 m/s means a stoppage". Fixes: player noise simulated as slow drift (speeds now
   1.55–1.78 m/s, the clean sample games' own 1.54–1.77), and every tempo feature made *relative to
   the half's median*, so no tracker's smoothing can move it again. Verified on the sample games:
   the predicted dead share is now the same (0.40–0.42, truth 0.45) with clean, drifting or jittery
   players. Second run: dead share 0.405, public-window check passed.
2. **Ball offset.** The report then showed the ball sitting farther from the nearest player on
   BAR-ATH (median 1.92 m) than anywhere in the simulation (1.1–1.8 m): single-camera ball tracking
   has slowly varying offsets (lofted balls projected to the ground, feet hiding the ball) that do
   not average out. We added exactly that error to the simulation (re-drawn every second,
   interpolated), sized so the simulated range (1.14–1.96 m) covers BAR-ATH's, and retrained. The
   real run sits closest to the *heavy* setting on every noise statistic.
3. **Tried and rejected: per-match calibration of the switch cost.** On BAR-ATH the decoder agreed
   with the ownership model less often (0.947) than in simulation (0.96–0.98), suggesting the switch
   cost was too high for the real evidence. We tried setting it per match so that agreement matched
   the simulation — label-free, so allowed — and measured it on the sample games first: neutral to
   negative (0.582 → 0.572–0.581), with erratic scales. Not shipped.

4. **Evidence cross-check** (`solution/crosscheck.py`). The per-frame pipeline (update 3) showed
   the decoder disagreeing with both tracking-based team models inside ball gaps far more than in
   simulation (0.81 against 0.94–0.97), following the Smart Tagging phases instead — a sign that a
   component did not transfer. Every run now prints, for each pair of evidence sources, how often
   they agree, next to the simulated range.
5. **The temporal CNN transfers.** On BAR-ATH the network agrees with the independent ownership
   and shape models exactly as much as in simulation (0.923 and 0.910, inside 0.906–0.941 and
   0.892–0.916), the decoder–ownership disagreement in ball gaps disappears (0.935), and the
   possession rate moves toward the simulated range (1.94 → 2.59 per minute). Predicted dead ball
   is 45%, plausible for a LaLiga match (ball in play ~55% of the time).

The public 5 minutes were used only as the alignment check they are meant to be (a score near 0
would mean misaligned frames or swapped teams); no version was chosen by it.

## 8. Reproduction

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
cache, and across machines on a public test bundle: `dev/make_fake_bundle.py`). The models,
network weights and constants are committed (`solution/models.json`, `solution/seqnet.npz`,
`solution/params.json`). To regenerate *them* from scratch — from public data only:

```bash
bash dev/fetch_sample_data.sh          # Metrica public sample games -> data/sample-data/
bash dev/fetch_dfl_data.sh             # DFL/IDSSE open data (CC BY 4.0) -> data/dfl/
python dev/evaluate.py --baseline      # the starter baseline on the degraded development cases
python dev/train_models.py             # per-frame models -> solution/models.json
python -m venv .torchenv && .torchenv/bin/pip install torch numpy pandas      # training only
.torchenv/bin/python dev/train_seqnet.py --logo --flip --xheavy --boundary 2.0 --more3 --steps 6000
                                       # held-out network outputs, for tuning
python dev/tune.py --rounds 2 --settings medium heavy --out solution/params.json
.torchenv/bin/python dev/train_seqnet.py --final --flip --xheavy --boundary 2.0 --more3 \
    --seeds 5 --steps 6000 --dilations 1 2 4 8 16 32 64 128   # -> solution/seqnet.npz (networks 0-4, ±51 s)
.torchenv/bin/python dev/train_seqnet.py --final --resume --flip --xheavy --boundary 2.0 --more3 \
    --seeds 10 --steps 12000           # adds networks 5-9 (10-network ensemble)
python dev/evaluate.py                 # held-out scores
python -m solution.test                # loader / decoder / model-width / writer self-tests
```

## 9. Other teams' public work

The competition repositories of other teams are public. We read them (code and write-ups only;
two forks that appeared to contain the gated match data were not opened). Two ideas we found in
common use there shaped our final weeks, and are credited here: the **DFL/IDSSE open dataset** as
external training data, and a **temporal convolutional network over per-frame features**. Both were
re-implemented independently and kept only after our own held-out tests showed a gain (§4, §6).
Nothing was copied; no other team's labels, models or outputs were used.
