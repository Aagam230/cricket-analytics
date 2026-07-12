# Model Card: IPL Match Winner Prediction

This model card documents the win-prediction model in the Cricket
Analytics & Match Prediction Platform, following the spirit of the
[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
framework: what the model does, how it was built, how well it actually
performs, and where it shouldn't be trusted.

**Last regenerated:** 2026-07-10, against the IPL dataset covering
2007/08–2025 (`data/raw/matches.csv`, `data/raw/deliveries.csv`).
Run `python -m src.models.train` to regenerate the underlying numbers and
update this document if the dataset or feature set changes.

---

## 1. Intended Use

**Task:** Binary classification — given two IPL teams, a venue, and toss
information for a hypothetical match, predict which team (`team1` or
`team2`) will win.

**Intended users:** Portfolio/demo audience exploring cricket analytics;
not intended for betting, wagering, or any decision with real financial
or personal stakes.

**Out of scope:** This model does *not* predict margins, scores, or
player performance, and does not use any in-match information (it only
uses features knowable *before* the first ball is bowled). It is not
fit for real-money prediction markets — see Section 5 (Limitations)
before drawing any conclusions from its output.

---

## 2. Data

- **Source:** IPL ball-by-ball dataset, 2007/08–2025 seasons (1,169
  matches, ~278,000 deliveries). See `README.md` → Dataset and
  `docs/INSTALLATION.md` for provenance.
- **Training population:** 1,146 matches with a decisive result (ties and
  no-results are excluded — there's no valid winner label to learn from).
- **Class balance:** 50.7% team1 wins / 49.3% team2 wins overall — the
  `team1`/`team2` labels are just "however the source data listed them,"
  not "home" vs. "away" or "stronger" vs. "weaker," so this is close to
  balanced by construction, not by design choice.

---

## 3. Features

All features are **leakage-safe**: every rolling statistic for a team is
computed using only that team's matches *strictly before* the match being
predicted (via `.shift(1)` on chronologically sorted data). No
current-match information (scores, wickets, margins) is ever used as a
feature.

| Feature | Description |
|---|---|
| `season`, `match_month` | Calendar context. |
| `team1_enc`, `team2_enc`, `venue_enc`, `toss_decision_enc` | Label-encoded categoricals. |
| `toss_winner_is_team1` | 1 if team1 won the toss. |
| `team1_career_win_pct`, `team2_career_win_pct`, `diff_career_win_pct` | Recency-weighted (exponential decay, halflife ≈30 matches) career win rate, and their difference. |
| `team1_recent_win_pct`, `team2_recent_win_pct`, `diff_recent_win_pct` | Win rate over each team's last 5 matches, and their difference. |
| `team1_venue_win_pct`, `team2_venue_win_pct`, `diff_venue_win_pct` | Win rate at this specific venue, shrunk toward career win rate by sample size (`K=5`), and their difference. |
| `team1_matches_played`, `team2_matches_played` | Career matches played prior to this match (sample-size signal). |
| `h2h_team1_win_pct`, `h2h_matches_played` | Recency-weighted (halflife ≈6 meetings) head-to-head win rate and raw meeting count. |

**Notably absent:** `is_day_night` (no day/night or start-time field
exists anywhere in the source dataset — see `src/utils/config.py`) and
`city_enc` (dropped 2026-07-10 as redundant with venue and a likely
source of encoding noise given the modest training-set size).

### 3.1 Why recency-weighting and shrinkage were added

IPL squads turn over substantially every season via trades and auctions.
A flat, all-time "career win %" mixes in performance from a roster that
may bear little resemblance to the team playing today. Recency weighting
(exponential decay) lets recent seasons dominate a team's own "career"
rate without discarding older history entirely. Venue shrinkage exists
because a team with 2 prior matches at a venue was previously getting a
venue win% treated with the same confidence as a team with 40 — shrinking
small samples toward the team's overall rate avoids noisy extremes (e.g.
1-for-1 = "100% win rate at this venue").

Under a fair, apples-to-apples comparison (same random train/test split,
old features vs. new), these changes produced a **modest but real
improvement** for the best-performing model (Logistic Regression):
test accuracy 56.1% → 58.3%, test ROC-AUC 0.609 → 0.605 (essentially
flat — the gain shows up more in accuracy/F1 than ranking quality). This
is a genuine, if small, improvement — not the headline result, though;
see Section 4.2.

---

## 4. Evaluation

### 4.1 Methodology — why a time-based split

Earlier evaluation used a random stratified train/test split. That
produces flattering numbers but doesn't reflect how this model would
actually be used: a random split can train on 2024 matches and test on
2009 matches, which is not the deployment scenario ("predict a match
that hasn't happened yet, given everything known so far").

The current default (`time_based_split=True` in
`src.models.train.train_and_compare_models`) instead holds out the most
recent ~20% of matches, **in chronological order**, as the test set:

- **Train:** 916 matches, seasons 2007–2022
- **Test:** 230 matches, seasons 2022–2025
- **Cross-validation:** 5-fold stratified CV, computed only within the
  training portion (never touches the time-based test set)
- **Class balance:** train 50.9% team1 wins, test 50.0% — the split does
  not introduce a class-imbalance confound, so weak results below can't
  be explained away by that.

Set `time_based_split=False` to reproduce the old random-split behavior
for comparison; it is no longer the default because it overstates
real-world performance.

### 4.2 Results (time-based split, current default)

Cross-validated on the training portion (916 matches, 5-fold), then
scored once on the held-out, chronologically later test portion (230
matches):

| Model | CV F1 | CV Accuracy | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** (best) | 0.530 | 0.511 | 0.529 | 0.517 | 0.545 | 0.209 | 0.302 | 0.538 |
| Random Forest | 0.528 | 0.509 | 0.490 | 0.522 | 0.541 | 0.287 | 0.375 | 0.512 |
| Decision Tree | 0.520 | 0.509 | 0.500 | 0.465 | 0.460 | 0.400 | 0.428 | 0.467 |
| Gradient Boosting | 0.511 | 0.489 | 0.492 | 0.522 | 0.513 | 0.835 | 0.636 | 0.497 |

*(XGBoost is included in the code as an optional 5th model but was not
available in the environment this table was generated in; install
`xgboost` and re-run `python -m src.models.train` to include it.)*

Best model selected by cross-validated F1: **Logistic Regression**.

**A confusion matrix note on Gradient Boosting:** its high test recall
(0.835) comes from a heavy bias toward predicting the positive class (91
of 115 team2-win matches were predicted as team1 wins) — a degenerate,
close-to-always-predict-one-class strategy that happens to score well on
recall/F1 in a roughly balanced test set, not evidence of genuine skill.
This is exactly the kind of thing a single metric can hide, which is why
this card reports the full confusion-matrix-derived metric set rather
than leading with accuracy or F1 alone.

Confusion matrices (`test_confusion_matrix`, format `[[TN, FP], [FN, TP]]`
where "positive" = team1 wins) are also persisted to
`models/model_metadata.json` and viewable per-model in the Streamlit
**Model Performance** page.

### 4.3 Honest interpretation

**Test-set ROC-AUC ranges from 0.467 to 0.538 across all four models —
close to 0.5, the value that indicates no better than random guessing.**
Cross-validated F1 (0.51–0.53) is similarly close to what a model that
always predicts the majority class would score on this near-balanced
dataset.

This is not a training bug, a data leak, or an implementation error — it
has been checked for all three: features are shift-safe, the split is
class-balanced on both sides, and predict-time feature computation
(`compute_live_features`) has been verified to mirror the training-time
logic exactly (see `src/features/engineering.py` module docstring). It
reflects a real property of the prediction task: **pre-match information
alone does not explain much of what decides an IPL match.**

Plausible reasons, not mutually exclusive:
- **Playing XI is not in this feature set.** Team lineups change match to
  match (injuries, rotation, conditions-based selection) and this dataset
  has no pre-match lineup information — the model only knows the
  franchise names, not who's actually playing.
- **In-match dynamics dominate limited-overs cricket.** A single over,
  a dropped catch, or a batter's form *on the day* can swing a T20 match
  far more than season-long team form — information that, by
  construction, isn't available before the first ball.
- **Small dataset.** 916 training matches is not a large sample for a
  20-feature model; cross-validation standard deviations (1.5–4.7 points
  of F1 across folds) suggest real sensitivity to which matches land in
  which fold.

**What would likely perform better:** a model predicting *in-match* win
probability from live ball-by-ball state (current score, wickets, overs
remaining, required run rate) has access to information that simply does
not exist before a ball is bowled, and would be expected to substantially
outperform this pre-match model — that's a different, larger project, not
a tweak to this one.

---

## 5. Limitations

- **No playing-XI data.** The model cannot account for injuries, rested
  players, or match-specific team selection — a known, structural gap
  given this dataset's schema (see `README.md` → Dataset).
- **No day/night, weather, or pitch-report data.** None of this exists in
  the source dataset.
- **Small, non-i.i.d. sample.** 1,146 matches across 18 seasons, with
  meaningful rule/format changes over that period (impact player rule,
  franchise rebrands/relocations, DLS-affected matches) that the model
  does not explicitly account for beyond team-name normalization.
- **`team1`/`team2` are not "home"/"away" or "favorite"/"underdog."**
  They're whatever order the source data assigned; the model has no
  home-advantage signal beyond whatever the venue/team win-rate features
  implicitly capture.
- **Not validated on data after the training cutoff.** Retrain
  (`python -m src.models.train`) after adding new-season data before
  trusting predictions for a season not represented in the table above.

---

## 6. Reproducing this card

```bash
python -m src.data.preprocess
python -m src.models.train
python -c "
import json
with open('models/model_metadata.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

Update the tables in Section 4 if the numbers change meaningfully after a
dataset refresh or feature change.

---

# Part 2: In-Match ("Live") Win-Probability Model

**Added 2026-07-12.** Section 4.3 above concludes that pre-match features
explain very little of an IPL outcome (test ROC-AUC ~0.5). This raises an
obvious follow-up question: is that a limitation of the *modeling*, or of
the *information available before a ball is bowled*? This section answers
that by solving a different, better-posed problem with the same
underlying data: given the match state *during* a second-innings chase
(score, wickets, overs, target), what's the win probability right now?

This is not a replacement for the pre-match model in Part 1 -- it's a
genuinely different task (predicting from live state vs. predicting from
pre-match information), included specifically to test whether the low
Part 1 numbers reflect "hard problem" or "insufficient information."

## 2.1 Task & Scope

**Task:** Binary classification, evaluated at ball-level granularity --
given the state of a second-innings chase at any point, predict whether
the batting (chasing) team will win.

**Scope:** Second-innings chases only. A first-innings equivalent ("will
this total be enough") is a different problem with no fixed target to
measure progress against, and is out of scope here. Matches with no
result, no recorded first-innings score, or decided under the D/L method
are excluded -- D/L-affected matches (19 of 1,169, ~1.6%) have no
revised-target/revised-overs data in this dataset, so their true target
and over-count can't be reconstructed; including them with the
*unadjusted* full-match target would silently mislabel those matches'
`runs_needed` / `overs_remaining` features.

## 2.2 Data

- **1,127 eligible matches** (of 1,169 total: excluded 23 no-result/tied,
  19 D/L-affected — some matches fall into both categories).
- **130,733 balls** (rows) — one row per ball of the second innings across
  all eligible matches.
- **Class balance:** 51.8% of balls belong to a chase the batting team
  ultimately won.

## 2.3 Features

| Feature | Description |
|---|---|
| `batting_team_enc`, `bowling_team_enc`, `venue_enc` | Label-encoded categoricals. |
| `target` | Runs required to win (first innings score + 1). |
| `runs_scored`, `runs_needed` | Chase progress at this ball. |
| `wickets_lost`, `wickets_in_hand` | Wickets lost / remaining. |
| `balls_bowled`, `balls_remaining` | Balls into the chase / balls left (out of 120). |
| `current_run_rate`, `required_run_rate`, `run_rate_diff` | Scoring rate so far vs. rate still needed, and their difference. |
| `batting_team_career_win_pct`, `bowling_team_career_win_pct` | Each team's PRE-MATCH recency-weighted career win rate (reused directly from Part 1's `src.features.engineering` logic) — included for completeness, expected to matter far less than the live state features. |

All features reflect state strictly *before* the ball being labeled (via
`.shift(1)`-style cumulative sums), so there's no within-ball leakage.

## 2.4 Evaluation Methodology — match-level, not ball-level

**This is the single most important correctness property of this
model.** Every ball within one match shares that match's eventual winner
and is highly correlated with every other ball in that match. Both
cross-validation and the train/test split are grouped by `match_id`:

- **Cross-validation:** `GroupKFold` (5 folds), grouped by match — no
  fold ever contains balls from a match that also has balls in another
  fold.
- **Train/test split:** match-level chronological split — 901 matches
  (104,378 balls) train, 226 matches (26,355 balls) test, with the test
  matches being the most recent chronologically (mirroring Part 1's
  time-based split rationale).

A ball-level random split was deliberately NOT used, since it would let
the model implicitly see other balls from the same match during training
and inflate every metric below in a way that wouldn't hold up on genuinely
unseen matches.

## 2.5 Results

| Model | CV F1 | CV Accuracy | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** (best) | 0.791 | 0.775 | 0.871 | 0.786 | 0.806 | 0.717 | 0.759 | **0.876** |
| Random Forest | 0.781 | 0.766 | 0.859 | 0.796 | 0.832 | 0.709 | 0.766 | 0.875 |
| Gradient Boosting | 0.766 | 0.749 | 0.841 | 0.786 | 0.829 | 0.686 | 0.750 | 0.861 |
| Decision Tree | 0.725 | 0.705 | 0.753 | 0.755 | 0.744 | 0.731 | 0.738 | 0.793 |

*(XGBoost is included in the code as an optional 5th model but was not
available in the environment this table was generated in.)*

Best model selected by cross-validated F1: **Logistic Regression**
(test ROC-AUC 0.876).

## 2.6 Honest Interpretation — this confirms the Part 1 hypothesis

**Test ROC-AUC of 0.876 vs. Part 1's 0.538, on the same underlying
matches, evaluated with the same rigor (match-grouped CV, chronological
test split).** This is a large, genuine gap, and it directly answers the
question posed in Section 4.3 of Part 1: pre-match features weren't weak
because the modeling was weak — they were weak because *most of what
determines an IPL match's outcome isn't knowable before the match
starts*. Once the model can see what's actually happening in the game
(score, wickets, run rate required), it becomes a substantially more
accurate predictor using a *simpler* model (plain Logistic Regression
beat every ensemble method here, same as in Part 1).

**A specific, interesting behavior worth noting:** the model was NOT
given ball-by-ball position within an over, bowler identity, or recent
scoring momentum (e.g. runs in the last 6 balls) — only cumulative state.
That it still reaches 0.876 ROC-AUC from cumulative state alone suggests
those omitted features would likely push accuracy higher still; they were
left out of this first pass to keep the feature set simple and
interpretable (see Section 2.7).

**Where the model is weakest:** examining the confusion matrices, recall
sits meaningfully below precision for the best models (e.g. Logistic
Regression: 80.6% precision vs. 71.7% recall) — the model is somewhat more
confident calling a win for the batting team than correctly catching
every eventual batting-team win. This is consistent with early-innings
balls being inherently harder to call (the model hasn't seen the match
lean either way yet) diluting recall on eventual winners more than
precision on confident predictions.

## 2.7 Limitations (Part 2)

- **No bowler-specific features.** The model doesn't know if the current
  bowler is a strong death-overs specialist or a part-timer — a real
  factor in late-innings win probability that a viewer watching the match
  would weight heavily.
- **No momentum/recency features within the innings.** A team scoring 20
  runs in the last over is treated identically to one that scored 20 runs
  spread evenly over the last 3 overs, as long as cumulative state
  matches — momentum is a plausible next feature to add.
- **First innings not modeled.** "Is this total competitive" is a
  meaningfully different, harder problem (no fixed target to measure
  against) and wasn't attempted here.
- **D/L-affected matches excluded entirely** (rather than modeled with
  approximated revised targets) — see Section 2.2.
- **Team-strength features are the same Part 1 pre-match statistics**,
  carrying over any of Part 1's own limitations (no playing-XI data, etc.)
  for that small piece of the feature set specifically.

## 2.8 Reproducing this section

```bash
python -m src.data.preprocess
python -m src.models.live_win_probability
python -c "
import json
with open('models/live_model_metadata.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

See the **Live Win Probability** page in the Streamlit dashboard for an
interactive match-replay chart and manual win-probability calculator
built on this model.

