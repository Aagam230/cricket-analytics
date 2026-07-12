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
