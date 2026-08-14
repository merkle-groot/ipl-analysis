<title>Ball-level wicket prediction — model evaluation</title>

# Model evaluation

> **Evaluation status:** exploratory. The 2025–26 period was excluded from model fitting, but it
> was inspected repeatedly during feature pruning and follow-up experiments. Notebook outputs are
> the canonical current results; small deltas need a future-season or rolling-origin confirmation.

How well three models fit the ball-level IPL data, how the numbers should be read, and what the
data turned out to say. Companion to [FEATURES.md](FEATURES.md) and the developer-oriented
[notebook fitting guide](NOTEBOOK_FITTING_GUIDE.md).

**Bottom line:** LightGBM and XGBoost both land at about a 2.3% log-loss improvement over the
base rate, ROC-AUC 0.618, on two seasons excluded from fitting. The headline edge is useful, but
small comparisons among follow-up variants are exploratory because those seasons were inspected
repeatedly.

---

## Evaluation protocol

### The split is temporal, never random

| split | seasons | balls | wicket rate |
|---|---|---|---|
| train | 2008–2023 | 243,656 | 0.0493 |
| validation | 2024 | 17,103 | 0.0516 |
| test | 2025–2026 | 34,798 | 0.0501 |

Two balls in the same over share a batter, bowler, phase and match situation. They are near
duplicates. A random split puts some of an over in train and the rest in test, letting the model
memorise the match and report a score it could never reproduce on a future game. Early stopping
watches 2024 only. The final seasons stay out of fitting, but later exploratory diagnostics reuse
them, so they should not be described as an untouched one-shot test.

### Accuracy is banned

At a 4.96% base rate, predicting "no wicket" on every ball scores 95.0% accuracy and is
worthless. Every number here is one of:

- Log loss, the primary metric. Rewards honest probabilities and punishes confident errors.
- Gain over null, meaning log loss against the constant-base-rate predictor, as a percentage.
  This is the only number that answers "did the model learn anything at all."
- ROC-AUC / PR-AUC, ranking quality. PR-AUC is compared against the base rate (0.0501),
  which is what random guessing scores on an imbalanced problem.
- Brier score and calibration curves: is a predicted 8% actually 8%.

### No class rebalancing

`scale_pos_weight`, SMOTE and friends are deliberately absent. They inflate the minority class
until the confusion matrix looks balanced, at the cost of destroying calibration. The model
starts claiming 40% on balls that are really 8%. With 14,649 positives there is no shortage of
signal to learn from. The deliverable here is a usable probability, not a yes/no call.

---

## Results

### Test set (2025–26), all models

| model | log loss | gain vs null | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| null (base rate) | 0.19887 | — | — | — |
| logistic + splines | 0.19531 | +1.79% | 0.6065 | 0.0798 |
| LightGBM | 0.19434 | +2.28% | 0.6181 | 0.0842 |
| XGBoost | 0.19432 | +2.29% | 0.6179 | 0.0840 |
| lgb + xgb averaged | 0.19430 | +2.30% | 0.6184 | 0.0842 |

### Fit across all three splits

| model | split | log loss | gain | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| logistic + splines | train | 0.19146 | +2.60% | 0.6260 | 0.0845 |
| | val | 0.19979 | +1.74% | 0.6024 | 0.0801 |
| | test | 0.19531 | +1.79% | 0.6065 | 0.0798 |
| LightGBM | train | 0.18883 | +3.93% | 0.6592 | 0.1026 |
| | val | 0.19901 | +2.13% | 0.6103 | 0.0886 |
| | test | 0.19434 | +2.28% | 0.6181 | 0.0842 |
| XGBoost | train | 0.18907 | +3.81% | 0.6549 | 0.1020 |
| | val | 0.19893 | +2.17% | 0.6113 | 0.0862 |
| | test | 0.19432 | +2.29% | 0.6179 | 0.0840 |

Overfitting check: the train→test ROC-AUC gap is 0.041 for LightGBM and 0.020 for the
logistic regression. Both are modest, and the gain-over-null holds up on unseen seasons
(+3.93% → +2.28%). The boosters were kept shallow deliberately, at 15 leaves / depth 4,
`min_child_samples=400`, `reg_lambda=10`, learning rate 0.02, because with this little signal an
unconstrained booster spends its capacity memorising noise. Early stopping landed at 252 trees
(LightGBM) and 282 (XGBoost).

Test beats validation for every model, which looks odd but is fine: 2024 had a higher wicket
rate (0.0516 vs 0.0501) and 17k balls is a small sample. It is season-to-season variation, not a
bug.

---

## Are the probabilities usable?

Calibration on the test set, LightGBM, by decile of predicted risk:

| decile | n | predicted | actual |
|---|---|---|---|
| 1 | 3,480 | 0.0287 | 0.0250 |
| 2 | 3,480 | 0.0336 | 0.0322 |
| 3 | 3,480 | 0.0374 | 0.0339 |
| 4 | 3,479 | 0.0412 | 0.0411 |
| 5 | 3,480 | 0.0449 | 0.0368 |
| 6 | 3,480 | 0.0490 | 0.0506 |
| 7 | 3,479 | 0.0539 | 0.0543 |
| 8 | 3,480 | 0.0604 | 0.0626 |
| 9 | 3,480 | 0.0714 | 0.0687 |
| 10 | 3,480 | 0.1034 | 0.0960 |

Good through nine deciles, with mild over-prediction at the top. LightGBM is better calibrated
than the logistic regression here: the LR's top decile predicts 0.1145 against 0.0943 actual,
roughly twice the error.

Discrimination in practical terms. Predictions span 0.021 to 0.233, and ranking by risk:

- Top 1% riskiest balls: 18.7% actual wicket rate, 3.7× the base rate. 65 wickets in 347 balls.
- Top 5%: 12.1%, a 2.4× lift.

So the model cannot tell you *which* ball takes a wicket, but it can reliably isolate the
deliveries that are three to four times more dangerous than average. For a broadcast win-probability
graphic or an in-play risk indicator, that is the useful form of the output.

---

## Diagnostics

### Ablation: what actually matters

Retrain LightGBM without a feature group and see whether the test loss moves. This is the real
test of importance.

| dropped | log loss | gain | ROC-AUC |
|---|---|---|---|
| (nothing) | 0.19434 | +2.28% | 0.6181 |
| `ground` | 0.19439 | +2.25% | 0.6177 |
| `ground` + both venue rates | 0.19434 | +2.28% | 0.6183 |
| `h2h_dismissals` + `h2h_balls` | 0.19447 | +2.21% | 0.6160 |

Dropping every venue feature changes the loss by nothing, to five decimal places. The
head-to-head pair is marginally positive: removing it costs 0.07%, so it earns its place,
barely.

### Permutation importance (top 10, % of total model skill)

| feature | Δ log loss | % of skill |
|---|---|---|
| `legal_balls` | +0.00112 | 24.7% |
| `bat_career_dismissal_rate` | +0.00080 | 17.7% |
| `over` | +0.00059 | 13.1% |
| `balls_since_wicket` | +0.00035 | 7.7% |
| `bat_career_balls` | +0.00033 | 7.2% |
| `rrr` | +0.00029 | 6.3% |
| `bowl_balls` | +0.00023 | 5.0% |
| `bat_balls` | +0.00020 | 4.4% |
| `ground` | +0.00018 | 3.9% |
| `bat_runs` | +0.00011 | 2.4% |

Twelve features have importance at or below zero, including all three bowler career features,
`toss_won`, `is_chase` and `crr`. Correlated features share credit, so `over` and `legal_balls`
individually understate the combined 37.8% that "how late is it" contributes.

---

## Model-by-model assessment

### Logistic regression + splines, the honest baseline

36 features expand to 103 columns. Cubic spline bases (6 knots) on nine features whose effect is
curved; plain standardised terms for the rest; median imputation with a missingness indicator for
the first-innings nulls.

**Fit:** +1.79% over null, ROC-AUC 0.6065. Recovers about three-quarters of what boosting
achieves.

The splines were necessary, not decorative. `bat_runs` is flat from 0–40 and then steps up past
50; a single linear coefficient fits one slope through that and captures neither half.

Its weaknesses are structural. It is additive, so it cannot express `over × wickets_in_hand`.
And with collinear inputs the coefficients become uninterpretable: `over` at 1.69 and
`legal_balls` at 0.53 measure the same underlying thing with offsetting signs.

That collinearity produced the sharpest methodological trap in the project. The first version of
the notebook plotted partial dependence, sweeping one feature and holding the rest at their
medians, and the "effect of the over" curve came out backwards, showing risk *falling* from over
1 to over 15. Nothing was wrong with the model; the plot was asking a nonsensical question, since
holding `legal_balls` and `phase` fixed while varying `over` describes deliveries that cannot
exist. It is now an observed-vs-predicted plot over real deliveries, which shows the model
tracking the true curve closely.

### LightGBM, the pick

**Fit:** +2.28% over null, ROC-AUC 0.6181, best calibration of the three.

Finds thresholds like `bat_runs >= 50` unaided, handles NaN natively (no imputation for the
first-innings chase columns), and takes `ground` as a native categorical instead of 37 one-hot
columns. Beats the LR by 0.5 percentage points of gain, a 28% relative improvement in skill over
the null.

### XGBoost, indistinguishable

**Fit:** +2.29% over null, ROC-AUC 0.6179. Within 0.00002 log loss of LightGBM.

Two independently-implemented boosters with different regularisation, different split-finding and
different categorical handling landing this close is informative: it means the ceiling is the
feature set, not the algorithm. No amount of hyperparameter tuning will move this much.
Averaging the two adds 0.01 percentage points, nothing.

One practical note: XGBoost errors outright on a categorical level it never saw in training,
which Mullanpur triggers (the ground debuted in 2024, after the training cut-off). Fixed by
sharing one category vocabulary across all three splits; only level *names* are shared, never
target information.

---

## Interesting findings

The modelling is the smaller half of what came out of this. In rough order of how much they
surprised me:

1. Set batters are more likely to get out, not less. Every "settling in" intuition is
inverted in T20. A batter past 41 balls faced gets out at 0.0706 per ball against 0.0428 for one
who has faced 0–2. Score tells the same story (0.0438 at single figures, 0.0771 past 100), and so
does partnership length (0.0459 within 3 balls of a wicket, 0.0586 past 60). Controlling for
phase resolves the mechanism: in death overs, being past 50 makes no difference (0.0759 either
way), while in middle overs it does (0.0576 vs 0.0393). It is licence, not nerves. A batter
with runs in the bank starts playing shots earlier, and the extra risk is the price of that.

2. The stadium doesn't matter for wickets. The true between-ground spread in wicket rate is
sd 0.0014 once binomial noise is subtracted; most of the apparent variation between venues is
sampling error. Dropping every venue feature changes test log loss by nothing. Yet the same
grounds range from 1.30 to 1.51 runs per ball, a large and real spread. Pitch strongly affects
scoring and barely affects wickets per ball, because teams adjust aggression to conditions:
on a slow pitch they score less but also risk less, and the wicket rate equilibrates.

3. Gain importance lied, and the ablation caught it. `ground` ranks third on LightGBM's gain
chart at 9.3% while contributing exactly zero. High-cardinality categoricals get many chances to
carve off noise and are credited for each one. If you take one methodological lesson from this
project: never report gain importance without an ablation.

4. "He's got his number" is not a thing. Among batter–bowler pairs with 24+ balls of history,
prior head-to-head dismissal rates spanning 0.000 to 0.080 produce subsequent rates of 0.0371 to
0.0394, no relationship. Worse, the raw dismissal count runs *backwards* (0.0464 at zero prior
dismissals, 0.0394 at two) through survivorship: accumulating dismissals against a bowler
requires a long career, which selects for good batters.

5. Recent form is a quality proxy, nothing more. Last-5-innings average looks strongly
predictive (worst quintile 0.0582 vs best 0.0398) until you split by career quality, at which
point it goes flat for established batters (0.0384 / 0.0409 / 0.0397 / 0.0386 / 0.0377). What
looked like form was batting position and player quality.

6. Dot-ball pressure does not exist at team level. Balls since the last team boundary: 0.0496
within 2 balls, 0.0487 past 20. Flat, and sloping the wrong way. It stays flat after controlling
for over and batter quality, so it is not being masked. The effect is absent. A boundary drought
turns out to be a low-aggression state (1.32 → 1.13 runs/ball as it lengthens), not a
building-pressure one. See FEATURES.md, `balls_since_boundary`.

7. Who is bowling barely matters; when he is bowling matters enormously. All three bowler
career features are ≤0.2% of model skill, and `bowl_career_runs_per_ball` is negative. Meanwhile
`over` and `legal_balls` together are 37.8%. Bowler quality does show a real gradient in
isolation (0.0467 → 0.0533 across career-wicket-rate quartiles) but adds nothing once phase is
known.

8. Chasing is not inherently riskier, being behind is. First-innings wicket rate 0.0496 vs
chase 0.0495, indistinguishable, and `is_chase` is 0.0% of skill. But a chase 6+ runs per over
behind the required rate produces wickets at 0.0725 against 0.0399 for one comfortably ahead.

9. The toss is worth nothing. 0.0495 vs 0.0497. Included specifically to check, kept as
documentation that it was checked.

---

## Limitations

The ceiling is low and it is the data's fault, not the model's. A ~2.3% log-loss gain is what
this feature set supports. Two well-regularised boosters agreeing to four decimal places says
tuning is not the constraint.

Missing information that would plausibly help, in order:

1. Bowler type (pace/spin). Absent from Cricsheet, derivable from career patterns. Now the
   most promising untested addition.
2. Ball-tracking / shot data. Would move this from "situational risk" to actual dismissal
   prediction, and is not in any free dataset.

`batting_position` was the top item here and has now been tested. It does not help, see
`starter.ipynb`. The raw effect is real (0.042 per ball for an opener vs 0.103 for a No. 10) but
`bat_career_dismissal_rate`, `team_wickets` and `bat_balls` already capture it between them.

Known weaknesses in what exists: the model over-predicts the top decile (0.1034 vs 0.0960);
the `bat_runs` shape check shows under-prediction for batters past 70 (0.067 predicted vs 0.078
actual); and `bat_career_dismissal_rate` conflates skill with role, since aggressive batters are
not worse batters.

---

## The pruned model (current best)

Dropping the dead features was one of the "cheap next experiments" below. It worked, and it is
now the best model in the project.

| model | features | log loss | gain over null | ROC-AUC |
|---|---|---|---|---|
| old LightGBM | 37 | 0.19439 | +2.25% | 0.6168 |
| pruned, 5-seed average | 14 | 0.19423 | +2.33% | 0.6192 |

Two changes: cut the 23 features scoring ≤0.4% of skill, then raise regularisation
(`reg_lambda` 10 → 30, `min_child_samples` 400 → 1000) because there is less signal left to
overfit. Averaging 5 seeds removes run-to-run wobble.

The absolute improvement is small. Seed-to-seed wobble is not a confidence interval for future
matches; the paired match bootstrap in `starter.ipynb` crosses zero for this comparison. Prefer
the pruned model for simplicity, not because this sample proves a generalisation gain.

This is a plateau, not a peak. XGBoost on the same 14 features scores 0.19425, and blending
the two gives 0.19423, identical to LightGBM alone. Sweeping `num_leaves` (7–31), learning rate
(0.01–0.02), `min_child_samples` (150–2000) and `reg_lambda` (10–100) produced nothing outside
0.19423–0.19438. Remaining gains are in new information, not in hyperparameters.

Cheap next experiments: a monotone constraint on `bat_runs` in LightGBM to stop the model
inventing wiggles in sparse regions, but *not* on `over`, whose true shape is not monotone.
It peaks at 0.043 in over 5 and falls to 0.033 in over 7 before the death-overs climb, on ~15k
balls per over, so the powerplay bump is structure rather than sparsity (see FEATURES.md,
`over`). Constraining it would force the model to flatten a real effect. Also worth trying:
`wicket_bowler` as the target, to remove run-outs as a separate causal process.
