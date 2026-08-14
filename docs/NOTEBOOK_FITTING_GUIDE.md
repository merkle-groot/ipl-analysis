# How the modelling notebooks fit IPL delivery data

This guide explains the training path in [`starter.ipynb`](../starter.ipynb) and
[`runs.ipynb`](../runs.ipynb). It is written for a developer who understands neural-network
basics—features, labels, loss, gradient descent, TensorFlow, and confusion matrices—but is new to
classical probabilistic models and gradient-boosted trees.

The short version is:

- `starter.ipynb` estimates the probability that a wicket occurs on a delivery.
- `runs.ipynb` estimates a six-class probability distribution for runs off the bat on a legal
  delivery: `P(0)`, `P(1)`, `P(2)`, `P(3)`, `P(4)`, and `P(6)`.
- Both begin with an interpretable linear model, then fit gradient-boosted trees to capture
  nonlinear effects and feature interactions.
- The latest runs model additionally averages five tree models and calibrates their class
  probabilities on a season that was not used to fit the trees.

Neither notebook trains one large neural network. The surrounding concepts are nevertheless the
same: convert rows into features and labels, minimize a loss on past data, select capacity using a
later validation period, and evaluate probabilities on future periods.

## 1. Project map

| File | Role |
|---|---|
| [`build_ball_data.py`](../build_ball_data.py) | Converts Cricsheet IPL JSON into one pre-ball feature row per delivery. |
| [`ball_data.parquet`](../ball_data.parquet) | The modelling table read by both notebooks. |
| [`starter.ipynb`](../starter.ipynb) | Exploratory training and evaluation for wicket probability. |
| [`runs.ipynb`](../runs.ipynb) | Exploratory training and evaluation for the run-outcome distribution. |
| [`model_baseline.py`](../model_baseline.py) | Shared feature lists, time split, and the standalone wicket logistic baseline. |
| [`model_trees.py`](../model_trees.py) | Standalone LightGBM/XGBoost wicket baselines. |
| [`model_runs.py`](../model_runs.py) | Run labels, outcome classes, feature conversion, and a standalone run model. |
| [`model_runs_calibrated.py`](../model_runs_calibrated.py) | Reproducible fit for the latest independently calibrated run model. |
| [`evaluation.py`](../evaluation.py) | Logit-bias calibration and match-clustered uncertainty estimates. |

The notebooks contain the narrative, plots, comparisons, and experiments. The `.py` modules hold
the reusable parts of the fitting logic. At present, the notebooks and scripts fit models in
memory; they do not save a production model artifact.

## 2. The common data path

```text
Cricsheet JSON matches
        |
        v
build_ball_data.py
        |
        v
ball_data.parquet: one row per delivery, pre-ball state + observed result
        |
        +-------------------------+
        |                         |
        v                         v
starter.ipynb                runs.ipynb
wicket: binary label         runs: six-class label, legal balls only
        |                         |
        v                         v
logistic baseline            multinomial logistic baseline
        |                         |
        v                         v
boosted trees                boosted trees -> ensemble -> calibration
```

The current parquet has 295,557 deliveries from 1,243 matches covering 2008–2026. A row represents
the state immediately before a ball plus fields describing what happened on that ball. Examples
of pre-ball features include:

- innings state: `team_runs`, `team_wickets`, `legal_balls`, `over`, and `phase`;
- batter state: runs, balls, strike rate, dot-ball percentage, and balls since a boundary;
- chase state: runs required, balls left, current rate, and required rate;
- historical state: career rates, venue rates, and batter–bowler history;
- matchup state: batting hand, bowling arm/type, and derived handedness/turn features.

The modelling code selects only pre-ball columns. The target is taken from the delivery result.
Historical aggregates are updated after a match, rather than after each delivery, so a row cannot
learn from the result of its own match through a career or head-to-head statistic. This timing rule
is the main defence against target leakage.

### Why the split follows time

The standard notebook split is:

| Split | Seasons | Wicket rows | Legal run rows | Purpose |
|---|---:|---:|---:|---|
| Training | 2008–2023 | 243,656 | 234,995 | Estimate model parameters. |
| Validation | 2024 | 17,103 | 16,299 | Choose boosting duration and compare candidates. |
| Evaluation | 2025–2026 | 34,798 | 33,171 | Estimate performance on later cricket. |

A random row split would be misleading. Consecutive balls share the same match, ground, batter,
bowler, innings state, and recent history. Putting some balls from an over into training and others
into validation is similar to putting near-duplicate images on both sides of an image-classification
split. A chronological split better represents the real question: can a model fitted on previous
seasons work on a future season?

The notebooks also assert that match IDs do not overlap across splits. This matters more than merely
checking row counts because the match is the natural correlation boundary.

### Fit, model selection, calibration, and evaluation are different operations

These terms are used precisely throughout the notebooks:

- **Fit** changes model parameters using labels. For logistic regression these are coefficients;
  for boosting these are tree splits and leaf values.
- **Model selection** chooses a configuration, such as the number of boosting rounds, based on a
  validation loss.
- **Calibration** changes the probability scale after the predictive model is fitted. It does not
  learn new ball-level feature relationships.
- **Evaluation** computes metrics without changing the fitted system.

Using different data for these jobs reduces the chance that a reported improvement is just a
decision tuned to the reporting set.

## 3. `starter.ipynb`: fitting wicket probability

### 3.1 Label and output

The target is the binary `wicket` column:

```text
y = 1  a wicket is recorded on this delivery
y = 0  no wicket is recorded
```

The model returns one number, `P(wicket | pre-ball state)`. It is a probability model, not merely a
wicket/no-wicket classifier. A caller may later convert that probability into a decision with a
threshold, but the notebook intentionally evaluates the probability before choosing any threshold.

### 3.2 Logistic regression with fixed nonlinear features

The first fitted model is a scikit-learn `Pipeline`:

```text
36 selected columns
        |
        +-- 9 curved numeric features
        |      median imputation + missing indicator
        |      -> cubic spline expansion
        |      -> standardization
        |
        +-- 26 other numeric features
        |      median imputation + missing indicator
        |      -> standardization
        |
        +-- phase
               one-hot encoding
        |
        v
103 transformed columns
        |
        v
logistic regression -> sigmoid -> P(wicket)
```

The preprocessing and estimator are one pipeline, so calling `fit` on the training rows first fits
the imputation values, spline basis, scaling statistics, and category encoding, then fits the
logistic coefficients. Validation and evaluation rows only pass through the already-fitted
transformations.

Nine columns use cubic splines because their relationship with risk is unlikely to be one straight
line. For example, the effect of `bat_runs` may be nearly flat over one range and change after a
batter is set. A spline converts one numeric feature into several fixed curved basis functions. The
logistic model can then learn a weighted combination of those curves while remaining linear in its
parameters.

In TensorFlow terms, this is approximately a fixed feature-engineering layer followed by:

```python
tf.keras.layers.Dense(1, activation="sigmoid",
                      kernel_regularizer=tf.keras.regularizers.l2(...))
```

There are no hidden learned layers. Scikit-learn's default `lbfgs` solver numerically minimizes
regularized binary cross-entropy. `C=1.0` is the inverse regularization strength: a smaller `C`
means a stronger penalty on large coefficients.

One implementation nuance is worth knowing. In the current spline branch,
`SimpleImputer(add_indicator=True)` creates missingness flags before `SplineTransformer`, so those
binary flags also pass through the spline expansion. This is valid but unnecessarily indirect. A
production refactor should route imputed numeric values and missingness indicators through separate
branches.

### 3.3 What the logistic fit establishes

This model is the sanity-check baseline. If a complex model cannot beat it out of time, the added
complexity has not paid for itself. It currently reaches about `0.19547` test log loss versus
`0.19887` for the constant-rate predictor, a 1.71% reduction.

The notebook then checks:

- calibration: whether balls predicted near 8% actually produce wickets near 8% of the time;
- ROC-AUC and PR-AUC: whether risky balls are ranked above safer balls;
- Brier score and log loss: whether probability magnitudes are useful;
- observed-versus-predicted shape: whether spline curves track patterns in real rows;
- coefficient magnitude: a diagnostic, not a causal interpretation.

The last warning matters because several clocks describe almost the same state. For example,
`over`, `legal_balls`, and `phase` are strongly related. Coefficients can compensate for one another,
so varying one while freezing the others can create impossible synthetic balls and misleading
partial-dependence plots.

### 3.4 LightGBM and XGBoost

The notebook next fits two gradient-boosted tree models on 37 raw features. `ground` is added as a
native categorical feature. There is no standardization or spline expansion: a tree can learn a
rule such as `bat_runs >= 50` directly and can combine it with another condition such as death-over
phase.

Gradient boosting still minimizes log loss, but it does not backpropagate through a neural network.
At each round it computes how the current probabilities should change to reduce loss, then fits a
small decision tree to those gradients. The new tree is added to the existing score. Useful neural
network analogies are:

| Boosting concept | Rough NN analogy | Important difference |
|---|---|---|
| `learning_rate` | optimizer step size | It scales a newly added tree. |
| one boosting round | one incremental optimization step | The step adds a new function/tree, not an update to every existing weight. |
| leaves/depth | model capacity | Capacity is piecewise decision rules, not hidden units. |
| `min_child_samples`, `reg_lambda` | regularization | They constrain leaf formation and values. |
| early stopping | early stopping on validation loss | This is a close analogy. |
| average of models | deep ensemble | Different random seeds vary row/feature subsampling rather than neural initialization alone. |

The configured upper bound is thousands of rounds, but 2024 log loss is monitored after each
round. Training stops when it has failed to improve for the configured patience. The final
2025–2026 period does not send gradients into the model and does not determine the stopping round.

Class rebalancing and SMOTE are deliberately absent. Wickets are rare, but artificially changing
their prevalence would make raw probabilities too high unless a separate correction were applied.
That might help a thresholded confusion matrix while hurting the actual deliverable: an honest risk
estimate.

LightGBM and XGBoost land very close to each other, at roughly `0.1943` test log loss. Averaging
their probabilities gives a tiny additional reduction. Agreement between independently implemented
boosters is also evidence that the remaining limitation is more likely the available signal than a
particular library.

### 3.5 Pruning and matchup experiments

The later cells are model-development experiments rather than a single linear training script:

1. Gain importance suggests features a tree used, but can over-credit high-cardinality or correlated
   columns.
2. Drop-column ablation refits a model without a feature group and observes the loss change.
3. Permutation importance breaks a feature at evaluation time and measures the damage. Correlated
   clocks are also permuted together so they cannot substitute for one another.
4. A smaller 14-feature LightGBM is fitted with stronger regularization and averaged across five
   seeds.
5. Batting position and bowler-style/matchup columns are tried as candidate additions.

The pruned wicket model is slightly better numerically and much smaller, but a paired match-level
bootstrap interval includes zero for its improvement over the earlier tree. The defensible reason
to prefer it is simplicity, not a claim of proven superior generalization. Bowler-style features do
not show a stable wicket-loss improvement in the explored period.

## 4. `runs.ipynb`: fitting the run distribution

### 4.1 Filtering and label construction

The runs notebook starts from the same parquet but calls `model_runs.label` before splitting:

1. Keep only `is_legal == 1`. Wides and no-balls are a different process and are excluded.
2. Read `runs_batter`, the runs credited to the batter rather than total extras.
3. Fold the extremely rare value 5 into class 4.
4. Map cricket values `[0, 1, 2, 3, 4, 6]` to contiguous class IDs `[0, 1, 2, 3, 4, 5]` required by
   the ML libraries.

Five runs off the bat occurs only 74 times in the full history, usually due to overthrows. Folding
it into 4 prevents an almost-empty seventh class, but it is an approximation. It slightly
understates expected runs for those rare rows.

The fitted output is a complete categorical distribution:

```text
[P(0), P(1), P(2), P(3), P(4), P(6)]
```

The probabilities sum to one. When a caller needs one numeric estimate, the notebook derives it
without fitting another regressor:

```text
E[runs] = 0*P(0) + 1*P(1) + 2*P(2) + 3*P(3) + 4*P(4) + 6*P(6)
```

This preserves the odd shape of cricket scoring. A continuous regressor would act as though every
value between zero and six were an ordinary possible result, despite most probability mass being
on 0, 1, 4, and 6.

The scope is conditional: this is `P(runs off bat | delivery is legal, pre-ball state)`. It is not
an unconditional next-delivery simulator. Such a simulator also needs a legality/extras model.

### 4.2 The null model and multinomial logistic baseline

The null distribution is the outcome frequency in the training seasons:

```text
0: 0.3805   1: 0.3839   2: 0.0653
3: 0.0032   4: 0.1174   6: 0.0498
```

Predicting this same vector for every ball is the minimum baseline. The first learned model uses
the same imputation, spline, scaling, and one-hot preprocessing as the wicket notebook, followed by
multinomial logistic regression.

In TensorFlow terms, the trainable part is approximately:

```python
tf.keras.layers.Dense(6, activation="softmax",
                      kernel_regularizer=tf.keras.regularizers.l2(...))
```

The loss is multiclass cross-entropy. Every row changes all six class logits: the observed class is
pushed up and competing classes are pushed down. The current test log loss is about `1.35260`, a
3.42% reduction from the test null loss of `1.40050`.

### 4.3 Multiclass boosted trees

LightGBM and XGBoost are then trained with multiclass objectives. Conceptually, LightGBM grows one
tree per class in each boosting round, so 258 rounds correspond to 1,548 class-specific trees. The
softmax of the six accumulated class scores becomes the probability vector.

The fitting loop is otherwise the same as the wicket version:

1. Fit on 2008–2023 rows.
2. Monitor 2024 multiclass log loss.
3. Stop adding trees after validation loss stops improving.
4. Generate probabilities for 2025–2026 without fitting on those labels.
5. Optionally average LightGBM and XGBoost probabilities row by row.

Diagnostics are class-aware. In addition to total log loss, the notebook examines dot-ball ranking,
boundary ranking, classwise loss/calibration, and expected-runs RMSE. A model can improve the full
distribution without much changing a particular derived metric.

### 4.4 Feature pruning and the era shift

Ablation and permutation tests reduce the tree input from 37 to 15 features, then five LightGBM
models with stronger regularization are fitted using seeds 0–4 and their probabilities are averaged.
The ensemble reduces sensitivity to random row and column subsampling.

The notebook then exposes an important distribution shift:

```text
training mean, 2008–2023: 1.2923 runs per legal ball
evaluation mean, 2025–2026: 1.5448 runs per legal ball
raw model mean prediction: 1.4755 runs per legal ball
```

The model ranks situations reasonably but inherits too much of the older scoring environment. This
is a calibration problem: the class levels have moved even when useful within-season relationships
remain.

An initial prior-shift correction and temperature scaling are explored. Temperature scales all
logit gaps with one parameter and mainly controls confidence. It does not directly express that,
for example, sixes became more common while threes became less common, so it gives little benefit
here. Per-class offsets are a better match for that failure mode.

### 4.5 Bowler style and matchup features

The final feature set combines the 15 retained state/history columns with:

```text
bowler_type, bowler_arm, bat_hand, same_handed, turn_into_batter
```

These allow the trees to fit rules that distinguish pace/spin, left/right arm, batter handedness,
and whether spin turns into the batter. Unlike the wicket experiment, this matchup information
improves run-distribution loss in the explored period.

### 4.6 The latest independently calibrated fit

The last notebook section delegates the reproducible fit to
[`model_runs_calibrated.py`](../model_runs_calibrated.py). It gives each time window exactly one
role:

| Seasons | Operation | Parameters affected |
|---|---|---|
| 2008–2022 | Fit five temporary selector models. | Tree splits and leaf values. |
| 2023 | Early-stop each selector and record its best round count. | Number of rounds only. |
| 2008–2023 | Refit five final tree models using the selected round counts. | Final tree splits and leaf values. |
| 2024 | Fit one regularized class-bias calibrator to the ensemble predictions. | Six class intercepts; no tree changes. |
| 2025–2026 | Evaluate the completed tree ensemble plus calibrator. | Nothing. |

This separation fixes a subtle issue in the earlier experiment: using 2024 both to choose the tree
count and fit calibration lets two decisions adapt to the same labels. It is not direct training on
the final period, but the calibration estimate is cleaner when it has its own data.

For each of five seeds, fitting happens twice. The temporary fit chooses a round count from 2023;
the model is then rebuilt on the larger 2008–2023 set for exactly that many rounds. The selected
counts are currently `[346, 383, 386, 339, 308]`. The five probability matrices are averaged before
calibration.

The calibrator learns only a class-specific bias:

```text
calibrated_p[k] = softmax(log(raw_p[k]) + bias[k])
```

Adding the same constant to every bias would leave the softmax unchanged, so the class-6 bias is
fixed to zero for identifiability. The other five values are optimized on 2024 by L-BFGS-B with an
L2 penalty. Regularization is particularly useful for class 3, which has few examples.

From a neural-network perspective, calibration is a tiny output layer over existing logits. Its
weight matrix is fixed to the identity and only the bias vector is trainable. It can correct class
prevalence, but it cannot learn that a specific ground, batter, or matchup behaves differently. It
also leaves within-class rankings unchanged.

The current combined model has:

| Metric on 2025–2026 legal balls | Result |
|---|---:|
| Multiclass log loss | `1.346543` |
| Reduction versus null loss | `3.853%` |
| Mean predicted runs | `1.5572` |
| Expected-runs RMSE | `1.8174` |
| Boundary ROC-AUC | `0.6158` |

It improves log loss by `0.001860` over its own uncalibrated style model. A paired bootstrap that
resamples whole matches gives a 95% interval of `[0.001262, 0.002474]`, which remains positive. It
also improves by `0.001031` over the earlier pruned prior-shift model, with interval
`[0.000297, 0.001765]`.

These are modest numerical gains, as expected for a mature ball-level model. Their value is that
the final probability vector is consistently more plausible, not that the predicted most-likely
class changes often.

## 5. How loss and metrics connect to fitting

### Binary log loss

For wicket label `y` and predicted probability `p`:

```text
loss = -mean(y*log(p) + (1-y)*log(1-p))
```

A confident wrong answer is penalized strongly. Predicting 0.01 on a wicket is much worse than
predicting 0.10 on that wicket. That behaviour is desirable when downstream code consumes the
probability itself.

### Multiclass log loss

For the six-class run model:

```text
loss = -mean(log(probability assigned to the outcome that occurred))
```

Only the probability of the observed class appears explicitly for a row, but because softmax
probabilities sum to one, increasing it necessarily reduces probability assigned elsewhere.

The notebooks report gain over the appropriate constant predictor:

```text
gain_percent = 100 * (null_loss - model_loss) / null_loss
```

This is easier to compare across targets than raw loss alone. It answers how much predictable
cross-entropy the model recovered beyond season-history frequencies.

### Why a confusion matrix is not the primary view

At roughly a 5% wicket rate, always predicting “no wicket” is about 95% accurate and produces a
simple-looking but useless confusion matrix. A confusion matrix also requires a threshold; changing
the threshold changes the matrix without changing the underlying probabilities.

Use a confusion matrix after specifying an operational decision and its costs—for example, when a
warning should trigger. For a simulator, live graphic, or expected-score calculation, log loss and
calibration are more direct.

Other reported metrics answer different questions:

| Metric | What it answers | What it does not answer |
|---|---|---|
| ROC-AUC | Are positive/high-value events ranked above others? | Are probabilities numerically correct? |
| PR-AUC | How useful is rare-event ranking? | Is the output calibrated? |
| Brier score | How close are binary probabilities to outcomes in squared-error terms? | Where errors occur by risk band. |
| Expected-runs RMSE | Is the mean of the predicted run distribution close to the observed number? | Is the six-class distribution shaped correctly? |
| Calibration table/curve | Does predicted frequency match observed frequency? | Does the model rank rows well? |

Single-ball expected-runs RMSE remains high because cricket outcomes are inherently noisy. Even an
excellent distribution cannot know whether the next good ball produces 0, 1, 4, or 6. Distribution
quality is the product; the expected value is only one summary of it.

### Why uncertainty is clustered by match

Ordinary row-level confidence intervals pretend 33,000 evaluation deliveries are independent. They
are not: all balls in a match share conditions, teams, selection, and tactics.

`evaluation.clustered_log_loss_gain` computes the per-row loss difference between two models, keeps
the predictions paired, and resamples whole matches. A positive interval means the candidate tends
to win across match resamples, rather than gaining from a handful of correlated balls. Averaging
seeds addresses fitting randomness; the match bootstrap addresses evaluation-sample uncertainty.
They are different questions.

## 6. Shared implementation details and caveats

### Categorical vocabularies

Tree helpers construct a common list of category names across the full table before creating pandas
categoricals. This prevents XGBoost from failing when a ground first appears after the training
cutoff. Only names such as a ground string are shared—not target averages or labels—so this does not
directly leak outcome information. A production service should instead persist the training
vocabulary and define explicit unknown-category behaviour.

### Missing values

The logistic pipelines median-impute numeric values and include missingness flags. Tree libraries
can route missing numeric values natively. Some nulls are meaningful—for example, required run rate
does not exist in the first innings—so missingness can itself contain useful state information.

### The two outputs are not a joint distribution

The wicket model estimates `P(wicket)`. The run model estimates `P(runs | legal delivery)`. They are
separate marginal models. Multiplying their outputs does not automatically produce a valid joint
distribution because runs, wickets, legality, and dismissal type are dependent. A complete innings
simulator needs an explicit factorization or a joint target, plus modelling for wides, no-balls,
extras, and strike changes.

### Evaluation status

The 2025–2026 rows are temporally excluded from parameter fitting, but they have been inspected
repeatedly during feature pruning and follow-up experiments. They are therefore an exploratory final
period, not a pristine one-shot holdout. Small model-to-model improvements should be confirmed on a
future IPL season or through a predeclared rolling-origin evaluation.

### What is still needed for production inference

The current code demonstrates fitting and evaluation but does not yet create a versioned artifact.
A production training path should save, together:

- preprocessing objects and category vocabularies;
- all fitted ensemble members and the calibration bias;
- exact ordered feature names, types, and null policy;
- outcome-to-class mapping and legal-delivery scope;
- training/calibration date windows, code/data version, and metrics;
- an inference contract and tests that prevent pre-ball/post-ball leakage.

The same feature builder, with the same timing semantics, must be used online. Reimplementing
features independently in a service is a common source of training/serving skew.

## 7. Reproducing the fits

From the project root:

```bash
uv sync
python build_ball_data.py
```

Then run `starter.ipynb` or `runs.ipynb` from top to bottom. The cells depend on variables and models
created earlier in the notebook, so executing isolated late cells in a fresh kernel will fail or,
worse, reuse stale state.

The latest run model also has a standalone entry point:

```bash
python model_runs_calibrated.py
```

That command refits the five-seed ensemble and calibrator and prints its round counts, class biases,
evaluation metrics, and match-bootstrap calibration interval. Because it refits LightGBM models, it
is expected to take substantially longer than merely loading the parquet or rendering notebook
plots.
