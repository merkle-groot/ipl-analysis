<title>Ball-level wicket prediction — feature reference</title>

# Feature reference

Every column in `ball_data.parquet`: what it is, why it was included, and what it contributes.
Built by `build_ball_data.py` from 1,243 Cricsheet IPL match files.

295,557 deliveries, 1,243 matches, 2008–2026, 63 columns, 4.96% base wicket rate.

Rates quoted below are the fraction of deliveries that produced a wicket, measured across the
whole dataset. Unless noted, the target is `wicket` (any dismissal excluding retired hurt/out).

Importance figures are permutation importance on the 2025–26 test set, as a share of total
LightGBM skill. They carry roughly ±1–2 points of shuffle noise, so small differences between
features are not meaningful. See `starter.ipynb` for the calculation.

---

## The three rules that keep it leak-free

Everything in this file depends on these holding.

1. State is emitted before it advances. Inside the delivery loop, the row is appended and
*then* the running counters update. A feature can never see the ball it is predicting.

2. Cross-match history updates only at match boundaries. Career, venue and head-to-head
counters accumulate into a queue during a match and are applied after the whole match is
emitted. A player's career dismissal rate never includes the match being predicted. Matches are
processed in date order, so history is strictly backward-looking.

3. Post-ball columns are in the file but are not features. `runs_batter`, `runs_total`,
`is_wide`, `is_noball`, `is_legal` and `dismissal_kind` describe what happened *on* the ball.
They are kept for building rolling aggregates and slicing analyses. Putting them in a feature
matrix is instant leakage.

Verified by: reconstructed innings totals match the raw JSON for 399 innings (0 mismatches);
every innings' first ball has all state at zero; 7,195 debut balls carry `bat_career_balls == 0`;
`venue_prior_balls` never changes mid-match.

---

## Targets

Three, because "a wicket fell" is ambiguous and the right choice depends on the question.

| column | rate | counts |
|---|---|---|
| `wicket` | 0.0496 | Any dismissal on the ball. Excludes retired hurt (19) and retired out (6), which are not dismissals in a predictive sense. The default. |
| `wicket_striker` | 0.0477 | The striker was dismissed. Excludes non-striker run-outs. |
| `wicket_bowler` | 0.0455 | Bowler-credited only: bowled, caught, c&b, lbw, stumped, hit wicket. Excludes all 1,194 run-outs and obstructing the field. |

Use `wicket_bowler` when the question is about bowling skill. Run-outs come from a different
process (fielding, calling, panic) and no bowler feature can explain them, so including them
adds noise. This also matters for evaluation: measuring a run-out-free feature against a target
that includes run-outs will understate that feature.

---

## Match state

The scoreboard as the bowler runs in.

### `over` (0–19) and `legal_balls` (0–119)

Why included: wicket probability changes more across the innings than with anything else.
Batters take risks in the last over that they would not take in the 8th.

What the data says: a powerplay bump, a lull, then a steep climb at the death. The rate opens
at 0.032 in over 1, rises to 0.043 in over 5, falls back to 0.033 in over 7 as the field goes out
and both sides consolidate, moves sideways through the middle overs (0.036–0.045, overs 8–12),
then climbs from over 13 to 0.085 in over 19 and 0.130 in over 20. A ball in the final over is
3.9× as likely to take a wicket as one in the 7th.

The climb from over 13 onward is close to monotone. The middle overs are not. Do not assume a
monotone shape when modelling this column.

**Verdict:** keep both. This is the largest effect in the model. Shuffled together they cost
68.2% of skill.

Two warnings about reading their individual numbers. `legal_balls` is `over` at ball resolution
(`legal_balls` ≈ 6 × `over`), so the two columns say nearly the same thing:

- The split between them is arbitrary. They score 24.7% and 13.1% individually, but a different
  random seed would divide it differently. That split is not a finding.
- Do not add the individual numbers. Each is measured with the other column still intact and
  covering for it, so both come out low. Their sum is 37.8%, which understates the pair by nearly
  half. Only the joint figure, 68.2%, is meaningful.

### `team_wickets` (0–9) and `wickets_in_hand` (10 − team_wickets)

Why included: to capture how deep into the batting order we are, and whether a collapse is
under way.

What the data says: the rate rises steadily as wickets fall, from 0.0401 with all 10 in hand to
0.0517 at 6, 0.0778 at 4 and 0.1016 at 2.

**Verdict:** real signal, but nearly all of it is already covered elsewhere. The pair is 1.1% of
skill shuffled together; `team_wickets` alone is 0.8%, `wickets_in_hand` alone is −0.1%.

`wickets_in_hand` is `10 - team_wickets`, so the two columns hold identical information. Beyond
that, `legal_balls` already captures most of what either says: wickets fall as the innings goes
on, so knowing the over tells you roughly how many are down.

Keep `team_wickets`, drop `wickets_in_hand`. Dropping it will not change predictions. It
removes a column that looks informative and is not.

Batting position was tested as a replacement and did not help. `team_wickets` correlates 0.88
with it, so measuring position directly looked like the better version of this feature. It is
not. Adding it made the model slightly worse. See "Not included, and why" below.

### `team_runs` and `crr` (current run rate)

Why included: scoring context. A side at 90/1 off 10 overs is in a different position from
one at 60/5.

What the data says: `team_runs` is 2.2% of skill. `crr` is −0.2%, meaning shuffling it very
slightly improves the model. It adds nothing.

**Verdict:** keep `team_runs`, drop `crr`. Its information is already in `team_runs` and
`legal_balls`.

### `balls_since_wicket`

Why included: partnership length. A settled pair should be a different proposition from two
new batters.

What the data says: the rate rises steadily the longer a partnership lasts, from 0.0459 within 3
balls of a wicket to 0.0509 by 8–15 balls, 0.0526 by 31–60 and 0.0586 past 60 balls.

This is the opposite of the usual assumption. Long partnerships are *more* likely to end on any
given ball, not less. Probably the same mechanism as the milestone effect under `bat_runs`: a
set pair is expected to accelerate.

**Verdict:** keep. 7.7% of skill, fourth overall, and the strongest of the situational features
after time in the innings.

### `balls_since_boundary`

Why included: to test the pressure hypothesis. Dot balls build pressure, pressure causes
wickets.

What the data says: nothing. 0.0496 within 2 balls of a boundary, 0.0503 at 3–5, 0.0493 at
6–10, 0.0486 at 11–20, 0.0487 past 20. The line is flat, and what slope there is runs the wrong
way.

This result is counterintuitive enough that it is worth showing it survives controls. Longer
droughts happen later in the innings (mean over 8.8 → 11.7) and to weaker batters (mean career
rate 0.0472 → 0.0493), both of which should *raise* the wicket rate. Holding both roughly fixed,
in overs 7–15, split by batter quality:

| batter | 0–2 balls since boundary | 3–10 | 11+ |
|---|---|---|---|
| good | 0.0377 | 0.0374 | 0.0373 |
| weaker | 0.0464 | 0.0479 | 0.0443 |

Still flat. The effect is not being masked by confounding; it is absent.

Why there is nothing to find. A boundary drought is not batters straining against pressure.
It is batters playing conservatively, which is what causes the drought:

| balls since boundary | runs/ball | dot % |
|---|---|---|
| 0–2 | 1.32 | 33% |
| 3–10 | 1.21 | 36% |
| 11+ | 1.13 | 38% |

The pressure hypothesis assumes a drought makes batters force the pace. The data says a drought
*is* them not forcing it. Lower aggression means fewer wickets, which cancels whatever pressure
effect exists.

**Verdict:** drop. 0.2% of skill. Team-level pressure is not visible in per-ball wicket
probability.

### `is_powerplay` and `phase` (powerplay / middle / death)

Why included: fielding restrictions change which shots are worth playing. `is_powerplay`
comes from the innings' own `powerplays` block, so it is correct for rain-shortened games rather
than assuming overs 1–6.

What the data says: `phase` is 1.0%, `is_powerplay` is 0.0%. Both are coarser versions of
`over`, and a tree can find those boundaries itself.

**Verdict:** keep, but not for the model. They stay because they make analyses easier to read and
because the logistic regression needs the step function.

---

## Striker state

### `bat_balls` — balls faced by the striker this innings

Why included: to capture batters settling in.

What the data says is the opposite of that:

| balls faced | 0–2 | 3–5 | 6–10 | 11–20 | 21–40 | 41+ |
|---|---|---|---|---|---|---|
| wicket rate | 0.0428 | 0.0461 | 0.0512 | 0.0499 | 0.0528 | 0.0706 |

Risk rises with time at the crease. A batter who has faced 41+ balls is 65% more likely to be
dismissed on the next ball than one who has just arrived. In Tests the new-batter effect is real
and runs the other way; in T20 it inverts, because a set batter is expected to accelerate and a
new one is allowed time to look at it.

**Verdict:** keep. 4.4% of skill. The feature works, but not for the reason it was added.

### `bat_is_new` (bat_balls < 6)

0.0443 when new vs 0.0522 when not, the same inversion.

**Verdict:** drop. 0.2% of skill, and redundant now that `bat_balls` is in as a continuous
feature.

### `bat_runs` — the striker's score

Why included: risk appetite scales with the platform a batter has built.

What the data says (target: `wicket_striker`):

| score | 0–9 | 20–29 | 40–49 | 50–59 | 80–99 | 100+ |
|---|---|---|---|---|---|---|
| rate | 0.0438 | 0.0488 | 0.0523 | 0.0629 | 0.0732 | 0.0771 |

There is a jump at the fifty: 0.0486 on 45–49 rising to 0.0646 on 50–54.

This is not milestone nerves. If it were, the rate would spike and then settle, but 55–59 sits at
0.0636, essentially unchanged. It is a persistent level shift. Splitting by phase explains it:
in the death overs the rate is 0.0759 whether or not the batter is past 50, and the gap appears
only in the middle overs (0.0576 vs 0.0393). A batter past fifty in the middle overs has licence
to attack that a batter on 40 does not.

**Verdict:** keep. 2.4% of skill. The shape is why the logistic regression needs a spline basis. A
single linear term fits one slope through a flat-then-step curve and misses both parts of it.

### `bat_sr`, `bat_dot_pct` — the striker's rate and dot-ball share so far

Why included: to distinguish a batter who is timing it from one who is stuck.

What the data says: `bat_dot_pct` runs backwards from the intuition. Batters under 25% dots
are dismissed at 0.0577; those above 55% at 0.0435. Playing freely is riskier than blocking.

**Verdict:** drop both. 0.4% and −0.0%. The information is already in `bat_runs` and `bat_balls`,
and early in an innings the ratio is computed on too few balls to mean anything.

### `bat_balls_since_boundary`

The batter-level version of the pressure hypothesis. Unlike the team version there is a faint
effect, 0.0490 at 0–2 balls rising to 0.0650 past 20, but permutation importance is 0.1%. The
tail is thin (1,569 balls) and the effect is probably batter quality rather than pressure.

**Verdict:** drop.

---

## Bowler state

### `bowl_balls` — balls bowled in this innings (spell position)

What the data says: the steepest-looking gradient in the dataset.

| spell | over 1 | over 2 | over 3 | over 4 |
|---|---|---|---|---|
| rate | 0.0368 | 0.0422 | 0.0533 | 0.0786 |

**Verdict:** keep, 5.0% of skill, but do not read it as a spell effect. It is confounded with
match phase. A bowler's 4th over is usually at the death, and death overs produce wickets
regardless of who is bowling. This is not evidence that bowlers improve as a spell goes on.

### `bowl_runs`, `bowl_wickets`, `bowl_econ`

In-innings figures for the bowler. Runs conceded follows scoring convention: runs off the bat
plus wides and no-balls, excluding byes and leg-byes, which are not charged to the bowler.

**Verdict:** drop all three. Each is ≤0.2% of skill. A four-over spell is too small a sample to
say anything the career features do not say better.

---

## Chase pressure

Null for all 153,254 first-innings balls by design. Trees handle the nulls natively; the logistic
regression imputes with a missingness indicator. Targets come from the innings' own `target`
block, so DLS-revised targets are respected rather than recomputed from the first-innings total.

### `rrr`, `runs_required`, `balls_left`, `rrr_minus_crr`, `target_runs`

What the data says (chases only), by how far the required rate exceeds the current rate:

| rrr − crr | < −3 | −3..0 | 0..3 | 3..6 | 6+ |
|---|---|---|---|---|---|
| rate | 0.0432 | 0.0399 | 0.0443 | 0.0502 | 0.0725 |

A chase that has fallen 6+ runs per over behind produces wickets at nearly double the rate of one
cruising home.

**Verdict:** keep `rrr`, 6.3% of skill, sixth overall. The strongest situational feature after
time in the innings and partnership length.

### `is_chase`, `innings`

First-innings rate 0.0496 vs chase 0.0495. The two innings are indistinguishable in aggregate,
and both features are 0.0% of skill.

**Verdict:** drop both. Chasing does not raise wicket risk. Being *behind* in a chase does, and
`rrr_minus_crr` already measures that.

---

## Career features

Per-player history from all earlier matches, Bayesian-shrunk toward the league average with a
120-ball prior, so a debutant reads as average rather than as whatever their first three balls
suggested. The shrinkage formula is `(events + prior × 120) / (balls + 120)`.

### `bat_career_dismissal_rate`

The batter's historical probability of dismissal per ball faced. Ranges from about 0.030 (KL
Rahul, Kohli, one dismissal per 33 balls) to about 0.062 (Maxwell, Russell, one per 16).

**Verdict:** keep. 17.7% of skill, second overall, and the most valuable player feature.

Two caveats:

- It measures role as much as skill. Russell is not a worse batter than Saha; he is paid to
  hit from the first ball.
- It is badly miscalibrated for inexperienced batters, which is exactly where it would help
  most. See below.

#### The debutant problem

A player with no history gets the prior, 0.0497, the league average. That assumption is wrong,
because players with no record are mostly tailenders and fringe selections rather than typical
batters:

| career balls | feature says | actual rate |
|---|---|---|
| 0 | 0.0500 | 0.0741 |
| 1–50 | 0.0558 | 0.0686 |
| 51–150 | 0.0547 | 0.0591 |
| 151–400 | 0.0513 | 0.0520 |
| 401–1000 | 0.0473 | 0.0466 |
| 1001–3000 | 0.0424 | 0.0418 |

Past ~150 career balls the feature is well calibrated. Below that it understates risk, by 48% at
debut. The same effect compresses the feature across the order: it spans 0.0439→0.0613 by batting
position where the true spread is 0.0421→0.0992.

Do not fix this in the shrinkage. Tested and rejected, see `bat_career_balls` below. The
model already corrects it, and better than a reshrink can.

### `bat_career_balls`

Career experience. Median 636, max 6,884 (Kohli).

It does two jobs. It is a feature in its own right, and it is the confidence weight on the
other career features: at 120 balls the prior still carries 47% of the estimate, at 2,000 balls
only 6%. Keeping it lets a tree learn "trust the dismissal rate when this is high, ignore it when
low", an interaction that cannot be expressed otherwise.

**Verdict:** keep. 7.2% of skill, fifth overall, and most of that is repair work. Its main job
is fixing the debutant miscalibration described above. The tree learns "when career balls is low,
the dismissal rate is understated, so raise the estimate."

Dropping it costs far more than its own signal is worth:

| model | log loss |
|---|---|
| pruned, with `bat_career_balls` | 0.19424 |
| without it | 0.19445 |
| without it, but with the shrinkage prior corrected | 0.19440 |

Removing it costs 0.00021, about 3× the seed noise. Correcting the prior instead recovers only a
quarter of that. The tree's correction is conditional on the rest of the state, so it beats any
fixed adjustment to the feature.

Tested and rejected: retuning the shrinkage. Prior weight k = 120 (current), 60, 30 and 10
all land between 0.19423 and 0.19425, no difference. Raising the prior *mean* for inexperienced
batters to their measured rate (0.0691, estimated on training years only) makes things worse:
0.19433 at k = 30. Baking the correction into the feature removes the model's freedom to apply it
conditionally. The shrinkage is fine as it is; leave it alone.

### `bat_career_sr`, `bowl_career_wicket_rate`, `bowl_career_runs_per_ball`, `bowl_career_balls`

Career strike rate, and the bowler's equivalents. Note that `bowl_career_runs_per_ball` is runs
per ball, not the conventional runs-per-over economy. Multiply by 6 (Narine 1.13 → 6.8 rpo;
league average 1.32 → 7.92 rpo).

**Verdict:** drop all four. `bat_career_sr` is 1.3%; the three bowler features are 0.2%, −0.0%
and −0.6%.

Bowler quality does show a real gradient in isolation, 0.0467 to 0.0533 across
`bowl_career_wicket_rate` quartiles, but none of it survives once the situational features are
in the model. When a bowler is bowling matters much more than who he is.

---

## Venue

### `ground`, `venue_wicket_rate`, `venue_runs_per_ball`, `venue_prior_balls`

`venue` (60 raw strings) is canonicalised to `ground` (37 real grounds) by stripping the city
suffix and merging renames: Feroz Shah Kotla → Arun Jaitley (2019), Subrata Roy Sahara →
Maharashtra CA, Zayed Cricket Stadium → Sheikh Zayed, plus the `M.Chinnaswamy` spelling.

Sardar Patel (Motera, 2010–15) is deliberately *not* merged into Narendra Modi Stadium (2021–).
The ground was demolished and rebuilt between them, so the pitch has no continuity.

The two rate features use the same expanding-window mechanism as the career features, with
per-feature shrinkage derived from measurement rather than guessed:

- `venue_wicket_rate` uses a prior weight of 24,000 balls, from the empirical-Bayes formula
  `k = p(1−p)/τ²`, where τ = 0.0014 is the true between-ground signal after subtracting binomial
  noise. This deliberately crushes the feature: the final spread across grounds is sd 0.00064.
- `venue_runs_per_ball` uses a prior weight of 1,500 balls, because this signal is real. Final
  spread is sd 0.0675, from Kingsmead at 1.30 runs/ball to Narendra Modi at 1.51.

**Verdict:** drop all of them. Removing `ground` and both rate features leaves test log loss
unchanged to five decimal places (0.19434 → 0.19434). `venue_wicket_rate` has negative
permutation importance (−0.6%).

`ground` looks important and is not. It shows up as 9.3% of LightGBM's gain importance, third
on that chart. Gain importance flatters high-cardinality categoricals: 37 levels means 37 chances
to carve off a slice of noise, and every split gets credited. The ablation above is the check
that matters.

The cricket behind this: the stadium strongly affects scoring (1.30–1.51 runs/ball) and
barely affects wickets per ball, because teams adjust their aggression to conditions. On a
slow pitch they score less but also take fewer risks. The wicket rate equilibrates; the run rate
does not.

---

## Head-to-head

### `h2h_dismissals`, `h2h_balls`

How many times this bowler has previously dismissed this striker (bowler-credited kinds only, so
run-outs are excluded), and how many balls that pairing has faced. 22.9% of balls have some
history; only 7.2% have 30+ balls of it.

What the data says: among pairings with 24+ balls of history, sorting by prior head-to-head
dismissal rate gives:

| prior h2h rate | 0.000 | 0.024 | 0.040 | 0.080 |
|---|---|---|---|---|
| subsequent rate | 0.0371 | 0.0354 | 0.0397 | 0.0394 |

An eight-fold difference in prior rate produces a half-percentage-point spread in what happens
next. The "he's got his number" effect is regression to the mean, the same result found for
batter–pitcher matchups in baseball.

The raw count is actively misleading. Wicket rate *falls* as prior dismissals rise: 0.0464 at
zero, 0.0436 at one, 0.0394 at two. That is survivorship. Accumulating 3 dismissals against a
bowler requires facing him across many seasons, which selects for long-career top-order batters,
who have below-average dismissal rates. `h2h_balls` exists so the model can form the ratio and
escape that confound.

**Verdict:** keep, but they earn nothing. Together 0.2% of skill; dropping both costs 0.07% of
the gain. They stay because they are cheap and correctly built.

---

## Context and identity

`match_id`, `date`, `season`, `year`, `venue`, `innings`, `batting_team`, `bowling_team`,
`batter`, `non_striker`, `bowler`, `over`, `ball_in_over` are grouping and joining keys. Only
`innings` and `over` are used as features; the rest exist for slicing, splitting and debugging.

`toss_won` / `toss_decision`: 0.0495 when the batting side won the toss vs 0.0497 when it lost.
No effect. Included to check, kept as a record that it was checked.

---

## Verdict summary

Permutation importance on the 2025–26 test set, as a share of total LightGBM skill.

| tier | features | share |
|---|---|---|
| Backbone | `over` + `legal_balls`, shuffled together | 68.2% |
| Strong | `bat_career_dismissal_rate` 17.7%, `balls_since_wicket` 7.7%, `bat_career_balls` 7.2%, `rrr` 6.3% | 38.9% |
| Useful | `bowl_balls` 5.0%, `bat_balls` 4.4%, `ground` 3.9%, `bat_runs` 2.4%, `team_runs` 2.2% | 17.9% |
| Marginal | `rrr_minus_crr` 1.6%, `bat_career_sr` 1.3%, `team_wickets` 1.1%, `phase` 1.0% | 5.0% |
| Dead (≤0.4%) | `runs_required`, `bat_sr`, `bowl_career_*`, `bat_is_new`, `h2h_*`, `balls_since_boundary`, `venue_*`, `bat_balls_since_boundary`, `bowl_wickets`, `balls_left`, `wickets_in_hand`, `is_powerplay`, `is_chase`, `innings`, `bat_dot_pct`, `toss_won`, `crr` | ~0% |

The shares do not sum to 100%, and should not. Each feature is measured against the full
model with everything else left intact, so overlapping features are counted more than once.

Correlated features share credit and are individually under-credited. The backbone row shows
the joint figure for `over` and `legal_balls` because their individual numbers (24.7% and 13.1%)
sum to 37.8%, barely half the truth. `team_wickets` / `wickets_in_hand` have the same problem at
a much smaller scale. Do not drop one of a correlated pair on the strength of this table alone.

---

## Not included, and why

Recent form (last-5-innings average). Tested and rejected. It looks strong at first, worst
quintile 0.0582 vs best 0.0398, but the effect disappears once you split by career quality. For
established batters the rate across form quintiles is 0.0384 / 0.0409 / 0.0397 / 0.0386 / 0.0377,
which is flat. Form was acting as a proxy for player quality and batting position, and
`bat_career_dismissal_rate` measures that better. Correlation with next-ball dismissal: form
−0.029, career rate +0.047.

`batting_position`. Built and tested. It does not help: adding it to the pruned model
moved test log loss from 0.19423 to 0.19430, slightly worse.

The raw effect is real and large: 0.042 per ball for an opener against 0.103 for a No. 10, a 2.4×
spread, bigger than the death-overs effect. But the model already has three routes to the same
fact. `bat_career_dismissal_rate` identifies tailenders by their record, `team_wickets` says how
far down the order the innings is, and `bat_balls` says how long this batter has been in.
Position is a fourth view of what those already describe.

It does not need to be added to `build_ball_data.py`. It derives from the existing columns in a
few lines (order of first appearance per innings; see `starter.ipynb`).

Bowler type (pace/spin). Not in Cricsheet. Derivable from a bowler's own career pattern. Now
the most promising untested addition, since batting position turned out to add nothing.

Pitch/weather data. Not available. Given that venue contributes nothing measurable, pitch data
would most likely improve run-rate prediction rather than wicket prediction.
