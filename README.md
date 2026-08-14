# IPL, delivery by delivery

Ball-level models and a statistical broadsheet built from nineteen seasons of Indian
Premier League cricket: 1,243 matches, 295,557 deliveries, of which 284,465 are legal.

**Site:** [ipl.primemodulus.com](https://ipl.primemodulus.com/) · **Author:** [@0x1379](https://x.com/0x1379)

Two models are fitted on the same data. One predicts whether a delivery takes a wicket,
the other predicts the full distribution of runs off the bat. Both are trained on 2008–2023,
tuned against 2024 and scored on 2025–26, split by time rather than at random, because balls
from the same over are near-duplicates and a random split would let a model memorise a match.

The published page leads with the player findings and then works through the places where the
numbers disagree with the commentary. It carries the two notebooks in full behind tabs.

## What the models are worth

| | Best loss | Beats guessing by | Ranking |
|---|---|---|---|
| Wickets | 0.19423 | +2.33% | ROC-AUC 0.619 |
| Runs | 1.34654 | +3.85% | boundary AUC 0.616 |

Whether a batter scores is roughly 1.7 times more predictable than whether they get out.
Both figures come from an exploratory backtest rather than a clean one-shot holdout: the
2025–26 seasons were inspected more than once while features were pruned and calibrators
compared. Model comparisons carry 95% intervals from resampling whole matches, and the small
differences should be treated as hypotheses until a future season confirms them.

## Layout

```
build_ball_data.py        Cricsheet JSON -> one row per delivery -> ball_data.parquet
fetch_bowler_style.py     bowling action and batting stance for all 577 bowlers
organize_by_season.py     sorts the raw match files by season
model_*.py                feature lists, baselines and the fitted models
evaluation.py             shared scoring helpers
starter.ipynb             wicket model, start to finish
runs.ipynb                run-distribution model, start to finish
site/                     the generator for the published page
public/index.html         the built page, deployed as-is
docs/                     feature notes, model notes, data-collection to-dos
player_style.csv          bowler and batter styles, joined on Cricsheet player id
```

## Reproducing it

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The raw match files are not in this repo. Download the IPL set from
[Cricsheet](https://cricsheet.org/downloads/) into `ipl_male_json/`, along with the player
register as `people.csv`, then build the delivery table:

```bash
uv run python organize_by_season.py
uv run python fetch_bowler_style.py     # writes player_style.csv, needs network
uv run python build_ball_data.py        # writes ball_data.parquet, ~7 MB
```

`ball_data.parquet`, `people.csv` and the raw JSON are all excluded from version control:
they are either large or downloadable, and every one of them is reproducible from the
scripts above.

## Rebuilding the page

The site is generated from the parquet, so `ball_data.parquet` has to exist first.

```bash
uv run python site/stats_par.py         # par-adjusted player tables
uv run python site/stats_myths.py       # the counter-intuitive splits
uv run python site/stats_outliers.py    # boom-or-bust, warm-up, six-proof bowlers
uv run python site/charts.py            # writes every chart as static SVG
uv run python site/build.py             # renders public/index.html
uv run python site/audit.py             # checks charts for overlapping labels
```

Intermediates land in `site/_data/` and are not committed. `build.py` also renders both
notebooks into the page, using a small renderer in `site/nb2html.py` rather than nbconvert,
because nbconvert ships its own stylesheet and, in some templates, external script tags.

`audit.py` is worth running after any chart change. It parses every `<text>` element out of
the generated SVG, estimates its bounding box, and reports collisions with other labels, with
plotted lines and with the viewBox edge. It has caught several bugs that were invisible in
review.

## A note on "par"

Raw strike rates and economy rates mostly describe the job a player was given rather than how
well they did it. Every delivery on the site is therefore scored against what the whole league
managed in the same season, the same phase of the innings and against the same type of bowling.
That removes the twenty per cent scoring inflation of the last nineteen years and most of the
batting-order effect. It does not remove the quality of the opposition, the ground, or the
state of the match.

## Deploying

The page is a single self-contained HTML file with no external requests, so any static host
will serve it. `vercel.json` points Vercel at `public/`; there is no build step, because the
generator needs the parquet that is deliberately not in the repo.

## Data

Ball-by-ball records from [Cricsheet](https://cricsheet.org), under the
[Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1-0/).
Bowling actions and batting stances were reconstructed separately, since Cricsheet records
neither; see `docs/BOWLER_STYLE_TODO.md` for how, and for the 25 uncapped players filled in
by hand.
