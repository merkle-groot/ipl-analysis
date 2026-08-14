"""Flatten Cricsheet IPL JSON into a ball-level table for wicket modelling.

One row per delivery. Every feature describes the state of the match *before*
the delivery is bowled, so the table can be fed to a classifier without
leaking the outcome. Run:

    python3 build_ball_data.py            # -> ball_data.parquet

Design notes worth knowing before you model with this:

* Target is `wicket`: any dismissal on the ball, excluding "retired hurt" and
  "retired out" (not dismissals in a predictive sense). `wicket_striker` and
  `wicket_bowler` are narrower targets - see build_row().
* Career features (batter/bowler history) are snapshots taken *before the
  current match*, built by walking matches in date order. A player's stats
  never include the match being predicted.
* Super overs are dropped entirely.
* Wides are not balls faced and not legal balls; no-balls are balls faced but
  not legal balls (standard scorecard convention).
"""

import json
from collections import defaultdict
from pathlib import Path

import polars as pl

SRC = Path("ipl_male_json")
OUT = Path("ball_data.parquet")

# Dismissals that are not "the batter got out to this ball".
NOT_OUT_KINDS = {"retired hurt", "retired out"}
# Dismissals credited to the bowler.
BOWLER_KINDS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}

# League-average per-ball rates, used as Bayesian priors so a player with 12
# career balls is shrunk toward average instead of showing a 0.0 or 1.0 rate.
PRIOR_BALLS = 120.0
PRIOR_DISMISSAL_RATE = 0.0497
PRIOR_STRIKE_RATE = 1.32  # runs per ball

# Grounds get their own, far heavier shrinkage. Measured across the 20
# well-sampled grounds, the between-ground spread in wicket rate is sd 0.0027,
# of which sd 0.0023 is binomial noise - the true signal is only tau = 0.0014.
# The empirical-Bayes weight is k = p(1-p)/tau^2 ~= 24,000 balls, so even a
# ground with 15k balls of history is pulled most of the way to league average.
# Runs per ball is a genuinely strong venue effect (1.19-1.52 across grounds),
# so it gets far lighter shrinkage.
PRIOR_BALLS_VENUE_WICKET = 24_000.0
PRIOR_BALLS_VENUE_RUNS = 1_500.0
PRIOR_RUNS_PER_BALL = 1.36

# Cricsheet writes the same ground under several names. Keys are the string
# left after dropping the ", City" suffix; values are the canonical ground.
# Deliberately NOT merged: Sardar Patel Stadium (Motera, 2010-15) into
# Narendra Modi Stadium (2021-) - the ground was demolished and rebuilt in
# between, so the pitch has no continuity.
VENUE_ALIASES = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",  # renamed 2019
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Zayed Cricket Stadium": "Sheikh Zayed Stadium",  # same Abu Dhabi ground
}


def canon_venue(venue: str) -> str:
    """'MA Chidambaram Stadium, Chepauk, Chennai' -> 'MA Chidambaram Stadium'."""
    base = venue.split(",")[0].strip()
    return VENUE_ALIASES.get(base, base)


def career_rate(events: float, balls: float, prior: float, weight: float = PRIOR_BALLS) -> float:
    return (events + prior * weight) / (balls + weight)


class Career:
    """Running per-player history, updated only at match boundaries."""

    def __init__(self) -> None:
        self.bat_balls = defaultdict(int)
        self.bat_runs = defaultdict(int)
        self.bat_outs = defaultdict(int)
        self.bowl_balls = defaultdict(int)
        self.bowl_runs = defaultdict(int)
        self.bowl_wkts = defaultdict(int)
        # (batter, bowler) -> times dismissed / balls faced in that matchup
        self.h2h_outs = defaultdict(int)
        self.h2h_balls = defaultdict(int)
        self.venue_balls = defaultdict(int)
        self.venue_runs = defaultdict(int)
        self.venue_wkts = defaultdict(int)

    def batter_features(self, p: str) -> dict:
        balls = self.bat_balls[p]
        return {
            "bat_career_balls": balls,
            "bat_career_dismissal_rate": career_rate(
                self.bat_outs[p], balls, PRIOR_DISMISSAL_RATE
            ),
            "bat_career_sr": career_rate(self.bat_runs[p], balls, PRIOR_STRIKE_RATE),
        }

    def h2h_features(self, batter: str, bowler: str) -> dict:
        return {
            "h2h_dismissals": self.h2h_outs[(batter, bowler)],
            "h2h_balls": self.h2h_balls[(batter, bowler)],
        }

    def venue_features(self, g: str) -> dict:
        balls = self.venue_balls[g]
        return {
            "venue_prior_balls": balls,
            "venue_wicket_rate": career_rate(
                self.venue_wkts[g], balls, PRIOR_DISMISSAL_RATE, PRIOR_BALLS_VENUE_WICKET
            ),
            "venue_runs_per_ball": career_rate(
                self.venue_runs[g], balls, PRIOR_RUNS_PER_BALL, PRIOR_BALLS_VENUE_RUNS
            ),
        }

    def bowler_features(self, p: str) -> dict:
        balls = self.bowl_balls[p]
        return {
            "bowl_career_balls": balls,
            "bowl_career_wicket_rate": career_rate(
                self.bowl_wkts[p], balls, PRIOR_DISMISSAL_RATE
            ),
            # runs per BALL, not the conventional runs-per-over economy: x6
            "bowl_career_runs_per_ball": career_rate(
                self.bowl_runs[p], balls, PRIOR_STRIKE_RATE
            ),
        }


def in_powerplay(powerplays: list, over: int) -> bool:
    return any(int(p["from"]) <= over <= int(p["to"]) for p in powerplays)


def phase_of(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over < 15:
        return "middle"
    return "death"


def overs_to_balls(overs: int | float | str) -> int:
    """Convert scorecard notation (``9.2`` = 9 overs + 2 balls) to balls.

    Cricsheet target overs use cricket notation rather than a decimal fraction.
    Multiplying 9.2 by six would incorrectly return 55 instead of 56.
    """
    text = str(overs)
    whole, dot, ball = text.partition(".")
    extra = int(ball) if dot else 0
    if not 0 <= extra <= 5:
        raise ValueError(f"invalid over notation: {overs!r}")
    return 6 * int(whole) + extra


def flatten_match(path: Path, career: Career) -> tuple[list[dict], list]:
    """Return (rows, career_updates) for one match."""
    match = json.loads(path.read_text())
    info = match["info"]
    rows: list[dict] = []
    updates: list = []  # (kind, player, ...) applied after the match

    match_id = path.stem
    date = min(info["dates"])
    season = str(info["season"])
    venue = info.get("venue")
    ground = canon_venue(venue) if venue else None
    teams = info.get("teams", [])
    toss_winner = info.get("toss", {}).get("winner")
    toss_decision = info.get("toss", {}).get("decision")

    for innings_no, inn in enumerate(match["innings"], start=1):
        if inn.get("super_over"):
            continue
        if innings_no > 2:  # super-over remnants
            continue

        batting_team = inn["team"]
        bowling_team = next((t for t in teams if t != batting_team), None)
        powerplays = inn.get("powerplays", [])

        target = inn.get("target")
        is_chase = target is not None
        target_runs = target["runs"] if is_chase else None
        target_balls = overs_to_balls(target["overs"]) if is_chase else None

        # --- innings running state (all pre-ball) ---
        team_runs = 0
        team_wkts = 0
        legal_balls = 0
        balls_since_boundary = 0
        balls_since_wicket = 0

        bat = defaultdict(lambda: {"balls": 0, "runs": 0, "dots": 0, "since_bdry": 0})
        bowl = defaultdict(lambda: {"balls": 0, "runs": 0, "wkts": 0})

        for ov in inn["overs"]:
            over = ov["over"]
            for ball_in_over, ball in enumerate(ov["deliveries"], start=1):
                batter = ball["batter"]
                bowler = ball["bowler"]
                extras = ball.get("extras", {})
                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                is_legal = not (is_wide or is_noball)
                runs_bat = ball["runs"]["batter"]
                runs_total = ball["runs"]["total"]

                wickets = [
                    w for w in ball.get("wickets", []) if w["kind"] not in NOT_OUT_KINDS
                ]
                striker_out = any(w["player_out"] == batter for w in wickets)
                bowler_credited = any(w["kind"] in BOWLER_KINDS for w in wickets)

                b = bat[batter]
                bl = bowl[bowler]
                balls_left = target_balls - legal_balls if is_chase else None
                runs_needed = target_runs - team_runs if is_chase else None

                row = {
                    # --- identity / grouping (not features) ---
                    "match_id": match_id,
                    "date": date,
                    "season": season,
                    "year": int(date[:4]),
                    "venue": venue,
                    "ground": ground,
                    "innings": innings_no,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "batter": batter,
                    "non_striker": ball["non_striker"],
                    "bowler": bowler,
                    "over": over,
                    "ball_in_over": ball_in_over,
                    # --- targets ---
                    "wicket": int(bool(wickets)),
                    "wicket_striker": int(striker_out),
                    "wicket_bowler": int(bowler_credited),
                    "dismissal_kind": wickets[0]["kind"] if wickets else None,
                    # --- ball outcome (NOT features: post-ball) ---
                    "runs_batter": runs_bat,
                    "runs_total": runs_total,
                    "is_legal": int(is_legal),
                    "is_wide": int(is_wide),
                    "is_noball": int(is_noball),
                    # --- match situation (pre-ball) ---
                    "team_runs": team_runs,
                    "team_wickets": team_wkts,
                    "wickets_in_hand": 10 - team_wkts,
                    "legal_balls": legal_balls,
                    "balls_remaining_innings": max(0, 120 - legal_balls),
                    "crr": team_runs / (legal_balls / 6) if legal_balls else 0.0,
                    "balls_since_boundary": balls_since_boundary,
                    "balls_since_wicket": balls_since_wicket,
                    "is_powerplay": int(in_powerplay(powerplays, over)),
                    "phase": phase_of(over),
                    "is_chase": int(is_chase),
                    # --- striker state (pre-ball) ---
                    "bat_balls": b["balls"],
                    "bat_runs": b["runs"],
                    "bat_dots": b["dots"],
                    "bat_dot_pct": b["dots"] / b["balls"] if b["balls"] else None,
                    "bat_sr": b["runs"] / b["balls"] if b["balls"] else None,
                    "bat_balls_since_boundary": b["since_bdry"],
                    "bat_is_new": int(b["balls"] < 6),
                    # --- bowler state this innings (pre-ball) ---
                    "bowl_balls": bl["balls"],
                    "bowl_runs": bl["runs"],
                    "bowl_wickets": bl["wkts"],
                    "bowl_econ": bl["runs"] / (bl["balls"] / 6) if bl["balls"] else None,
                    # --- chase pressure (None in the 1st innings) ---
                    "target_runs": target_runs,
                    "runs_required": runs_needed,
                    "balls_left": balls_left,
                    "rrr": (
                        runs_needed / (balls_left / 6)
                        if is_chase and balls_left and balls_left > 0
                        else None
                    ),
                    # --- context ---
                    "toss_won": int(toss_winner == batting_team) if toss_winner else None,
                    "toss_decision": toss_decision,
                }
                if row["rrr"] is not None:
                    row["rrr_minus_crr"] = row["rrr"] - row["crr"]
                else:
                    row["rrr_minus_crr"] = None

                row.update(career.batter_features(batter))
                row.update(career.bowler_features(bowler))
                row.update(career.venue_features(ground))
                row.update(career.h2h_features(batter, bowler))
                rows.append(row)

                # --- advance state (strictly after the row is emitted) ---
                team_runs += runs_total
                team_wkts += len(wickets)
                if is_legal:
                    legal_balls += 1

                if runs_bat in (4, 6):
                    balls_since_boundary = 0
                else:
                    balls_since_boundary += 1
                balls_since_wicket = 0 if wickets else balls_since_wicket + 1

                if not is_wide:  # wides are not faced
                    b["balls"] += 1
                    b["runs"] += runs_bat
                    if runs_total == 0:
                        b["dots"] += 1
                    b["since_bdry"] = 0 if runs_bat in (4, 6) else b["since_bdry"] + 1
                    updates.append(("bat_ball", batter, runs_bat))
                    updates.append(("h2h_ball", batter, bowler))
                if striker_out:
                    updates.append(("bat_out", batter))

                if is_legal:
                    bl["balls"] += 1
                # wides/no-balls are charged to the bowler, byes/legbyes are not
                conceded = runs_bat + extras.get("wides", 0) + extras.get("noballs", 0)
                bl["runs"] += conceded
                bl["wkts"] += int(bowler_credited)
                updates.append(("bowl_ball", bowler, conceded, int(is_legal),
                                int(bowler_credited)))
                updates.append(("venue_ball", ground, runs_total, int(is_legal),
                                len(wickets)))
                # bowler-credited kinds exclude run outs and obstructing the
                # field, so this only counts the striker falling to the bowler
                if bowler_credited and striker_out:
                    updates.append(("h2h_out", batter, bowler))

    return rows, updates


STYLE_COLS = ["bowler_type", "bowler_arm", "spin_kind", "bat_hand",
              "same_handed", "turn_into_batter"]


def add_player_style(df: pl.DataFrame) -> pl.DataFrame:
    """Join bowling style and batting hand, and build the matchup from them.

    None of this is derived from the ball data, so there is nothing to leak: a
    bowler's action and a batter's stance are fixed properties of the player,
    known before the season starts. Built by fetch_bowler_style.py.

    The matchup columns are the point of the exercise. Handedness on its own
    says little - an arm only means something once you know who is facing it:

    * `same_handed` - right-arm to a right-hander, left to a left. For pace this
      is mostly about the angle the ball arrives on.
    * `turn_into_batter` - which way a spinner's ball moves relative to the bat.
      Finger spin turns into a batter of the same handedness as the bowling arm
      (off-break to a right-hander, orthodox to a left-hander); wrist spin turns
      into the opposite one (leg-break to a left-hander). Null for pace, where
      turn is not the variable.
    """
    path = Path("player_style.csv")
    if not path.exists():
        print("! player_style.csv missing - run fetch_bowler_style.py; "
              "style columns will be null")
        return df.with_columns([pl.lit(None, pl.String).alias(c) for c in STYLE_COLS[:4]]
                               + [pl.lit(None, pl.Int64).alias(c) for c in STYLE_COLS[4:]])

    style = pl.read_csv(path).with_columns(
        [pl.col(c).replace("", None) for c in
         ("bowler_type", "bowler_arm", "spin_kind", "bat_hand")])

    bowl = style.select(pl.col("player").alias("bowler"), "bowler_type",
                        "bowler_arm", "spin_kind")
    bat = style.select(pl.col("player").alias("batter"), pl.col("bat_hand"))

    finger = pl.col("spin_kind") == "finger"
    same = pl.col("bowler_arm") == pl.col("bat_hand")
    out = (df.join(bowl, on="bowler", how="left")
             .join(bat, on="batter", how="left")
             .with_columns(
                 same.cast(pl.Int64).alias("same_handed"),
                 pl.when(pl.col("bowler_type") != "spin").then(None)
                   .when(pl.col("spin_kind").is_null()).then(None)
                   .otherwise((finger == same).cast(pl.Int64))
                   .alias("turn_into_batter")))

    for col, what in [("bowler_type", "deliveries"), ("bat_hand", "deliveries")]:
        missing = out.filter(pl.col(col).is_null()).height
        if missing:
            print(f"! {missing:,} {what} ({100 * missing / out.height:.2f}%) "
                  f"have no {col} - rerun fetch_bowler_style.py")
    return out


def main() -> None:
    files = sorted(SRC.glob("*.json"))
    # Date order matters: career features must only ever see earlier matches.
    dated = sorted(
        ((min(json.loads(f.read_text())["info"]["dates"]), f) for f in files),
        key=lambda t: (t[0], t[1].stem),
    )

    career = Career()
    all_rows: list[dict] = []

    for _, path in dated:
        rows, updates = flatten_match(path, career)
        all_rows.extend(rows)
        for u in updates:  # applied only after the whole match is emitted
            if u[0] == "bat_ball":
                career.bat_balls[u[1]] += 1
                career.bat_runs[u[1]] += u[2]
            elif u[0] == "bat_out":
                career.bat_outs[u[1]] += 1
            elif u[0] == "bowl_ball":
                career.bowl_runs[u[1]] += u[2]
                career.bowl_balls[u[1]] += u[3]
                career.bowl_wkts[u[1]] += u[4]
            elif u[0] == "h2h_ball":
                career.h2h_balls[(u[1], u[2])] += 1
            elif u[0] == "h2h_out":
                career.h2h_outs[(u[1], u[2])] += 1
            elif u[0] == "venue_ball":
                career.venue_runs[u[1]] += u[2]
                career.venue_balls[u[1]] += u[3]
                career.venue_wkts[u[1]] += u[4]

    # infer_schema_length=None: the first innings leaves every chase column
    # null, so a short inference window types them as Null and then fails.
    df = pl.DataFrame(all_rows, infer_schema_length=None)
    df = add_player_style(df)
    df.write_parquet(OUT)

    print(f"{df.height:,} deliveries from {df['match_id'].n_unique():,} matches")
    print(f"wicket rate: {df['wicket'].mean():.4f}")
    print(f"seasons: {df['year'].min()}-{df['year'].max()}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
