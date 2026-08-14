"""Hunt for counter-intuitive effects. Everything is measured within phase where the
clock would otherwise confound it."""
import json

import polars as pl

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "_data"
DATA.mkdir(parents=True, exist_ok=True)
PARQUET = ROOT / "ball_data.parquet"


d = pl.read_parquet(str(PARQUET)).filter(pl.col("is_legal") == 1)
out = {}
P = ["powerplay", "middle", "death"]


def rate(fr):
    return dict(
        balls=fr.height,
        rpb=fr["runs_batter"].mean(),
        bnd=fr.select((pl.col("runs_batter") >= 4).mean()).item(),
        six=fr.select((pl.col("runs_batter") == 6).mean()).item(),
        dot=fr.select((pl.col("runs_batter") == 0).mean()).item(),
        wkt=fr["wicket"].mean(),
    )


# ---- 1. "due for a boundary": dry-spell length vs next-ball boundary, within phase
rows = []
for ph in P:
    f = d.filter(pl.col("phase") == ph)
    for lo, hi, lab in [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 9, "6-9"),
                        (10, 14, "10-14"), (15, 99, "15+")]:
        s = f.filter(pl.col("balls_since_boundary").is_between(lo, hi))
        if s.height < 400:
            continue
        rows.append(dict(phase=ph, bucket=lab, **rate(s)))
out["dry_spell"] = rows

# same, pooled, plus the batter's own dry spell
rows = []
for lo, hi, lab in [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 9, "6-9"),
                    (10, 14, "10-14"), (15, 99, "15+")]:
    s = d.filter(pl.col("bat_balls_since_boundary").is_between(lo, hi))
    rows.append(dict(bucket=lab, **rate(s)))
out["dry_spell_bat"] = rows

# ---- 2. the set batter
rows = []
for lo, hi, lab in [(0, 4, "0-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, 29, "20-29"),
                    (30, 39, "30-39"), (40, 59, "40-59"), (60, 199, "60+")]:
    s = d.filter(pl.col("bat_runs").is_between(lo, hi))
    rows.append(dict(bucket=lab, **rate(s)))
out["set_batter"] = rows

# balls faced, not runs
rows = []
for lo, hi, lab in [(0, 0, "1st"), (1, 2, "2nd-3rd"), (3, 5, "4th-6th"), (6, 11, "7th-12th"),
                    (12, 23, "13th-24th"), (24, 999, "25th+")]:
    s = d.filter(pl.col("bat_balls").is_between(lo, hi))
    rows.append(dict(bucket=lab, **rate(s)))
out["bat_balls"] = rows

# ---- 3. after a wicket: is the new batter actually vulnerable?
rows = []
for ph in P:
    f = d.filter(pl.col("phase") == ph)
    for lo, hi, lab in [(0, 2, "0-2"), (3, 5, "3-5"), (6, 11, "6-11"), (12, 23, "12-23"), (24, 999, "24+")]:
        s = f.filter(pl.col("balls_since_wicket").is_between(lo, hi))
        if s.height < 400:
            continue
        rows.append(dict(phase=ph, bucket=lab, **rate(s)))
out["since_wicket"] = rows

# ---- 4. momentum: what follows a six / a four / a dot, within phase
prev = d.sort(["match_id", "innings", "over", "ball_in_over"]).with_columns(
    pl.col("runs_batter").shift(1).over(["match_id", "innings"]).alias("prev_runs"),
    pl.col("over").shift(1).over(["match_id", "innings"]).alias("prev_over"),
).filter(pl.col("prev_over") == pl.col("over"))   # same over only: same bowler, same field
rows = []
for v, lab in [(0, "after a dot"), (1, "after a single"), (4, "after a four"), (6, "after a six")]:
    s = prev.filter(pl.col("prev_runs") == v)
    rows.append(dict(bucket=lab, **rate(s)))
rows.append(dict(bucket="all balls", **rate(prev)))
out["momentum"] = rows

# ---- 5. toss
rows = []
for v, lab in [(1, "toss winner batting"), (0, "toss loser batting")]:
    s = d.filter(pl.col("toss_won") == v)
    rows.append(dict(bucket=lab, **rate(s)))
out["toss"] = rows

# ---- 6. handedness matchup, within phase
rows = []
for v, lab in [(1, "same-handed"), (0, "opposite-handed")]:
    s = d.filter(pl.col("same_handed") == v)
    rows.append(dict(bucket=lab, **rate(s)))
out["handed"] = rows
rows = []
for bt in ["pace", "spin"]:
    for v, lab in [(1, "same"), (0, "opposite")]:
        s = d.filter((pl.col("same_handed") == v) & (pl.col("bowler_type") == bt))
        rows.append(dict(type=bt, bucket=lab, **rate(s)))
out["handed_type"] = rows

# ---- 7. chasing vs setting, by phase
rows = []
for ph in P:
    for v, lab in [(0, "setting"), (1, "chasing")]:
        s = d.filter((pl.col("phase") == ph) & (pl.col("is_chase") == v))
        rows.append(dict(phase=ph, bucket=lab, **rate(s)))
out["chase"] = rows

# ---- 8. pace vs spin at the death (the one captains get wrong?)
rows = []
for ph in P:
    for bt in ["pace", "spin"]:
        s = d.filter((pl.col("phase") == ph) & (pl.col("bowler_type") == bt))
        rows.append(dict(phase=ph, type=bt, **rate(s)))
out["type_phase"] = rows

# how often is spin actually used
rows = []
for ov in range(20):
    s = d.filter(pl.col("over") == ov)
    rows.append(dict(over=ov, spin_share=s.select((pl.col("bowler_type") == "spin").mean()).item(),
                     spin_rpb=s.filter(pl.col("bowler_type") == "spin")["runs_batter"].mean(),
                     pace_rpb=s.filter(pl.col("bowler_type") == "pace")["runs_batter"].mean()))
out["spin_use"] = rows

# ---- 9. h2h: do past dismissals predict the next one?
h = d.filter(pl.col("h2h_balls") >= 12)
rows = []
for lo, hi, lab in [(0, 0, "never out to him"), (1, 1, "out once before"), (2, 9, "out 2+ times")]:
    s = h.filter(pl.col("h2h_dismissals").is_between(lo, hi))
    rows.append(dict(bucket=lab, **rate(s)))
out["h2h_pred"] = rows

# ---- 10. last over: dots go UP
rows = []
for ov in [15, 16, 17, 18, 19]:
    s = d.filter(pl.col("over") == ov)
    rows.append(dict(over=ov, **rate(s)))
out["last_overs"] = rows

# ---- 11. required rate pressure in a chase
rows = []
ch = d.filter((pl.col("is_chase") == 1) & (pl.col("balls_left") >= 12) & pl.col("rrr").is_not_null())
for lo, hi, lab in [(0, 6, "under 6"), (6, 8, "6-8"), (8, 10, "8-10"), (10, 12, "10-12"),
                    (12, 15, "12-15"), (15, 99, "15+")]:
    s = ch.filter((pl.col("rrr") >= lo) & (pl.col("rrr") < hi))
    if s.height < 500:
        continue
    rows.append(dict(bucket=lab, **rate(s)))
out["rrr"] = rows

# ---- 12. dismissal mix by bowler type
rows = []
for bt in ["pace", "spin"]:
    s = d.filter((pl.col("bowler_type") == bt) & pl.col("dismissal_kind").is_not_null())
    tot = s.height
    for k in ["caught", "bowled", "lbw", "run out", "stumped", "caught and bowled"]:
        rows.append(dict(type=bt, kind=k, share=s.filter(pl.col("dismissal_kind") == k).height / tot))
out["dismissal_mix"] = rows

json.dump(out, open(str(DATA / "contra.json"), "w"), indent=1, default=str)
for k, v in out.items():
    print("==", k)
    for r in v:
        print(" ", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()})
