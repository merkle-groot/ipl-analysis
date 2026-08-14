"""More outliers: boom-or-bust batting, slow starters, chase specialists,
dot-ball bowlers vs wicket hunters, six-proof bowlers, death specialists."""
import json

import polars as pl

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "_data"
DATA.mkdir(parents=True, exist_ok=True)
PARQUET = ROOT / "ball_data.parquet"


d = pl.read_parquet(str(PARQUET)).filter(pl.col("is_legal") == 1)
cell = ["year", "phase", "bowler_type"]
base = d.group_by(cell).agg(
    pl.col("runs_batter").mean().alias("e_runs"),
    pl.col("wicket_striker").mean().alias("e_wkt"),
    pl.col("wicket_bowler").mean().alias("e_wktb"),
    (pl.col("runs_batter") >= 4).mean().alias("e_bnd"),
    (pl.col("runs_batter") == 6).mean().alias("e_six"),
    (pl.col("runs_batter") == 0).mean().alias("e_dot"),
)
j = d.join(base, on=cell, how="left")
out = {}

BAT_MIN, BOWL_MIN = 1500, 1500

bat = (
    j.group_by("batter").agg(
        pl.len().alias("balls"),
        (100 * pl.col("runs_batter").mean()).alias("sr"),
        (100 * pl.col("e_runs").mean()).alias("par_sr"),
        (pl.col("runs_batter") == 0).mean().alias("dot"),
        pl.col("e_dot").mean().alias("par_dot"),
        (pl.col("runs_batter") >= 4).mean().alias("bnd"),
        pl.col("e_bnd").mean().alias("par_bnd"),
        (pl.col("runs_batter") == 6).mean().alias("six"),
        (pl.col("runs_batter") == 4).mean().alias("four"),
        (pl.col("runs_batter").is_between(1, 3)).mean().alias("rot"),
        pl.col("wicket_striker").sum().alias("outs"),
        pl.col("e_wkt").mean().alias("par_wkt"),
    )
    .filter(pl.col("balls") >= BAT_MIN)
    .with_columns(
        (pl.col("outs") / pl.col("balls")).alias("out_rate"),
        (100 * (pl.col("dot") - pl.col("par_dot"))).alias("dot_vs_par"),
        (100 * (pl.col("bnd") - pl.col("par_bnd"))).alias("bnd_vs_par"),
        (pl.col("six") / (pl.col("six") + pl.col("four"))).alias("six_share"),
    )
    .with_columns(
        (100 * (pl.col("par_wkt") - pl.col("out_rate")) / pl.col("par_wkt")).alias("surv"),
        (pl.col("dot_vs_par") + pl.col("bnd_vs_par")).alias("boom"),
    )
)

out["survive_top"] = bat.sort("surv", descending=True).head(10).to_dicts()
out["survive_bot"] = bat.sort("surv").head(8).to_dicts()
out["boom"] = bat.sort("boom", descending=True).head(10).to_dicts()
out["grind"] = bat.sort("boom").head(8).to_dicts()
out["six_share"] = bat.filter(pl.col("bnd") > 0.12).sort("six_share", descending=True).head(10).to_dicts()
out["four_share"] = bat.filter(pl.col("bnd") > 0.12).sort("six_share").head(8).to_dicts()
out["rotators"] = bat.sort("rot", descending=True).head(10).to_dicts()
out["bat_all"] = bat.to_dicts()

# slow starters: strike rate over the first ten balls of an innings vs after
early = j.filter(pl.col("bat_balls") < 10).group_by("batter").agg(
    pl.len().alias("n_e"), (100 * pl.col("runs_batter").mean()).alias("sr_e"))
late = j.filter(pl.col("bat_balls") >= 10).group_by("batter").agg(
    pl.len().alias("n_l"), (100 * pl.col("runs_batter").mean()).alias("sr_l"))
warm = (early.join(late, on="batter")
        .filter((pl.col("n_e") >= 400) & (pl.col("n_l") >= 700))
        .with_columns((pl.col("sr_l") - pl.col("sr_e")).alias("gain")))
out["slow_start"] = warm.sort("gain", descending=True).head(10).to_dicts()
out["fast_start"] = warm.sort("gain").head(8).to_dicts()
out["warm_all"] = warm.to_dicts()

# chase specialists
setg = j.filter(pl.col("is_chase") == 0).group_by("batter").agg(
    pl.len().alias("n_s"), (100 * (pl.col("runs_batter") - pl.col("e_runs")).mean()).alias("raa_s"))
chas = j.filter(pl.col("is_chase") == 1).group_by("batter").agg(
    pl.len().alias("n_c"), (100 * (pl.col("runs_batter") - pl.col("e_runs")).mean()).alias("raa_c"))
ch = (setg.join(chas, on="batter").filter((pl.col("n_s") >= 700) & (pl.col("n_c") >= 700))
      .with_columns((pl.col("raa_c") - pl.col("raa_s")).alias("gap")))
out["chase_good"] = ch.sort("gap", descending=True).head(8).to_dicts()
out["chase_bad"] = ch.sort("gap").head(8).to_dicts()

# ---- bowlers
bowl = (
    j.group_by("bowler").agg(
        pl.len().alias("balls"),
        pl.col("bowler_type").first().alias("type"),
        (6 * pl.col("runs_batter").mean()).alias("econ"),
        (6 * pl.col("e_runs").mean()).alias("par_econ"),
        (pl.col("runs_batter") == 0).mean().alias("dot"),
        pl.col("e_dot").mean().alias("par_dot"),
        (pl.col("runs_batter") == 6).mean().alias("six"),
        pl.col("e_six").mean().alias("par_six"),
        (pl.col("runs_batter") >= 4).mean().alias("bnd"),
        pl.col("e_bnd").mean().alias("par_bnd"),
        pl.col("wicket_bowler").sum().alias("wkts"),
        pl.col("wicket_bowler").mean().alias("wkt_rate"),
        pl.col("e_wktb").mean().alias("par_wkt"),
    )
    .filter(pl.col("balls") >= BOWL_MIN)
    .with_columns(
        (100 * (pl.col("dot") - pl.col("par_dot"))).alias("dot_vs_par"),
        (100 * (pl.col("wkt_rate") - pl.col("par_wkt")) / pl.col("par_wkt")).alias("wkt_pct"),
        (100 * (pl.col("par_six") - pl.col("six")) / pl.col("par_six")).alias("six_saved"),
        (100 * (pl.col("par_bnd") - pl.col("bnd")) / pl.col("par_bnd")).alias("bnd_saved"),
        (pl.col("balls") / pl.col("wkts")).alias("sr_balls"),
    )
)
out["dot_bowl"] = bowl.sort("dot_vs_par", descending=True).head(10).to_dicts()
out["wkt_bowl"] = bowl.sort("wkt_pct", descending=True).head(10).to_dicts()
out["sixproof"] = bowl.sort("six_saved", descending=True).head(10).to_dicts()
out["sixprone"] = bowl.sort("six_saved").head(8).to_dicts()
out["bowl_all"] = bowl.to_dicts()

# bowlers by phase against par
rows = []
for ph in ["powerplay", "middle", "death"]:
    g = (j.filter(pl.col("phase") == ph).group_by("bowler").agg(
            pl.len().alias("balls"),
            (6 * pl.col("runs_batter").mean()).alias("econ"),
            (6 * pl.col("e_runs").mean()).alias("par_econ"),
            pl.col("bowler_type").first().alias("type"))
         .filter(pl.col("balls") >= 500)
         .with_columns((100 * (pl.col("par_econ") - pl.col("econ")) / 6).alias("saved")))
    for r in g.sort("saved", descending=True).head(6).to_dicts():
        rows.append(dict(phase=ph, **r))
out["bowl_phase_best"] = rows

# how a bowler's wickets are taken: share not requiring a fielder
mix = (d.filter(pl.col("wicket_bowler") == 1)
       .group_by("bowler").agg(
           pl.len().alias("w"),
           (pl.col("dismissal_kind").is_in(["bowled", "lbw"])).mean().alias("unaided"))
       .filter(pl.col("w") >= 60).sort("unaided", descending=True))
out["unaided"] = mix.head(10).to_dicts()
out["aided"] = mix.tail(6).to_dicts()

json.dump(out, open(str(DATA / "outliers.json"), "w"), default=str)

def show(k, cols, n=10):
    print("==", k)
    for r in out[k][:n]:
        print("  ", " ".join(f"{c}={r[c]:.3f}" if isinstance(r[c], float) else f"{c}={r[c]}" for c in cols))

show("boom", ["batter", "balls", "sr", "dot_vs_par", "bnd_vs_par", "boom"])
show("grind", ["batter", "balls", "sr", "dot_vs_par", "bnd_vs_par", "boom"])
show("six_share", ["batter", "balls", "six", "four", "six_share"])
show("four_share", ["batter", "balls", "six", "four", "six_share"])
show("slow_start", ["batter", "n_e", "sr_e", "sr_l", "gain"])
show("fast_start", ["batter", "n_e", "sr_e", "sr_l", "gain"])
show("chase_good", ["batter", "raa_s", "raa_c", "gap"])
show("chase_bad", ["batter", "raa_s", "raa_c", "gap"])
show("dot_bowl", ["bowler", "balls", "type", "dot", "par_dot", "dot_vs_par", "econ"])
show("sixproof", ["bowler", "balls", "type", "six", "par_six", "six_saved"])
show("sixprone", ["bowler", "balls", "type", "six", "six_saved"])
show("unaided", ["bowler", "w", "unaided"])
show("aided", ["bowler", "w", "unaided"])
show("survive_top", ["batter", "balls", "surv", "sr"])
show("bowl_phase_best", ["phase", "bowler", "balls", "econ", "par_econ", "saved"], 18)
