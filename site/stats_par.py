import json

import polars as pl

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "_data"
DATA.mkdir(parents=True, exist_ok=True)
PARQUET = ROOT / "ball_data.parquet"

df = pl.read_parquet(str(PARQUET)).filter(pl.col("is_legal") == 1)

# era + situation adjusted baseline: league mean within (year, phase, bowler_type)
cell = ["year", "phase", "bowler_type"]
base = df.group_by(cell).agg(
    pl.col("runs_batter").mean().alias("exp_runs"),
    pl.col("wicket_striker").mean().alias("exp_wkt"),
    pl.col("wicket_bowler").mean().alias("exp_wkt_b"),
    (pl.col("runs_batter") >= 4).mean().alias("exp_bnd"),
    (pl.col("runs_batter") == 0).mean().alias("exp_dot"),
)
d = df.join(base, on=cell, how="left")

out = {}


def bat(frame, key, min_balls):
    g = (
        frame.group_by(key)
        .agg(
            pl.len().alias("balls"),
            pl.col("runs_batter").sum().alias("runs"),
            pl.col("runs_batter").mean().alias("rpb"),
            pl.col("exp_runs").mean().alias("exp_rpb"),
            (pl.col("runs_batter") >= 4).mean().alias("bnd"),
            pl.col("exp_bnd").mean().alias("exp_bnd"),
            (pl.col("runs_batter") == 0).mean().alias("dot"),
            pl.col("exp_dot").mean().alias("exp_dot"),
            pl.col("wicket_striker").sum().alias("outs"),
            pl.col("exp_wkt").mean().alias("exp_wkt"),
        )
        .filter(pl.col("balls") >= min_balls)
        .with_columns(
            (100 * (pl.col("rpb") - pl.col("exp_rpb"))).alias("raa100"),
            (100 * pl.col("rpb")).alias("sr"),
            (100 * pl.col("exp_rpb")).alias("exp_sr"),
            (pl.col("outs") / pl.col("balls")).alias("out_rate"),
        )
        .with_columns(
            (100 * (pl.col("exp_wkt") - pl.col("out_rate")) / pl.col("exp_wkt")).alias("surv_pct")
        )
    )
    return g


bats = bat(d, "batter", 1500).sort("raa100", descending=True)
out["bat_top"] = bats.head(12).to_dicts()
out["bat_bottom"] = bats.tail(8).to_dicts()

# pace vs spin split for batters
pace = bat(d.filter(pl.col("bowler_type") == "pace"), "batter", 700).select(
    ["batter", "balls", "sr", "raa100"]
).rename({"balls": "b_pace", "sr": "sr_pace", "raa100": "raa_pace"})
spin = bat(d.filter(pl.col("bowler_type") == "spin"), "batter", 500).select(
    ["batter", "balls", "sr", "raa100"]
).rename({"balls": "b_spin", "sr": "sr_spin", "raa100": "raa_spin"})
ps = pace.join(spin, on="batter").with_columns((pl.col("raa_pace") - pl.col("raa_spin")).alias("diff"))
out["pace_specialists"] = ps.sort("diff", descending=True).head(8).to_dicts()
out["spin_specialists"] = ps.sort("diff").head(8).to_dicts()
out["pace_spin_all"] = ps.to_dicts()

# phase splits
for ph in ["powerplay", "middle", "death"]:
    out[f"bat_{ph}"] = bat(d.filter(pl.col("phase") == ph), "batter", 600).sort(
        "raa100", descending=True
    ).head(8).to_dicts()

# bowlers
bowl = (
    d.group_by("bowler")
    .agg(
        pl.len().alias("balls"),
        pl.col("runs_batter").mean().alias("rpb"),
        pl.col("exp_runs").mean().alias("exp_rpb"),
        pl.col("wicket_bowler").sum().alias("wkts"),
        pl.col("wicket_bowler").mean().alias("wkt_rate"),
        pl.col("exp_wkt_b").mean().alias("exp_wkt"),
        (pl.col("runs_batter") >= 4).mean().alias("bnd"),
        pl.col("exp_bnd").mean().alias("exp_bnd"),
        (pl.col("runs_batter") == 0).mean().alias("dot"),
        pl.col("exp_dot").mean().alias("exp_dot"),
        pl.col("bowler_type").first().alias("type"),
    )
    .filter(pl.col("balls") >= 1200)
    .with_columns(
        (100 * (pl.col("exp_rpb") - pl.col("rpb"))).alias("saved100"),
        (6 * pl.col("rpb")).alias("econ"),
        (6 * pl.col("exp_rpb")).alias("exp_econ"),
        (100 * (pl.col("wkt_rate") - pl.col("exp_wkt")) / pl.col("exp_wkt")).alias("wkt_pct"),
    )
)
out["bowl_top"] = bowl.sort("saved100", descending=True).head(12).to_dicts()
out["bowl_wkt_top"] = bowl.sort("wkt_pct", descending=True).head(10).to_dicts()
out["bowl_bottom"] = bowl.sort("saved100").head(6).to_dicts()
out["bowl_all"] = bowl.to_dicts()

# named players
NAMED_BAT = ["RG Sharma", "V Kohli", "MS Dhoni"]
NAMED_BOWL = ["JJ Bumrah", "R Ashwin"]
out["named_bat"] = bats.filter(pl.col("batter").is_in(NAMED_BAT)).to_dicts()
out["named_bowl"] = bowl.filter(pl.col("bowler").is_in(NAMED_BOWL)).to_dicts()

# named batter by phase and by bowler type
rows = []
for p in NAMED_BAT:
    for ph in ["powerplay", "middle", "death"]:
        s = d.filter((pl.col("batter") == p) & (pl.col("phase") == ph))
        if s.height < 100:
            continue
        rows.append(
            dict(
                player=p,
                phase=ph,
                balls=s.height,
                sr=100 * s["runs_batter"].mean(),
                exp_sr=100 * s["exp_runs"].mean(),
                bnd=s.select((pl.col("runs_batter") >= 4).mean()).item(),
                dot=s.select((pl.col("runs_batter") == 0).mean()).item(),
            )
        )
out["named_phase"] = rows

rows = []
for p in NAMED_BAT:
    for bt in ["pace", "spin"]:
        s = d.filter((pl.col("batter") == p) & (pl.col("bowler_type") == bt))
        rows.append(
            dict(
                player=p,
                type=bt,
                balls=s.height,
                sr=100 * s["runs_batter"].mean(),
                exp_sr=100 * s["exp_runs"].mean(),
                outs=int(s["wicket_striker"].sum()),
                bpd=s.height / max(1, int(s["wicket_striker"].sum())),
            )
        )
out["named_type"] = rows

# named bowlers by phase
rows = []
for p in NAMED_BOWL:
    for ph in ["powerplay", "middle", "death"]:
        s = d.filter((pl.col("bowler") == p) & (pl.col("phase") == ph))
        if s.height < 100:
            continue
        rows.append(
            dict(
                player=p,
                phase=ph,
                balls=s.height,
                econ=6 * s["runs_batter"].mean(),
                exp_econ=6 * s["exp_runs"].mean(),
                wkts=int(s["wicket_bowler"].sum()),
                wkt_rate=s["wicket_bowler"].mean(),
                exp_wkt=s["exp_wkt_b"].mean(),
                dot=s.select((pl.col("runs_batter") == 0).mean()).item(),
            )
        )
out["named_bowl_phase"] = rows

# head to head among the five
h2h = []
for b in NAMED_BAT + ["SK Raina", "AB de Villiers", "DA Warner", "KL Rahul", "SA Yadav"]:
    for bo in NAMED_BOWL + ["SP Narine", "YS Chahal", "Rashid Khan", "B Kumar"]:
        s = d.filter((pl.col("batter") == b) & (pl.col("bowler") == bo))
        if s.height < 40:
            continue
        h2h.append(
            dict(
                batter=b,
                bowler=bo,
                balls=s.height,
                runs=int(s["runs_batter"].sum()),
                sr=100 * s["runs_batter"].mean(),
                outs=int(s["wicket_striker"].sum()),
            )
        )
out["h2h"] = sorted(h2h, key=lambda r: -r["balls"])

# career trajectory of named batters by year
traj = (
    d.filter(pl.col("batter").is_in(NAMED_BAT))
    .group_by(["batter", "year"])
    .agg(pl.len().alias("balls"), (100 * pl.col("runs_batter").mean()).alias("sr"))
    .filter(pl.col("balls") >= 150)
    .sort(["batter", "year"])
)
out["named_traj"] = traj.to_dicts()

# league scoring by year (era drift) and by over
out["era"] = (
    d.group_by("year")
    .agg(
        pl.col("runs_batter").mean().alias("rpb"),
        (pl.col("runs_batter") == 6).mean().alias("six"),
        (pl.col("runs_batter") == 4).mean().alias("four"),
        (pl.col("runs_batter") == 0).mean().alias("dot"),
        pl.col("wicket").mean().alias("wkt"),
    )
    .sort("year")
    .to_dicts()
)
out["by_over"] = (
    d.group_by("over")
    .agg(
        pl.col("runs_batter").mean().alias("rpb"),
        (pl.col("runs_batter") >= 4).mean().alias("bnd"),
        (pl.col("runs_batter") == 0).mean().alias("dot"),
        pl.col("wicket").mean().alias("wkt"),
    )
    .sort("over")
    .to_dicts()
)
out["type_phase"] = (
    d.group_by(["phase", "bowler_type"])
    .agg(
        pl.len().alias("balls"),
        pl.col("runs_batter").mean().alias("rpb"),
        (pl.col("runs_batter") >= 4).mean().alias("bnd"),
        pl.col("wicket").mean().alias("wkt"),
    )
    .sort(["phase", "bowler_type"])
    .to_dicts()
)
out["turn"] = (
    d.filter(pl.col("bowler_type") == "spin")
    .group_by("turn_into_batter")
    .agg(
        pl.len().alias("balls"),
        pl.col("runs_batter").mean().alias("rpb"),
        (pl.col("runs_batter") >= 4).mean().alias("bnd"),
        pl.col("wicket").mean().alias("wkt"),
    )
    .to_dicts()
)

# per-ground scoring and wicket rates, for the venue chart
grounds = (d.group_by("ground")
             .agg(pl.len().alias("n"),
                  pl.col("runs_batter").mean().alias("rpb"),
                  pl.col("wicket").mean().alias("wkt"))
             .filter(pl.col("n") >= 3000).sort("rpb"))
json.dump(grounds.to_dicts(), open(DATA / "grounds.json", "w"))

with open(str(DATA / "stats.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
print("ok")
