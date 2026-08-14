import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "_data"
DATA.mkdir(parents=True, exist_ok=True)
PARQUET = ROOT / "ball_data.parquet"

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb2html

D = str(DATA) + "/"
S = json.load(open(D + "stats.json"))
C = json.load(open(D + "charts.json"))
O = json.load(open(D + "outliers.json"))
tpl = open(HERE / "page.tpl.html").read()

NAME = {
    "V Sehwag": "Virender Sehwag", "AD Russell": "Andre Russell", "CH Gayle": "Chris Gayle",
    "GJ Maxwell": "Glenn Maxwell", "AB de Villiers": "AB de Villiers", "N Pooran": "Nicholas Pooran",
    "DR Smith": "Dwayne Smith", "DA Warner": "David Warner", "SR Watson": "Shane Watson",
    "SK Raina": "Suresh Raina", "JC Buttler": "Jos Buttler", "YBK Jaiswal": "Yashasvi Jaiswal",
    "WP Saha": "Wriddhiman Saha", "AM Rahane": "Ajinkya Rahane", "JP Duminy": "JP Duminy",
    "KS Williamson": "Kane Williamson", "MK Pandey": "Manish Pandey", "JH Kallis": "Jacques Kallis",
    "RA Jadeja": "Ravindra Jadeja", "AR Patel": "Axar Patel", "BB McCullum": "Brendon McCullum",
    "MS Dhoni": "MS Dhoni", "F du Plessis": "Faf du Plessis", "RG Sharma": "Rohit Sharma",
    "PA Patel": "Parthiv Patel", "Q de Kock": "Quinton de Kock", "SA Yadav": "Suryakumar Yadav",
    "H Klaasen": "Heinrich Klaasen", "SE Marsh": "Shaun Marsh", "YK Pathan": "Yusuf Pathan",
    "B Sai Sudharsan": "Sai Sudharsan", "N Rana": "Nitish Rana", "V Kohli": "Virat Kohli",
    "JJ Bumrah": "Jasprit Bumrah", "R Ashwin": "R Ashwin", "SL Malinga": "Lasith Malinga",
    "SP Narine": "Sunil Narine", "JC Archer": "Jofra Archer", "DW Steyn": "Dale Steyn",
    "B Kumar": "Bhuvneshwar Kumar", "Mustafizur Rahman": "Mustafizur Rahman",
    "M Muralitharan": "Muttiah Muralitharan", "Rashid Khan": "Rashid Khan",
    "KH Pandya": "Krunal Pandya", "Imran Tahir": "Imran Tahir", "YS Chahal": "Yuzvendra Chahal",
    "K Rabada": "Kagiso Rabada", "KV Sharma": "Karn Sharma", "MM Patel": "Munaf Patel",
    "CV Varun": "Varun Chakravarthy", "A Mishra": "Amit Mishra", "A Nehra": "Ashish Nehra",
    "KL Rahul": "KL Rahul",
    "Yuvraj Singh": "Yuvraj Singh", "KA Pollard": "Kieron Pollard", "N Rana": "Nitish Rana",
    "Shubman Gill": "Shubman Gill", "G Gambhir": "Gautam Gambhir", "RD Gaikwad": "Ruturaj Gaikwad",
    "SR Tendulkar": "Sachin Tendulkar", "R Dravid": "Rahul Dravid", "S Dhawan": "Shikhar Dhawan",
    "SPD Smith": "Steve Smith", "HH Pandya": "Hardik Pandya", "MP Stoinis": "Marcus Stoinis",
    "DA Miller": "David Miller", "MEK Hussey": "Mike Hussey", "S Badrinath": "S Badrinath",
    "RA Tripathi": "Rahul Tripathi", "DPMD Jayawardene": "Mahela Jayawardene", "M Vijay": "Murali Vijay",
    "AC Gilchrist": "Adam Gilchrist", "Mohammed Siraj": "Mohammed Siraj", "Z Khan": "Zaheer Khan",
    "M Prasidh Krishna": "Prasidh Krishna", "Arshdeep Singh": "Arshdeep Singh",
    "JD Unadkat": "Jaydev Unadkat", "R Bhatia": "Rajat Bhatia", "Kuldeep Yadav": "Kuldeep Yadav",
    "MA Starc": "Mitchell Starc", "IK Pathan": "Irfan Pathan", "JR Hazlewood": "Josh Hazlewood",
    "SN Thakur": "Shardul Thakur", "KK Ahmed": "Khaleel Ahmed", "CH Morris": "Chris Morris",
    "PJ Cummins": "Pat Cummins", "PP Chawla": "Piyush Chawla",
}
nm = lambda k: NAME.get(k, k)
sign = lambda v, d=1: ('<span class="pos">+' if v >= 0 else '<span class="neg">&minus;') + f"{abs(v):.{d}f}</span>"
SPOT = {"V Kohli", "RG Sharma", "MS Dhoni", "JJ Bumrah", "R Ashwin"}


def tr(key, cells):
    cls = ' class="spot"' if key in SPOT else ""
    return f"<tr{cls}>" + "".join(cells) + "</tr>"


def name_cell(k, sub=None):
    s = f"<small>{sub}</small>" if sub else ""
    return f'<td class="name">{nm(k)}{s}</td>'


rows = []
for r in S["bat_top"][:10]:
    rows.append(tr(r["batter"], [name_cell(r["batter"]), f'<td>{r["balls"]:,}</td>',
                                 f'<td>{r["sr"]:.1f}</td>', f'<td>{r["exp_sr"]:.1f}</td>',
                                 f'<td>{sign(r["raa100"])}</td>']))
bat_top = "\n".join(rows)

rows = []
for r in S["bat_bottom"][::-1][:8]:
    rows.append(tr(r["batter"], [name_cell(r["batter"]), f'<td>{r["balls"]:,}</td>',
                                 f'<td>{r["sr"]:.1f}</td>', f'<td>{r["exp_sr"]:.1f}</td>',
                                 f'<td>{sign(r["raa100"])}</td>']))
bat_bottom = "\n".join(rows)


def spec_rows(key):
    out = []
    for r in S[key]:
        out.append(tr(r["batter"], [
            name_cell(r["batter"], f'{r["b_pace"]:,} / {r["b_spin"]:,} balls'),
            f'<td>{r["sr_pace"]:.0f}</td>', f'<td>{r["sr_spin"]:.0f}</td>',
            f'<td>{sign(r["diff"], 0)}</td>']))
    return "\n".join(out)


rows = []
for r in S["bowl_top"][:10]:
    rows.append(tr(r["bowler"], [name_cell(r["bowler"], r["type"]), f'<td>{r["balls"]:,}</td>',
                                 f'<td>{r["econ"]:.2f}</td>', f'<td>{r["exp_econ"]:.2f}</td>',
                                 f'<td>{sign(r["saved100"])}</td>']))
bowl_top = "\n".join(rows)

rows = []
for r in S["bowl_wkt_top"][:10]:
    rows.append(tr(r["bowler"], [name_cell(r["bowler"], r["type"]), f'<td>{r["balls"]:,}</td>',
                                 f'<td>{r["wkts"]}</td>', f'<td>{r["econ"]:.2f}</td>',
                                 f'<td>{sign(r["wkt_pct"], 0)}%</td>']))
bowl_wkt = "\n".join(rows)

rows = []
for r in S["h2h"]:
    if r["balls"] < 78:
        continue
    rows.append(tr(r["batter"], [f'<td class="name">{nm(r["batter"])}</td>',
                                 f'<td style="text-align:left">{nm(r["bowler"])}</td>',
                                 f'<td>{r["balls"]}</td>', f'<td>{r["runs"]}</td>',
                                 f'<td>{r["sr"]:.0f}</td>', f'<td>{r["outs"]}</td>']))
h2h = "\n".join(rows)

rows = []
for r in O["unaided"][:7] + O["aided"][:3][::-1]:
    rows.append(tr(r["bowler"], [name_cell(r["bowler"]), f'<td>{r["w"]}</td>',
                                 f'<td>{r["unaided"]*100:.0f}%</td>']))
unaided = "\n".join(rows)

PHN = {"powerplay": "Overs 1-6", "middle": "Overs 7-16", "death": "Overs 17-20"}
rows = []
for ph in ["powerplay", "middle", "death"]:
    for r in [x for x in O["bowl_phase_best"] if x["phase"] == ph][:4]:
        rows.append(tr(r["bowler"], [name_cell(r["bowler"]),
                                     f'<td style="text-align:left">{PHN[ph]}</td>',
                                     f'<td>{r["econ"]:.2f}</td>',
                                     f'<td>{sign(r["saved"])}</td>']))
phasebest = "\n".join(rows)


def chase_rows(key):
    out = []
    for r in O[key]:
        out.append(tr(r["batter"], [name_cell(r["batter"]),
                                    f'<td>{sign(r["raa_s"], 0)}</td>',
                                    f'<td>{sign(r["raa_c"], 0)}</td>',
                                    f'<td>{sign(r["gap"], 0)}</td>']))
    return "\n".join(out)


NBS = str(ROOT) + "/"
nb_runs = nb2html.render(
    NBS + "runs.ipynb", "The working, unedited",
    "The unedited working behind the run figures: how the outcome of each delivery was modelled, "
    "what was tried, and what failed. Code, output and figures exactly as they ran.")
nb_wickets = nb2html.render(
    NBS + "starter.ipynb", "The working, unedited",
    "The companion notebook, predicting whether a delivery takes a wicket rather than how many "
    "runs it concedes. Same data, same splits, a much harder problem.")

page = tpl
for k, v in [("bat_top", bat_top), ("bat_bottom", bat_bottom),
             ("pace_spec", spec_rows("pace_specialists")), ("spin_spec", spec_rows("spin_specialists")),
             ("bowl_top", bowl_top), ("unaided", unaided), ("phasebest", phasebest),
             ("chase_good", chase_rows("chase_good")),
             ("chase_bad", chase_rows("chase_bad")),
             ("nb_runs", nb_runs), ("nb_wickets", nb_wickets),
             ("pygments_css", nb2html.style_css())]:
    page = page.replace("{{" + k + "}}", v)
for k, v in C.items():
    page = page.replace("{{" + k + "}}", v)

assert "{{" not in page, page[page.index("{{"):page.index("{{") + 60]
open(str(ROOT / "public" / "index.html"), "w").write(page)
print("wrote", len(page), "bytes")
