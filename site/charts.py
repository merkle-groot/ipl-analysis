"""Chart set for the second edition: fan-facing, counter-intuitive results only."""
import json

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "_data"
DATA.mkdir(parents=True, exist_ok=True)
PARQUET = ROOT / "ball_data.parquet"


B = str(DATA) + "/"
S = json.load(open(B + "stats.json"))
K = json.load(open(B + "contra.json"))
G = json.load(open(B + "grounds.json"))

INK, SPOT, RULE, INK2 = "var(--ink)", "var(--spot)", "var(--rule)", "var(--ink-2)"
UT = 'font-family="var(--ff-util)"'
LAB = f'{UT} font-size="10" letter-spacing=".08em" fill="{INK2}"'
LABI = f'{UT} font-size="10" letter-spacing=".08em" fill="{INK}"'
LABS = f'{UT} font-size="10" letter-spacing=".08em" fill="{SPOT}"'
NUM = f'{UT} font-size="10.5" fill="{INK}"'
NUMD = f'{UT} font-size="10.5" fill="{INK2}"'
out = {}


def n(x):
    return f"{x:.1f}"


class Ax:
    def __init__(self, w=700, h=300, l=48, r=18, t=26, b=44):
        self.w, self.h, self.l, self.r, self.t, self.b = w, h, l, r, t, b
        self.g = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
                  f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">']

    def band(self, i, k):
        """centre of the i-th of k equal bands"""
        return self.l + (i + 0.5) * (self.w - self.l - self.r) / k

    def lin(self, v, lo, hi):
        return self.l + (v - lo) / (hi - lo) * (self.w - self.l - self.r)

    def y(self, v, lo, hi):
        return self.t + (hi - v) / (hi - lo) * (self.h - self.t - self.b)

    def add(self, s):
        self.g.append(s)

    def hgrid(self, vals, lo, hi, fmt="{:.2f}", colour=None):
        for v in vals:
            yy = self.y(v, lo, hi)
            self.add(f'<line x1="{self.l}" y1="{n(yy)}" x2="{self.w-self.r}" y2="{n(yy)}" '
                     f'stroke="{RULE}" stroke-width=".5" stroke-dasharray="1 3"/>')
            c = colour or INK2
            self.add(f'<text x="{self.l-7}" y="{n(yy+3.5)}" text-anchor="end" {UT} '
                     f'font-size="10.5" fill="{c}">{fmt.format(v)}</text>')

    def baseline(self):
        self.add(f'<line x1="{self.l}" y1="{n(self.h-self.b)}" x2="{self.w-self.r}" '
                 f'y2="{n(self.h-self.b)}" stroke="{INK}" stroke-width="1.2"/>')

    def xcats(self, labels, dy=15):
        k = len(labels)
        for i, s in enumerate(labels):
            self.add(f'<text x="{n(self.band(i,k))}" y="{n(self.h-self.b+dy)}" '
                     f'text-anchor="middle" {NUMD}>{s}</text>')

    def xtitle(self, s, dy=32):
        self.add(f'<text x="{n((self.l+self.w-self.r)/2)}" y="{n(self.h-self.b+dy)}" '
                 f'text-anchor="middle" {LAB}>{s}</text>')

    def line(self, pts, colour=INK, width=2.2, dash=None, dots=2.4):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{n(x)},{n(y)}" for x, y in pts)
        self.add(f'<polyline points="{p}" fill="none" stroke="{colour}" '
                 f'stroke-width="{width}" stroke-linejoin="round"{d}/>')
        if dots:
            for x, y in pts:
                self.add(f'<circle cx="{n(x)}" cy="{n(y)}" r="{dots}" fill="{colour}"/>')

    def note(self, x, y, s, anchor="middle", style=None):
        self.add(f'<text x="{n(x)}" y="{n(y)}" text-anchor="{anchor}" {style or LAB}>{s}</text>')

    def done(self, key):
        self.g.append("</svg>")
        out[key] = "\n".join(self.g)


def legend(ax, items, x, y, gap=110):
    """items: (label, colour, dashed)"""
    for i, (lab, col, dash) in enumerate(items):
        xx = x + i * gap
        d = ' stroke-dasharray="5 3"' if dash else ""
        ax.add(f'<line x1="{n(xx)}" y1="{n(y)}" x2="{n(xx+18)}" y2="{n(y)}" stroke="{col}" '
               f'stroke-width="2.2"{d}/>')
        ax.add(f'<text x="{n(xx+23)}" y="{n(y+3.5)}" {UT} font-size="10" '
               f'letter-spacing=".07em" fill="{INK}">{lab}</text>')


PH = {"powerplay": (INK, None), "middle": (SPOT, None), "death": (INK2, "5 3")}
PHL = {"powerplay": "OVERS 1-6", "middle": "OVERS 7-16", "death": "OVERS 17-20"}

# ---------------------------------------------------------------- 1. the dry spell
ax = Ax(h=332, b=74)
lo, hi = 0.10, 0.26
ax.hgrid([.10, .14, .18, .22, .26], lo, hi)
ax.baseline()
cats = ["0", "1-2", "3-5", "6-9", "10-14", "15+"]
ax.xcats(cats)
ax.xtitle("BALLS SINCE THE LAST BOUNDARY BY EITHER BATTER")
for ph, (col, dash) in PH.items():
    rows = [r for r in K["dry_spell"] if r["phase"] == ph]
    pts = [(ax.band(cats.index(r["bucket"]), 6), ax.y(r["bnd"], lo, hi)) for r in rows]
    ax.line(pts, col, 2.2, dash)
ax.note(ax.l + 4, ax.t + 12, "CHANCE THE NEXT BALL GOES TO THE BOUNDARY", "start", LAB)
legend(ax, [(PHL["powerplay"], INK, False), (PHL["middle"], SPOT, False),
            (PHL["death"], INK2, True)], ax.w - ax.r - 340, ax.h - ax.b + 52, 112)
ax.done("dry")

# ---------------------------------------------------------------- 2. the bunny myth
ax = Ax(h=250, b=56, l=60)
lo, hi = 0.0, 0.06
ax.hgrid([0, .02, .04, .06], lo, hi)
ax.baseline()
rows = K["h2h_pred"]
labs = ["NEVER DISMISSED<tspan> HIM BEFORE</tspan>", "DISMISSED HIM<tspan> ONCE</tspan>",
        "DISMISSED HIM<tspan> TWICE OR MORE</tspan>"]
for i, r in enumerate(rows):
    cx = ax.band(i, 3)
    bw = 66
    yy = ax.y(r["wkt"], lo, hi)
    ax.add(f'<rect x="{n(cx-bw/2)}" y="{n(yy)}" width="{bw}" height="{n(ax.y(lo,lo,hi)-yy)}" fill="{INK}"/>')
    ax.note(cx, yy - 8, f'{r["wkt"]*100:.2f}%', "middle", LABI)
    parts = ["NEVER OUT TO HIM", "OUT ONCE BEFORE", "OUT TWICE OR MORE"][i]
    ax.note(cx, ax.h - ax.b + 16, parts, "middle", NUMD)
    ax.note(cx, ax.h - ax.b + 30, f'{r["balls"]:,} balls'.upper(), "middle", LAB)
yline = ax.y(rows[0]["wkt"], lo, hi)
ax.add(f'<line x1="{ax.l}" y1="{n(yline)}" x2="{ax.w-ax.r}" y2="{n(yline)}" stroke="{SPOT}" '
       f'stroke-width="1.4" stroke-dasharray="5 3"/>')
ax.note(ax.l + 4, ax.t + 12, "CHANCE THIS BALL TAKES HIS WICKET", "start", LAB)
ax.done("h2h")

# ---------------------------------------------------------------- 3. the set batter
ax = Ax(h=300, b=52, r=34)
cats = [r["bucket"] for r in K["set_batter"]]
k = len(cats)
slo, shi = 0.9, 2.0
wlo, whi = 0.040, 0.080
ax.hgrid([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], slo, shi, "{:.1f}")
ax.baseline()
ax.xcats(cats)
ax.xtitle("RUNS THE BATTER ALREADY HAS")
ax.line([(ax.band(i, k), ax.y(r["rpb"], slo, shi)) for i, r in enumerate(K["set_batter"])], INK)
ax.line([(ax.band(i, k), ax.y(r["wkt"], wlo, whi)) for i, r in enumerate(K["set_batter"])],
        SPOT, 2.0, "5 3", 2.2)
for v in [.04, .05, .06, .07, .08]:
    ax.note(ax.w - ax.r + 4, ax.y(v, wlo, whi) + 3.5, f"{v*100:.0f}%", "start",
            f'{UT} font-size="10.5" fill="{SPOT}"')
ax.note(ax.band(1, k), ax.y(1.62, slo, shi), "RUNS PER BALL", "start", LABI)
ax.note(ax.band(0, k) + 6, ax.y(0.9, slo, shi) - 12, "CHANCE OF BEING OUT", "start", LABS)
ax.done("setbat")

# ---------------------------------------------------------------- 4. the new batter
ax = Ax(h=330, b=74)
lo, hi = 0.025, 0.095
ax.hgrid([.03, .05, .07, .09], lo, hi, "{:.0%}")
ax.baseline()
cats = ["0-2", "3-5", "6-11", "12-23", "24+"]
ax.xcats(cats)
ax.xtitle("BALLS SINCE THE LAST WICKET FELL")
for ph, (col, dash) in PH.items():
    rows = [r for r in K["since_wicket"] if r["phase"] == ph]
    pts = [(ax.band(cats.index(r["bucket"]), 5), ax.y(r["wkt"], lo, hi)) for r in rows]
    ax.line(pts, col, 2.2, dash)
ax.note(ax.l + 4, ax.t + 12, "CHANCE THIS BALL TAKES A WICKET", "start", LAB)
legend(ax, [(PHL["powerplay"], INK, False), (PHL["middle"], SPOT, False),
            (PHL["death"], INK2, True)], ax.w - ax.r - 340, ax.h - ax.b + 52, 112)
x0 = ax.band(0, 5)
ax.add(f'<line x1="{n(x0)}" y1="{n(ax.y(0.088, lo, hi))}" x2="{n(x0)}" y2="{n(ax.h-ax.b)}" '
       f'stroke="{SPOT}" stroke-width="1" stroke-dasharray="3 3"/>')
ax.note(x0 + 8, ax.y(0.0285, lo, hi), "A NEW PAIR AT THE CREASE", "start", LABS)
ax.done("newbat")

# ---------------------------------------------------------------- 5. momentum
ax = Ax(h=290, b=54, l=54)
allr = [r for r in K["momentum"] if r["bucket"] == "all balls"][0]
rows = [r for r in K["momentum"] if r["bucket"] != "all balls"] + [allr]
k = len(rows)
lo, hi = 0.0, 0.28
ax.hgrid([0, .07, .14, .21, .28], lo, hi, "{:.0%}")
ax.baseline()
for i, r in enumerate(rows):
    cx = ax.band(i, k)
    bw = 30
    for j, (val, col, hatch) in enumerate([(r["bnd"], INK, False), (r["wkt"], SPOT, False)]):
        yy = ax.y(val, lo, hi)
        x = cx - bw + j * bw
        ax.add(f'<rect x="{n(x)}" y="{n(yy)}" width="{bw-3}" height="{n(ax.y(lo,lo,hi)-yy)}" fill="{col}"/>')
        ax.note(x + (bw - 3) / 2, yy - 6, f"{val*100:.1f}", "middle",
                LABI if j == 0 else LABS)
    cap = "ANY BALL" if r["bucket"] == "all balls" else r["bucket"].replace("after a ", "").upper()
    ax.note(cx, ax.h - ax.b + 16, cap, "middle", NUMD)
ax.xtitle("WHAT HAPPENED ON THE PREVIOUS BALL OF THE SAME OVER")
sep = (ax.band(3, k) + ax.band(4, k)) / 2
ax.add(f'<line x1="{n(sep)}" y1="{ax.t}" x2="{n(sep)}" y2="{n(ax.h-ax.b)}" stroke="{RULE}" stroke-width=".7"/>')
legend(ax, [("BOUNDARY NEXT BALL", INK, False), ("WICKET NEXT BALL", SPOT, False)], ax.l + 250, ax.t - 8, 190)
ax.done("momentum")

# ---------------------------------------------------------------- 6. spin at the death
ax = Ax(h=344, b=90)
lo, hi = 0.95, 1.95
ax.hgrid([1.0, 1.2, 1.4, 1.6, 1.8], lo, hi, "{:.1f}")
sp = K["spin_use"]
xs = lambda i: ax.lin(i, 0, 19)
ax.add(f'<line x1="{n(xs(0))}" y1="{n(ax.h-ax.b)}" x2="{n(xs(19))}" y2="{n(ax.h-ax.b)}" '
       f'stroke="{INK}" stroke-width="1.2"/>')
ax.line([(xs(r["over"]), ax.y(r["pace_rpb"], lo, hi)) for r in sp], INK, 2.2, None, 2.2)
ax.line([(xs(r["over"]), ax.y(r["spin_rpb"], lo, hi)) for r in sp], SPOT, 2.2, "5 3", 2.2)
ax.note(xs(9), ax.y(1.42, lo, hi) - 6, "PACE", "middle", LABI)
ax.note(xs(9), ax.y(1.16, lo, hi) + 12, "SPIN", "middle", LABS)
# usage strip
by = ax.h - ax.b + 46
bh = 30
ax.add(f'<line x1="{n(xs(0))}" y1="{n(by)}" x2="{n(xs(19))}" y2="{n(by)}" stroke="{INK}" stroke-width="1"/>')
bw = (xs(19) - xs(0)) / 20 * 0.66
for r in sp:
    h = r["spin_share"] * bh
    ax.add(f'<rect x="{n(xs(r["over"])-bw/2)}" y="{n(by-h)}" width="{n(bw)}" height="{n(h)}" '
           f'fill="{SPOT}" opacity="0.8"/>')
ax.note(ax.l, by - bh - 5, "SHARE OF OVERS GIVEN TO SPIN", "start", LAB)
for i in range(0, 20, 2):
    ax.note(xs(i), by + 15, str(i + 1), "middle", NUMD)
ax.note((xs(0) + xs(19)) / 2, by + 31, "OVER OF THE INNINGS", "middle", LAB)
ax.note(ax.l + 4, ax.t + 12, "RUNS CONCEDED PER BALL", "start", LAB)
ax.done("spin")

# ---------------------------------------------------------------- 7. the last over
ax = Ax(h=290, b=52)
rows = K["last_overs"]
k = len(rows)
lo, hi = 0.0, 0.36
ax.hgrid([0, .09, .18, .27, .36], lo, hi, "{:.0%}")
ax.baseline()
ax.xcats([str(r["over"] + 1) for r in rows])
ax.xtitle("OVER OF THE INNINGS")
for key, col, dash, lab, at, anchor, dy in [
        ("six", INK, None, "BALLS HIT FOR SIX", 1, "start", -10),
        ("dot", INK2, "5 3", "BALLS THAT SCORE NOTHING", 0, "start", 17),
        ("wkt", SPOT, None, "BALLS THAT TAKE A WICKET", 4, "end", -11)]:
    pts = [(ax.band(i, k), ax.y(r[key], lo, hi)) for i, r in enumerate(rows)]
    ax.line(pts, col, 2.2, dash)
    style = LABI if col == INK else (LABS if col == SPOT else LAB)
    ox = -6 if anchor == "start" else 6
    ax.note(pts[at][0] + ox, pts[at][1] + dy, lab, anchor, style)
ax.done("lastover")

# ---------------------------------------------------------------- 8. chasing
ax = Ax(h=270, b=52, l=54)
lo, hi = 1.15, 1.70
ax.hgrid([1.2, 1.3, 1.4, 1.5, 1.6, 1.7], lo, hi, "{:.1f}")
ax.baseline()
phs = ["powerplay", "middle", "death"]
for i, ph in enumerate(phs):
    cx = ax.band(i, 3)
    rows = {r["bucket"]: r for r in K["chase"] if r["phase"] == ph}
    bw = 46
    for j, key in enumerate(["setting", "chasing"]):
        v = rows[key]["rpb"]
        yy = ax.y(v, lo, hi)
        x = cx - bw + j * bw
        col = INK if key == "setting" else SPOT
        ax.add(f'<rect x="{n(x)}" y="{n(yy)}" width="{bw-4}" height="{n(ax.y(lo,lo,hi)-yy)}" fill="{col}"/>')
        ax.note(x + (bw - 4) / 2, yy - 6, f"{v:.2f}", "middle", LABI if j == 0 else LABS)
    ax.note(cx, ax.h - ax.b + 16, PHL[ph], "middle", NUMD)
legend(ax, [("BATTING FIRST", INK, False), ("CHASING", SPOT, False)], ax.l + 300, ax.t - 8, 150)
ax.note(ax.l + 4, ax.t + 12, "RUNS PER BALL", "start", LAB)
ax.done("chase")

# ---------------------------------------------------------------- 9. required rate
ax = Ax(h=290, b=52, r=34)
rows = K["rrr"]
k = len(rows)
slo, shi = 1.15, 1.55
wlo, whi = 0.030, 0.090
ax.hgrid([1.2, 1.3, 1.4, 1.5], slo, shi, "{:.1f}")
ax.baseline()
ax.xcats([r["bucket"].upper() for r in rows])
ax.xtitle("RUNS PER OVER STILL REQUIRED")
ax.line([(ax.band(i, k), ax.y(r["rpb"], slo, shi)) for i, r in enumerate(rows)], INK)
ax.line([(ax.band(i, k), ax.y(r["wkt"], wlo, whi)) for i, r in enumerate(rows)], SPOT, 2.0, "5 3", 2.2)
for v in [.04, .06, .08]:
    ax.note(ax.w - ax.r + 4, ax.y(v, wlo, whi) + 3.5, f"{v*100:.0f}%", "start",
            f'{UT} font-size="10.5" fill="{SPOT}"')
ax.note(ax.band(0, k), ax.y(1.32, slo, shi) - 8, "RUNS PER BALL SCORED", "start", LABI)
ax.note(ax.band(0, k), ax.y(0.052, wlo, whi) + 14, "CHANCE OF BEING OUT", "start", LABS)
ceil = ax.y(rows[-2]["rpb"], slo, shi)
ax.add(f'<line x1="{n(ax.band(3,k))}" y1="{n(ceil)}" x2="{n(ax.w-ax.r)}" y2="{n(ceil)}" '
       f'stroke="{INK}" stroke-width="1" stroke-dasharray="2 3"/>')
ax.note(ax.band(3, k) - 8, ceil + 4, "SCORING STOPS RESPONDING", "end", LAB)
ax.done("rrr")

# ---------------------------------------------------------------- 10. sixes not fours
ax = Ax(h=300, b=52)
era = S["era"]
k = len(era)
lo, hi = 0.02, 0.15
ax.hgrid([.03, .06, .09, .12, .15], lo, hi, "{:.0%}")
ax.baseline()
xs = lambda i: ax.lin(i, 0, k - 1)
ax.line([(xs(i), ax.y(r["four"], lo, hi)) for i, r in enumerate(era)], INK2, 2.2, "5 3", 2.2)
ax.line([(xs(i), ax.y(r["six"], lo, hi)) for i, r in enumerate(era)], SPOT, 2.4, None, 2.4)
ax.note(xs(3), ax.y(era[3]["four"], lo, hi) - 10, "FOURS", "middle", LAB)
ax.note(xs(3), ax.y(era[3]["six"], lo, hi) + 16, "SIXES", "middle", LABS)
for i, r in enumerate(era):
    if r["year"] % 3 == 2 or i in (0, k - 1):
        ax.note(xs(i), ax.h - ax.b + 15, str(r["year"])[2:], "middle", NUMD)
ax.note(ax.l + 4, ax.t + 12, "SHARE OF ALL DELIVERIES", "start", LAB)
ax.note(xs(k - 1) - 6, ax.y(era[-1]["six"], lo, hi) - 10, "+79% SINCE 2008", "end", LABS)
ax.note(xs(k - 1) - 6, ax.y(era[-1]["four"], lo, hi) + 16, "+6%", "end", LAB)
ax.done("sixes")

# ---------------------------------------------------------------- 11. grounds
ax = Ax(h=300, b=52, l=56)
lo, hi = 0.044, 0.060
ax.hgrid([.045, .050, .055, .060], lo, hi, "{:.1%}")
ax.baseline()
xlo, xhi = 1.13, 1.54
for v in [1.15, 1.25, 1.35, 1.45]:
    xx = ax.lin(v, xlo, xhi)
    ax.add(f'<line x1="{n(xx)}" y1="{ax.t}" x2="{n(xx)}" y2="{n(ax.h-ax.b)}" stroke="{RULE}" '
           f'stroke-width=".5" stroke-dasharray="1 3"/>')
    ax.note(xx, ax.h - ax.b + 15, f"{v:.2f}", "middle", NUMD)
ax.xtitle("RUNS PER BALL AT THIS GROUND")
for r in G:
    ax.add(f'<circle cx="{n(ax.lin(r["rpb"],xlo,xhi))}" cy="{n(ax.y(r["wkt"],lo,hi))}" '
           f'r="4" fill="{INK}" opacity="0.7"/>')
mw = sum(r["wkt"] for r in G) / len(G)
yy = ax.y(mw, lo, hi)
ax.add(f'<line x1="{ax.l}" y1="{n(yy)}" x2="{ax.w-ax.r}" y2="{n(yy)}" stroke="{SPOT}" '
       f'stroke-width="1.4" stroke-dasharray="5 3"/>')
ax.note(ax.w - ax.r - 4, yy - 8, "NO SLOPE AT ALL", "end", LABS)
ax.note(ax.l + 4, ax.t + 12, "CHANCE OF A WICKET PER BALL AT THIS GROUND", "start", LAB)
ax.note(ax.lin(1.157, xlo, xhi), ax.y(0.0573, lo, hi) - 10, "KINGSMEAD", "middle", LAB)
ax.note(ax.lin(1.509, xlo, xhi) - 6, ax.y(0.0577, lo, hi) - 10, "MULLANPUR", "end", LAB)
ax.done("grounds")

# ---------------------------------------------------------------- 12. pace/spin scatter
ax = Ax(w=700, h=430, l=52, r=20, t=26, b=48)
NAMES = {
    "MS Dhoni": ("MS DHONI", 0, -12), "RG Sharma": ("ROHIT SHARMA", 0, -12),
    "V Kohli": ("VIRAT KOHLI", 0, 16), "AB de Villiers": ("de VILLIERS", 0, -12),
    "GJ Maxwell": ("MAXWELL", 0, 16), "N Pooran": ("POORAN", 0, -12),
    "H Klaasen": ("KLAASEN", 0, -12), "SA Yadav": ("SURYAKUMAR", 0, 16),
    "CH Gayle": ("GAYLE", 0, -12), "BB McCullum": ("McCULLUM", 0, 16),
    "SE Marsh": ("S MARSH", 0, 16), "JC Buttler": ("BUTTLER", 0, -12),
    "F du Plessis": ("du PLESSIS", 50, 4), "YK Pathan": ("Y PATHAN", 0, -12),
}
xlo, xhi, ylo, yhi = 100, 180, 95, 180
for v in range(100, 181, 20):
    xx = ax.lin(v, xlo, xhi)
    ax.add(f'<line x1="{n(xx)}" y1="{ax.t}" x2="{n(xx)}" y2="{n(ax.h-ax.b)}" stroke="{RULE}" '
           f'stroke-width=".5" stroke-dasharray="1 3"/>')
    ax.note(xx, ax.h - ax.b + 15, str(v), "middle", NUMD)
    yy = ax.y(v, ylo, yhi)
    ax.add(f'<line x1="{ax.l}" y1="{n(yy)}" x2="{ax.w-ax.r}" y2="{n(yy)}" stroke="{RULE}" '
           f'stroke-width=".5" stroke-dasharray="1 3"/>')
    ax.note(ax.l - 7, yy + 3.5, str(v), "end", NUMD)
a, b_ = 100, 180
ax.add(f'<line x1="{n(ax.lin(a,xlo,xhi))}" y1="{n(ax.y(a,ylo,yhi))}" x2="{n(ax.lin(b_,xlo,xhi))}" '
       f'y2="{n(ax.y(b_,ylo,yhi))}" stroke="{INK}" stroke-width="1" stroke-dasharray="4 3"/>')
ax.note(ax.lin(112, xlo, xhi), ax.y(119, ylo, yhi), "EQUAL AGAINST BOTH", "middle", LAB)
ax.note(ax.lin(174, xlo, xhi), ax.y(110, ylo, yhi), "MUCH BETTER AGAINST PACE", "end", LABS)
ax.note(ax.lin(108, xlo, xhi), ax.y(170, ylo, yhi), "MUCH BETTER AGAINST SPIN", "start", LABS)
for r in S["pace_spin_all"]:
    x, y = ax.lin(r["sr_pace"], xlo, xhi), ax.y(r["sr_spin"], ylo, yhi)
    if not (ax.l <= x <= ax.w - ax.r and ax.t <= y <= ax.h - ax.b):
        continue
    named = r["batter"] in NAMES
    ax.add(f'<circle cx="{n(x)}" cy="{n(y)}" r="{3.0 if named else 2.0}" '
           f'fill="{INK if named else RULE}" opacity="{1 if named else 0.7}"/>')
    if named:
        lab, dx, dy = NAMES[r["batter"]]
        ax.note(x + dx, y + dy, lab, "middle", f'{UT} font-size="10" letter-spacing=".07em" fill="{INK}"')
ax.xtitle("STRIKE RATE AGAINST PACE", 33)
ax.add(f'<text transform="translate(14,{n((ax.t+ax.h-ax.b)/2)}) rotate(-90)" text-anchor="middle" '
       f'{LAB}>STRIKE RATE AGAINST SPIN</text>')
ax.done("scatter")

# ---------------------------------------------------------------- 13. big three by phase
ax = Ax(h=292, b=64, t=42)
ph = {(r["player"], r["phase"]): r for r in S["named_phase"]}
players = ["RG Sharma", "V Kohli", "MS Dhoni"]
disp = {"RG Sharma": "ROHIT SHARMA", "V Kohli": "VIRAT KOHLI", "MS Dhoni": "MS DHONI"}
phases = ["powerplay", "middle", "death"]
lo, hi = 80, 200
ax.hgrid([100, 125, 150, 175, 200], lo, hi, "{:.0f}")
ax.baseline()
gw = (ax.w - ax.l - ax.r) / 3
for pi, p in enumerate(players):
    x0 = ax.l + pi * gw
    if pi:
        ax.add(f'<line x1="{n(x0)}" y1="{ax.t-16}" x2="{n(x0)}" y2="{n(ax.h-ax.b)}" '
               f'stroke="{RULE}" stroke-width=".7"/>')
    ax.note(x0 + gw / 2, ax.t - 20, disp[p], "middle",
            f'{UT} font-size="11.5" letter-spacing=".14em" fill="{INK}"')
    for qi, q in enumerate(phases):
        r = ph.get((p, q))
        cx = x0 + gw * (qi + 0.5) / 3
        bw = gw / 3 * 0.42
        ax.note(cx, ax.h - ax.b + 15, PHL[q].replace("OVERS ", ""), "middle", NUMD)
        if not r:
            ax.note(cx, ax.y(103, lo, hi), "72 BALLS", "middle", LAB)
            continue
        yy = ax.y(r["sr"], lo, hi)
        ax.add(f'<rect x="{n(cx-bw)}" y="{n(yy)}" width="{n(bw)}" height="{n(ax.y(lo,lo,hi)-yy)}" fill="{INK}"/>')
        ye = ax.y(r["exp_sr"], lo, hi)
        ax.add(f'<rect x="{n(cx)}" y="{n(ye)}" width="{n(bw)}" height="{n(ax.y(lo,lo,hi)-ye)}" '
               f'fill="url(#hatch)" stroke="{RULE}" stroke-width=".6"/>')
        ax.note(cx - bw / 2, yy - 6, f'{r["sr"]:.0f}', "middle", LABI)
legend(ax, [("THE PLAYER", INK, False), ("LEAGUE PAR", RULE, False)],
       ax.w - ax.r - 232, ax.h - ax.b + 36, 130)
ax.done("phase")

# ---------------------------------------------------------------- 14. bumrah / ashwin
ax = Ax(h=260, b=44, t=42)
bp = {(r["player"], r["phase"]): r for r in S["named_bowl_phase"]}
bowlers = ["JJ Bumrah", "R Ashwin"]
bdisp = {"JJ Bumrah": "JASPRIT BUMRAH", "R Ashwin": "R ASHWIN"}
lo, hi = 5.0, 10.2
ax.hgrid([6, 7, 8, 9, 10], lo, hi, "{:.0f}")
ax.baseline()
gw = (ax.w - ax.l - ax.r) / 2
for pi, p in enumerate(bowlers):
    x0 = ax.l + pi * gw
    if pi:
        ax.add(f'<line x1="{n(x0)}" y1="{ax.t-16}" x2="{n(x0)}" y2="{n(ax.h-ax.b)}" '
               f'stroke="{RULE}" stroke-width=".7"/>')
    ax.note(x0 + gw / 2, ax.t - 20, bdisp[p], "middle",
            f'{UT} font-size="11.5" letter-spacing=".14em" fill="{INK}"')
    for qi, q in enumerate(phases):
        r = bp[(p, q)]
        cx = x0 + gw * (qi + 0.5) / 3
        bw = gw / 3 * 0.38
        yy = ax.y(r["econ"], lo, hi)
        ax.add(f'<rect x="{n(cx-bw)}" y="{n(yy)}" width="{n(bw)}" height="{n(ax.y(lo,lo,hi)-yy)}" fill="{INK}"/>')
        ye = ax.y(r["exp_econ"], lo, hi)
        ax.add(f'<rect x="{n(cx)}" y="{n(ye)}" width="{n(bw)}" height="{n(ax.y(lo,lo,hi)-ye)}" '
               f'fill="url(#hatch)" stroke="{RULE}" stroke-width=".6"/>')
        ax.note(cx - bw / 2, yy - 6, f'{r["econ"]:.1f}', "middle", LABI)
        ax.note(cx + bw / 2, ye - 6, f'&#8722;{r["exp_econ"]-r["econ"]:.1f}', "middle", LABS)
        ax.note(cx, ax.h - ax.b + 15, PHL[q].replace("OVERS ", ""), "middle", NUMD)
ax.note(ax.l + 4, ax.t - 20, "RUNS PER OVER", "start", LAB)
ax.done("bowlphase")

json.dump(out, open(B + "charts.json", "w"))
print("charts:", ", ".join(f"{k}({len(v)//100})" for k, v in out.items()))

# ================================================================ second batch
NAMES_LONG = {
    "SR Watson": "Shane Watson", "DA Miller": "David Miller", "N Pooran": "Nicholas Pooran",
    "KA Pollard": "Kieron Pollard", "AB de Villiers": "AB de Villiers", "CH Gayle": "Chris Gayle",
    "S Badrinath": "S Badrinath", "MEK Hussey": "Mike Hussey", "GJ Maxwell": "Glenn Maxwell",
    "KS Williamson": "Kane Williamson", "YBK Jaiswal": "Yashasvi Jaiswal", "HH Pandya": "Hardik Pandya",
    "RA Tripathi": "Rahul Tripathi", "SPD Smith": "Steve Smith", "G Gambhir": "Gautam Gambhir",
    "PA Patel": "Parthiv Patel", "JH Kallis": "Jacques Kallis", "DPMD Jayawardene": "Mahela Jayawardene",
    "JJ Bumrah": "Jasprit Bumrah", "DW Steyn": "Dale Steyn", "SL Malinga": "Lasith Malinga",
    "SP Narine": "Sunil Narine", "B Kumar": "Bhuvneshwar Kumar", "Arshdeep Singh": "Arshdeep Singh",
    "Z Khan": "Zaheer Khan", "KH Pandya": "Krunal Pandya", "JC Archer": "Jofra Archer",
    "KV Sharma": "Karn Sharma", "AD Russell": "Andre Russell", "JD Unadkat": "Jaydev Unadkat",
    "R Bhatia": "Rajat Bhatia", "Kuldeep Yadav": "Kuldeep Yadav", "PP Chawla": "Piyush Chawla",
    "K Rabada": "Kagiso Rabada",
}

O = json.load(open(B + "outliers.json"))

# ---------------------------------------------------------------- 15. all or nothing
ax = Ax(w=700, h=420, l=54, r=22, t=26, b=52)
xlo, xhi, ylo, yhi = -8.0, 9.0, -4.0, 10.0
LBL = {
    "AD Russell": ("RUSSELL", 0, -13), "Yuvraj Singh": ("YUVRAJ", 0, -13),
    "CH Gayle": ("GAYLE", 0, -13), "N Pooran": ("POORAN", 0, -13),
    "GJ Maxwell": ("MAXWELL", 0, 18), "KA Pollard": ("POLLARD", 0, -13),
    "SR Watson": ("WATSON", 0, 18), "DA Warner": ("WARNER", 0, -13),
    "V Kohli": ("KOHLI", 0, 18), "Shubman Gill": ("GILL", 0, 18),
}
for v in range(-8, 10, 4):
    xx = ax.lin(v, xlo, xhi)
    ax.add(f'<line x1="{n(xx)}" y1="{ax.t}" x2="{n(xx)}" y2="{n(ax.h-ax.b)}" stroke="{RULE}" '
           f'stroke-width=".5" stroke-dasharray="1 3"/>')
    ax.note(xx, ax.h - ax.b + 15, f"{v:+d}", "middle", NUMD)
for v in range(-4, 11, 2):
    yy = ax.y(v, ylo, yhi)
    ax.add(f'<line x1="{ax.l}" y1="{n(yy)}" x2="{ax.w-ax.r}" y2="{n(yy)}" stroke="{RULE}" '
           f'stroke-width=".5" stroke-dasharray="1 3"/>')
    ax.note(ax.l - 7, yy + 3.5, f"{v:+d}", "end", NUMD)
zx, zy = ax.lin(0, xlo, xhi), ax.y(0, ylo, yhi)
ax.add(f'<line x1="{n(zx)}" y1="{ax.t}" x2="{n(zx)}" y2="{n(ax.h-ax.b)}" stroke="{INK}" stroke-width=".8"/>')
ax.add(f'<line x1="{ax.l}" y1="{n(zy)}" x2="{n(ax.w-ax.r)}" y2="{n(zy)}" stroke="{INK}" stroke-width=".8"/>')
ax.note(zx + 6, ax.t + 12, "MORE DOTS THAN PAR", "start", LAB)
ax.note(ax.lin(8.6, xlo, xhi), ax.y(5.6, ylo, yhi), "ALL OR NOTHING", "end", LABS)
ax.note(ax.l + 6, ax.h - ax.b - 10, "FEW DOTS, FEW BOUNDARIES", "start", LAB)
for r in O["bat_all"]:
    x, y = ax.lin(r["dot_vs_par"], xlo, xhi), ax.y(r["bnd_vs_par"], ylo, yhi)
    if not (ax.l <= x <= ax.w - ax.r and ax.t <= y <= ax.h - ax.b):
        continue
    named = r["batter"] in LBL
    ax.add(f'<circle cx="{n(x)}" cy="{n(y)}" r="{3.0 if named else 2.0}" '
           f'fill="{INK if named else RULE}" opacity="{1 if named else 0.7}"/>')
    if named:
        lab, dx, dy = LBL[r["batter"]]
        ax.note(x + dx, y + dy, lab, "middle", f'{UT} font-size="10" letter-spacing=".07em" fill="{INK}"')
ax.xtitle("DOT BALLS ABOVE OR BELOW PAR, PERCENTAGE POINTS", 33)
ax.add(f'<text transform="translate(15,{n((ax.t+ax.h-ax.b)/2)}) rotate(-90)" text-anchor="middle" '
       f'{LAB}>BOUNDARIES ABOVE OR BELOW PAR</text>')
ax.done("boom")

# ---------------------------------------------------------------- 16. slow starters
slow = O["slow_start"][:9]
fast = O["fast_start"][:3][::-1]
rows = slow + fast
ax = Ax(w=700, h=44 + 26 * len(rows), l=150, r=52, t=34, b=40)
lo, hi = 85, 200
for v in [100, 125, 150, 175, 200]:
    xx = ax.lin(v, lo, hi)
    ax.add(f'<line x1="{n(xx)}" y1="{ax.t-6}" x2="{n(xx)}" y2="{n(ax.h-ax.b)}" stroke="{RULE}" '
           f'stroke-width=".5" stroke-dasharray="1 3"/>')
    ax.note(xx, ax.h - ax.b + 15, str(v), "middle", NUMD)
rh = (ax.h - ax.b - ax.t) / len(rows)
for i, r in enumerate(rows):
    y = ax.t + (i + 0.5) * rh
    x1, x2 = ax.lin(r["sr_e"], lo, hi), ax.lin(r["sr_l"], lo, hi)
    ax.add(f'<line x1="{n(x1)}" y1="{n(y)}" x2="{n(x2)}" y2="{n(y)}" stroke="{RULE}" stroke-width="2.4"/>')
    ax.add(f'<circle cx="{n(x1)}" cy="{n(y)}" r="4" fill="{SPOT}"/>')
    ax.add(f'<circle cx="{n(x2)}" cy="{n(y)}" r="4" fill="{INK}"/>')
    ax.note(ax.l - 10, y + 3.5, NAMES_LONG.get(r["batter"], r["batter"]).upper(), "end",
            f'{UT} font-size="10.5" letter-spacing=".06em" fill="{INK}"')
    ax.note(ax.w - ax.r + 6, y + 3.5, f'{r["gain"]:+.0f}', "start",
            f'{UT} font-size="10.5" fill="{INK if r["gain"] > 20 else INK2}"')
legend(ax, [("FIRST 10 BALLS", SPOT, False), ("AFTER THAT", INK, False)], ax.l, ax.t - 14, 150)
ax.note(ax.w - ax.r + 6, ax.t - 14, "SWING", "start", LAB)
ax.xtitle("STRIKE RATE", 32)
ax.done("warmup")

# ---------------------------------------------------------------- 17. six-proof bowlers
top = O["sixproof"][:8]
bot = O["sixprone"][:6][::-1]
rows = top + bot
ax = Ax(w=700, h=40 + 24 * len(rows), l=176, r=44, t=22, b=40)
mx = 52.0
zero = ax.l + (ax.w - ax.l - ax.r) * 0.52
rh = (ax.h - ax.b - ax.t) / len(rows)
for i, r in enumerate(rows):
    y = ax.t + i * rh
    v = r["six_saved"]
    w = abs(v) / mx * (ax.w - ax.l - ax.r) * 0.46
    x = zero if v >= 0 else zero - w
    col = INK if v >= 0 else SPOT
    ax.add(f'<rect x="{n(x)}" y="{n(y+rh*0.18)}" width="{n(max(w,1.5))}" height="{n(rh*0.54)}" fill="{col}"/>')
    tx = x + w + 6 if v >= 0 else x - 6
    ax.note(tx, y + rh * 0.62, f"{v:+.0f}%", "start" if v >= 0 else "end",
            f'{UT} font-size="10.5" fill="{INK}"')
    ax.note(ax.l - 10, y + rh * 0.62, NAMES_LONG.get(r["bowler"], r["bowler"]).upper(), "end",
            f'{UT} font-size="10.5" letter-spacing=".06em" fill="{INK}"')
ax.add(f'<line x1="{n(zero)}" y1="{ax.t}" x2="{n(zero)}" y2="{n(ax.h-ax.b+2)}" stroke="{INK}" stroke-width="1.2"/>')
ax.note(zero, ax.h - ax.b + 16, "PAR", "middle", LAB)
ax.note(ax.l, ax.h - ax.b + 16, "CONCEDES MORE SIXES", "start", LABS)
ax.note(ax.w - ax.r, ax.h - ax.b + 16, "CONCEDES FEWER SIXES", "end", LAB)
ax.done("sixproof")

json.dump(out, open(B + "charts.json", "w"))
print("total charts:", len(out))
