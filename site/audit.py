"""Check every chart for text colliding with other text, with plotted lines, or with
the edge of the viewBox. Line collisions are what the first version of this missed."""
import json
import re

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "_data"
DATA.mkdir(parents=True, exist_ok=True)
PARQUET = ROOT / "ball_data.parquet"


B = str(DATA) + "/"
C = json.load(open(B + "charts.json"))
TXT = re.compile(r"<text ([^>]*)>(.*?)</text>", re.S)
POLY = re.compile(r'<polyline points="([^"]+)"')
LINE = re.compile(r"<line ([^>]*)/>")
RECT = re.compile(r"<rect ([^>]*)/>")


def at(a, k, d=None):
    m = re.search(rf'{k}="([^"]*)"', a)
    return m.group(1) if m else d


def seg_box_hit(p, q, box):
    """crude: sample the segment and test the points"""
    x0, y0, x1, y1 = box
    for i in range(21):
        t = i / 20
        x = p[0] + (q[0] - p[0]) * t
        y = p[1] + (q[1] - p[1]) * t
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


problems = 0
for name, svg in C.items():
    vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    boxes = []
    for a, t in TXT.findall(svg):
        if "transform" in a:
            continue
        txt = re.sub(r"<[^>]+>", "", t)
        fs = float(at(a, "font-size", "10"))
        ls = at(a, "letter-spacing", "0em")
        extra = float(ls.replace("em", "")) * fs if "em" in ls else 0
        w = len(txt) * (fs * 0.5 + extra)
        x, y = float(at(a, "x", "0")), float(at(a, "y", "0"))
        anc = at(a, "text-anchor", "start")
        x0 = x if anc == "start" else (x - w / 2 if anc == "middle" else x - w)
        boxes.append([x0, y - fs * 0.78, x0 + w, y + fs * 0.22, txt])

    # text vs text
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a_, b_ = boxes[i], boxes[j]
            if a_[0] < b_[2] and b_[0] < a_[2] and a_[1] < b_[3] and b_[1] < a_[3]:
                print(f'{name}: text "{a_[4]}" over "{b_[4]}"')
                problems += 1

    # text vs plotted polylines (data lines, not gridlines)
    segs = []
    for pts in POLY.findall(svg):
        p = [tuple(float(v) for v in q.split(",")) for q in pts.split()]
        segs += list(zip(p, p[1:]))
    # solid rules and legend swatches count too; dotted gridlines do not
    for a in LINE.findall(svg):
        if at(a, "stroke-dasharray") == "1 3":
            continue
        segs.append(((float(at(a, "x1", "0")), float(at(a, "y1", "0"))),
                     (float(at(a, "x2", "0")), float(at(a, "y2", "0")))))
    for bx in boxes:
        pad = [bx[0] - 1, bx[1] - 1, bx[2] + 1, bx[3] + 1]
        for p, q in segs:
            if seg_box_hit(p, q, pad):
                print(f'{name}: line crosses text "{bx[4]}"')
                problems += 1
                break

    # text vs filled bars
    for a in RECT.findall(svg):
        if at(a, "fill", "") in ("none", ""):
            continue
        rx, ry = float(at(a, "x", "0")), float(at(a, "y", "0"))
        rw, rh = float(at(a, "width", "0")), float(at(a, "height", "0"))
        if rw > 200 and rh > 100:      # background wash, ignore
            continue
        for bx in boxes:
            if rx < bx[2] - 2 and bx[0] < rx + rw - 2 and ry < bx[3] - 2 and bx[1] < ry + rh - 2:
                print(f'{name}: bar covers text "{bx[4]}"')
                problems += 1

    for bx in boxes:
        if bx[2] > vb[2] + 1 or bx[0] < -1 or bx[3] > vb[3] + 1 or bx[1] < -1:
            print(f'{name}: out of bounds "{bx[4]}"')
            problems += 1

print("problems:", problems)
