"""Bowling style and batting hand for every player in ball_data.parquet.

Cricsheet records neither. Both come from the same place - the `bowling` and
`batting` fields of the en.wikipedia infobox - so they are fetched together.

    python3 fetch_bowler_style.py             # writes player_style.csv
    python3 fetch_bowler_style.py --refresh   # re-download people.csv first

Batting hand is wanted for everyone who faced or backed up a ball, not just
bowlers, so the player list is batters + non-strikers + bowlers (807 people
against 577 bowlers). Handedness matters mostly as a *matchup*: an off-break
turns away from a left-hander and into a right-hander, which is the same
delivery posing a different problem depending on who is on strike.

`people.csv` is Cricsheet's register, cached in the project verbatim so a rebuild
does not depend on the network. Refresh it when new players appear.

Manual labels: 25 bowlers have no usable source and were filled in by hand from
docs/BOWLER_STYLE_TODO.md. They are re-applied on every run from MANUAL below,
so re-running this script no longer discards them.

The route, and why it is this one:

* Cricsheet's own register (cricsheet.org/register/people.csv) maps every player
  identifier to a Cricinfo ID. Coverage on our bowlers is 577/577.
* Cricinfo itself is behind Akamai and refuses API requests, so the ID is used
  only as a join key, not as a source.
* Wikidata has a `bowling style` property (P2545) and it is unusable: 1,512
  items worldwide, 43 of our 577 bowlers, 10% of balls, and the value
  distribution is visibly bot-damaged (835 "left-arm orthodox spin" against 2
  "off spin", plus values like "orthogonal direction to the right").
* Wikidata *is* useful for its Cricinfo ID -> en.wikipedia sitelink, which
  resolves 558/577 bowlers. The `bowling` field of the infobox on those articles
  is free text but highly regular, and normalises to pace/spin for 552 of them,
  98.7% of deliveries.

The remaining 25 - uncapped domestic players with no article, plus a handful
whose infobox omits the field - are in MANUAL below. Coverage is now 577/577,
every delivery, so downstream code never has to handle a missing type.
"""

import collections
import csv
import os
import sys
import json
import re
import time
import urllib.parse
import urllib.request

import polars as pl

UA = {"User-Agent": "cricProject/1.0 (bowling-style lookup)"}
REGISTER = "https://cricsheet.org/register/people.csv"
SPARQL = "https://query.wikidata.org/sparql"
WIKI = "https://en.wikipedia.org/w/api.php"

# Bowlers with no usable public source, filled in by hand. See
# docs/BOWLER_STYLE_TODO.md for the evidence behind each one.
MANUAL = {
    "P Awana": ("pace", "right"),
    "DS Rathi": ("spin", "right"),
    "Prince Yadav": ("pace", "right"),
    "V Nigam": ("spin", "right"),
    "Brijesh Sharma": ("pace", "right"),
    "Shivang Kumar": ("spin", "left"),
    "Yash Raj Punja": ("spin", "right"),
    "PP Hinge": ("pace", "right"),
    "DP Vijaykumar": ("pace", "right"),
    "Ashok Sharma": ("pace", "right"),
    "SR Dubey": ("pace", "left"),
    "Gagandeep Singh": ("pace", "right"),
    "M Tiwari": ("pace", "right"),
    "Abhinandan Singh": ("pace", "right"),
    "Naman Dhir": ("spin", "right"),
    "K Santokie": ("pace", "left"),
    "Krish Bhagat": ("pace", "right"),
    "VS Yeligati": ("spin", "right"),
    "MA Khote": ("pace", "right"),
    "MB Parmar": ("spin", "right"),
    "RS Ghosh": ("pace", "right"),
    "Mohit Rathee": ("spin", "right"),
    "T Vijay": ("spin", "left"),
    "RA Shaikh": ("pace", "left"),
    "AC Gilchrist": ("spin", "right"),
}

# Batting hand for players with no article. Fill from docs/BATTING_HAND_TODO.md.
# Applied on every run, so a re-fetch never discards it.
MANUAL_HAND: dict[str, str] = {}

# Order matters: "slow left-arm orthodox" contains neither "spin" nor a pace
# word, and "right-arm medium off-break" is a spinner who is described with a
# pace word, so spin evidence is tested first and wins ties.
SPIN = r"orthodox|off.?break|off.?spin|leg.?break|leg.?spin|googly|chinaman|slow left|wrist.?spin|finger.?spin|spin"
PACE = r"fast|medium|pace|seam|swing"


def get(url, params=None, post=None):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = urllib.parse.urlencode(post).encode() if post else None
    req = urllib.request.Request(url, data=data, headers=UA)
    return urllib.request.urlopen(req, timeout=60).read()


def register(refresh=False):
    """Cricsheet's player register, cached in the project as people.csv."""
    if refresh or not os.path.exists("people.csv"):
        with open("people.csv", "wb") as f:
            f.write(get(REGISTER))
        print("  downloaded people.csv")
    return list(csv.DictReader(open("people.csv")))


def cricinfo_ids(bowlers, refresh=False):
    """Cricsheet name -> Cricinfo ID, via the match registries and the register."""
    import glob

    registry = {}
    for f in glob.glob("ipl_male_json/*.json"):
        registry.update(json.load(open(f))["info"].get("registry", {}).get("people", {}))

    rows = register(refresh)
    key = {r["identifier"]: r["key_cricinfo"] for r in rows}
    return {b: key.get(registry.get(b, ""), "") for b in bowlers}


def wikipedia_titles(ids):
    """Cricinfo ID -> en.wikipedia article title, through Wikidata's P2697."""
    values = " ".join(f'"{i}"' for i in ids if i)
    q = (f"SELECT ?ci ?article WHERE {{ VALUES ?ci {{ {values} }} "
         "?p wdt:P2697 ?ci . OPTIONAL { ?article schema:about ?p ; "
         "schema:isPartOf <https://en.wikipedia.org/> } }")
    res = json.loads(get(SPARQL, post={"query": q, "format": "json"}))
    out = {}
    for r in res["results"]["bindings"]:
        if "article" in r:
            slug = r["article"]["value"].rsplit("/", 1)[-1]
            out[r["ci"]["value"]] = urllib.parse.unquote(slug).replace("_", " ")
    return out


def search_titles(names, ids, unique, have):
    """Fallback for players Wikidata does not link: find the article by name.

    Wikidata's Cricinfo ID (P2697) is missing on recently-created items, so
    players who debuted in the last season or two drop out of the sitelink
    route even when they clearly have an article - Priyansh Arya, 491 balls
    faced, was one.

    A name match alone is not safe: searching "Sakib Hussain cricketer" returns
    a footballer whose article happens to contain the same digits, and it would
    have been read as "Right footed". So a candidate is accepted only if it is
    a player article (infobox cricketer) *and* cites this exact Cricinfo ID
    inside an ESPNcricinfo link or template, not merely somewhere in the text.
    """
    out = {}
    for name in names:
        cid = ids.get(name, "")
        if not cid or name in have:
            continue
        try:
            res = json.loads(get(WIKI, params={
                "action": "query", "list": "search", "format": "json", "srlimit": 3,
                "srsearch": f"{unique.get(name, name)} cricketer"}))
            cands = [h["title"] for h in res["query"]["search"]]
            if not cands:
                continue
            pages = json.loads(get(WIKI, params={
                "action": "query", "prop": "revisions", "rvprop": "content",
                "rvslots": "main", "format": "json", "titles": "|".join(cands)}))
            for page in pages["query"]["pages"].values():
                try:
                    text = page["revisions"][0]["slots"]["main"]["*"]
                except (KeyError, IndexError):
                    continue
                cricinfo_ref = re.search(
                    rf"(espncricinfo[^\n]{{0,120}}?|\{{\{{\s*(esp[nc]|cricinfo)[^}}]{{0,120}}?)\b{re.escape(cid)}\b",
                    text, re.I)
                if cricinfo_ref and re.search(r"\{\{\s*infobox cricketer", text, re.I):
                    out[name] = page["title"]
                    break
        except Exception as exc:                      # a flaky lookup must not
            print(f"  ! search failed for {name}: {exc}")   # kill the whole run
        time.sleep(0.2)
    return out


def infobox_fields(titles, batch=25):
    """Raw `bowling =` and `batting =` fields from each article's infobox."""
    out = {}
    titles = list(titles)
    for i in range(0, len(titles), batch):
        res = json.loads(get(WIKI, params={
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json", "titles": "|".join(titles[i:i + batch])}))
        for page in res["query"]["pages"].values():
            try:
                text = page["revisions"][0]["slots"]["main"]["*"]
            except (KeyError, IndexError):
                continue
            got = {}
            for field in ("bowling", "batting"):
                m = re.search(rf"\|\s*{field}\s*=\s*(.*?)(?=\n\s*\|\s*\w+\s*=|\n\s*\}}\}})",
                              text, re.I | re.S)
                if m:
                    got[field] = m.group(1)
            if got:
                out[page["title"]] = got
        time.sleep(0.3)
    return out


def clean(raw):
    """Infobox values carry lists, links and refs; keep the words."""
    s = re.sub(r"<ref[^>]*>.*?</ref>|<[^>]+>", " ", raw, flags=re.S)
    s = re.sub(r"\{\{\s*(ubl|unbulleted list|plainlist|hlist)\s*\|", " ", s, flags=re.I)
    s = re.sub(r"\{\{[^{}]*\}\}", " ", s)
    s = s.replace("[[", " ").replace("]]", " ").replace("'''", " ")
    s = re.sub(r"\|", " ", s)          # after templates: pipes separate list items
    return re.sub(r"\s+", " ", s).strip(" *-–,;")


def classify(text):
    s = text.lower()
    spin, pace = bool(re.search(SPIN, s)), bool(re.search(PACE, s))
    if spin:
        return "spin"
    return "pace" if pace else None


def spin_kind(text):
    """Finger or wrist spin - the half of the matchup that decides which way it turns.

    Off-break and left-arm orthodox are finger spin; leg-break, googly, chinaman
    and anything called wrist spin are wrist spin. The distinction is what makes
    `turn_into_batter` computable downstream: finger spin turns into a batter of
    the same handedness as the bowler's arm, wrist spin into the opposite one.
    """
    s = text.lower()
    if re.search(r"leg.?break|legbreak|leg.?spin|googly|chinaman|wrist.?spin|unorthodox", s):
        return "wrist"
    if re.search(r"off.?break|off.?spin|orthodox|finger.?spin", s):
        return "finger"
    return None


def bowling_arm(text):
    """Arm, stated or implied.

    Most infoboxes say it outright, in either the "-arm" or the "-hand" form
    (the "-hand" form is ambiguous in a batting field, but this only ever reads
    the bowling one). Leg-spin entries usually do not say it at all - "Legbreak
    googly" and nothing else - and Rashid Khan alone is 3,572 deliveries of that.

    The spin terms carry the arm definitionally, so the fallback is not a guess:
    wrist spin from a left-armer is called left-arm unorthodox or chinaman, never
    a leg break, and left-arm finger spin is orthodox, never an off break. Both
    of those left-arm terms are matched above, so they never reach the fallback.
    """
    s = text.lower()
    if re.search(r"left.?arm|left.?hand|slow left|chinaman|orthodox|unorthodox", s):
        return "left"
    if re.search(r"right.?arm|right.?hand", s):
        return "right"
    if re.search(r"leg.?break|legbreak|leg.?spin|googly|off.?break|off.?spin", s):
        return "right"
    return None


def player_counts(df):
    """Every player who bowled, faced, or backed up a ball, with their volume."""
    bowled = df.group_by("bowler").agg(pl.len().alias("bowled")).rename({"bowler": "player"})
    faced = df.group_by("batter").agg(pl.len().alias("faced")).rename({"batter": "player"})
    return (bowled.join(faced, on="player", how="full", coalesce=True)
                  .with_columns(pl.col("bowled").fill_null(0), pl.col("faced").fill_null(0))
                  .with_columns((pl.col("bowled") + pl.col("faced")).alias("balls"))
                  .sort("balls", descending=True))


def batting_hand(text):
    """Left/right from the `batting` field, which is near-universally populated.

    The field is terse and regular ("Right-handed", "Left-hand bat"), so this is
    a much cleaner parse than the bowling one. Ambidextrous switch-hitters are
    listed by their stance, which is what matters for the matchup anyway.
    """
    s = text.lower()
    if re.search(r"left.?hand", s):
        return "left"
    return "right" if re.search(r"right.?hand", s) else None


def main() -> None:
    df = pl.read_parquet("ball_data.parquet")
    counts = player_counts(df)
    bowlers = counts["player"].to_list()
    total = int(counts["bowled"].sum())
    print(f"{len(bowlers)} players ({counts.filter(pl.col('bowled') > 0).height} of them bowled)")

    ids = cricinfo_ids(bowlers, refresh="--refresh" in sys.argv)
    print(f"  cricinfo id:  {sum(1 for v in ids.values() if v)}")
    titles = wikipedia_titles(set(ids.values()))
    print(f"  wikipedia:    {len(titles)} via wikidata sitelink")
    linked = {n for n in bowlers if titles.get(ids.get(n, ""))}
    unique = {r["name"]: r["unique_name"] for r in register()}
    found = search_titles(bowlers, ids, unique, linked)
    for n, t in found.items():
        titles[ids[n]] = t
    print(f"                +{len(found)} via verified name search")
    raw = infobox_fields(set(titles.values()))
    print(f"  infobox:      {len(raw)}")

    rows = []
    for name, bowled, faced in zip(counts["player"], counts["bowled"], counts["faced"]):
        fields = raw.get(titles.get(ids.get(name, ""), ""), {})
        text = clean(fields.get("bowling", "")) or ""
        bat_text = clean(fields.get("batting", "")) or ""
        btype, arm = classify(text), bowling_arm(text)
        if not btype and name in MANUAL:
            btype, arm = MANUAL[name]
            text = "manual (docs/BOWLER_STYLE_TODO.md)"
        if not bowled:                       # never bowled: leave the bowling side empty
            btype = arm = None
        kind = spin_kind(text) if btype == "spin" else None
        rows.append({"player": name, "bowled": int(bowled), "faced": int(faced),
                     "bowler_type": btype or "", "bowler_arm": arm or "",
                     "spin_kind": kind or "",
                     "bat_hand": batting_hand(bat_text) or MANUAL_HAND.get(name, ""),
                     "source_text": text, "bat_source_text": bat_text})

    with open("player_style.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    faced_total = sum(r["faced"] for r in rows)
    got = [r for r in rows if r["bowler_type"]]
    covered = sum(r["bowled"] for r in got)
    print(f"\nbowling style: {len(got)}/{counts.filter(pl.col('bowled') > 0).height} bowlers, "
          f"{covered:,} deliveries ({100 * covered / total:.1f}%)")
    print("  " + "  ".join(f"{k}:{v}" for k, v in
                           collections.Counter(r["bowler_type"] for r in got).items()))
    manual = sum(1 for r in rows if r["source_text"].startswith("manual"))
    print(f"  of which {manual} from MANUAL (hand-filled, no public source)")

    spinners = [r for r in rows if r["bowler_type"] == "spin"]
    kinds = collections.Counter(r["spin_kind"] or "unknown" for r in spinners)
    print("  spin kind: " + "  ".join(f"{k}:{v}" for k, v in kinds.most_common()))

    hands = [r for r in rows if r["bat_hand"]]
    if MANUAL_HAND:
        print(f"  {sum(1 for r in rows if r['player'] in MANUAL_HAND and r['bat_hand'])} "
              "from MANUAL_HAND (hand-filled)")
    bat_covered = sum(r["faced"] for r in hands)
    print(f"batting hand:  {len(hands)}/{len(rows)} players, "
          f"{bat_covered:,} balls faced ({100 * bat_covered / faced_total:.1f}%)")
    print("  " + "  ".join(f"{k}:{v}" for k, v in
                           collections.Counter(r["bat_hand"] for r in hands).items()))
    missing = sorted((r for r in rows if not r["bat_hand"]),
                     key=lambda r: -r["faced"])[:5]
    if missing:
        print("  no batting hand (most balls faced first): "
              + ", ".join(f"{r['player']} ({r['faced']})" for r in missing))
    print("\nwrote player_style.csv")


if __name__ == "__main__":
    main()
