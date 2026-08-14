"""Copy Cricsheet IPL match files into per-season folders named by match number.

    ipl_male_json/335982.json  ->  ipl_by_season/2008/001.json

Playoff matches carry a `stage` ("Final", "Qualifier 1", ...) instead of a
match_number, so they are named after the stage and sorted after the league
games. The three seasons with two semi-finals get a trailing 1/2 by date.
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

SRC = Path("ipl_male_json")
DST = Path("ipl_by_season")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def main() -> None:
    # (year, name) -> list of source files, so duplicate names can be numbered
    planned = defaultdict(list)

    for src in sorted(SRC.glob("*.json")):
        info = json.loads(src.read_text())["info"]
        date = min(info["dates"])
        # The season string is unreliable as a folder name: it is sometimes a
        # split season ("2007/08" is IPL 2008, "2020/21" is IPL 2020), so take
        # the year the season actually started playing.
        year = date[:4]
        event = info.get("event", {})

        if "match_number" in event:
            name = f"{int(event['match_number']):03d}"
        else:
            name = slug(event.get("stage", "unknown"))

        planned[(year, name)].append((date, src))

    for (year, name), matches in planned.items():
        matches.sort()  # by date, so semi_final -> semi_final_1, semi_final_2
        out_dir = DST / year
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, (_, src) in enumerate(matches, start=1):
            stem = name if len(matches) == 1 else f"{name}_{i}"
            shutil.copy2(src, out_dir / f"{stem}.json")

    total = sum(len(v) for v in planned.values())
    print(f"copied {total} files into {len(set(y for y, _ in planned))} season folders")


if __name__ == "__main__":
    main()
