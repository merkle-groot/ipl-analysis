<title>Bowler style: unresolved players</title>

# Bowler style — unresolved players

25 of the 577 bowlers in `ball_data.parquet` could not be resolved by
`fetch_bowler_style.py`. Between them they bowled 3,872 deliveries, 1.3% of the data.

Two different reasons, and the second group is the quicker win:

- 19 have no English Wikipedia article. Uncapped domestic players, most of
  them 2025–26 debutants. Cricinfo is the only realistic source.
- 6 have an article whose infobox carries no `bowling` field. The article body or
  the categories usually settle it without leaving Wikipedia; the link is in the table.

**Status: done.** The `fixes` dict below was filled in and now lives in `MANUAL` in
`fetch_bowler_style.py`, which re-applies it on every run, so a re-fetch no longer discards it.
Bowling style is 577/577 bowlers, 100% of deliveries. This page is kept as the record of where
those 25 labels came from, and as the template if new uncapped players appear.

Two of them, `V Nigam` and `Mohit Rathee`, were later picked up automatically once the fetcher
gained a name-search fallback, and the auto-derived labels agreed with the manual ones, a small
but real check on both.

The output file is `player_style.csv` (bowling style *and* batting hand); the old
`bowler_style.csv` is gone. See [BATTING_HAND_TODO.md](BATTING_HAND_TODO.md) for the
still-open list on the batting side.

## The players

`mean over` / `PP%` / `death%` describe when they actually bowled. That is evidence, not an
answer: a bowler used in overs 7–15 is more likely spin, one used in the powerplay and at the
death more likely pace. But leg-spinners bowl at the death and seamers bowl in the middle.
Sample sizes below ~50 balls are too small to read anything into.

| # | bowler | balls | seasons | team | wkts | econ | mean over | PP% | death% | lookup |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **P Awana** | 774 | 2012–2014 | Kings XI Punjab | 43 | 8.11 | 10.3 | 36% | 24% | [cricinfo](https://www.espncricinfo.com/ci/content/player/323131.html) · **[wikipedia](https://en.wikipedia.org/wiki/Parvinder_Awana)** |
| 2 | **DS Rathi** | 561 | 2025–2026 | Lucknow Super Giants | 19 | 8.92 | 10.3 | 25% | 16% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1460529.html) |
| 3 | **Prince Yadav** | 474 | 2025–2026 | Lucknow Super Giants | 25 | 8.75 | 11.2 | 31% | 30% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1350768.html) |
| 4 | **V Nigam** | 301 | 2025–2026 | Delhi Capitals | 13 | 9.53 | 10.2 | 19% | 13% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1449074.html) |
| 5 | **Brijesh Sharma** | 297 | 2026 | Rajasthan Royals | 14 | 9.13 | 12.0 | 27% | 35% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1515046.html) |
| 6 | **Shivang Kumar** | 240 | 2026 | Sunrisers Hyderabad | 11 | 9.4 | 10.4 | 3% | 6% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1512089.html) |
| 7 | **Yash Raj Punja** | 186 | 2026 | Rajasthan Royals | 9 | 9.06 | 10.9 | 3% | 10% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1515186.html) |
| 8 | **PP Hinge** | 168 | 2026 | Sunrisers Hyderabad | 14 | 10.96 | 10.2 | 42% | 35% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1216167.html) |
| 9 | **DP Vijaykumar** | 159 | 2008 | Deccan Chargers | 4 | 7.74 | 7.2 | 65% | 19% | [cricinfo](https://www.espncricinfo.com/ci/content/player/272949.html) · **[wikipedia](https://en.wikipedia.org/wiki/Paidikalva_Vijaykumar)** |
| 10 | **Ashok Sharma** | 136 | 2026 | Gujarat Titans | 6 | 10.28 | 12.5 | 18% | 39% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1299879.html) |
| 11 | **SR Dubey** | 87 | 2026 | Kolkata Knight Riders | 5 | 7.59 | 8.3 | 68% | 32% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1206492.html) · **[wikipedia](https://en.wikipedia.org/wiki/Saurabh_Dubey_(Maharashtra_cricketer))** |
| 12 | **Gagandeep Singh** | 84 | 2008 | Kings XI Punjab | 3 | 10.14 | 12.9 | 7% | 29% | [cricinfo](https://www.espncricinfo.com/ci/content/player/28758.html) · **[wikipedia](https://en.wikipedia.org/wiki/Gagandeep_Singh_(cricketer,_born_1981))** |
| 13 | **M Tiwari** | 75 | 2025–2026 | Delhi Capitals | 4 | 8.4 | 11.6 | 0% | 17% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1460385.html) |
| 14 | **Abhinandan Singh** | 60 | 2026 | Royal Challengers Bengaluru | 3 | 12.2 | 9.7 | 40% | 17% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1449085.html) |
| 15 | **Naman Dhir** | 53 | 2024–2025 | Mumbai Indians | 1 | 8.6 | 11.6 | 11% | 23% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1287032.html) |
| 16 | **K Santokie** | 51 | 2014 | Mumbai Indians | 3 | 10.71 | 11.2 | 39% | 49% | [cricinfo](https://www.espncricinfo.com/ci/content/player/314622.html) |
| 17 | **Krish Bhagat** | 36 | 2026 | Mumbai Indians | 0 | 10.5 | 11.5 | 50% | 50% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1454586.html) |
| 18 | **VS Yeligati** | 32 | 2008 | Mumbai Indians | 0 | 11.44 | 12.6 | 19% | 19% | [cricinfo](https://www.espncricinfo.com/ci/content/player/307448.html) |
| 19 | **MA Khote** | 30 | 2008 | Mumbai Indians | 2 | 10.4 | 16.0 | 0% | 40% | [cricinfo](https://www.espncricinfo.com/ci/content/player/30220.html) |
| 20 | **MB Parmar** | 19 | 2010 | Kolkata Knight Riders | 0 | 10.42 | 8.6 | 37% | 0% | [cricinfo](https://www.espncricinfo.com/ci/content/player/240734.html) · **[wikipedia](https://en.wikipedia.org/wiki/Monish_Parmar)** |
| 21 | **RS Ghosh** | 18 | 2026 | Chennai Super Kings | 1 | 8.0 | 11.7 | 0% | 0% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1339053.html) |
| 22 | **Mohit Rathee** | 12 | 2023 | Punjab Kings | 0 | 14.5 | 14.0 | 0% | 0% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1349361.html) |
| 23 | **T Vijay** | 12 | 2026 | Delhi Capitals | 0 | 14.5 | 6.0 | 50% | 0% | [cricinfo](https://www.espncricinfo.com/ci/content/player/1292527.html) |
| 24 | **RA Shaikh** | 6 | 2009 | Mumbai Indians | 0 | 11.0 | 3.0 | 100% | 0% | [cricinfo](https://www.espncricinfo.com/ci/content/player/279460.html) |
| 25 | **AC Gilchrist** | 1 | 2013 | Kings XI Punjab | 1 | 0.0 | 20.0 | 0% | 100% | [cricinfo](https://www.espncricinfo.com/ci/content/player/5390.html) · **[wikipedia](https://en.wikipedia.org/wiki/Adam_Gilchrist)** |

### Fill these in

```python
fixes = {
    "P Awana": ("pace", "right"),                   # 774 balls, Kings XI Punjab
    "DS Rathi": ("spin", "right"),                  # 561 balls, Lucknow Super Giants
    "Prince Yadav": ("pace", "right"),              # 474 balls, Lucknow Super Giants
    "V Nigam": ("spin", "right"),                   # 301 balls, Delhi Capitals
    "Brijesh Sharma": ("pace", "right"),            # 297 balls, Rajasthan Royals
    "Shivang Kumar": ("spin", "left"),             # 240 balls, Sunrisers Hyderabad
    "Yash Raj Punja": ("spin", "right"),            # 186 balls, Rajasthan Royals
    "PP Hinge": ("pace", "right"),                  # 168 balls, Sunrisers Hyderabad
    "DP Vijaykumar": ("pace", "right"),             # 159 balls, Deccan Chargers
    "Ashok Sharma": ("pace", "right"),              # 136 balls, Gujarat Titans
    "SR Dubey": ("pace", "left"),                  # 87 balls, Kolkata Knight Riders
    "Gagandeep Singh": ("pace", "right"),           # 84 balls, Kings XI Punjab
    "M Tiwari": ("pace", "right"),                  # 75 balls, Delhi Capitals
    "Abhinandan Singh": ("pace", "right"),          # 60 balls, Royal Challengers Bengaluru
    "Naman Dhir": ("spin", "right"),                # 53 balls, Mumbai Indians
    "K Santokie": ("pace", "left"),                # 51 balls, Mumbai Indians
    "Krish Bhagat": ("pace", "right"),              # 36 balls, Mumbai Indians
    "VS Yeligati": ("spin", "right"),               # 32 balls, Mumbai Indians
    "MA Khote": ("pace", "right"),                  # 30 balls, Mumbai Indians
    "MB Parmar": ("spin", "right"),                 # 19 balls, Kolkata Knight Riders
    "RS Ghosh": ("pace", "right"),                  # 18 balls, Chennai Super Kings
    "Mohit Rathee": ("spin", "right"),              # 12 balls, Punjab Kings
    "T Vijay": ("spin", "left"),                   # 12 balls, Delhi Capitals
    "RA Shaikh": ("pace", "left"),                 # 6 balls, Mumbai Indians
    "AC Gilchrist": ("spin", "right"),              # 1 ball, Kings XI Punjab, keeper, skip
}
```

## Notes

- The Cricinfo IDs are from Cricsheet's register and are reliable. The pages open fine in a
  browser; it is only scripted requests that Akamai blocks, which is why this list is manual.
- Leaving a row blank is safe: an empty `bowler_type` stays a missing category, which LightGBM
  handles natively. The 12 players below 60 balls are not worth much effort.
