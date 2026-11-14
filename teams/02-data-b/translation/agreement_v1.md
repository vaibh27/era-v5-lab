# Translation rating: agreement (v1, 13 Nov 2026)

Part of #216. Two native readers rate the same translated documents from the #212 review pack, each on their own,
from 1 to 5 on TSAI's rubric. A language passes when both readers rated at least 20 of the same documents and their
scores are within 1 point on 80% or more of them (TSAI, G2, 31 Oct). Made with `translation/score_ratings.py`.

Sheets used (13 Nov, 18:45 IST):
- Kannada (#402): `kn_keshavamurali.csv` (#419, merged) and `kn_subramanyanaik.csv` (#425).
- Tamil (#403): `ta_Dharshanthaigu.csv` (#432). The second reader's sheet is not in yet.
- Telugu (#404): `te_shankarpandala.csv` (#429). The second reader's sheet is not in yet.
- Hindi (#401): no sheets yet.

This file is updated when more sheets come in.

## Result

| Language | Readers | Shared documents | Within 1 point | Result |
|---|---|---|---|---|
| kn | keshavamurali, subramanyanaik | 25 | 19 (76%) | FAIL |

## Notes
- Kannada fails: 19 of 25 within 1 point (76%). The second sheet is lower on 21 of the 25 and never higher (means 3.60
  and 2.52), so the two readers differ by a steady offset, not at random.
- Tamil, Telugu and Hindi can't be checked until each has two sheets.
