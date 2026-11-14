# Data A results

Reports written by our scripts: counts, document IDs, digests and short summaries. No document text is ever committed here.

| Folder | What is in it | Start with |
|---|---|---|
| [spec_v1/](spec_v1/README.md) | Reports on the cleaner and its sheets: the harm tests of `clean.py` and of every rule, the validation of the sheet, the Indic integration, and F2 | [cleaner/harm_tests.md](spec_v1/cleaner/harm_tests.md) |

`dedup/` and `audit/` results get their own folders here when those topics start. Results that TSAI returns for a job request are not here: they come back under [jobs/results/](../jobs/README.md).

Do not edit a report by hand. Each one records the hashes of the code, the sheet and the inputs it was made with, so an edited report stops matching them. Regenerate it with the script that the folder's index names, and check the hashes before applying its result to a new build.
