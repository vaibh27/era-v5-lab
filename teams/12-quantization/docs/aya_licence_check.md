# aya_collection Indic subset licence check

Rotation task: licence check of the 12 aya_collection Indic subsets (`sft-indic-aya-collection`).

| # | Subset | Upstream source | Licence | Origin | Usable? |
|---|---|---|---|---|---|
| 1 | templated_hindi_headline | Hindi-article-generation | CC BY-NC-SA 4.0 | Web-scraped Hindi news headlines/articles | Excluded (non-commercial) |
| 2 | templated_hindi_news | Hindi-article-summarization | CC BY-SA 4.0 | Hindi Text Short/Large Summarization corpus (Kaggle) | Needs TSAI decision (share-alike) |
| 3 | templated_bengali_news | News-summary-instruct | CC0-1.0 | Bengali news summarization | Usable |
| 4 | templated_tamil_stories | TamilStories | Apache 2.0 | Scraped Tamil short stories, Aya Open Science | Usable |
| 5 | templated_tamil_thirukkural | Thirukkural-instruct | Apache 2.0 | Thirukkural verses + meanings, Aya Open Science | Usable |
| 6 | templated_telugu_food | Telugu-food-recipes | Apache 2.0 | Web-scraped Telugu recipes, Aya Open Science | Usable |
| 7 | templated_telugu_jokes | Telugu-jokes | Apache 2.0 | Web-scraped Telugu jokes, Aya Open Science | Usable |
| 8 | templated_telugu_news | Telugu-news-articles | Apache 2.0 | Web-scraped Telugu news, Aya Open Science | Usable |
| 9 | templated_telugu_poems | Telugu-poems | Apache 2.0 | Web-scraped Telugu poetry, Aya Open Science | Usable |
| 10 | templated_telugu_riddles | TeluguRiddles | Apache 2.0 | Web-scraped riddles, Aya Open Science | Usable |
| 11 | templated_indic_sentiment | IndicSentiment-inst | MIT | Translation of IndicSentiment benchmark | Usable |
| 12 | templated_indic_paraphrase | IndicXParaphrase-inst | MIT | Translation of IndicXParaphrase benchmark | Usable |

## Summary

- **Excluded (1):** templated_hindi_headline — CC BY-NC-SA 4.0 (non-commercial)
- **Needs TSAI decision (1):** templated_hindi_news — CC BY-SA 4.0 (share-alike)
- **Usable (10):** all others are Apache 2.0, CC0-1.0 or MIT

**Key finding:** the Aya collection wrapper is Apache 2.0, but the templated content derives from the upstream datasets above — the wrapper licence does not override the upstream terms. Only the two Hindi subsets carry restrictions.

templated_urdu_news_headline is not in this set (it is in `sft-indic-aya-collection-other`, excluded).
