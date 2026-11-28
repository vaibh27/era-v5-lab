# en-nemotroncc-synth audit and medicine follow-ups

Reviewer: `patilvijay23`. Review recorded 21 Nov 2026, completing the Week 3 audit assignment. The CSVs contain **76 unique reviewed documents: 48 full reads (Part_1 28, Part_2 20) and 28 additional partial reads (Part_2)**. Full reads exceed the original 25 + 10 allocation. The additional review inspected every observed phone mask in Part_2, not every document in full. These counts describe the supplied labels, not an estimate of whole-pack quality.

## Inputs and replay

The comparison is **raw versus cleaned Parquet**, not raw versus decoded shards. Raw files are under `my-audit/pre/en-nemotroncc-synth/Part_1|Part_2/`; cleaned files are under `my-audit/post/en-nemotroncc-synth/`. Documents were matched by `doc_id`. The selection seed and viewer revision were not recorded with these labels.

The supplied `spec_v1_1/clean.py`, SHA-256 `6a57276c0e83312b09facfba656cca28db0ef59ca851bcfbfa3ef63f16fdd1f2`, was replayed on all 48 raw documents with rule-level instrumentation. **48/48 replay results match the supplied post data exactly**, including the one document absent after cleaning. This establishes which rules caused the changes in this sample. It does not establish that those changes were desirable, or clear the corresponding token shards.

Local evidence:

- [Part_1 labels](day_D6_en-nemotroncc-synth_Part_1.csv).
- [Part_2 labels](day_D6_en-nemotroncc-synth_Part_2.csv).

The normalized label CSVs are included next to this report. Validation results and input hashes are recorded below. Raw/cleaned Parquets, local backups, temporary validation scripts and run logs are outside this minimal submission.

## Final labels

| Part | Full reads | Partial reads | Unique docs | clean | fix | drop |
|---|---:|---:|---:|---:|---:|---:|
| Part_1 | 28 | 0 | 28 | 21 | 7 | 0 |
| Part_2 | 20 | 28 | 48 | 16 | 31 | 1 |
| **Total** | **48** | **28** | **76** | **37** | **38** | **1** |

`fix` includes confirmed residual disease, incorrect transformations, two mixed-content cases awaiting judgment, and five phone-contact ownership review flags. Those five are explicitly marked REVIEW REQUIRED; they are not counted as confirmed phone-masking errors. `clean` on a partial read clears only the inspected phone mask, not the entire document. `drop` is the reviewer's recommendation for an output that was retained; it does not mean that the cleaner already dropped it.

The three residual-boilerplate rows carry C1. Incorrect removal/masking of otherwise valid English text is recorded by rule and effect in `note`, with an empty disease field where no source-disease code fits. In particular, valid titles are not labelled C5 spam, binary digits are not labelled F1 personal data, and the Indic-specific G codes are not forced onto English damage.

## Findings and corresponding medicine work

All medicines below are **proposed updates or decisions**, not approved replacements. The running cleaner/specification was not changed. Six executable replacements and their full-data/gold/regression results are included below; review and approval remain necessary before integration.

| Finding | Documents | Responsible rule | Medicine work |
|---|---|---|---|
| Valid Topic heading removed | `a9092dc10e8938a1`, `d1f7ebf967426312`, `9e9afb31ec4eecd5`, `b3b84f2e80df609c` | `DA-C5-005` | M1: distinguish a single subject heading from a keyword/tag list. |
| Site chrome partially retained | `8c0b0e0c559afe0b`, `0297eb260363a338`, `b57a2a8fbf16fdc6` | `DA-C1-006`; `DA-C2-003` removes repeat copies but leaves one | M2: targeted C1 patterns and metadata handling, preserving article/comment content. |
| Substantive article wrongly dropped | `5b4d0f975990e808` | `DA-C3-003` | M3: replace the terminal-word-only truncation decision. |
| Incorrect phone masks in Part_2 | 22 documents / 25 masks; five additional documents need ownership review | `DA-F1-005` | M4: protect technical values and evidenced public contacts; keep ambiguous ownership cases masked. See phone-mask review below. |
| Repeated record values deleted | `7523ebd6d772fada` | `DA-C2-003` | M5: preserve fields belonging to separate records; do not globally deduplicate them. |
| Mixed factual/meta bullet removed | `f8027db45cfbebb2`, `93bf1542853fd276` | `DA-B4-007` | M6: review clause/sentence-level removal; preserve useful assertions. Final scope needs review. |
| Low-information synthetic task summary retained | `780f33d558dd9f87` | `DA-B4-007` removes a meta line but leaves the task summary | M7: decide the admissible-quality/drop policy; do not infer a general drop rule from one example. |

### M1 Update DA-C5-005 for subject headings

**Cause.** The case-insensitive prefix regex includes `topics?` alongside tags, keywords and filed-under markers. Every matching line is removed; the rule does not inspect whether its value is a list. All four audited documents containing this prefix lose that line. This does not mean every document in the pack has a Topic heading.

**Proposed medicine.** Preserve informative single-topic titles. Remove `Topic:`/`Topics:` from the unconditional prefix test, or restrict removal to separately evidenced tag-list/navigation structures. Use a new proposal/replacement identity through the normal spec process; do not edit generated `clean.py` directly.

**Regression/harm requirements.** Preserve the four reported headings, including `Topic: Japanese Used Cars`, singular/plural/case variants and legitimate headings with commas or colons. Still remove actual tag/keyword boilerplate. Test all four English pack scopes and gold sets. Gold and full-data results for the scoped candidate are in Validation results below.

### M2 Extend C1 cleanup without deleting meaningful comments

**Cause.** `DA-C1-006` recognizes particular chrome lines, such as `Share this:` and `Loading...`, but misses adjacent `Like this:`, image controls, advertisement/byline furniture and comment metadata. `DA-C2-003` removes several duplicate `Click to enlarge.` lines while retaining one; repetition removal is not equivalent to boilerplate removal.

**Proposed medicine.** Add narrowly scoped, whole-line or known-block C1 patterns for the observed chrome. Handle username/timestamp presentation lines with structural context. Remove every confirmed chrome occurrence; retain substantive comments and article text. A comment body is not boilerplate merely because it follows a username.

**Regression/harm requirements.** Recheck the three reported documents. Include substantive prose mentioning the same phrases, meaningful user comments, names/dates in article content, image captions and advertisements adjacent to useful text. Validate against English gold and report rule timing. No new rule ID is reserved here; gold and full-data results are recorded below.

### M3 Replace DA-C3-003's terminal-word drop heuristic

**Cause.** The history article is 3,558 characters long and ends with a complete answer containing “depending on whom you side with”. Because the last token is `with` and the text has no terminal punctuation, the rule drops the entire document. A dangling-word list cannot distinguish a broken fragment from a valid final preposition in this sentence.

**Proposed medicine.** Do not drop an otherwise substantial document from its final token alone. Require stronger, validated truncation evidence; consider preserving the body or trimming only a demonstrably incomplete tail where appropriate. Decide how to treat unpunctuated appended Q&A separately from article truncation.

**Regression/harm requirements.** Preserve this article and grammatical sentences ending with prepositions. Include genuinely truncated short fragments, bracketed truncation markers and otherwise complete documents with appended answers. Record recovered good documents as intended fixes, not gold harm. Run applicable gold and performance checks before adoption.

### M4 Update F1 handling of technical values and public contacts

**Cause.** All **33 `<PHONE>` placeholders across 29 Part_2 documents** were compared to raw text. The numeric regex mistakes grit sequences, binary digits, gaming friend codes and parts of IP examples for phones. It can cross two adjacent IP examples inside shell code. The organisation-context vocabulary/window also misses clearly identified public service contacts, sometimes masking only one of repeated occurrences.

**Assessment.** **25 masks across 22 documents** are assessed as incorrect: 11 non-phone masks in nine documents and 14 masks in 13 public-contact/historical-example documents. **Six masks in five documents** require ownership/policy review; they remain masked in the proposal. **Two masks in two documents** appear appropriate and are retained as controls. These are judgments from the supplied text, not independently verified phone ownership.

**Proposed medicine.** Extend the scoped DA-F1-008 replacement to preserve contextual binary, grit and friend-code examples, protect whole IP examples including adjacent code values, and recognize the evidenced public roles/providers without an ID/number allowlist. Keep private-contact and ambiguous ownership cases masked. Preserve email/address behavior. The historical PE6-5000 example is retained as instructional content. This does not establish a universal consistency policy for numbers in mixed contexts; coordinate with fork #75.

**Regression/harm requirements.** Restore all 25 assessed incorrect masks, retain all eight reviewed ambiguous/private masks, test real phone-shaped strings inside technical prose, IP punctuation/cross-value boundaries, explicit private-contact cues and other-pack no-op scope. Run synth gold, all supplied training parts and held-out data again for the expanded executable block. Earlier binary-only figures must not be presented as validation of this revision.

### M5 Update DA-C2-003 for structured records

**Cause.** Different spell records legitimately repeat Cost, Element, Range and Cool-Down fields. Global repeated-line removal deletes later fields, losing which values apply to which spell. This is not duplicate boilerplate.

**Proposed medicine.** Preserve repeated labelled fields under separate record headings; confine deduplication to evidenced repetitive furniture or equivalent duplicate blocks. Coordinate with fork #75's English C2 replacement: its code/config exclusions do not prove this record case is protected.

**Regression/harm requirements.** Keep all original fields in `7523ebd6d772fada`, repeated key/value lists, product/specification records and worked examples, while still removing genuine repeated share/navigation lines. Test scope, idempotence, gold harm and speed before integration.

### M6 Review DA-B4-007's whole-bullet deletion

**Cause.** This rule detects source-availability meta-comments. In a bullet or key/value line it deletes the whole line, rather than only the matching sentence. In the stainless-steel document, that also deletes the assertion that 316L costs more than 304. In the care-arrangements document, it removes the final explanation and leaves its heading without content.

**Proposed medicine.** Preserve useful assertions when they share a bullet with removable source commentary. Assess sentence/clause-level edits and remove a heading only when it truly becomes empty. The care-arrangements example includes an inference attributed to the source; a reviewer should decide whether to retain it, not have the audit silently rewrite or certify it.

**Control case.** `1fcdf6476cbe3c6d` loses an appended Q&A whose answer explicitly says the requested wage is not provided. That is the intended meta-removal behavior, and its verdict remains `clean`. It must stay a negative-control case for a narrower replacement.

**Regression/harm requirements.** Test the two mixed-content bullets and the control case, separate-line Q&A, inline meta-comments, orphan headings and minimum-length handling. The two `fix` findings remain review flags until the desired editing policy is agreed. Gold and full-data results are recorded below.

### M7 Decide the synthetic-summary quality policy

The reviewer recommends dropping `780f33d558dd9f87`, which describes an AI email-writing request without providing the webinar content needed to fulfill it. The cleaner removes one missing-source meta line, reducing 460 characters to 356, but keeps the rest. This is not established SEO spam or personal data, so neither C5 nor F1 is assigned.

**Proposed next step.** Agree whether such incomplete synthetic task summaries belong in this source, then collect additional positives and useful counterexamples. If a rule is approved, define scoped signals and a drop/fix action and validate it against English gold and useful instruction/task descriptions. The single audit judgment is not enough to deploy an automatic quality filter.

## Remaining decisions

1. Review M6's two mixed-content cases and M7's one drop recommendation; the notes distinguish these judgments from replay-confirmed rule behavior.
2. Review the executable M1–M6 replacement cards and validation below, and resolve M7's policy. The six scoped replacements below have gold/full-data evidence; none has been deployed.
3. Arrange the second-reader check on at least 5 of the 48 documents. Include the history drop, binary false positive and mixed-content removal cases.
4. If claiming these findings apply to the frozen-p1536 shards or corpus-v1, compare those actual versions separately. This report establishes raw/cleaned behavior only.

No raw documents, personal data values, model bytes or signed download links are included in this report. Offsets in the normalized notes are explicitly raw or post where supplied; no new offsets were guessed.

## Executable medicine proposals

The following cards follow [Aneesha's PR #71](https://github.com/patilvijay23/era-v5-lab/pull/71) format (reviewed head `4d17b696`). They are scoped to `en-nemotroncc-synth`, copy the replaced approved rule, and make the narrow change described below. IDs are provisional until the team checks the shared registry and other open proposals. Each block is standalone and is loaded by the existing harm-test loader. These are replacements, not extra rules to append after destructive originals.


### Medicine 1 DA-C5-007 C5 subject headings versus tag lists

1. **Disease:** C5; observed source disease or false-positive behavior of the corresponding rule, as distinguished in the finding table above. **Replaces:** `DA-C5-005` on en-nemotroncc-synth only. **Status:** PROPOSAL.
2. **Medicine:** Preserve all Topic:/Topics: lines in this scoped replacement; keep the existing tag/keyword/filed-under removal. This intentionally trades some tag-list recall for preserving informative headings.

```python
SCOPE = {"DA-C5-005": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-C5-005"
RULE_ID = "DA-C5-007"
SOURCE_RULE_ID = "DA-C5-005"
CODE = "C5"
ACTION = "fix"
STAGE = "35"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

C5_PREFIX = re.compile(r"^(?:tags?|keywords?|tagged(?: with| in)?|filed under)\s*:", re.I)


def medicine_c5(text):
    lines = text.split("\n")
    keep = [ln for ln in lines if not C5_PREFIX.search(ln.strip())]
    if len(keep) == len(lines):
        return ("keep", text)
    new = "\n".join(keep).strip("\n")
    return ("drop", text) if not new.strip() else ("fix", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return medicine_c5(doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-C5-005"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** preserve the original rule's fix/drop contract, with the scoped change above. Run at the original rule's pipeline position; leave other packs on their existing rules.
4. **Harm and full-pack tests:** results are reported in the validation section below; a passing gold result is separate from rule adoption.
5. **Side effects:** Topic-prefixed tag lists are now retained too. New list-specific detection requires separate labelled evidence; a heading containing commas is not automatically spam.
6. **Integration:** reviewer pending. Apply only after spec approval, rebuild and pin the cleaner, and rerun affected pipeline/shard checks. Never run the old destructive rule first and attempt to repair its output afterward.


### Medicine 2 DA-C1-013 C1 image controls and comment metadata

1. **Disease:** C1; observed source disease or false-positive behavior of the corresponding rule, as distinguished in the finding table above. **Replaces:** `DA-C1-006` on en-nemotroncc-synth only. **Status:** PROPOSAL.
2. **Medicine:** Extend the existing whole-line widget list with Like this and Click to enlarge; remove username/relative-age presentation lines; strip an Advertisement: prefix while preserving its following byline text. Retain substantive comments.

```python
SCOPE = {"DA-C1-006": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-C1-006"
RULE_ID = "DA-C1-013"
SOURCE_RULE_ID = "DA-C1-006"
CODE = "C1"
ACTION = "fix"
STAGE = "35"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

WIDGET = re.compile(r"(?i)(?:like this:?|click to enlarge[.!]?|primary tabs|search for:?|related (?:documents|posts|articles):?|click to expand\W*|read more\W*|\[?read more\W*\]?|share this:?|share on (?:other sites|[a-z ]+)|previous|next|menu|contents|advertisement|loading\W*|link copied|the link has been copied!?|post navigation|quantity|breadcrumb|logged|registered|banned|show more|you may also like[a-z ]*|subscribe to [a-z ]+|skip to (?:main )?content|back to top)")

COMMENT_META = re.compile(r"[A-Za-z0-9_-]{3,40} (?:over )?(?:a|an|\d+) (?:years?|months?|days?|hours?) ago", re.I)

def c1(t):
    """Delete lines that consist only of a known widget or menu label ('Primary tabs', 'Search for:', 'Related posts', 'Share this:' ...)."""
    t = re.sub(r"(?m)^Advertisement:(?=\S)", "", t)
    return "\n".join(l for l in t.split("\n")
                     if not WIDGET.fullmatch(l.strip()) and not COMMENT_META.fullmatch(l.strip()))


def _std(out, text):  # added by Data A
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(c1(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-C1-006"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** preserve the original rule's fix/drop contract, with the scoped change above. Run at the original rule's pipeline position; leave other packs on their existing rules.
4. **Harm and full-pack tests:** results are reported in the validation section below; a passing gold result is separate from rule adoption.
5. **Side effects:** A meaningful standalone sentence matching username-plus-age syntax can be mistaken for metadata. Advertisement handling removes only that marker, not arbitrary following author/article text. Gold and contextual review remain necessary.
6. **Integration:** reviewer pending. Apply only after spec approval, rebuild and pin the cleaner, and rerun affected pipeline/shard checks. Never run the old destructive rule first and attempt to repair its output afterward.


### Medicine 3 DA-C3-008 C3 conservative truncation handling

1. **Disease:** C3; observed source disease or false-positive behavior of the corresponding rule, as distinguished in the finding table above. **Replaces:** `DA-C3-003` on en-nemotroncc-synth only. **Status:** PROPOSAL.
2. **Medicine:** Withdraw whole-document drops based solely on the final token. Preserve the existing explicit bracketed ellipsis-marker behavior. This recovers the complete answer ending in depending on whom you side with.

```python
SCOPE = {"DA-C3-003": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-C3-003"
RULE_ID = "DA-C3-008"
SOURCE_RULE_ID = "DA-C3-002"
CODE = "C3"
ACTION = "drop"
STAGE = "30"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

_MARKER = re.compile(r"\[\s*(?:…|\.\.\.)\s*\]$")

_DANGLING = frozenset(
    "a an the and or but of to in on at for with from by as is are was were "
    "that this these those which who because however".split()
)

_TOKEN = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'")  # the characters of [A-Za-z0-9']
_LONGEST = max(map(len, _DANGLING))  # a last token longer than every dangling word cannot be one

def medicine_c3(text):
    tail = text.rstrip()
    if not tail:
        return text
    if tail.endswith("]") and _MARKER.search(tail):  # the marker ends the text, so the text must end in "]"
        return ("drop", "")
    if tail.endswith((".", "!", "?", '"', "'", ")", ":", "…")):
        return text
    # The terminal-word heuristic is deliberately withdrawn.
    return text


def _std(out, text):  # added by Data A
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(medicine_c3(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-C3-003"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** preserve the original rule's fix/drop contract, with the scoped change above. Run at the original rule's pipeline position; leave other packs on their existing rules.
4. **Harm and full-pack tests:** results are reported in the validation section below; a passing gold result is separate from rule adoption.
5. **Side effects:** Genuinely truncated documents previously rejected only by the terminal-word heuristic will now be kept. The full-pack recovery count must be reported and reviewed; zero gold harm alone does not validate every recovered document.
6. **Integration:** reviewer pending. Apply only after spec approval, rebuild and pin the cleaner, and rerun affected pipeline/shard checks. Never run the old destructive rule first and attempt to repair its output afterward.


### Medicine 4 DA-F1-008 F1 technical values and public contacts

1. **Disease:** F1; observed source disease or false-positive behavior of the corresponding rule, as distinguished in the finding table above. **Replaces:** `DA-F1-005` on en-nemotroncc-synth only. **Status:** PROPOSAL.
2. **Medicine:** Protect contextual binary, sandpaper-grit and 3DS friend-code examples; protect IP examples including matches crossing adjacent code values. Extend phone-only recognition of evidenced public service/provider contacts and the historical phoneword example. Ambiguous ownership cases keep baseline masking. Email/address behavior remains unchanged. No document-ID or phone-number allowlist is used.

```python
SCOPE = {"DA-F1-005": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-F1-005"
RULE_ID = "DA-F1-008"
SOURCE_RULE_ID = "DA-F1-005"
CODE = "F1"
ACTION = "fix"
STAGE = "40"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")

WEBMAIL = re.compile(r"@(?:gmail|googlemail|yahoo|ymail|hotmail|outlook|live|msn|aol|icloud|me|mac|protonmail|proton|"
                     r"rediffmail|zoho|gmx|yandex|mail|comcast|verizon|att|sbcglobal|bellsouth|earthlink|btinternet)\.", re.I)

PERSONAL_LOCAL = re.compile(r"[._\-]|\d|^.{13,}$")

ROLE_LOCAL = re.compile(r"^(?:info|contact|contactus|enquir(?:y|ies)|inquir(?:y|ies)|sales|support|help|helpdesk|admin|"
                        r"office|hello|hi|press|media|news|newsletter|editor|editors|team|jobs|careers|hr|marketing|"
                        r"billing|accounts|orders|service|services|customerservice|webmaster|postmaster|noreply|"
                        r"no-reply|abuse|privacy|legal|admissions|registrar|events|bookings|reservations)$", re.I)

NOT_REAL = re.compile(r"@(?:example|test|domain|email)\.(?:com|org|net)$", re.I)

PHONE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.\-‐‑–]?)?(?:\(\d{2,4}\)[\s.\-‐‑–]?|\d{2,4}[\s.\-‐‑–])\d{3,4}[\s.\-‐‑–]\d{3,4}(?![\w-])")

TOLL_FREE = re.compile(r"^(?:\+?1[\s.-]?)?\(?8(?:00|88|77|66|55|44|33)\)?")

STREET = (r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Ter|Close|"
          r"Crescent|Cres|Highway|Hwy|Parkway|Pkwy|Circle|Cir|Square|Sq)")

ADDRESS = re.compile(r"\b\d{1,5}[A-Za-z]?\s+(?:[NSEW]\.?\s+)?(?:[A-Z][a-z]+\.?\s+){1,4}" + STREET + r"\b\.?"
                     r"(?:,?\s*(?:Apt|Unit|Suite|Ste|#)\.?\s*\w+)?"
                     r"(?:,\s*(?:[A-Z][a-z]+\s?){1,3},\s*[A-Z]{2}\.?\s*\d{5}(?:-\d{4})?)?")

CONTACT_LINE = 120

ORGANISATION = re.compile(
    r"(?i)\b(?:inc|llc|ltd|plc|corp|co\.|company|warehouse|store|shop|market|restaurant|cafe|café|bar|grill|bistro|"
    r"hotel|inn|motel|resort|spa|salon|clinic|hospital|surgery|pharmacy|cent(?:er|re)|office|offices|university|"
    r"college|school|academy|library|museum|church|gallery|theat(?:er|re)|cinema|bank|agency|association|society|"
    r"council|department|ministry|embassy|tourist|information|headquarters|hq|branch|showroom|dealership|"
    r"open (?:mon|tue|wed|thu|fri|sat|sun|daily|7|24)\w*|hours|menu|reservations|fax|"
    r"on (?:urbanspoon|yelp|tripadvisor|zomato|foursquare)|"
    # trades and services that put their number on their own page
    r"pros?|plumb\w*|roofing|hvac|electric(?:al|ians?)|landscap\w*|cleaning|movers?|moving|storage|rentals?|hire|"
    r"repairs?|construction|builders?|contractors?|auto|motors|towing|insurance|realty|real estate|mortgage|loans?|"
    r"law|lawyers?|attorneys?|legal|dental|dentistry|orthodont\w*|chiropract\w*|physiotherap\w*|therapy|"
    r"veterinar\w*|vet|clinics?|practice|surgery|medical|health(?:care)?|consult(?:ation|ing|ants?)?|services?|"
    r"solutions|group|partners|associates|studio|chalet|lodge|villa|apartments?|cottage|camping|farm|winery|"
    r"brewery|bakery|club|bbs|board|optometr\w*|optic\w*|opticians?|spectacles|suite)\b")

ORG_VOICE = re.compile(r"(?i)\b(?:call us|contact us|email us|e-mail us|visit us|reach us|text us|give us a call|"
                       r"our (?:office|team|staff|clinic|store|shop|experts?|technicians?|specialists?|services?|"
                       r"company|firm|practice|location)|one of our|we (?:offer|provide|specialize|specialise|serve)|need help\?)|☎")

PUBLIC_ROLE = re.compile(r"(?i)\b(?:reporter|editor|journalist|correspondent|spokes(?:man|woman|person)|press|media|"
                         r"publicist|author of|professor|prof\.|dr\.|lecturer|faculty|researcher|principal investigator|"
                         r"media contact|press contact|for more information|high[- ]res|images|photos)\b")

DIGIT = re.compile(r"\d")

def _email(m, org):
    addr = m.group()
    local = addr.split("@", 1)[0]
    if NOT_REAL.search(addr) or ROLE_LOCAL.match(local):
        return addr
    if WEBMAIL.search(addr):  # a webmail address belongs to a person
        return "<EMAIL>"
    # a name-like address at an organisation's domain is a work address when the context is an organisation's
    return addr if org or not PERSONAL_LOCAL.search(local) else "<EMAIL>"

BINARY_CONTEXT = re.compile(r"\b(?:binary|bits?|base[ -]?2)\b", re.I)
PHONE_CONTEXT = re.compile(r"\b(?:phone|telephone|tel|mobile|call|contact)\b", re.I)
BINARY_DIGITS = re.compile(r"[01 ]{8,64}")
IP_EXAMPLE = re.compile(r"(?<![\w.])[0-9Oo]{1,3}(?:\.[0-9Oo]{1,3}){3}(?!\w|\.\w)")
IP_CONTEXT = re.compile(r"\b(?:ip|subnet|broadcast|net\s*mask)\b", re.I)
GRIT_CONTEXT = re.compile(r"\b(?:sand(?:ing|paper)?|grits?)\b", re.I)
FRIEND_CONTEXT = re.compile(r"\b(?:3DS|friend\s*code|Pok[eé]mon)\b", re.I)
PRIVATE_CUE = re.compile(r"\b(?:my (?:personal |private |home )?(?:phone|number)|personal (?:phone|number)|private (?:phone|number)|contact me|call me|mobile)\b", re.I)
PUBLIC_INSTITUTION = re.compile(r"\b(?:charities|youth bureau|citizens for|schools|heating\s*(?:&|and)\s*air conditioning)\b", re.I)
PUBLIC_CONTACT = re.compile(r"\b(?:contact|call|registration|waitlist|inquiries|sponsors?|booking)\b", re.I)

def _technical_number(line, match, previous):
    value = match.group().strip()
    # Protect complete IP examples before phone matching, including a match that
    # crosses adjacent IP examples. Keep invalid examples unchanged, not repaired.
    if IP_CONTEXT.search(line):
        spans = [m.span() for m in IP_EXAMPLE.finditer(line)]
        if spans and all(line[p].isspace() or any(a <= p < b for a, b in spans)
                         for p in range(match.start(), match.end())):
            return True
    if PHONE_CONTEXT.search(line):
        return False
    if (BINARY_DIGITS.fullmatch(value) is not None
            and len(value.replace(" ", "")) >= 8
            and BINARY_CONTEXT.search(line[max(0, match.start() - 80):match.start()])):
        return True
    if re.fullmatch(r"\d{2,4}-\d{2,4}-\d{2,4}", value):
        grits = [int(v) for v in value.split("-")]
        if (16 <= grits[0] < grits[1] < grits[2] <= 3000
                and GRIT_CONTEXT.search(previous[-200:] + line[:match.start()])):
            return True
    return (re.fullmatch(r"\d{4}-\d{4}-\d{4}", value) is not None
            and re.search(r"\bFC\s*$", line[:match.start()], re.I) is not None
            and FRIEND_CONTEXT.search(previous[-200:] + line) is not None)

def _public_phone(text, lines, i, match):
    line = lines[i]
    if PRIVATE_CUE.search(line):
        return False
    previous = "\n".join(lines[max(0, i - 2):i])
    if (PUBLIC_INSTITUTION.search(line)
            or (i > 0 and PUBLIC_INSTITUTION.search(lines[i - 1])
                and len(lines[i - 1]) < 160)):
        return True
    if re.search(r"\bEducation Director\b", line, re.I):
        return True
    if (re.search(r"\bDirector of Guidance and Testing for .{1,80}\bSchools\b", previous, re.I)
            and re.search(r"inquiries regarding .{1,100}Assessment.*contact Mr\.", line, re.I)):
        return True
    if re.search(r"supported by .{1,80}\bResearch\b.*sponsors", line, re.I):
        return True
    if re.search(r"Checking Voter Registration Status", previous, re.I) and PUBLIC_CONTACT.search(line):
        return True
    if (PUBLIC_CONTACT.search(line)
            and re.search(r"\bMuseum\b", text)
            and re.search(r"\b(?:program|session)\b", line, re.I)):
        return True
    if (PUBLIC_CONTACT.search(line)
            and re.search(r"(?:\b[\w.+-]+@[\w.-]+\.edu\b|\b(?:info|registration|bookings)@[\w.-]+\b)", line, re.I)):
        return True
    if re.search(r"^Provider:\s*\S", previous, re.M) and re.match(r"Contact:", line, re.I):
        return True
    return ("Oldest Continuously Operating Phone Number" in text
            and "PE6-5000" in text and "number can be dialed as" in line)

def medicine_f1(text):
    lines = text.split("\n")
    org_page = []  # ORG_VOICE on the whole text: searched once, and only when a contact line asks
    has_org = {}  # line number -> an organisation word on that line: searched once, and only when a contact line asks

    def on_org_page():
        if not org_page:
            org_page.append(bool(ORG_VOICE.search(text)))
        return org_page[0]

    def org_near(i):  # same answer as searching the joined 13-line window, as no match spans a newline
        for j in range(max(0, i - 6), min(len(lines), i + 7)):
            if j not in has_org:
                has_org[j] = ORGANISATION.search(lines[j]) is not None
            if has_org[j]:
                return True
        return False

    out = []
    for i, line in enumerate(lines):
        if "@" not in line and not DIGIT.search(line):  # no EMAIL, PHONE or ADDRESS can match
            out.append(line)
            continue
        new = EMAIL.sub(lambda m: _email(m, on_org_page() or org_near(i) or bool(PUBLIC_ROLE.search(
            "\n".join(lines[max(0, i - 2):i + 1])))), line) if "@" in line else line
        if PHONE.search(new) or ADDRESS.search(new):
            if not (on_org_page() or org_near(i)):
                new = PHONE.sub(lambda m: m.group() if TOLL_FREE.match(m.group()) or _technical_number(new, m, "\n".join(lines[max(0, i - 2):i])) or _public_phone(text, lines, i, m) else "<PHONE>", new)
                if len(line.strip()) <= CONTACT_LINE:
                    new = ADDRESS.sub("<ADDRESS>", new)
        out.append(new)
    return "\n".join(out)


def _std(out, text):  # added by Data A
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(medicine_f1(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-F1-005"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** preserve the original rule's fix/drop contract, with the scoped change above. Run at the original rule's pipeline position; leave other packs on their existing rules.
4. **Harm and full-pack tests:** results are reported in the validation section below; a passing gold result is separate from rule adoption.
5. **Side effects:** Technical-context and public-role heuristics can retain private numbers if their surrounding prose is misleading. Explicit phone/contact cues defeat the grit/binary/friend-code exemption; private-contact cues defeat the added public-contact exemption. Existing baseline organisation exemptions remain unchanged. Sparse/ambiguous ownership cases are not automatically unmasked. This does not implement a universal per-number consistency policy; coordinate with fork #75. IP examples are preserved, not fact-corrected.
6. **Integration:** reviewer pending. Apply only after spec approval, rebuild and pin the cleaner, and rerun affected pipeline/shard checks. Never run the old destructive rule first and attempt to repair its output afterward.


### Medicine 5 DA-C2-008 C2 repeated structured fields

1. **Disease:** C2; observed source disease or false-positive behavior of the corresponding rule, as distinguished in the finding table above. **Replaces:** `DA-C2-003` on en-nemotroncc-synth only. **Status:** PROPOSAL.
2. **Medicine:** Exclude labelled key/value fields from global repeated-line deletion. Preserve each spell record's Cost/Element/Range/Cool-Down values; retain existing deduplication for unlabelled repeated lines.

```python
SCOPE = {"DA-C2-003": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-C2-003"
RULE_ID = "DA-C2-008"
SOURCE_RULE_ID = "DA-C2-003"
CODE = "C2"
ACTION = "fix"
STAGE = "35"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

def _merge(spans):
    """Overlapping spans joined into one, keeping the rule of the first."""
    out = []
    for s, e, rule in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]), out[-1][2])
        else:
            out.append((s, e, rule))
    return out

C2_MIN_REPEAT, C2_MIN_CHARS = 3, 10
RECORD_FIELD = re.compile(r"^\s*(?:[-*•]\s+)?[A-Za-z][A-Za-z0-9 /_-]{0,35}:\s+\S")

def c2_spans(text):
    """C2: later copies of a line repeated C2_MIN_REPEAT or more times in the document."""
    seen, lines, pos = {}, [], 0
    for line in text.split("\n"):
        key = " ".join(line.split())
        if len(key) >= C2_MIN_CHARS and not RECORD_FIELD.match(line):
            lines.append((pos, pos + len(line), key))
            seen[key] = seen.get(key, 0) + 1
        pos += len(line) + 1
    out, first = [], set()
    for s, e, key in lines:
        if seen[key] >= C2_MIN_REPEAT:
            if key in first:
                out.append((s, e, "C2-repeated-line"))
            first.add(key)
    return out


def _std(out, text):  # added by Data A
    # the author's result (spans to remove) in the shared form, as their clean() reads it: (spans, True) from
    # c1_spans -> drop; a whole-document "B4-mostly-code" or "C5-list-page" span -> drop; else the spans go
    if isinstance(out, tuple):
        out, flagged = out
        if flagged: return ("drop", text)
    if out and out[0][2] in ("B4-mostly-code", "C5-list-page"): return ("drop", text)
    new = text
    for s, e, _ in reversed(_merge(out)):
        new = new[:s] + new[e:]
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(c2_spans(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-C2-003"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** preserve the original rule's fix/drop contract, with the scoped change above. Run at the original rule's pipeline position; leave other packs on their existing rules.
4. **Harm and full-pack tests:** results are reported in the validation section below; a passing gold result is separate from rule adoption.
5. **Side effects:** Repeated metadata such as Views: or Share: can now survive; C1 needs to recognize verified furniture separately. This does not fix all code/formula preservation issues in the English C2 rule and is separate from fork #75.
6. **Integration:** reviewer pending. Apply only after spec approval, rebuild and pin the cleaner, and rerun affected pipeline/shard checks. Never run the old destructive rule first and attempt to repair its output afterward.


### Medicine 6 DA-B4-015 B4 mixed factual and source-meta bullets

1. **Disease:** B4; observed source disease or false-positive behavior of the corresponding rule, as distinguished in the finding table above. **Replaces:** `DA-B4-007` on en-nemotroncc-synth only. **Status:** PROPOSAL.
2. **Medicine:** Remove only matching meta sentences from mixed bullets/key-value lines, preserving factual sentences. Retain an ambiguous but it implies/suggests clause unchanged. Keep existing removal of unanswerable appended Q&A and pure meta lines.

```python
SCOPE = {"DA-B4-007": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-B4-007"
RULE_ID = "DA-B4-015"
SOURCE_RULE_ID = "DA-B4-007"
CODE = "B4"
ACTION = "fix"
STAGE = "40"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re
ANSWER_LINE = re.compile('^(?:Answer|A)\\s*[:.]|\\bAnswer:\\s')  # added by Data A: from maheshsv's medicine.md
KEY_VALUE_LINE = re.compile('^[^:\\n]{1,60}:\\s')  # added by Data A: from maheshsv's medicine.md
LIST_MARK = re.compile('^(?:[-*•–—]|\\d+[.)]|[a-z][.)])\\s')  # added by Data A: from maheshsv's medicine.md
META = re.compile('(?i)\\bnot (?:explicitly |specifically |directly )?(?:provided|stated|mentioned|specified|given|available|included|present|disclosed|named) in the (?:given |provided |original |above |source )?(?:text|passage|article|document|excerpt|content|context)|\\bthe (?:text|passage|article|document) does not (?:explicitly |specifically )?(?:provide|mention|specify|state|say|give|include|indicate|name)\\b|\\bthe (?:given|provided) (?:text|passage) (?:also )?(?:discusses|mentions|describes|states|says|provides|refers|covers|explains|includes|lists|notes)\\b')  # added by Data A: from maheshsv's medicine.md
MIN_KEEP_CHARS = 200
# Preserve an ambiguous source inference rather than inventing a stronger assertion.
MIXED_INFERENCE = re.compile(r"\bbut (?:it |the (?:text|passage|article) )?(?:implies|suggests|indicates)\b", re.I)  # added by Data A: from maheshsv's medicine.md

def remove_meta_comments(text: str) -> tuple[str | None, bool]:
    """Delete the bullet, `Key: value` line or Question/Answer pair that carries a rewriter meta-comment; in running
    prose delete only the sentence."""
    if not META.search(text):
        return text, False
    lines = text.split("\n")
    drop = set()
    for i, ln in enumerate(lines):
        if not META.search(ln):
            continue
        s = ln.strip()
        if s.startswith("Question:") or ANSWER_LINE.match(s):
            drop.add(i)                                   # drop the whole QA pair, whichever line carries it
            if s.startswith("Question:") and i + 1 < len(lines) and ANSWER_LINE.match(lines[i + 1].strip()):
                drop.add(i + 1)
            if ANSWER_LINE.match(s) and not s.startswith("Question:") and i > 0 and lines[i - 1].strip().startswith("Question:"):
                drop.add(i - 1)
        elif LIST_MARK.match(s) or KEY_VALUE_LINE.match(s):
            if MIXED_INFERENCE.search(s):
                continue
            sentences = re.split(r"(?<=[.!?])\s+", ln)
            kept = [sentence for sentence in sentences if not META.search(sentence)]
            if kept:
                lines[i] = " ".join(kept)
            else:
                drop.add(i)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", ln)
            lines[i] = " ".join(x for x in sentences if not META.search(x))
    new = "\n".join(ln for i, ln in enumerate(lines) if i not in drop)
    new = re.sub(r"\n{3,}", "\n\n", new).strip()
    if len(new) < MIN_KEEP_CHARS:
        return None, True
    return new, True


def _std(out, text):  # added by Data A
    # the author's result (new_text or None, changed) in the shared form: None -> drop, else fix/keep by the text
    new, _ = out
    if new is None: return ("drop", text)
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(remove_meta_comments(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-B4-007"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** preserve the original rule's fix/drop contract, with the scoped change above. Run at the original rule's pipeline position; leave other packs on their existing rules.
4. **Harm and full-pack tests:** results are reported in the validation section below; a passing gold result is separate from rule adoption.
5. **Side effects:** Some source commentary intentionally remains in ambiguous mixed-inference bullets. Sentence segmentation is heuristic; source facts in a single matching sentence can still be lost. This is conservative partial repair, not a factuality scorer.
6. **Integration:** reviewer pending. Apply only after spec approval, rebuild and pin the cleaner, and rerun affected pipeline/shard checks. Never run the old destructive rule first and attempt to repair its output afterward.

## All audited documents

`full` and `partial` describe reading coverage, not confidence in the whole pack. The existing full read of `24e830d71e5bc566` is retained; its additional phone-context review does not create a duplicate row. All 28 newly added IDs are `partial`, with blank evidence/hex and blank disease codes for masking damage or ownership questions rather than source F1 personal-data disease.

| Part | doc_id | Read | Verdict | Diseases | Note |
|---|---|---|---|---|---|
| 1 | `8c0b0e0c559afe0b` | full | fix | C1 | Residual boilerplate: DA-C1-006 removes Share this but leaves Like this at post offset 998. Extend the scoped site-chrome medicine; preserve article text. |
| 1 | `b949f7934ece26f6` | full | clean | — | No issue reported in the manual review. |
| 1 | `5b4d0f975990e808` | full | fix | — | Incorrect whole-document drop by DA-C3-003: the final complete answer ends with depending on whom you side with. A terminal preposition alone is not proof of truncation. Restore the article; investigate a context-aware rule replacement. |
| 1 | `c0700a42d38ffcfb` | full | clean | — | No issue reported in the manual review. |
| 1 | `b7da3294657a4aa6` | full | clean | — | No issue reported in the manual review. |
| 1 | `0297eb260363a338` | full | fix | C1 | Residual site chrome: Click to enlarge at post offset 1810 and advertisement/byline furniture at post offset 1920. DA-C2-003 removes repeated copies but leaves one; add scoped C1 removal, not broader repeat deletion. |
| 1 | `6edfb805e854f8cd` | full | clean | — | No issue reported in the manual review. |
| 1 | `80ec379221b6d231` | full | clean | — | No issue reported in the manual review. |
| 1 | `42e3205b05893f8f` | full | clean | — | No issue reported in the manual review. |
| 1 | `f8027db45cfbebb2` | full | fix | — | Potential content loss: DA-B4-007 deletes the complete Cost Difference bullet because it also contains a source-availability caveat. Preserve the factual stainless-steel comparison; review sentence-level removal. Raw 976 chars, post 794 chars. |
| 1 | `10070ea935885063` | full | clean | — | No issue reported in the manual review. |
| 1 | `b57a2a8fbf16fdc6` | full | fix | C1 | Residual comment metadata: username/relative timestamp at post offset 412 remains after DA-C1-006 removes Loading. Remove only presentation metadata if approved; preserve substantive user comments. |
| 1 | `9a8d21351d48e7a4` | full | clean | — | No issue reported in the manual review. |
| 1 | `a9092dc10e8938a1` | full | fix | — | Replay-confirmed Topic heading removal by DA-C5-005, also present in three other audited documents. The rule deletes every Topic/Topics-prefixed line without testing for a keyword list. Review as a false-positive removal, not C5 spam in the source. |
| 1 | `1fcdf6476cbe3c6d` | full | clean | — | Resolved question: DA-B4-007 removes an appended Q&A whose answer explicitly says the requested wage is not provided. This matches the intended rewriter-meta cleanup; not counted as confirmed harmful removal. |
| 1 | `0f23da54cecc5a58` | full | clean | — | No issue reported in the manual review. |
| 1 | `fc10b82da7fb24ac` | full | clean | — | No issue reported in the manual review. |
| 1 | `91ab3364ac1512d7` | full | clean | — | No issue reported in the manual review. |
| 1 | `a2e0f82c86a5e538` | full | clean | — | No issue reported in the manual review. |
| 1 | `acfa25d6c34e5bf5` | full | clean | — | No issue reported in the manual review. |
| 1 | `d1f7ebf967426312` | full | fix | — | DA-C5-005 removes the valid Topic heading Text Analysis of Political Speech. Proposed update: preserve single-topic titles; remove only evidenced tag/keyword-list boilerplate. |
| 1 | `add97e507366e20e` | full | clean | — | No issue reported in the manual review. |
| 1 | `e2e54ae8ca53fb13` | full | clean | — | No issue reported in the manual review. |
| 1 | `2917191a7028177c` | full | clean | — | No issue reported in the manual review. |
| 1 | `0f790b3e51447cfd` | full | clean | — | No issue reported in the manual review. |
| 1 | `04c8254eec115f84` | full | clean | — | No issue reported in the manual review. |
| 1 | `12ce8decc266bfcf` | full | clean | — | No issue reported in the manual review. |
| 1 | `eec2c60c12f1d3f5` | full | clean | — | No issue reported in the manual review. |
| 2 | `d511dd903a4a067c` | full | clean | — | No issue reported in the manual review. |
| 2 | `e46c58b45d4011ce` | full | clean | — | No issue reported in the manual review. |
| 2 | `8a6a1d17705d675c` | full | clean | — | No issue reported in the manual review. |
| 2 | `c74967ec522fd90c` | full | clean | — | No issue reported in the manual review. |
| 2 | `24e830d71e5bc566` | full | fix | — | Confirmed false-positive phone masking by DA-F1-005. A 12-bit binary illustration is incorrectly replaced by `<PHONE>`. Additional partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `063e8bbd4f779a03` | full | clean | — | No issue reported in the manual review. |
| 2 | `780f33d558dd9f87` | full | drop | — | Reviewer recommends dropping this low-information synthetic task summary: it describes a request for a webinar email while the necessary webinar content is missing. DA-B4-007 removes one meta line but retains the task summary. Quality-policy decision pending; no unsupported disease code assigned. |
| 2 | `9e9afb31ec4eecd5` | full | fix | — | DA-C5-005 removes Topic: Japanese Used Cars at raw offset 0. This is a single subject heading; the rule does not distinguish it from a tag/keyword list. |
| 2 | `282a9f774a954f79` | full | clean | — | No issue reported in the manual review. |
| 2 | `7523ebd6d772fada` | full | fix | — | DA-C2-003 removes repeated Cost, Element, Range and Cool-Down fields across different spell records, including the field near raw offset 400. These repetitions express independent record values; preserve structured records rather than deduplicating lines globally. |
| 2 | `9b60a97bd619362d` | full | clean | — | No issue reported in the manual review. |
| 2 | `96a659fe3e369e94` | full | clean | — | No issue reported in the manual review. |
| 2 | `7566e41e459e6255` | full | clean | — | No issue reported in the manual review. |
| 2 | `dded0be038079859` | full | clean | — | No issue reported in the manual review. |
| 2 | `b415a2af9acab8c0` | full | clean | — | No issue reported in the manual review. |
| 2 | `79034f4823c0a497` | full | clean | — | No issue reported in the manual review. |
| 2 | `b3b84f2e80df609c` | full | fix | — | DA-C5-005 removes the single Topic heading at raw offset 0 describing governments and banks during financial globalization. Preserve informative subject headings. |
| 2 | `055ecf0228699d9e` | full | clean | — | No issue reported in the manual review. |
| 2 | `de5b3e2dd5b7fbb0` | full | clean | — | No issue reported in the manual review. |
| 2 | `93bf1542853fd276` | full | fix | — | Potential content loss: DA-B4-007 deletes the complete final bullet under Extended family relationships because it contains a source-availability caveat, leaving a heading without its explanation. Review at sentence/clause level before changing the medicine. |
| 2 | `e8f908a42871e4b6` | partial | fix | — | Confirmed false-positive phone masking by DA-F1-005. Sandpaper grit sequences 80-120-180 and 100-150-220 are incorrectly replaced by two `<PHONE>` tokens. Partial phone-context review of raw versus supplied post; 2 observed mask(s). |
| 2 | `4bc070c79215632d` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: WNCCEIB organisation contact. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `1f472835ef145fb9` | partial | fix | — | Confirmed false-positive phone masking by DA-F1-005. A Nintendo 3DS friend code (FC) is incorrectly replaced by `<PHONE>`; it is not a telephone number. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `fa933cc36ed916ff` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: Gee & Missler Heating & Air Conditioning business contact. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `c27ffe1aed1ee536` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: school assessment director official inquiry contact. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `17003ddc9fb7a9d6` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: Catholic Charities and City-County Youth Bureau contacts (two masks; the second source number is incomplete, which is not a reason to relabel it private). Partial phone-context review of raw versus supplied post; 2 observed mask(s). |
| 2 | `020037fc56eaceee` | partial | fix | — | Confirmed false-positive phone masking by DA-F1-005. The first three octets of broadcast address 255.255.255.254 are replaced, leaving `<PHONE>`.254. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `364298590b01a61e` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: Burraja cultural-program contact. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `452886dff1756ff9` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: university tournament sponsorship contact. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `8aecb4bed3222e59` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: public voter-registration status service; repeated occurrences are treated inconsistently. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `487169495fd63645` | partial | fix | — | REVIEW REQUIRED; not a confirmed false positive. Named volunteer coordinator in an institutional setting; public versus personal line is not established. Retain existing masking until resolved. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `417036b98dd9380a` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: museum-program registration/waitlist; repeated occurrences are treated inconsistently. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `1643b537b787cb99` | partial | fix | — | Confirmed false-positive phone masking by DA-F1-005. Two subnet-mask examples lose their first three octets, leaving `<PHONE>`.0. Preserve numeric examples; the questionable 225-prefixed source example is a separate factual issue. Partial phone-context review of raw versus supplied post; 2 observed mask(s). |
| 2 | `0fdc213d4606fc4f` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: Penn LPS Online course registration. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `b00929f7a291d0c8` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: Education Director booking contact with extension. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `23111b1b824081cd` | partial | fix | — | REVIEW REQUIRED; not a confirmed false positive. Unattributed number above a health article; organisation ownership is not established. Retain existing masking until resolved. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `07c8d8752910472e` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: Get Trainee training-provider contact. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `53b84b1caa2f39f1` | partial | fix | — | REVIEW REQUIRED; not a confirmed false positive. Two named church-program contacts; personal versus institutional ownership is not established. Retain existing masking until resolved. Partial phone-context review of raw versus supplied post; 2 observed mask(s). |
| 2 | `d6f8a9fc304391e3` | partial | clean | — | Direct personal contact: the observed phone mask appears appropriate. Other occurrences/formats were not certified; this is a masking-review control, not a full-document clearance. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `9c8ae5f6ab4bbca8` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: Paleorama workshop registration. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `4ee1673075b7d128` | partial | fix | — | Confirmed false-positive phone masking by DA-F1-005. The phone regex crosses two adjacent IP test values in shell code, consuming the end of one and start of the next. Preserve the original code, including intentionally invalid IP test cases. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `ba1bae6cb7f032c5` | partial | fix | — | Incorrect masking under the F1 keep-organisations/public-contact policy: historical phone-number/phoneword explanation (PE6-5000), with no private person identified. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `05d5bae84c860471` | partial | fix | — | Confirmed false-positive phone masking by DA-F1-005. The last three octets of illustrative IP address 162.69.239.132 are replaced, leaving 162.`<PHONE>`. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `6f6a0e1ab208e97b` | partial | fix | — | Confirmed false-positive phone masking by DA-F1-005. The last three octets of a server IP address are incorrectly replaced, leaving 139.`<PHONE>`. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `8988e16c5bbdda86` | partial | clean | — | Personal-record listing: the observed phone mask appears appropriate. Whole-document drop eligibility and other privacy fields were not reviewed; this is not a full-document clearance. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `cef3cbc54c914320` | partial | fix | — | REVIEW REQUIRED; not a confirmed false positive. Event-registration number with no organisation identified. Retain existing masking until resolved. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `d6e8da1b50823198` | partial | fix | — | REVIEW REQUIRED; not a confirmed false positive. Advertised professional contact for a named practitioner; sole-practitioner policy needs confirmation. Retain existing masking until resolved. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |
| 2 | `af4fc4c9232d331d` | partial | fix | — | Confirmed false-positive phone masking by DA-F1-005. A subnet mask is partially replaced, leaving `<PHONE>`.0. Partial phone-context review of raw versus supplied post; 1 observed mask(s). |

## Validation results

The executable medicine blocks above were tested, not a separate handwritten implementation. Code-block SHA-256: `2a05cd19d01d2fa2583236eec70aee8836be3a6f9c3803525bbe1fcd8c231391`. The approved cleaner remains unchanged. Candidate code was inserted at each replaced rule's original position for this pack only; other packs retain the baseline pipeline. The complete supplied training parts were processed without a document cap. No candidate Parquets or token shards were published.

### Complete training parts

| Part | Raw docs | Baseline drops | Candidate drops | Recovered | Newly dropped | Different final outputs | Baseline/post mismatches | Candidate second-pass changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 147,754 | 160 | 122 | 38 | 0 | 416 | 4 | 3 |
| 2 | 49,365 | 58 | 39 | 19 | 0 | 157 | 2 | 1 |
| **Total** | **197,119** | **218** | **161** | **57** | **0** | **573** | **6** | **4** |

Recovery means a document dropped by the baseline is kept by the candidate pipeline. These are intended candidates for review, not proof every recovered fragment is desirable. A second-pass change is reported rather than silently excluded from the idempotence result. All four second-pass failures also occur with the unchanged baseline: DA-B4-010 and DA-B1-002 remove more text, while DA-C3-001 drops two documents. No newly observed candidate second-pass failures were found. The six training baseline/post mismatches are the unchanged DA-D7-001 confirmed-error drop-list IDs retained in the supplied post data; reconcile its version/provenance.

### Per-medicine comparisons

Each candidate was compared to its original rule at the exact text that rule saw in the baseline pipeline. These counts overlap and are not additive. Earlier baseline drops mean later rules are not reached in this isolated comparison; the combined run above includes the downstream processing of recovered documents.

| Replacement | Original | Training docs reached | Different rule outputs | Isolated second-pass changes | Synth gold changed/dropped |
|---|---|---:|---:|---:|---|
| `DA-C5-007` | `DA-C5-005` | 196,917 | 271 | 0 | 0 / 0 of 326, PASS |
| `DA-C1-013` | `DA-C1-006` | 196,917 | 50 | 0 | 0 / 0 of 326, PASS |
| `DA-C3-008` | `DA-C3-003` | 197,058 | 57 | 0 | 0 / 0 of 326, PASS |
| `DA-F1-008` | `DA-F1-005` | 196,901 | 34 | 0 | 0 / 0 of 326, PASS |
| `DA-C2-008` | `DA-C2-003` | 196,917 | 76 | 0 | 0 / 0 of 326, PASS |
| `DA-B4-015` | `DA-B4-007` | 196,916 | 86 | 0 | 0 / 0 of 326, PASS |

The official harm loader reports **0 changed / 0 dropped** on all **326 synth gold documents**, for every candidate individually and all six in card order. A separate check through the entire approved pipeline with replacements at their real positions also gives **0 changed / 0 dropped**, identical to the baseline on all 326 gold docs. This demonstrates gold preservation, not complete precision/recall on all nongold text. Gold is unchanged.

### Audited examples and remaining judgments

| doc_id | Candidate outcome | Validation interpretation |
|---|---|---|
| `5b4d0f975990e808` | drop → keep; final output differs | Previously dropped history article is retained. |
| `f8027db45cfbebb2` | fix → fix; final output differs | Factual steel-cost assertion survives; source-meta sentence is removed. |
| `1fcdf6476cbe3c6d` | fix → fix; final output unchanged | Unanswerable appended Q&A remains removed as the negative control. |
| `24e830d71e5bc566` | fix → keep; final output differs | Technical binary illustration is preserved. |
| `780f33d558dd9f87` | fix → fix; final output unchanged | No automatic quality drop was introduced; reviewer recommendation remains a policy decision. |
| `7523ebd6d772fada` | fix → keep; final output differs | Repeated values belonging to separate spell records are preserved. |
| `93bf1542853fd276` | fix → keep; final output differs | Ambiguous mixed-inference sentence is retained unchanged; requires the stated policy review. |

All **76 unique audit IDs** were found. Candidate output differs on **33/76**: 12 of the original full reads and 21 new partial reads. None of the 37 `clean` labels changes. The five partial ownership-review cases and two partial private-contact controls retain the observed masking. The submitted labels describe the supplied post data, not the proposed output.

For the 29-document Part_2 phone review specifically, the revised candidate restores **all 25 assessed incorrect masks in 22 documents** and retains **all eight ambiguous/private masks in seven documents**. The existing binary full read is included in those 22; it is not counted again as a partial row.

### Held-out reference discrepancy

The available held-out raw and supplied post Parquets were also tested. Both are byte-identical and contain **1,975 documents**. Nevertheless, the approved cleaner on the raw file changes 66 and drops 5, so **71 baseline/post replay comparisons differ**. Treat the supplied held-out post as raw-equivalent, not as verified cleaned output. This discrepancy is independent of the proposed medicines.

On this held-out part the candidate pipeline changes 58, drops 3, recovers 2 relative to baseline and newly drops 0. Ten final outcomes differ from baseline. There are 0 combined or isolated second-pass changes. Do not mix this reference discrepancy with candidate gold harm or the training-part replay result.

### Runtime and input identities

Intel Core Ultra 5 226V, WSL2 Linux, four visible CPUs, one process per run; Python 3.12.12, PyArrow 25.0.0, pytest 9.2.0 and ftfy 6.3.1. The expanded-F1 rerun processes the two training parts in separate, overlapping processes, comparing baseline, combined pipeline, second pass and isolated revised F1. The other five unchanged cards retain their earlier isolated-rule evidence. The timings below measure only the first baseline/candidate pipeline call per document, excluding I/O, hashing and the other validations. They are one full-data pass, not a best-of-three microbenchmark; small differences should not be overinterpreted.

| Part | Baseline seconds | Candidate seconds | Baseline docs/s | Candidate docs/s |
|---|---:|---:|---:|---:|
| 1 | 61.83 | 67.89 | 2,390 | 2,176 |
| 2 | 16.46 | 17.86 | 2,998 | 2,765 |
| **Total** | **78.29** | **85.75** | **2,518** | **2,299** |

The measured candidate throughput is **0.91× baseline**. Validation scan wall times: Part_1 **270.6 s**, Part_2 **71.5 s**; these overlap and are not additive. Process contention and a single pass limit performance conclusions.

| Input | Raw SHA-256 | Post SHA-256 |
|---|---|---|
| 1 | `a601748f22e32bfb839bdb76e8e645f3beae510460857eb206fbdf3ce53a2427` | `8d0d8aa581f0d56bd3984e9d010c0f1dbe37ff9e4aea799155eb7d45e4722648` |
| 2 | `03a9efeb92e210dcb5b32d350cdee60fa032d21ea9454cefc913b0785f55ae9f` | `30fedabc6fc8befc6d2395e6a5439f74b0c31fed3bdfc0e1ae2ee05ab8f61971` |
| heldout | `3fcbb50614d90f680168e0c95479e364e995a6ebbbdfc56df90746b58c38aff8` | `3fcbb50614d90f680168e0c95479e364e995a6ebbbdfc56df90746b58c38aff8` |

### Validation record and gold-check reproduction

The tables above retain the results of the local full-data validation. The temporary full-data harness, focused tests and machine-generated logs are omitted from this minimal submission. **53 focused tests passed on the initial binary-only revision**. For this expanded F1 revision, **six focused check groups passed**, covering all 29 reviewed documents, seven technical examples, seven phone/privacy-counterexamples, other-pack no-op scope, unchanged email/address behavior and all 326 gold documents through the real pipeline. The official harm loader also passes all six current cards individually and together with zero changed/dropped gold documents. Earlier test counts are not presented as a rerun of the removed test suite. Repeating the full-data experiment requires recreating that harness and using the pinned cleaner and input hashes above.

The repository's existing harm-test loader can rerun the executable cards directly from this report:

```bash
python -I teams/01-data-a/harmtest/harm_test.py --pack <diagnosis-packs>/en-nemotroncc-synth.parquet --gold teams/01-data-a/gold/en-nemotroncc-synth_gold.csv --rules teams/01-data-a/audit/day_D6_en-nemotroncc-synth.md --rule all
```

The executable-block hash above identifies the six tested candidates. The approved cleaner and original input data were not modified.

### Adoption gate

These are validated proposal artifacts, not an enabled cleaner release. Before adoption, review nongold changes and recovered documents, any idempotence failures and runtime regression, the F1 privacy trade-off, and the five unresolved public/private ownership cases and ambiguous M6/M7 policy cases. Confirm rule IDs and interactions with fork #75, then approve the spec replacement, rebuild/pin the cleaner and rerun affected pipeline/shard checks. The full run does not substitute for a second human reader or production/corpus-v1 verification.
