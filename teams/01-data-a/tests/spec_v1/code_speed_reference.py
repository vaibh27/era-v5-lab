"""Frozen speed references from merged PR34, commit 9299ff2, clean.py 11da63a27cca.

Keep these old rule bodies unchanged while testing behavior-preserving optimizations.
"""
import re

_CODE_PACKS = frozenset("code-cpp code-go code-java code-js-ts code-python code-rust".split())

def e3_reference():
    # DA-E3-004, specv1.md, medicine 24 (code)
    RULE_ID = "DA-E3-004"
    SOURCE_RULE_ID = "DA-E3-004"
    CODE = "E3"
    ACTION = "fix"
    STAGE = "40"
    STATUS = "CANDIDATE"
    LANGS = {'cpp', 'go', 'java', 'javascript', 'python', 'rust', 'typescript'}
    ORIGIN = "code-go, code-rust, code-js-ts, code-java, code-cpp, code-python"  # packs it runs on
    PACKS = _CODE_PACKS  # the math packs share the "en" tag, so the pack is checked too
    SOURCE = "new rule, written for this sheet"  # pack whose medicine this came from
    import math
    import re

    PLACEHOLDER = "EXAMPLE_SECRET"

    KNOWN = [re.compile(p) for p in (
        r"\bAIza[0-9A-Za-z_\-]{35}\b",                                   # Google API key
        r"\bAKIA[0-9A-Z]{16}\b",                                         # AWS access key
        r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",                               # GitHub token
        r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",                             # GitHub fine-grained token
        r"\bsk-[A-Za-z0-9_\-]{20,}\b",                                   # sk- API key
        r"\b[rs]k_live_[0-9A-Za-z]{16,}\b",                              # Stripe live key
        r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b",             # SendGrid key
        r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b",                            # Slack token
        r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",  # JWT
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    )]

    ASSIGNED = re.compile(
        r"""(?ix)
        (?P<head>
          \b[\w\-]*(?:secret|token|api[_\-]?key|app[_\-]?key|auth[_\-]?key|access[_\-]?key|client[_\-]?key|private[_\-]?key|passw(?:or)?d|pwd|\bkey)[\w\-]*
          ["']?
          (?:\s*:\s*&?(?:'\w+\s+)?[\w<>\[\]]{1,20})?   # Rust and TypeScript types: SECRET: &'static str = "..."
          \s*(?::=|=>|=|:)\s*
          (?P<quote>["'`])
        )
        (?P<value>[^"'`\s\\]{16,200})
        (?P=quote)
        """)

    SDK_KEY = re.compile(r"""(?P<head>\bnew\s+Pusher\(\s*(?P<quote>["'`]))[0-9a-fA-F]{16,}(?P=quote)""")

    DUMMY = re.compile(r"(?i)example|sample|dummy|test|fake|mock|placeholder|changeme|change_me|your|xxxx|0000|1234|abcd|secret|password|token|<|>|\$\{|\{\{|%s")

    def _entropy(v):
        n = len(v)
        return -sum(c / n * math.log2(c / n) for c in (v.count(ch) for ch in set(v)))

    def _looks_random(v):
        """A real key is a run of mixed letters and digits, not a word, an identifier, a path, a version or a dummy."""
        if DUMMY.search(v) or "." in v or v.count("/") > 2:
            return False
        if not (re.search(r"[0-9]", v) and re.search(r"[A-Za-z]", v)):  # enum names and plain words have no digits
            return False
        if sum(ch.isdigit() for ch in v) < 0.15 * len(v):                # user_profile_settings_v2
            return False
        return _entropy(v) >= 3.0

    def medicine_e3(text):
        new = text
        for rx in KNOWN:
            new = rx.sub(PLACEHOLDER, new)
        new = SDK_KEY.sub(lambda m: m.group("head") + PLACEHOLDER + m.group("quote"), new)
        new = ASSIGNED.sub(lambda m: m.group("head") + PLACEHOLDER + m.group("quote") if _looks_random(m.group("value")) else m.group(0), new)
        return new


    def _std(out, text):
        # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep
        if out is None or out is True: return ("drop", text)
        if out is False: return ("keep", text)
        if isinstance(out, str): return ("fix" if out != text else "keep", out)
        action, new = out
        if action == "drop": return ("drop", text)
        return ("fix" if new != text else "keep", new)


    def medicine(doc):
        if doc.get("lang_label") not in LANGS and doc.get("pack_id") not in PACKS:
            return ("keep", doc["text"])
        return _std(medicine_e3(doc["text"]), doc["text"])

    _spec_source_medicine = medicine

    SPEC_PACKS = _CODE_PACKS

    def medicine(doc):
        if doc.get("pack_id") not in SPEC_PACKS:
            return ("keep", doc["text"])
        out = _spec_source_medicine(doc)
        return out
    return medicine

_F1_EMAIL = re.compile(r"(?<![\w.+%-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})(?![\w-])")

_F1_ROLE = re.compile(r"(?i)^(info|contact|contacts|support|help|helpdesk|admin|administrator|sales|office|press|media|hello|enquiries|inquiries|"
                      r"webmaster|postmaster|noreply|no-reply|donotreply|service|services|customerservice|billing|careers|jobs|hr|legal|security|abuse|team|git|root)$")

_F1_SAFE_DOMAIN = re.compile(r"(?i)(^|\.)(example\.(com|org|net)|localhost|test|invalid|users\.noreply\.github\.com)$|(^|\.)(gov|mil)(\.[a-z]{2})?$")

_F1_FILE_END = re.compile(r"(?i)\.(js|ts|py|go|rs|java|c|h|cpp|hpp|css|html|md|json|png|jpg|svg|gif)$")

_F1_PHONE = re.compile(r"(?<![\w.+/#=-])(?:(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,5}\)?[ .-]){1,3}\d{3,5}|\+?\d{10,13})(?![\w/-]|\.\d)")

_F1_PHONE_WORD = re.compile(r"(?i)\b(tel|telephone|phone|mobile|cell|fax|whatsapp|call me)\b[^\n]{0,30}$")

_F1_ADDRESS = re.compile(r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][A-Za-z]*\.?\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Terrace|Crescent|Nagar|Marg|Colony)\b\.?")

_F1_ORG = re.compile(r"(?i)\b(inc|ltd|llc|corp|corporation|company|university|college|school|hospital|clinic|department|dept|ministry|council|office|"
                     r"embassy|police|bank|library|museum|hotel|restaurant|customer service|support|helpline|hotline|toll[- ]free|headquarters|hq|reception|sales)\b")

def _f1_is_phone(m, line):
    digits = sum(c.isdigit() for c in m.group())
    if not 10 <= digits <= 13:
        return False
    if m.group().startswith("+") or _F1_PHONE_WORD.search(line[:m.start()]):
        return True
    return bool(re.fullmatch(r"\(?\d{3}\)?[ -]\d{3}-\d{4}", m.group()))

def f1_hits_reference(line, previous):
    """(start, end, tag) of a private person's details on one line. An organisation on this line or the one above keeps its phone and address."""
    out = [(m.start(), m.end(), "<EMAIL>") for m in _F1_EMAIL.finditer(line)
           if not (_F1_ROLE.match(m.group(1)) or _F1_SAFE_DOMAIN.search(m.group(2)) or _F1_FILE_END.search(m.group(2)))]
    if not (_F1_ORG.search(line) or _F1_ORG.search(previous)):
        for rx, tag in ((_F1_PHONE, "<PHONE>"), (_F1_ADDRESS, "<ADDRESS>")):
            for m in rx.finditer(line):
                if (tag != "<PHONE>" or _f1_is_phone(m, line)) and not any(s < m.end() and m.start() < e for s, e, _ in out):
                    out.append((m.start(), m.end(), tag))
    return sorted(out)
