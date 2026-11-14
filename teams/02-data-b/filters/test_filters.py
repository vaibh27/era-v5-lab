"""Tests for the filters in medicine.md (standard library only, made-up text, no pack data).

Run:  python test_filters.py        (from this folder)
The code is read out of medicine.md itself, so the tests run the code the harm tool runs.
"""
import os
import re
import sys
import tempfile
import textwrap
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import harm_test_indic as ht  # noqa: E402
import run_harm  # noqa: E402

MEDICINE = HERE / "medicine.md"
ZWJ, ZWNJ = chr(0x200D), chr(0x200C)


def load_namespaces():
    """{disease code: names} - the python blocks of each section run on their own, as harm_test_indic.py runs them."""
    src = MEDICINE.read_text(encoding="utf-8")
    marks = list(ht.SECTION.finditer(src))
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(src)
        ns = {"re": re, "__name__": "medicine"}
        for fm in ht.FENCE.finditer(src, m.start(), end):
            exec(compile(textwrap.dedent(fm.group(2)), f"medicine.md {m.group(1)}", "exec"), ns)
        out[m.group(1)] = ns
    return out


NS_BY_CODE = load_namespaces()
NS = NS_BY_CODE["D1"]
d1 = NS["medicine_d1"]
decide = NS["_d1_decision"]
counts = NS["_script_counts"]

# One clean sentence in each script (made up). Joiners are written as escapes, never as raw characters.
SENTENCES = {
    "Devanagari": "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं। ",
    "Bengali": "বাংলা ভাষা অত্যন্ত সুন্দর এবং সমৃদ্ধ একটি ভাষা। ",
    "Gujarati": "ગુજરાતી ભાષા ખૂબ જ સુંદર અને સમૃદ્ધ ભાષા છે. ",
    "Odia": "ଓଡ଼ିଆ ଭାଷା ଅତ୍ୟନ୍ତ ସୁନ୍ଦର ଏବଂ ସମୃଦ୍ଧ ଭାଷା ଅଟେ। ",
    "Tamil": "தமிழ் மொழி மிகவும் பழமையான இனிய மொழி ஆகும். ",
    "Telugu": "తెలుగు భాష చాలా అందమైనది మరియు ప్రాచీనమైనది. ",
    "Kannada": "ಕನ್ನಡ ಭಾಷೆ ತುಂಬಾ ಸುಂದರ ಮತ್ತು ಪ್ರಾಚೀನ ಭಾಷೆಯಾಗಿದೆ. ",
    "Malayalam": "മലയാളം ഭാഷ വളരെ മനോഹരവും പുരാതനവുമായ ഭാഷയാണ്. ",
}
ENGLISH = "The quick brown fox jumps over the lazy dog while the river flows quietly. "

# G1: (broken, fixed) pieces for each script the filter has an owner's code for. All written as code points.
PIECES = {
    "Malayalam": [  # #116: a space before a vowel sign; a doubled virama; the old two-letter long vowel
        ("\u0D1C\u0D28 \u0D3F", "\u0D1C\u0D28\u0D3F"),
        ("\u0D38\u0D30\u0D4D\u0D4D\u0D15\u0D4D\u0D15\u0D3E\u0D30\u0D4D", "\u0D38\u0D30\u0D4D\u0D15\u0D4D\u0D15\u0D3E\u0D30\u0D4D"),
        ("\u0D12\u0D3E", "\u0D13"),
    ],
    "Kannada": [("\u0C95 \u0CBE", "\u0C95\u0CBE")],  # #47
    "Devanagari": [("\u0915 \u093E", "\u0915\u093E")],  # #50
    "Tamil": [  # #169: legacy ர turned into ா; a sign typed before its consonant
        ("\u0BA4\u0BAE\u0BBF\u0BB4\u0BBE\u0BCD", "\u0BA4\u0BAE\u0BBF\u0BB4\u0BB0\u0BCD"),
        ("\u0B95 \u0BC6\u0B95", "\u0B95 \u0B95\u0BC6"),
    ],
    "Bengali": [("\u0995 \u09BE", "\u0995\u09BE"), ("\u0995 \u09CD", "\u0995\u09CD")],  # Data B fallback: space before a sign or a virama
    "Gurmukhi": [("\u0A15 \u0A3E", "\u0A15\u0A3E")],  # Data B fallback
    "Gujarati": [("\u0A95 \u0ABE", "\u0A95\u0ABE")],  # Data B fallback
    "Telugu": [("\u0C15 \u0C3E", "\u0C15\u0C3E"), ("\u0C15\n\u0C4D", "\u0C15\u0C4D")],  # Data B fallback; a line break too
    "Odia": [  # #86: a lone AA used as a danda; the sign E typed before its letter, also with a nukta
        ("\u0B15 \u0B3E ", "\u0B15 \u0964 "),
        ("\u0B15 \u0B47\u0B38", "\u0B15 \u0B38\u0B47"),
        ("\u0B15 \u0B47\u0B21\u0B3C", "\u0B15 \u0B21\u0B3C\u0B47"),
    ],
}
# Visarga used as a colon, after a space: clean in every script (TSAI, #240); the filter must leave it as written.
VISARGA = {"Malayalam": "\u0D15 \u0D03", "Kannada": "\u0C95 \u0C83", "Devanagari": "\u0930\u093E\u092E \u0903",
           "Tamil": "\u0B95 \u0B83", "Odia": "\u0B15 \u0B03", "Bengali": "\u0995 \u0983", "Gujarati": "\u0A95 \u0A83",
           "Telugu": "\u0C15 \u0C03"}
# Words with ZWJ or ZWNJ and no defect, for each script.
JOINER_WORDS = {
    "Malayalam": ["\u0D28\u0D4D" + ZWJ, "\u0D15\u0D4D" + ZWNJ + "\u0D15", "\u0D7B"],
    "Kannada": ["\u0C95\u0CB0\u0CCD" + ZWNJ + "\u0CA8\u0CBE", "\u0C95\u0CCD" + ZWJ + "\u0CB7"],
    "Devanagari": ["\u0915\u094D" + ZWJ + "\u092F\u093E", "\u0915\u094D" + ZWNJ + "\u0937"],
    "Tamil": ["\u0B95\u0BCD" + ZWJ + "\u0BB7", "\u0B95\u0BCD" + ZWNJ + "\u0BB7"],
    "Odia": ["\u0B15\u0B4D" + ZWNJ + "\u0B37", "\u0B15\u0B4D" + ZWJ + "\u0B37"],
    "Bengali": ["\u09B0\u09CD" + ZWNJ + "\u0995", "\u0995\u09CD" + ZWJ + "\u09B7"],
    "Gurmukhi": ["\u0A15\u0A4D" + ZWNJ + "\u0A37", "\u0A15\u0A4D" + ZWJ + "\u0A37"],
    "Gujarati": ["\u0A95\u0ACD" + ZWNJ + "\u0AB7", "\u0A95\u0ACD" + ZWJ + "\u0AB7"],
    "Telugu": ["\u0C15\u0C4D" + ZWNJ + "\u0C37", "\u0C15\u0C4D" + ZWJ + "\u0C37"],
}
# A document of each script with every defect above and the joiner words: the filter must fix it and keep every joiner.
DEFECTS = {s: " ".join([b for b, _ in PIECES[s]] + JOINER_WORDS[s]) + " " for s in PIECES}
# A1: the invisible junk characters, and a Telugu document holding each of them next to ZWJ and ZWNJ.
A1_CHARS = "\u200b\ufeff\u00ad\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
A1_JUNK = ("\u0C24\u0C46\u0C32\u0C41" + "".join(c + "\u0C17" for c in A1_CHARS) + " \u0C15\u0C4D" + ZWNJ + "\u0C37 \u0C15\u0C4D"
           + ZWJ + "\u0C37 ")


def doc(script, n=3):
    return SENTENCES[script] * n


def mix(indic_text, latin_share):
    """indic_text plus Latin letters, so that Latin is about latin_share of all letters."""
    h = sum(counts(indic_text).values())
    return indic_text + " " + "x" * round(h * latin_share / (1 - latin_share))


def with_script(name):
    return mock.patch.dict(os.environ, {"D1_SCRIPT": name})


def without_script():
    env = {k: v for k, v in os.environ.items() if k != "D1_SCRIPT"}
    return mock.patch.dict(os.environ, env, clear=True)


class LoadsTest(unittest.TestCase):
    def test_tool_reads_the_rules_in_run_order(self):
        rules, uses_doc = ht.load_rules(MEDICINE)
        self.assertEqual([c for c, _ in rules], ["D1", "A1", "G1"])
        self.assertEqual([f.__name__ for _, f in rules], ["medicine_d1", "medicine_a1", "medicine_g1"])
        self.assertEqual(uses_doc, {})


class CleanTextTest(unittest.TestCase):
    def test_clean_text_in_own_script_is_kept_for_every_script(self):
        for script in SENTENCES:
            with self.subTest(script=script), with_script(script):
                self.assertFalse(d1(doc(script)))
                self.assertEqual(decide(doc(script), script)[0], "keep")

    def test_a_few_english_words_never_drop_a_good_page(self):
        for script in SENTENCES:
            for share in (0.10, 0.20, 0.25):
                with self.subTest(script=script, latin=share), with_script(script):
                    self.assertFalse(d1(mix(doc(script), share)))

    def test_mixed_page_between_30_and_50_percent_is_flagged_and_kept(self):
        with with_script("Telugu"):
            text = mix(doc("Telugu"), 0.60)  # 40% Telugu
            verdict, share, _ = decide(text, "Telugu")
            self.assertEqual(verdict, "flag")
            self.assertTrue(0.30 < share < 0.50)
            self.assertFalse(d1(text))

    def test_numbers_punctuation_dandas_and_joiners_are_not_counted(self):
        noise = "१२३ 456 ।। ॥ ... " + ZWJ + ZWNJ + " -- !!"
        self.assertEqual(sum(counts(noise).values()), 0)
        with with_script("Telugu"):
            self.assertFalse(d1(noise * 10))  # nothing to judge: kept

    def test_short_documents_are_kept(self):
        with with_script("Telugu"):
            self.assertFalse(d1("Hello world"))  # 10 letters, under 20
            self.assertFalse(d1(""))


class WrongScriptTest(unittest.TestCase):
    def test_another_indic_script_is_dropped(self):
        with with_script("Telugu"):
            self.assertTrue(d1(doc("Devanagari")))
            self.assertTrue(d1(doc("Tamil")))
        with with_script("Kannada"):
            self.assertTrue(d1(doc("Malayalam")))

    def test_english_page_is_dropped(self):
        with with_script("Bengali"):
            self.assertTrue(d1(ENGLISH * 3))

    def test_mostly_other_script_with_some_pack_script_is_dropped(self):
        with with_script("Telugu"):
            self.assertTrue(d1(doc("Devanagari") + SENTENCES["Telugu"]))  # about 20% Telugu

    def test_hindi_in_a_marathi_pack_is_kept(self):
        """Hindi, Marathi and Sanskrit share Devanagari: script share cannot see this, and it must not drop it."""
        with with_script("Devanagari"):
            self.assertFalse(d1(doc("Devanagari")))

    def test_exactly_at_the_limits(self):
        with with_script("Telugu"):
            text = mix(doc("Telugu"), 0.5)  # Telugu 50%: keep
            self.assertEqual(decide(text, "Telugu")[0], "keep")


class NoPackScriptTest(unittest.TestCase):
    """Without D1_SCRIPT the rule asks only for text that is mostly in some Indic script."""

    def test_indic_text_is_kept_in_any_script(self):
        with without_script():
            for script in SENTENCES:
                self.assertFalse(d1(doc(script)), script)

    def test_english_is_dropped(self):
        with without_script():
            self.assertTrue(d1(ENGLISH * 3))

    def test_cjk_is_dropped(self):
        with without_script():
            self.assertTrue(d1("这是一个很长的中文句子用来测试脚本比例的过滤器是否有效" * 2))


class SettingTest(unittest.TestCase):
    def test_unknown_script_is_an_error_not_a_silent_keep(self):
        with with_script("Klingon"), self.assertRaises(ValueError):
            d1(doc("Telugu"))

    def test_script_name_is_not_case_sensitive_and_oriya_is_odia(self):
        with with_script("telugu"):
            self.assertFalse(d1(doc("Telugu")))
        with with_script("Oriya"):
            self.assertFalse(d1(doc("Odia")))
            self.assertTrue(d1(doc("Tamil")))

    def test_pack_scripts(self):
        self.assertEqual(run_harm.script_of_pack("ind-kn-fw2"), "Kannada")
        self.assertEqual(run_harm.script_of_pack("ind-mr-sangraha-v"), "Devanagari")
        self.assertEqual(run_harm.script_of_pack("ind-or-fw2"), "Odia")
        with self.assertRaises(ValueError):
            run_harm.script_of_pack("cadj-commits")


class JoinerAndFormTest(unittest.TestCase):
    """ZWJ and ZWNJ before and after the filter: the rule returns a bool and never new text."""
    SAMPLES = {
        "Malayalam": "ജനങ്ങള്" + ZWJ + " കേരളത്തിലെ ഭാഷ വളരെ മനോഹരമാണ് " + "ന്" + ZWJ + " ",
        "Kannada": "ಕರ್" + ZWNJ + "ನಾಟಕ ರಾಜ್ಯದ ಭಾಷೆ ಕನ್ನಡ ಆಗಿದೆ " + "ಕ್" + ZWJ + "ಷ ",
        "Telugu": "ప్రా" + ZWNJ + "చీన తెలుగు భాష చాలా అందమైనది మరియు ప్రాచీనమైనది ",
        "Devanagari": "क्" + ZWJ + "या भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं " + "क्" + ZWNJ + "ष ",
    }

    def test_counts_before_and_after_are_the_same(self):
        for script, text in self.SAMPLES.items():
            text = text * 2
            with self.subTest(script=script), with_script(script):
                before = (text.count(ZWJ), text.count(ZWNJ))
                out = d1(text)
                self.assertIsInstance(out, bool)  # a bool, so no text can have been changed
                self.assertFalse(out)  # clean text: kept, so it leaves the filter as it came in
                after = (text.count(ZWJ), text.count(ZWNJ))
                self.assertEqual(before, after)
                self.assertGreater(sum(before), 0)  # the sample does hold joiners

    def test_dropped_documents_are_not_counted_as_joiner_changes(self):
        with with_script("Telugu"):
            text = self.SAMPLES["Malayalam"] * 3  # wrong script: dropped whole
            self.assertTrue(d1(text))

    def test_nfc_form_is_never_changed_and_nfkc_is_never_used(self):
        decomposed = unicodedata.normalize("NFD", "কো বৌ ") + "ক্" + ZWNJ + "ষ "  # ো and ৌ split in NFD
        text = (SENTENCES["Bengali"] + decomposed) * 3
        self.assertFalse(unicodedata.is_normalized("NFC", text))  # the input is not NFC; it must still be judged
        with with_script("Bengali"):
            self.assertFalse(d1(text))
        code = "\n".join(m.group(2) for m in ht.FENCE.finditer(MEDICINE.read_text(encoding="utf-8")))
        self.assertEqual(set(re.findall(r"normalize\(\s*[\"'](\w+)[\"']", code)), {"NFC"})  # NFC only, never NFKC or NFKD

    def test_nonbreaking_space_and_compat_characters_are_untouched(self):
        text = SENTENCES["Telugu"] * 3 + chr(0x00A0) + "Ａｂｃ" + chr(0x2026)
        with with_script("Telugu"):
            self.assertIsInstance(d1(text), bool)


def nfd_count(text, ch):
    return unicodedata.normalize("NFD", text).count(ch)


class A1Test(unittest.TestCase):
    """A1: the author's code (#114, adinathauti). It removes invisible junk and keeps ZWJ and ZWNJ."""
    a1 = staticmethod(NS_BY_CODE["A1"]["medicine_a1"])

    def test_each_junk_character_is_removed(self):
        for ch in A1_CHARS:
            with self.subTest(char=hex(ord(ch))):
                self.assertEqual(self.a1("\u0C24\u0C46" + ch + "\u0C32"), "\u0C24\u0C46\u0C32")

    def test_zwj_and_zwnj_are_kept_and_counted(self):
        out = self.a1(A1_JUNK)
        for ch in A1_CHARS:
            self.assertNotIn(ch, out)
        self.assertEqual((out.count(ZWJ), out.count(ZWNJ)), (A1_JUNK.count(ZWJ), A1_JUNK.count(ZWNJ)))
        self.assertEqual((A1_JUNK.count(ZWJ), A1_JUNK.count(ZWNJ)), (1, 1))

    def test_every_script_keeps_its_joiners(self):
        for script, words in JOINER_WORDS.items():
            text = " ".join(words) + "\u200b"
            with self.subTest(script=script):
                out = self.a1(text)
                self.assertEqual(out, " ".join(words))
                self.assertEqual((out.count(ZWJ), out.count(ZWNJ)), (text.count(ZWJ), text.count(ZWNJ)))

    def test_clean_text_is_unchanged_and_second_run_changes_nothing(self):
        for script in SENTENCES:
            self.assertEqual(self.a1(doc(script)), doc(script), script)
        self.assertEqual(self.a1(self.a1(A1_JUNK)), self.a1(A1_JUNK))

    def test_nfc_form_and_nukta_are_not_touched(self):
        text = "\u0B21\u0B3C \u0B15\u0BCA \u0B15\u0B47\u0B3E"  # a nukta, and decomposed vowel signs
        self.assertEqual(self.a1(text), text)

    def test_junk_next_to_a_joiner_goes_and_the_joiner_stays(self):
        """The case the issue is about: A1 must take U+200B out of a word that holds a ZWJ or a ZWNJ."""
        for script, words in JOINER_WORDS.items():
            for word in words:
                for junk in A1_CHARS:
                    for noisy in (junk + word, word + junk, word[:-1] + junk + word[-1]):
                        with self.subTest(script=script, junk=hex(ord(junk))):
                            self.assertEqual(self.a1(noisy), word)

    def test_only_the_characters_of_the_disease_list_are_removed(self):
        """DISEASES.md lists exactly these characters. Others that look invisible are not A1 in v1 and stay."""
        for ch in ("\u200e", "\u200f", "\u2060", "\u061c", "\u034f", "\u2028", "\u00a0", "\u0000"):
            with self.subTest(char=hex(ord(ch))):
                self.assertEqual(self.a1("\u0C24" + ch + "\u0C32"), "\u0C24" + ch + "\u0C32")
        ours = NS_BY_CODE["A1"]["_A1_RE"]
        self.assertEqual({chr(c) for c in range(0x110000) if ours.match(chr(c))}, set(A1_CHARS))

    def test_same_characters_as_both_authors_rules(self):
        """Spec v1 holds two A1 rules: DB-A1-001 (PR #114, the one used) and DB-A1-002 (PR #109). Same 12 characters,
        and the same output on random text. The author's own text is read from indic_rules/medicine_indic.md."""
        spec = HERE.parent / "indic_rules" / "medicine_indic.md"
        if not spec.exists():
            self.skipTest("indic_rules/medicine_indic.md is not here")
        text = spec.read_text(encoding="utf-8")
        i = text.index("_A1_RE = re.compile(")
        author114 = {"re": re}
        exec(text[i:text.index("def fix_a1", i)], author114)
        author114_set = {chr(c) for c in range(0x110000) if author114["_A1_RE"].match(chr(c))}
        j = text.index("_A1 = set(")
        author109 = {}
        exec(text[j:text.index("\n", j)], author109)
        self.assertEqual(author114_set, set(A1_CHARS))
        self.assertEqual(author109["_A1"], set(A1_CHARS))
        import random
        rnd = random.Random(7)
        pool = [chr(c) for c in range(0x20, 0x7F)] + [chr(c) for c in range(0x0C00, 0x0C80)] \
            + [chr(c) for c in range(0x2000, 0x2070)] + ["\ufeff", "\u00ad"]
        for _ in range(2000):
            s = "".join(rnd.choice(pool) for _ in range(rnd.randint(0, 60)))
            self.assertEqual(self.a1(s), "".join(c for c in s if c not in author109["_A1"]))
    def test_a1_runs_before_g1_so_junk_in_a_cut_word_does_not_hide_the_cut(self):
        """In the file's order A1 comes first. A zero-width space between a consonant and its sign is a G1 cut too."""
        g1 = NS_BY_CODE["G1"]["medicine_g1"]
        for script, (consonant, sign) in {"Kannada": ("\u0C95", "\u0CBE"), "Telugu": ("\u0C15", "\u0C3E"),
                                          "Malayalam": ("\u0D15", "\u0D3E"), "Devanagari": ("\u0915", "\u093E")}.items():
            text = consonant + " \u200b" + sign
            with self.subTest(script=script):
                self.assertEqual(g1(self.a1(text)), consonant + sign)
                self.assertEqual(g1(text), text)  # G1 alone cannot see through the junk: the order matters


class G1Test(unittest.TestCase):
    """G1: the owners' code for five scripts and Data B's fallback for four, chosen by the script of the text."""
    g1 = staticmethod(NS_BY_CODE["G1"]["medicine_g1"])
    script_of = staticmethod(NS_BY_CODE["G1"]["_g1_script"])

    def test_each_owners_fix(self):
        for script, pieces in PIECES.items():
            for broken, fixed in pieces:
                with self.subTest(script=script, broken=ascii(broken)):
                    self.assertEqual(self.script_of(broken), script)
                    self.assertEqual(self.g1(broken), fixed)

    def test_second_run_changes_nothing(self):
        for script, text in DEFECTS.items():
            with self.subTest(script=script):
                self.assertEqual(self.g1(self.g1(text)), self.g1(text))

    def test_clean_text_is_unchanged(self):
        for script in SENTENCES:
            with self.subTest(script=script):
                self.assertEqual(self.g1(doc(script)), doc(script))

    def test_visarga_after_a_space_stays_as_written(self):
        for script, text in VISARGA.items():
            with self.subTest(script=script):
                self.assertEqual(self.g1(text), text)
                self.assertEqual(self.g1(doc(script) + text), doc(script) + text)

    def test_fallback_changes_only_a_space_before_a_sign(self):
        """Bengali, Gurmukhi, Gujarati, Telugu: no owner's code; the fallback does one thing and nothing more."""
        for name, (c, sign, other_sign, visarga, anusvara, letter) in {
                "Telugu": ("\u0C15", "\u0C3E", "\u0C4D", "\u0C03", "\u0C02", "\u0C16"),
                "Bengali": ("\u0995", "\u09BE", "\u09CD", "\u0983", "\u0982", "\u0996"),
                "Gujarati": ("\u0A95", "\u0ABE", "\u0ACD", "\u0A83", "\u0A82", "\u0A96"),
                "Gurmukhi": ("\u0A15", "\u0A3E", "\u0A4D", "\u0A03", "\u0A02", "\u0A16")}.items():
            with self.subTest(script=name):
                self.assertEqual(self.g1(c + " " + sign + " " + c), c + sign + " " + c)
                self.assertEqual(self.g1(c + "\t" + other_sign + letter), c + other_sign + letter)
                for same in (c + " " + visarga, c + " " + anusvara, c + " " + letter, c + sign + " " + letter,
                             sign + " " + sign, c + "  ", c + "\u200d" + sign):
                    self.assertEqual(self.g1(same), same)

    def test_the_script_with_most_letters_decides(self):
        telugu_with_a_little_devanagari = doc("Telugu") + "\u0915 \u093E"
        self.assertEqual(self.g1(telugu_with_a_little_devanagari), telugu_with_a_little_devanagari)
        hindi_with_english = doc("Devanagari") + ENGLISH + "\u0915 \u093E"
        self.assertEqual(self.script_of(hindi_with_english), "Devanagari")
        self.assertTrue(self.g1(hindi_with_english).endswith("\u0915\u093E"))
        self.assertIsNone(self.script_of(ENGLISH))
        self.assertEqual(self.g1(ENGLISH), ENGLISH)

    def test_zwj_zwnj_nfc_and_compatibility_characters_in_a_fixed_document(self):
        for script, text in DEFECTS.items():
            with self.subTest(script=script):
                new = self.g1(text)
                self.assertNotEqual(new, text)  # the defects were fixed
                self.assertEqual((new.count(ZWJ), new.count(ZWNJ)), (text.count(ZWJ), text.count(ZWNJ)))
                self.assertGreater(text.count(ZWJ) + text.count(ZWNJ), 0)
                self.assertEqual(ht.audit(text, new), [])

    def test_nukta_and_malayalam_chillu_are_not_changed(self):
        """#211 asks for the nukta and chillu check, against Team 3's normalizer (#134)."""
        for ch, script in ((chr(0x0B3C), "Odia"), (chr(0x093C), "Devanagari"), (chr(0x09BC), "Bengali")):
            text = DEFECTS.get(script, "") + ("\u0915" + ch + " " if script == "Devanagari" else "")
            with self.subTest(nukta=hex(ord(ch))):
                self.assertEqual(nfd_count(self.g1(text), ch), nfd_count(text, ch))
        ml = DEFECTS["Malayalam"] + "".join(chr(c) for c in range(0x0D7A, 0x0D80))  # the six atomic chillus
        new = self.g1(ml)
        for c in range(0x0D7A, 0x0D80):
            self.assertEqual(new.count(chr(c)), ml.count(chr(c)))
        self.assertEqual(new.count("\u0D28\u0D4D" + ZWJ), ml.count("\u0D28\u0D4D" + ZWJ))  # a ZWJ chillu stays

    def test_every_script_has_code_and_only_four_use_the_fallback(self):
        ns = NS_BY_CODE["G1"]
        scripts = ns["_SCRIPTS"]
        self.assertTrue(all(fn for _, fn in scripts.values()))
        owners = {"Kannada": "_g1_kannada", "Devanagari": "_g1_devanagari", "Malayalam": "_g1_malayalam",
                  "Tamil": "_g1_tamil", "Odia": "_g1_odia"}
        for name, fn_name in owners.items():
            self.assertIs(scripts[name][1], ns[fn_name])
        self.assertEqual(set(scripts) - set(owners), set(ns["_GENERIC"]))
        self.assertEqual(set(ns["_GENERIC"]), {"Bengali", "Gurmukhi", "Gujarati", "Telugu"})


class EveryFilterTest(unittest.TestCase):
    """The #214 rule for each filter in medicine.md: a unit test, a before-and-after ZWJ/ZWNJ count, NFC only.
    A filter added to medicine.md without joiner samples in JOINER_SAMPLES fails here until it has them."""
    # code -> [(D1_SCRIPT for the run, [documents holding ZWJ and ZWNJ])]
    JOINER_SAMPLES = {
        "D1": [(s, [JoinerAndFormTest.SAMPLES[s] * 2]) for s in ("Malayalam", "Kannada", "Telugu")],
        # A1 and G1: each document holds a defect the filter fixes, and joiners it must keep
        "A1": [(s, [DEFECTS[s]]) for s in DEFECTS] + [("Telugu", [A1_JUNK])],
        "G1": [(s, [DEFECTS[s]]) for s in DEFECTS],
    }

    def samples_of(self, code):
        return self.JOINER_SAMPLES.get(code, [])

    def test_every_filter_has_joiner_samples_and_keeps_every_joiner(self):
        rules, _ = ht.load_rules(MEDICINE)
        self.assertTrue(rules)
        for code, fn in rules:
            self.assertIsNotNone(fn, f"{code} has no function in medicine.md")
            self.assertEqual(fn.__name__, "medicine_" + code.lower())
            samples = self.samples_of(code)
            self.assertTrue(samples, f"{code}: add joiner samples to JOINER_SAMPLES")
            for script, texts in samples:
                for text in texts:
                    with self.subTest(code=code, script=script), with_script(script):
                        before = (text.count(ZWJ), text.count(ZWNJ))
                        self.assertGreater(sum(before), 0)
                        verdict, new = ht.classify(text, fn(text))
                        if verdict != "drop":  # a dropped document leaves whole; a kept or fixed one keeps every joiner
                            self.assertEqual(before, (new.count(ZWJ), new.count(ZWNJ)))
                            self.assertEqual(ht.audit(text, new), [])

    def test_only_nfc_in_the_code(self):
        code = "\n".join(m.group(2) for m in ht.FENCE.finditer(MEDICINE.read_text(encoding="utf-8")))
        forms = re.findall(r"normalize\(\s*[\"'](\w+)[\"']", code)
        self.assertTrue(set(forms) <= {"NFC"}, forms)
        self.assertNotRegex(code, r"NFKC|NFKD")

    def test_joiner_report_counts_before_and_after(self):
        kept = JoinerAndFormTest.SAMPLES["Telugu"] * 2       # 1 ZWNJ in each copy: 2
        foreign = JoinerAndFormTest.SAMPLES["Malayalam"] * 2  # wrong script for Telugu: dropped with its joiners
        docs = [("p", "a", kept), ("p", "b", foreign)]
        with with_script("Telugu"):
            lines = run_harm.joiner_report(MEDICINE, docs)
        self.assertEqual(len(lines), 3)  # one line for each filter: D1, A1, G1
        d1_line = [l for l in lines if "before and after D1:" in l][0]
        zb = kept.count(ZWJ) + foreign.count(ZWJ)
        nb = kept.count(ZWNJ) + foreign.count(ZWNJ)
        self.assertIn(f"ZWJ {zb} to {kept.count(ZWJ)}, ZWNJ {nb} to {kept.count(ZWNJ)}", d1_line)
        self.assertIn("(1 dropped documents held", d1_line)
        self.assertTrue(all("**PASS**" in l for l in lines), lines)
        a1_line = [l for l in lines if "before and after A1:" in l][0]  # A1 drops nothing and keeps all joiners
        self.assertIn(f"ZWJ {zb} to {zb}, ZWNJ {nb} to {nb} (0 dropped", a1_line)

    def test_joiner_report_fails_a_filter_that_changes_joiners(self):
        bad = "def medicine_d1(text):\n    return text.replace(chr(0x200D), '')\n"
        md = "1. **Disease:** D1 test\n```python\n" + bad + "```\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.md"
            path.write_text(md, encoding="utf-8")
            lines = run_harm.joiner_report(path, [("p", "a", "ab" + ZWJ + "cd")])
        self.assertIn("ZWJ 1 to 0", lines[0])
        self.assertIn("**FAIL**", lines[0])


class HarmToolTest(unittest.TestCase):
    """Run the rule through harm_test_indic.py's own machinery on made-up documents."""

    def run_rule(self, script, texts):
        rules, _ = ht.load_rules(MEDICINE)
        fn = dict(rules)["D1"]
        docs = [("pack", f"d{i}", t) for i, t in enumerate(texts)]
        with with_script(script):
            res = ht.run_rule("D1", [fn], docs, 5.0)
        return res, ht.verdict_of(res, len(docs), 1)

    def test_clean_gold_set_passes_and_joiners_are_ok(self):
        clean = [doc("Telugu") + (ZWNJ if i % 2 else ZWJ) for i in range(120)] + [mix(doc("Telugu"), 0.2)]
        res, verdict = self.run_rule("Telugu", clean)
        self.assertEqual((res["changed"], res["dropped"], res["joiners"], res["form"], res["errors"]), ([], [], [], [], []))
        self.assertEqual(verdict, "PASS")

    def test_one_foreign_document_in_a_hundred_is_allowed_two_are_not(self):
        clean = [doc("Kannada")] * 98
        res, verdict = self.run_rule("Kannada", clean + [doc("Tamil"), doc("Devanagari")])
        self.assertEqual(len(res["dropped"]), 2)  # 2 of 100: over max(1, 1%) is a FAIL
        self.assertEqual(verdict, "FAIL")
        res, verdict = self.run_rule("Kannada", clean + [doc("Kannada"), doc("Tamil")])
        self.assertEqual(len(res["dropped"]), 1)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(res["joiners"], [])  # a drop is not a joiner change

    def test_a_crash_is_a_failure(self):
        res, verdict = self.run_rule("Klingon", [doc("Telugu")])
        self.assertEqual(len(res["errors"]), 1)
        self.assertEqual(verdict, "FAIL")


class RunHarmTest(unittest.TestCase):
    def test_empty_gold_sets_are_written_as_not_testable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold = tmp / "gold"
            gold.mkdir()
            (gold / "gold_coverage.csv").write_text("pack_id\nind-ta-fw2\nind-ml-fw2\n", encoding="utf-8")
            (gold / "ind-ml-fw2_gold.csv").write_text("doc_id,reader_1,reader_2\n", encoding="utf-8")  # header only
            code = run_harm.main(["--gold-dir", str(gold), "--out-dir", str(tmp)])
            self.assertEqual(code, 0)
            for pack in ("ind-ta-fw2", "ind-ml-fw2"):
                text = (tmp / f"harm_{pack}.md").read_text(encoding="utf-8")
                self.assertIn("gold set empty, not testable", text)

    def test_missing_pack_file_is_exit_code_2_and_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold = tmp / "gold"
            gold.mkdir()
            (gold / "ind-kn-fw2_gold.csv").write_text("doc_id,reader_1,reader_2\nabc,a,b\n", encoding="utf-8")
            code = run_harm.main(["--gold-dir", str(gold), "--out-dir", str(tmp), "--packs-dir", str(tmp / "none"),
                                  "--packs", "ind-kn-fw2"])
            self.assertEqual(code, 2)
            self.assertFalse((tmp / "harm_ind-kn-fw2.md").exists())

    def test_jsonl_pack_is_tested_with_the_pack_script(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold = tmp / "gold"
            gold.mkdir()
            ids = [f"d{i}" for i in range(100)]
            (gold / "ind-te-fw2_gold.csv").write_text("doc_id,reader_1,reader_2\n" + "\n".join(f"{i},a,b" for i in ids),
                                                      encoding="utf-8")
            rows = [{"doc_id": i, "text": doc("Telugu")} for i in ids]
            (tmp / "ind-te-fw2.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            code = run_harm.main(["--gold-dir", str(gold), "--out-dir", str(tmp), "--packs-dir", str(tmp),
                                  "--packs", "ind-te-fw2"])
            self.assertEqual(code, 0)
            out = (tmp / "harm_ind-te-fw2.md").read_text(encoding="utf-8")
            self.assertIn("Changed 0 of 100", out)
            self.assertNotIn("D1_SCRIPT", os.environ)


if __name__ == "__main__":
    unittest.main(verbosity=2)
