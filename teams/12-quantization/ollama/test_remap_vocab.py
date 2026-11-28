"""Tests for remap_vocab.py on the frozen p1536 tokenizer.

    python -m pytest teams/12-quantization/ollama/test_remap_vocab.py -q

tokenizer.json is not in the repository (teams/03-tokenizer/release/README.md says how to get it); set
P1536_TOKENIZER to its path, or put it at teams/03-tokenizer/release/tokenizer.json. Without it the tests skip.
"""
import copy, json, os, sys, unicodedata
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from remap_vocab import SCRIPTS, remap  # noqa: E402

TOK = Path(os.environ.get("P1536_TOKENIZER", HERE.parents[1] / "03-tokenizer/release/tokenizer.json"))
pytestmark = pytest.mark.skipif(not TOK.exists(), reason=f"p1536 tokenizer.json not found at {TOK}")

PUNJABI = ["ਪੰਜਾਬ ਦੀ ਭਾਸ਼ਾ ਬਾਰੇ ਜ਼ਰੂਰੀ ਗੱਲ",
           "ਖ਼ੁਸ਼ੀ ਅਤੇ ਫ਼ਿਲਮ ਦਾ ਸ਼ਹਿਰ"]
BENGALI = "আমার নাম জয় এবং আমি বাংলায়"


@pytest.fixture(scope="module")
def original():
    return json.loads(TOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def remapped(original):
    return remap(copy.deepcopy(original), ("gurmukhi",))


def nfc_only(tok_json):
    from tokenizers import Tokenizer
    t = copy.deepcopy(tok_json)
    t["normalizer"] = {"type": "NFC"}       # what a GGUF carries
    return Tokenizer.from_str(json.dumps(t))


def test_ids_never_change(original, remapped):
    old, new = original["model"]["vocab"], remapped[0]["model"]["vocab"]
    assert sorted(old.values()) == sorted(new.values())
    assert len(set(new)) == len(new)                               # no two ids share a string


def test_only_gurmukhi_tokens_change(original, remapped):
    from remap_vocab import byte_map
    b2u = byte_map()
    enc = lambda s: "".join(b2u[b] for b in s.encode())
    comp = [enc(c) for c in SCRIPTS["gurmukhi"]]
    split = [enc(s) for s in SCRIPTS["gurmukhi"].values()]
    inv = {i: s for s, i in remapped[0]["model"]["vocab"].items()}
    for s, i in original["model"]["vocab"].items():
        if inv[i] != s:
            assert any(c in s for c in comp + split), s           # renamed or retired: only Gurmukhi tokens


def test_report(remapped):
    report = remapped[1]
    assert report["unbuildable_ids"] == [] and report["pairs"] == len(SCRIPTS["gurmukhi"])


@pytest.mark.parametrize("text", PUNJABI)
@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_punjabi_gets_training_ids(original, remapped, text, form):
    from tokenizers import Tokenizer
    train = Tokenizer.from_str(json.dumps(original))              # training: the full normalizer
    typed = unicodedata.normalize(form, text)
    assert nfc_only(remapped[0]).encode(typed, add_special_tokens=False).ids == \
        train.encode(typed, add_special_tokens=False).ids


def test_bengali_untouched_by_default(original, remapped):
    assert nfc_only(remapped[0]).encode(BENGALI, add_special_tokens=False).ids == \
        nfc_only(original).encode(BENGALI, add_special_tokens=False).ids
