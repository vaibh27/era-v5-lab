"""FLORES-200 devtest: tokens per word and bytes per token of the frozen p1536 against published tokenizers (G3, 28 Nov,
Task 5). FLORES is used to report on the frozen tokenizer, never to choose one.

    python teams/03-tokenizer/harness/flores_task5.py --flores <flores200_dataset> --p1536 release/tokenizer.json \
        --base <BrahmicTokenizer-131K>/tokenizer.json --published <dir of downloaded HF tokenizers> \
        --out teams/03-tokenizer/results/flores_task5

Languages: the 11 Indic languages of the capstone and Sanskrit, plus English, French, German, Spanish and Russian
(European reference) and Chinese, Japanese and Korean (East Asian reference); devtest split, 1,012 sentences each.

Word rule (one rule, all languages): a word is a maximal run of non-whitespace characters (Python str.split()).
Chinese and Japanese do not put spaces between words, so their tokens/word is not comparable: read bytes/token there.
tokens/word = tokens of all sentences / words of all sentences; bytes/token = UTF-8 bytes / tokens. Each sentence is
encoded alone with add_special_tokens=False (no bos, eos, [CLS] or [SEP]).
Lossless: share of sentences where decode(encode(x)) == x after the tokenizer's own normalizer (p1536: NFC + v1 fixes).

Published tokenizers are read from --published/<org>__<name>/tokenizer.json (hf_hub_download); the Hub commit and the
file sha256 of each are recorded. MuRIL ships only a WordPiece vocab.txt and is read with BertWordPieceTokenizer.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

P1536 = "e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b"
BASE = "01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff"
INDIC = [("Hi", "hin_Deva"), ("Mr", "mar_Deva"), ("Bn", "ben_Beng"), ("As", "asm_Beng"), ("Ta", "tam_Taml"),
         ("Te", "tel_Telu"), ("Kn", "kan_Knda"), ("Ml", "mal_Mlym"), ("Gu", "guj_Gujr"), ("Pa", "pan_Guru"),
         ("Or", "ory_Orya")]
OTHER = [("Sa", "san_Deva"), ("En", "eng_Latn"), ("Fr", "fra_Latn"), ("De", "deu_Latn"), ("Es", "spa_Latn"),
         ("Ru", "rus_Cyrl"), ("Zh", "zho_Hans"), ("Ja", "jpn_Jpan"), ("Ko", "kor_Hang")]
NO_SPACES = {"Zh", "Ja"}
PUBLISHED = [("Sarvam-1", "sarvamai__sarvam-1"), ("Sarvam-M", "sarvamai__sarvam-m"),
             ("Krutrim-2", "krutrim-ai-labs__Krutrim-2-instruct"), ("IndicBERTv2", "ai4bharat__IndicBERTv2-MLM-only"),
             ("MuRIL", "google__muril-base-cased"), ("Qwen2.5", "Qwen__Qwen2.5-7B")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Tiktoken:
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("o200k_base")

    def ids(self, text):
        return self.enc.encode_ordinary(text)

    def lossless(self, text, ids):
        return self.enc.decode(ids) == text

    vocab = 200019


class HF:
    def __init__(self, tok):
        self.tok = tok
        self.vocab = tok.get_vocab_size(with_added_tokens=True)

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def lossless(self, text, ids):
        norm = self.tok.normalizer.normalize_str(text) if self.tok.normalizer else text
        return self.tok.decode(ids, skip_special_tokens=False) == norm


def load(args):
    from tokenizers import Tokenizer
    from tokenizers.implementations import BertWordPieceTokenizer
    if sha(args.p1536) != P1536:
        sys.exit("--p1536 is not the frozen p1536")
    toks = {"p1536 (frozen)": (HF(Tokenizer.from_file(str(args.p1536))), {"sha256": P1536})}
    if args.base:
        if sha(args.base) != BASE:
            sys.exit("--base is not BrahmicTokenizer-131K")
        toks["BrahmicTokenizer-131K (base)"] = (HF(Tokenizer.from_file(str(args.base))), {"sha256": BASE})
    toks["o200k_base"] = (Tiktoken(), {"source": "tiktoken"})
    pub = Path(args.published) if args.published else None
    for name, folder in PUBLISHED if pub else []:
        d = pub / folder
        rev = (d / "REVISION").read_text().strip() if (d / "REVISION").exists() else None
        if (d / "tokenizer.json").exists():
            t = Tokenizer.from_file(str(d / "tokenizer.json"))
            toks[name] = (HF(t), {"hub": folder.replace("__", "/"), "revision": rev, "sha256": sha(d / "tokenizer.json")})
        elif (d / "vocab.txt").exists():
            t = BertWordPieceTokenizer(str(d / "vocab.txt"), lowercase=False, strip_accents=False)
            toks[name] = (HF(t._tokenizer), {"hub": folder.replace("__", "/"), "revision": rev, "sha256": sha(d / "vocab.txt"),
                                             "note": "WordPiece vocab.txt, BertWordPieceTokenizer(lowercase=False)"})
    return toks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--p1536", type=Path, required=True)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--published", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    toks = load(a)
    langs = INDIC + OTHER
    texts = {}
    for code, fl in langs:
        f = a.flores / "devtest" / f"{fl}.devtest"
        texts[code] = (f.read_text(encoding="utf-8").splitlines(), sha(f))
    res = {}
    for name, (t, meta) in toks.items():
        res[name] = {"meta": {**meta, "vocab": t.vocab}, "langs": {}}
        for code, _ in langs:
            lines, _ = texts[code]
            ntok = nw = nb = ok = 0
            for s in lines:
                ids = t.ids(s)
                ntok += len(ids)
                nw += len(s.split())
                nb += len(s.encode("utf-8"))
                ok += t.lossless(s, ids)
            res[name]["langs"][code] = {"tokens": ntok, "words": nw, "bytes": nb,
                                        "tokens_per_word": round(ntok / nw, 4), "bytes_per_token": round(nb / ntok, 3),
                                        "lossless": round(ok / len(lines), 4)}
        ind = [res[name]["langs"][c] for c, _ in INDIC]
        res[name]["indic_mean_tokens_per_word"] = round(sum(x["tokens_per_word"] for x in ind) / len(ind), 4)
        print(f"{name}: Indic mean {res[name]['indic_mean_tokens_per_word']}", flush=True)

    # where p1536 loses: languages where a published tokenizer needs fewer tokens
    p = res["p1536 (frozen)"]["langs"]
    losses = []
    for code, _ in langs:
        better = sorted(((r["langs"][code]["tokens"], n) for n, r in res.items()
                         if n != "p1536 (frozen)" and r["langs"][code]["lossless"] >= 0.98
                         and r["langs"][code]["tokens"] < p[code]["tokens"]))
        if better:
            losses.append({"lang": code, "p1536_tokens": p[code]["tokens"],
                           "better": [{"tokenizer": n, "tokens": k, "fewer": round(1 - k / p[code]["tokens"], 4)}
                                      for k, n in better]})
    out = {"word_rule": "maximal runs of non-whitespace (str.split()); no word segmentation for Zh, Ja",
           "flores": {c: {"file": f"devtest/{fl}.devtest", "sha256": texts[c][1], "sentences": len(texts[c][0])}
                      for c, fl in langs},
           "tokenizers": res, "p1536_loses": losses,
           "environment": {"python": platform.python_version(), "script_sha256": sha(__file__)}}
    a.out.with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out, langs, a.out.with_suffix(".md"))


def write_md(o, langs, path):
    R = o["tokenizers"]
    names = list(R)
    codes = [c for c, _ in langs]
    md = [f"# FLORES-200 devtest: p1536 `{P1536[:8]}…` vs published tokenizers (Task 5)", "",
          "Made by `harness/flores_task5.py`. FLORES is used to report on the frozen tokenizer, not to choose one.", "",
          f"- **Word rule (all languages):** {o['word_rule']}.",
          "- **Encoding:** each of the 1,012 devtest sentences per language is encoded alone, with `add_special_tokens=False`.",
          "- **Mean:** the 11 Indic languages (Hi … Or). Sanskrit and the reference languages are not in it.", "",
          "## Tokens per word (lower is better)", "",
          "| Tokenizer | Vocab | " + " | ".join(codes) + " | **Indic mean** |", "|---|---:|" + "---:|" * len(codes) + "---:|"]
    for n in names:
        md.append(f"| {n} | {R[n]['meta']['vocab']:,} | " +
                  " | ".join(f"{R[n]['langs'][c]['tokens_per_word']:.3f}" + ("*" if c in NO_SPACES else "") for c in codes) +
                  f" | **{R[n]['indic_mean_tokens_per_word']:.3f}** |")
    md += ["", "\\* Chinese and Japanese have no spaces between words, so tokens per word is not comparable there. Read bytes per token.", "",
           "## Bytes per token (higher is better)", "",
           "| Tokenizer | " + " | ".join(codes) + " |", "|---|" + "---:|" * len(codes)]
    for n in names:
        md.append(f"| {n} | " + " | ".join(f"{R[n]['langs'][c]['bytes_per_token']:.2f}" for c in codes) + " |")
    md += ["", "## Lossless (share of sentences where decode(encode(x)) equals x after the tokenizer's normalizer)", "",
           "| Tokenizer | " + " | ".join(codes) + " |", "|---|" + "---:|" * len(codes)]
    for n in names:
        md.append(f"| {n} | " + " | ".join(f"{R[n]['langs'][c]['lossless']:.3f}" for c in codes) + " |")
    md += ["", "## Where p1536 loses", "",
           "A lossless tokenizer here (at least 98% of sentences round trip) needs fewer tokens for the same text.",
           "- **WordPiece encoders are left out.** IndicBERTv2 and MuRIL lose text (0–75% of sentences round trip): "
           "they split on whitespace and punctuation first and map unknown words to [UNK]. Their counts are not comparable "
           "with a byte-level BPE, and they are encoder vocabularies, not ones for a generative model.",
           "- **Krutrim-2 and Sarvam-M give the same counts everywhere.** Both use Mistral's Tekken vocabulary (131,072 ids); "
           "the merges are identical and only 10 control-token names differ ([IMG], <pad>, [PREFIX], …).", "",
           "| Lang | p1536 tokens | Fewer tokens with |", "|---|---:|---|"]
    for l in o["p1536_loses"]:
        md.append(f"| {l['lang']} | {l['p1536_tokens']:,} | " +
                  ", ".join(f"{b['tokenizer']} {b['tokens']:,} (−{b['fewer'] * 100:.1f}%)" for b in l["better"]) + " |")
    md += ["", "## Tokenizers", "", "| Tokenizer | Source | sha256 of the file |", "|---|---|---|"]
    for n in names:
        m = R[n]["meta"]
        src = m.get("hub", m.get("source", "G3 release" if "p1536" in n else "BrahmicTokenizer-131K"))
        if m.get("revision"):
            src += f" @ `{m['revision'][:12]}`"
        md.append(f"| {n} | {src} | `{m.get('sha256', '-')[:16]}…` |")
    md += ["", "FLORES files: `flores200_dataset/devtest/<lang>.devtest`; their sha256 values are in the JSON."]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[8:20]))


if __name__ == "__main__":
    sys.exit(main())
