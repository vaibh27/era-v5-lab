"""Tests for the two proposed code-go medicines in audit/code-go_audit_smitasasindran.md (DA-A6-004 and DA-E4-011).

    python3 -m pytest teams/01-data-a/tests/audit/test_code_go_medicines.py -q      (from the repository root)

The rules are loaded from the cards in the report the way harmtest/harm_test.py loads them. All text is made up, except the
two real code-go examples that the review asked for, which are not stored here: they are checked by doc_id in the report's
validation section, against the Parquet files, and are not part of this test.
"""
import sys
from pathlib import Path

import pytest

TEAM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEAM / "harmtest"))
import harm_test  # noqa: E402  (only its loader is used)

REPORT = TEAM / "audit" / "code-go_audit_smitasasindran.md"
RULES, _ = harm_test.load_rules(REPORT)
RULES = RULES["RULES_BY_ID"]
A6, E4 = RULES["DA-A6-004"], RULES["DA-E4-011"]

CY_E = "\u0435"  # Cyrillic e, a look-alike of the Latin e
CY_C = "\u0441"  # Cyrillic es, a look-alike of the Latin c
CY_C_UP = "\u0421"  # the capital, a look-alike of the Latin C


def doc(text, url="pkg/x/file.go", pack="code-go", lang="go"):
    return {"doc_id": "0" * 16, "text": text, "url": url, "lang_label": lang, "pack_id": pack}


def test_the_report_holds_exactly_the_two_new_rules():
    assert sorted(RULES) == ["DA-A6-004", "DA-E4-011"]


# ---------------------------------------------------------------- DA-A6-004

class TestA6:
    def test_a_single_look_alike_in_a_latin_word_is_folded(self):
        text = f"func r{CY_E}trieve() {{}}\nvar {CY_C}Type string\nconst PV{CY_C_UP}Report = 1\n"
        assert A6(doc(text)) == ("fix", "func retrieve() {}\nvar cType string\nconst PVCReport = 1\n")

    def test_plain_text_and_pure_cyrillic_words_are_kept(self):
        for text in ("func retrieve() {}\n", "// \u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440\n"):
            assert A6(doc(text)) == ("keep", text)

    @pytest.mark.parametrize("url", ["pkg/x/file_test.go", "pkg/x/testdata/struct_name.go", "testdata/a.go",
                                     "a/b/testdata/c/d.go"])
    def test_a_go_test_file_or_testdata_file_is_kept(self, url):
        text = f"type T{CY_E}stStruct struct{{}} // want `identifier`\n"
        assert A6(doc(text, url=url)) == ("keep", text)

    @pytest.mark.parametrize("url", ["pkg/x/file.go", "pkg/contest.go", "pkg/x/mytestdata/a.go", "pkg/attest_go", "", None])
    def test_other_paths_are_still_fixed(self, url):
        text = f"type T{CY_E}stStruct struct{{}}\n"
        assert A6(doc(text, url=url)) == ("fix", "type TestStruct struct{}\n")

    def test_the_same_text_in_the_other_code_packs(self):
        text = f"def r{CY_E}trieve(): pass\n"
        for pack, lang in (("code-python", "python"), ("code-rust", "rust"), ("code-java", "java")):
            assert A6(doc(text, url="a.py", pack=pack, lang=lang)) == ("fix", "def retrieve(): pass\n")
        assert A6(doc(text, url="tests/test_a.py", pack="code-python", lang="python"))[0] == "fix"  # only Go test paths are skipped

    @pytest.mark.parametrize("text", [
        '"\\n\u0418\u043b\u0438 \u043e\u0442\u043f\u0440\u0430\u0432\u044c"',                 # "\nИли отправь": a Go escape stuck to Russian
        '"server=IP_\u0441\u0435\u0440\u0432\u0435\u0440\u0430;"',                            # a Latin prefix stuck to a Russian word
        '"%s\u041f\u043e\u0433\u043e\u0434\u0430: %s"',                                      # a format verb stuck to a Russian word
        '"test1\u0444\u044b\u0432\u0430"',
    ])
    def test_real_cyrillic_words_stuck_to_a_latin_character_are_kept(self, text):
        assert A6(doc(text)) == ("keep", text)

    def test_two_cyrillic_letters_in_a_row_are_not_folded(self):
        text = f"var w{CY_E}{CY_C}x = 1\n"  # a typo with two adjacent look-alikes: a known cost, recorded in the card
        assert A6(doc(text)) == ("keep", text)

    def test_other_languages_and_packs_are_untouched(self):
        text = f"retri{CY_E}ve\n"
        assert A6(doc(text, pack="en-dclm", lang="en")) == ("keep", text)

    def test_a_large_russian_text_is_fast(self):
        import time
        text = "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440 \\n\u0418\u043b\u0438 x1 " * 40000
        start = time.perf_counter()
        assert A6(doc(text)) == ("keep", text)
        assert time.perf_counter() - start < 2.0  # about 0.07 s

    def test_idempotent(self):
        text = f"type T{CY_E}st struct{{}}\nvar {CY_C}Type int\n"
        once = A6(doc(text))[1]
        assert A6(doc(once)) == ("keep", once)


# ---------------------------------------------------------------- DA-E4-011

BSD3 = ("// Copyright 2014 The Foo Authors. All rights reserved.\n"
        "// Use of this source code is governed by a BSD-style\n"
        "// license that can be found in the LICENSE file.\n")
MIT2 = "// Copyright (c) 2020 Foo\n// Licensed under the MIT License.\n"
BODY = "\npackage foo\n\nimport \"fmt\"\n\nfunc F() { fmt.Println(1) }\n"


class TestE4:
    def test_the_three_line_go_header_is_removed(self):
        assert E4(doc(BSD3 + BODY)) == ("fix", BODY.lstrip("\n"))

    def test_the_two_line_licensed_under_header_is_removed(self):
        assert E4(doc(MIT2 + BODY)) == ("fix", BODY.lstrip("\n"))

    def test_the_sentence_wrapped_a_different_way_is_removed(self):
        text = ("// Copyright 2018 AUTHOR. All rights reserved.\n"
                "// Use of this source code is governed by a MIT\n"
                "// license that can be found in the LICENSE file.\n")
        assert E4(doc(text + BODY)) == ("fix", BODY.lstrip("\n"))

    def test_build_lines_and_dataset_lines_above_the_header_stay(self):
        text = "<reponame>a/b\n//go:build linux\n// +build linux\n\n" + BSD3 + BODY
        assert E4(doc(text)) == ("fix", "<reponame>a/b\n//go:build linux\n// +build linux\n\n" + BODY)

    def test_a_long_header_is_still_removed_and_a_short_one_behind_it_goes_in_the_same_pass(self):
        long_header = "".join(f"// Licensed under the Apache License line {i}\n" for i in range(5)) + "\n"
        out = E4(doc(long_header + BSD3 + BODY))
        assert out == ("fix", BODY.lstrip("\n"))
        assert E4(doc(out[1])) == ("keep", out[1])

    def test_a_restrictive_licence_still_drops(self):
        assert E4(doc("// Copyright 2020 Foo\n// Free for personal use only.\n" + BODY))[0] == "drop"

    # ---- negative cases: substantive comments are never removed
    @pytest.mark.parametrize("text", [
        # the first line is not a copyright line: a package doc comment that quotes the licence sentence
        "// Package foo parses licences.\n// Use of this source code is governed by a BSD-style\n// license that can be found in the LICENSE file.\n",
        # a copyright line, then the licence words and a substantive sentence in the same line
        "// Copyright 2020 Foo\n// Licensed under the MIT license, but this generator also embeds the vendored notices of every dependency, read them first.\n",
        # a copyright line and a comment that only mentions the LICENSE file
        "// Copyright 2020 Foo. See the LICENSE file for third-party notices\n// that this tool prints at start.\n",
        # a copyright line that is the only content
        "// Copyright 2020 Foo\n",
        # two copyright lines
        "// Copyright 2020 Foo\n// Copyright 2021 Bar\n",
        # a header that is not at the top: code comes first
        "var x = 1\n" + BSD3,
        # a header that follows another comment, which is kept
        "// Generated by a tool, edit with care.\n\n" + BSD3,
    ])
    def test_substantive_comments_stay(self, text):
        assert E4(doc(text + BODY)) == ("keep", text + BODY)

    def test_a_comment_inside_a_function_is_not_a_header(self):
        text = "package foo\n\nfunc F() {\n" + "".join("\t" + l for l in BSD3.splitlines(True)) + "}\n"
        assert E4(doc(text)) == ("keep", text)

    def test_many_comment_blocks_stay_linear(self):
        # 20,000 separate comment blocks above the first func: a quadratic loop takes seconds, this one about 25 ms
        import time
        text = "".join(f"// note {i}\n\n" for i in range(20000)) + "package a\n\nfunc F() {}\n"
        start = time.perf_counter()
        assert E4(doc(text)) == ("keep", text)
        assert time.perf_counter() - start < 1.0

    def test_many_stacked_short_headers_are_fast(self):
        import time
        text = BSD3 * 2000 + "\npackage a\n\nfunc F() {}\n"
        start = time.perf_counter()
        assert E4(doc(text))[0] == "fix"
        assert time.perf_counter() - start < 2.0  # about 0.12 s; the old rule takes 0.025 s

    def test_other_packs_are_untouched(self):
        assert E4(doc(BSD3 + BODY, pack="code-python", lang="python")) == ("keep", BSD3 + BODY)

    def test_idempotent(self):
        out = E4(doc(BSD3 + BODY))[1]
        assert E4(doc(out)) == ("keep", out)
