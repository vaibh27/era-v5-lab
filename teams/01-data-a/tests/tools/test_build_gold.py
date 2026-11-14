"""Run with: python -m unittest discover -s teams/01-data-a/tests/tools -v"""
import base64
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'tools' / 'build_gold.py'
spec = importlib.util.spec_from_file_location('g1_build_gold', SCRIPT)
bg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bg
spec.loader.exec_module(bg)

HEADER = 'doc_id,read,verdict,diseases,evidence,hex,note\n'


def sample(*records):
    return HEADER + '\n'.join(records) + '\n'


def sub(pack, handle, *records, stamp='', pr=0):
    return bg.Submission(pack, handle, 'fixture-' + handle, sample(*records), (stamp, pr))


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


class BuilderTests(unittest.TestCase):
    def test_d3_only_clean_is_gold_eligible_but_fix_is_not(self):
        # D3 is label-only for CLEAN votes; leave all FIX verdicts excluded.
        cases = [
            ("full", "clean", "", True, False),
            ("full", "clean", "D3", True, False),
            ("full", "clean", " d3 ; D3 ", True, False),
            ("full", "fix", "D3", False, True),
            ("full", "fix", " D3 ; D3 ", False, True),
            ("full", "clean", "D3;A1", False, True),
            ("full", "fix", "A1;D3", False, True),
            ("full", "fix", "", False, True),
            ("full", "drop", "D3", False, True),
            ("partial", "clean", "D3", False, False),
            ("partial", "fix", "D3", False, True),
        ]
        for read, verdict, diseases, expected_clean, expected_negative in cases:
            with self.subTest(read=read, verdict=verdict, diseases=diseases):
                vote = bg.Vote(read, verdict, diseases)
                self.assertEqual(vote.clean, expected_clean)
                self.assertEqual(vote.negative, expected_negative)

    def test_d3_only_clean_votes_contribute_to_gold_and_fix_blocks(self):
        docs = [
            sub('sci-finepdfs', 'alice',
                'A,full,clean,D3,,,', 'B,full,clean,D3,,,',
                'C,full,clean,,,,', 'D,full,fix,A1;D3,,,',
                'E,full,fix,D3,,,'),
            sub('sci-finepdfs', 'bob',
                'A,full,clean,,,,', 'B,full,fix,D3,,,',
                'C,full,clean,D3,,,', 'D,full,clean,,,,',
                'E,full,clean,D3,,,'),
            sub('sci-finepdfs', 'carol',
                'B,full,clean,,,,', 'D,full,clean,,,,',
                'E,full,clean,,,,'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            warnings = []
            report = bg.build_gold(docs, Path(tmp), warnings=warnings)
            rows = read_csv(Path(tmp) / 'sci-finepdfs_gold.csv')
            self.assertEqual([r['doc_id'] for r in rows], ['A', 'C'])
            self.assertEqual(report[0]['docs_with_2_clean_votes'], 5)
            self.assertEqual(report[0]['disputed_gold_candidates'], 3)  # B/D/E
            self.assertFalse(any('clean with non-D3' in w for w in warnings))

    def test_d3_fix_veto_and_duplicate_safety(self):
        good = 'X,full,clean,,,,'
        for third_vote, should_include in [
            ('X,full,clean,D3,,,', True),
            ('X,partial,clean,D3,,,', True),
            ('X,full,fix,D3,,,', False),
            ('X,partial,fix,D3,,,', False),
            ('X,full,drop,D3,,,', False),
            ('X,full,fix,D3;C1,,,', False),
            ('X,full,clean,D3;C1,,,', False),
        ]:
            with self.subTest(third_vote=third_vote):
                docs = [sub('sci-finepdfs', 'one', good),
                        sub('sci-finepdfs', 'two', good),
                        sub('sci-finepdfs', 'three', third_vote)]
                with tempfile.TemporaryDirectory() as tmp:
                    bg.build_gold(docs, Path(tmp))
                    self.assertEqual(len(read_csv(Path(tmp) / 'sci-finepdfs_gold.csv')),
                                     int(should_include))
        # A conflicting duplicate from one reviewer is always disputed.
        docs = [sub('sci-finepdfs', 'one', good, 'X,full,fix,D3,,,'),
                sub('sci-finepdfs', 'two', good),
                sub('sci-finepdfs', 'three', good)]
        for allow_disputed in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                bg.build_gold(docs, Path(tmp), allow_disputed=allow_disputed)
                self.assertEqual(read_csv(Path(tmp) / 'sci-finepdfs_gold.csv'), [])

    def test_gold_requires_two_distinct_full_clean_votes(self):
        docs = [
            sub('code-go', 'alice', '001,full,clean,,,,', '002,partial,clean,,,,',
                '003,full,clean,,,,', '004,full,clean,,,,'),
            sub('code-go', 'bob', '001,full,clean,,,,', '002,full,clean,,,,',
                '003,full,fix,A1,reason,,', '004,full,clean,,,,'),
            sub('code-go', 'carol', '004,full,drop,C1,reason,,'),
            sub('ind-hi-fw2', 'dev', '001,full,clean,,,,'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = bg.build_gold(docs, Path(tmp), target=2)
            self.assertEqual([r['pack_id'] for r in report], ['code-go'])
            self.assertEqual(read_csv(Path(tmp) / 'code-go_gold.csv'), [
                {'doc_id': '001', 'reader_1': 'alice', 'reader_2': 'bob'}
            ])
            self.assertEqual(report[0]['docs_with_2_clean_votes'], 2)
            self.assertEqual(report[0]['disputed_gold_candidates'], 1)
            self.assertEqual(report[0]['gold_docs'], 1)
            self.assertEqual(report[0]['shortfall'], 1)
            self.assertEqual(len(read_csv(Path(tmp) / 'gold_coverage.csv')), 1)

    def test_allow_disputed(self):
        docs = [sub('en-dclm','one','01,full,clean,,,,'),
                sub('en-dclm','two','01,full,clean,,,,'),
                sub('en-dclm','three','01,full,fix,A1,,,'),]
        with tempfile.TemporaryDirectory() as tmp:
            bg.build_gold(docs, Path(tmp), allow_disputed=True)
            self.assertEqual(len(read_csv(Path(tmp) / 'en-dclm_gold.csv')), 1)

    def test_duplicate_votes_and_versions_not_two_reviewers(self):
        old = sub('math-finemath', 'same', 'a,full,clean,,,,')
        updated = sub('math-finemath', 'SAME', 'a,full,clean,,,,',
                      'b,full,clean,,,,', stamp='2026-10-26T12:00:00Z', pr=12)
        other = sub('math-finemath','other','b,full,clean,,,,')
        with tempfile.TemporaryDirectory() as tmp:
            report = bg.build_gold([old,updated,other],Path(tmp))
            self.assertEqual(report[0]['reviewers'],2)
            self.assertEqual(read_csv(Path(tmp)/'math-finemath_gold.csv'), [
                {'doc_id':'b','reader_1':'other','reader_2':'SAME'}
            ])

    def test_conflicting_rows_from_one_reviewer_do_not_count(self):
        docs=[sub('en-test','alice','abc,full,clean,,,,','abc,full,fix,A1,,,'),
              sub('en-test','bob','abc,full,clean,,,,'),
              sub('en-test','carol','abc,full,clean,,,,')]
        warnings=[]
        with tempfile.TemporaryDirectory() as tmp:
            report=bg.build_gold(docs,Path(tmp),warnings=warnings)
            self.assertEqual(report[0]['gold_docs'],0)  # alice's fix remains a dispute
            self.assertEqual(report[0]['disputed_gold_candidates'],1)
            self.assertEqual(report[0]['docs_with_2_clean_votes'],1)  # bob + carol
            self.assertEqual(read_csv(Path(tmp)/'en-test_gold.csv'), [])
            self.assertTrue(any('conflicting duplicate' in x for x in warnings))

    def test_duplicate_fix_or_drop_always_blocks_gold(self):
        # The disputed row may appear before or after clean, and duplicate
        # conflicts must remain excluded even with --allow-disputed.
        for bad in ('fix', 'drop'):
            for clean_first in (True, False):
                with self.subTest(bad=bad, clean_first=clean_first):
                    good = 'abc,full,clean,,,,'
                    negative = f'abc,full,{bad},A1,,,'
                    alice_rows = (good, negative) if clean_first else (negative, good)
                    docs = [sub('en-test', 'alice', *alice_rows),
                            sub('en-test', 'bob', good),
                            sub('en-test', 'carol', good)]
                    for allow_disputed in (False, True):
                        with tempfile.TemporaryDirectory() as tmp:
                            report = bg.build_gold(docs, Path(tmp),
                                                   allow_disputed=allow_disputed)
                            self.assertEqual(report[0]['gold_docs'], 0)
                            self.assertEqual(report[0]['disputed_gold_candidates'], 1)
                            self.assertEqual(read_csv(Path(tmp)/'en-test_gold.csv'), [])

    def test_duplicate_conflict_retains_marker_after_more_rows(self):
        good='abc,full,clean,,,,'
        docs=[sub('en-test','alice',good,'abc,full,fix,A1,,,',good),
              sub('en-test','bob',good),sub('en-test','carol',good)]
        votes=bg.parse_labels(docs[0],[])
        self.assertTrue(votes['abc'].conflicted_duplicate)
        self.assertTrue(votes['abc'].negative)
        self.assertFalse(votes['abc'].clean)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(bg.build_gold(docs,Path(tmp))[0]['gold_docs'],0)

    def test_clean_with_diseases_and_partial_are_not_clean(self):
        docs=[sub('en-test','one','A,full,clean,A1,,,','B,partial,clean,,,,'),
              sub('en-test','two','A,full,clean,,,,','B,full,clean,,,,')]
        with tempfile.TemporaryDirectory() as tmp:
            report=bg.build_gold(docs,Path(tmp))
            self.assertEqual(report[0]['gold_docs'],0)

    def test_explicit_pack_list_yields_empty_gold_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            report = bg.build_gold([],p,expected_packs={'code-rust'})
            self.assertEqual(report[0]['gold_docs'],0)
            self.assertEqual((p/'code-rust_gold.csv').read_text(), 'doc_id,reader_1,reader_2\n')

    def test_scans_local_submissions_and_ignores_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'diagnosis'/'submissions'/'code-go'/'alice'
            p.mkdir(parents=True)
            (p/'labels.csv').write_text(sample('ABC,full,clean,,,,'), encoding='utf-8')
            local=bg.local_submissions(Path(tmp))
            self.assertEqual(len(local),1)
            self.assertEqual(local[0].pack_id, 'code-go')
            self.assertEqual(local[0].handle, 'alice')

    def test_github_pr_blob_loading(self):
        class StubAPI:
            def paged(self, endpoint):
                if 'pulls?state=open' in endpoint:
                    return iter([{'number':12,'updated_at':'2026-10-26T00:00:00Z',
                                  'base':{'ref':'main'}}])
                if 'pulls/12/files' in endpoint:
                    return iter([{'filename':'diagnosis/submissions/code-go/Alice/labels.csv',
                                  'status':'added','sha':'abc123'},
                                 {'filename':'diagnosis/submissions/code-go/Alice/medicine.md',
                                  'status':'added','sha':'other'}])
                raise AssertionError(endpoint)
            def get(self, endpoint):
                if endpoint.endswith('git/blobs/abc123'):
                    return {'encoding':'base64','content':base64.b64encode(sample('one,full,clean,,,,').encode()).decode()}
                raise AssertionError(endpoint)
        pr=bg.github_pr_submissions(StubAPI(), 'theschoolofai/era-v5-lab')
        self.assertEqual(len(pr),1)
        self.assertEqual(pr[0].handle,'Alice')
        self.assertEqual(bg.parse_labels(pr[0],[])['one'].clean,True)

    def test_malformed_csv_header_skipped_without_losing_other_reviewers(self):
        broken = bg.Submission('en-test', 'alice', 'bad-file',
                               'doc_id,read,verdict\na,full,clean\n', ('', 0))
        valid = [sub('en-test', 'bob', 'a,full,clean,,,,'),
                 sub('en-test', 'carol', 'a,full,clean,,,,'),
                 sub('code-go', 'dave', 'b,full,clean,,,,'),
                 sub('code-go', 'erin', 'b,full,clean,,,,')]
        warnings = []
        with tempfile.TemporaryDirectory() as tmp:
            result = bg.build_gold([broken] + valid, Path(tmp), warnings=warnings)
            packs = {p['pack_id']: p for p in result}
            self.assertEqual(packs['en-test']['skipped_label_files'], 1)
            self.assertEqual(packs['en-test']['label_files'], 2)
            self.assertEqual(packs['en-test']['reviewers'], 2)
            self.assertEqual(packs['en-test']['gold_docs'], 1)
            self.assertEqual(packs['code-go']['skipped_label_files'], 0)
            self.assertEqual(packs['code-go']['gold_docs'], 1)
            self.assertEqual(len(read_csv(Path(tmp) / 'en-test_gold.csv')), 1)
            self.assertEqual(read_csv(Path(tmp) / 'gold_coverage.csv')[1]['skipped_label_files'], '1')
            self.assertTrue(any('Skipping invalid labels CSV' in w and 'bad-file' in w for w in warnings))

    def test_large_field_over_original_128k_limit_is_accepted(self):
        self.assertGreaterEqual(csv.field_size_limit(), 512 * 1024)
        long_evidence = 'x' * (512 * 1024)
        raw = sample('a,full,fix,A1,"' + long_evidence + '",,',
                     'b,full,clean,,,,')
        large = bg.Submission('en-test', 'alice', 'PR #120 large evidence', raw, ('', 0))
        other = sub('en-test', 'bob', 'b,full,clean,,,,')
        with tempfile.TemporaryDirectory() as tmp:
            warnings = []
            rows = bg.build_gold([large, other], Path(tmp), warnings=warnings)
            self.assertEqual(rows[0]['skipped_label_files'], 0)
            self.assertEqual(rows[0]['gold_docs'], 1)
            self.assertEqual(read_csv(Path(tmp) / 'en-test_gold.csv')[0]['doc_id'], 'b')

    def test_unparseable_oversized_sheet_is_skipped_not_partially_used(self):
        previous_limit = csv.field_size_limit()
        try:
            # Force the same parser failure as the original 128 KiB limit,
            # without allocating an unreasonably large fixture.
            csv.field_size_limit(128)
            broken = bg.Submission('math-finemath', 'bad', 'oversized-sheet',
                sample('a,full,clean,,,,', 'b,full,fix,A1,"' + 'x' * 512 + '",,'), ('', 0))
            valid = [sub('math-finemath', 'alice', 'a,full,clean,,,,'),
                     sub('math-finemath', 'bob', 'a,full,clean,,,,')]
            with tempfile.TemporaryDirectory() as tmp:
                warnings = []
                result = bg.build_gold([broken] + valid, Path(tmp), warnings=warnings)
                self.assertEqual(result[0]['skipped_label_files'], 1)
                self.assertEqual(result[0]['label_files'], 2)
                self.assertEqual(result[0]['gold_docs'], 1)
                self.assertTrue(any('field larger than field limit' in w for w in warnings))
        finally:
            csv.field_size_limit(previous_limit)


if __name__ == '__main__':
    unittest.main()
