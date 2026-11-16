"""Smoke check fixture contract; add tests for your implementation."""
import json
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
class FixtureTest(unittest.TestCase):
    def test_fixture_gold_resolves(self):
        chunks=json.loads((ROOT/'data/example_chunks.json').read_text(encoding='utf-8'))
        questions=json.loads((ROOT/'data/example_questions.json').read_text(encoding='utf-8'))
        for q in questions:
            for g in q['gold']:
                self.assertTrue(any(c['source_id']==g['source_id'] and c['page']==g['page'] and g['evidence'] in c['text'] for c in chunks))

    def test_day6_bridged_data_and_benchmark(self):
        chunks = json.loads((ROOT / 'data/chunks.json').read_text(encoding='utf-8'))
        questions = json.loads((ROOT / 'data/questions.json').read_text(encoding='utf-8'))
        self.assertEqual(len(chunks), 154)
        for c in chunks:
            for key in ('id', 'text', 'source_id', 'source', 'page', 'metadata'):
                self.assertIn(key, c)
                self.assertIsNotNone(c[key])
        dev_ans = [q for q in questions if q['split'] == 'dev' and q.get('answerable', True)]
        dev_unans = [q for q in questions if q['split'] == 'dev' and not q.get('answerable', True)]
        test_ans = [q for q in questions if q['split'] == 'test' and q.get('answerable', True)]
        test_unans = [q for q in questions if q['split'] == 'test' and not q.get('answerable', True)]
        self.assertGreaterEqual(len(dev_ans), 15)
        self.assertGreaterEqual(len(dev_unans), 2)
        self.assertGreaterEqual(len(test_ans), 5)
        self.assertGreaterEqual(len(test_unans), 2)
        for q in dev_ans + test_ans:
            self.assertTrue(q.get('gold'))
            for g in q['gold']:
                self.assertTrue(
                    any(c['source_id'] == g['source_id'] and c['page'] == g['page'] and g['evidence'] in c['text'] for c in chunks),
                    f"Gold evidence failed to resolve for {q['id']}",
                )
        for q in dev_unans + test_unans:
            self.assertEqual(q.get('expected_behavior'), 'search_and_abstain')
            self.assertTrue(q.get('absence_reason'))
if __name__ == '__main__': unittest.main()
