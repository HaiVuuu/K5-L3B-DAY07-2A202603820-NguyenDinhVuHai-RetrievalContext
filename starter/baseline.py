"""Retrieval baselines (Dense, Lexical BM25/TF-IDF, Hybrid RRF) for Day 7."""
import argparse
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

CHAPTER_MAP = {
    1: 'Chapter 1: The Boy Who Lived',
    2: 'Chapter 2: The Vanishing Glass',
    3: 'Chapter 3: The Letters from No One',
    4: 'Chapter 4: The Keeper of the Keys',
}


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def infer_chapter_filter(question, user_filter=None):
    """Infer metadata filter from explicit chapter cues in query or user-supplied filter."""
    if isinstance(user_filter, dict) and user_filter:
        return dict(user_filter), 'user_filter'
    q_low = unicodedata.normalize('NFC', question or '').casefold()
    m = re.search(r'\bchapter\s+([1-4])\b', q_low)
    if m:
        return {'chapter': CHAPTER_MAP[int(m.group(1))]}, 'query_chapter_number'
    for chap in CHAPTER_MAP.values():
        title = chap.split(': ', 1)[1].casefold()
        if title in q_low:
            return {'chapter': chap}, 'query_chapter_title'
    return None, None


def _matches_filter(chunk, flt):
    if not flt:
        return True
    meta = chunk.get('metadata', {})
    return all(chunk.get(k, meta.get(k)) == v for k, v in flt.items())


def validate(chunks, questions):
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('chunks phải là JSON array không rỗng')
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions phải là JSON array không rỗng')
    ids = set()
    for c in chunks:
        for key in ('id', 'text', 'source_id', 'source', 'page', 'metadata'):
            if key not in c or c[key] is None:
                raise ValueError(f'chunk thiếu {key}')
        if not all(isinstance(c[k], str) and c[k].strip() for k in ('id', 'text', 'source_id', 'source')):
            raise ValueError('chunk id/text/source_id/source phải là chuỗi không rỗng')
        if not isinstance(c['metadata'], dict) or c['id'] in ids:
            raise ValueError('metadata phải là object và ID chunk phải duy nhất')
        ids.add(c['id'])
    ids.clear()
    for q in questions:
        if not isinstance(q.get('id'), str) or not q['id'].strip() or q['id'] in ids:
            raise ValueError('question id phải là chuỗi không rỗng và duy nhất')
        ids.add(q['id'])
        if q.get('split') not in ('dev', 'test') or not isinstance(q.get('question'), str) or not q['question'].strip():
            raise ValueError(f"question {q['id']}: split hoặc question không hợp lệ")
        if q.get('answerable', True) is False:
            continue
        if not isinstance(q.get('expected_answer'), str) or not q['expected_answer'].strip():
            raise ValueError(f"question {q['id']}: expected_answer phải là chuỗi không rỗng")
        if not isinstance(q.get('gold'), list) or not q['gold']:
            raise ValueError(f"question {q['id']}: gold phải là danh sách không rỗng")
        for g in q['gold']:
            if not all(key in g and g[key] is not None for key in ('source_id', 'page', 'evidence')) or not isinstance(g['evidence'], str) or not g['evidence']:
                raise ValueError(f"question {q['id']}: gold không hợp lệ")
            if not any(hit(c, [g]) for c in chunks):
                raise ValueError(f"question {q['id']}: gold không nằm trọn trong chunk đúng nguồn/trang")


def hit(chunk, gold):
    if not gold:
        return False
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page'] and g['evidence'] in chunk['text'] for g in gold)


def _format_results(items, strategy, top_k):
    ordered = sorted(items, key=lambda c: (-c['score'], c['id']))[:top_k]
    return [{**c, 'chunk_id': c['id'], 'rank': idx + 1, 'strategy': strategy} for idx, c in enumerate(ordered)]


class TfidfIndex:
    strategy = 'tfidf'

    def __init__(self, chunks):
        self.chunks = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts):
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {term: value / norm for term, value in vector.items()}

    def search(self, question, top_k=3, filters=None):
        query = self._vector(Counter(tokenize(question)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if not _matches_filter(chunk, filters):
                continue
            score = sum(value * vector.get(term, 0) for term, value in query.items())
            if score > 0:
                results.append({**chunk, 'score': score})
        return _format_results(results, self.strategy, top_k)


class Bm25Index:
    strategy = 'bm25'

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c['text']) for c in chunks]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.counts = [Counter(t) for t in self.doc_tokens]
        df = Counter(term for row in self.counts for term in row)
        n = len(chunks)
        self.idf = {term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, question, top_k=3, filters=None):
        q_terms = tokenize(question)
        results = []
        for chunk, tf_map, dl in zip(self.chunks, self.counts, self.doc_lens):
            if not _matches_filter(chunk, filters):
                continue
            score = 0.0
            norm_denom = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            for term in q_terms:
                tf = tf_map.get(term, 0)
                if tf > 0:
                    score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1.0)) / (tf + norm_denom)
            if score > 0:
                results.append({**chunk, 'score': score})
        return _format_results(results, self.strategy, top_k)


class DenseIndex:
    strategy = 'dense'

    def __init__(self, chunks, chroma_dir='chroma_db', collection_name='hp_strat_b', model_name='all-MiniLM-L6-v2'):
        import chromadb
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.chunk_map = {c['id']: c for c in chunks}
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_collection(collection_name)

    def search(self, question, top_k=3, filters=None):
        if not question or not question.strip():
            return []
        emb = self.model.encode([question], normalize_embeddings=True).tolist()
        kwargs = {'query_embeddings': emb, 'n_results': min(max(top_k, 1), len(self.chunks))}
        if filters:
            kwargs['where'] = filters
        res = self.collection.query(**kwargs)
        ids = res['ids'][0] if res.get('ids') else []
        dists = res['distances'][0] if res.get('distances') else []
        items = []
        for cid, dist in zip(ids, dists):
            if cid in self.chunk_map:
                items.append({**self.chunk_map[cid], 'score': float(1.0 - dist)})
        return _format_results(items, self.strategy, top_k)


class HybridRrfIndex:
    strategy = 'hybrid'

    def __init__(self, dense_index, lexical_index, rrf_k=60, candidate_k=20):
        self.dense = dense_index
        self.lexical = lexical_index
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.chunk_map = {c['id']: c for c in lexical_index.chunks}

    def search(self, question, top_k=3, filters=None):
        inferred_filter, _ = infer_chapter_filter(question, filters)
        d_hits = self.dense.search(question, top_k=self.candidate_k, filters=inferred_filter)
        l_hits = self.lexical.search(question, top_k=self.candidate_k, filters=inferred_filter)
        scores = {}
        for rank, item in enumerate(d_hits, start=1):
            scores[item['id']] = scores.get(item['id'], 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, item in enumerate(l_hits, start=1):
            scores[item['id']] = scores.get(item['id'], 0.0) + 1.0 / (self.rrf_k + rank)
        fused = [{**self.chunk_map[cid], 'score': sc} for cid, sc in scores.items() if cid in self.chunk_map]
        return _format_results(fused, self.strategy, top_k)


def pack_context(results, budget=800):
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


def evaluate(index, questions, budget=800, top_k=3):
    traces = []
    strategy = getattr(index, 'strategy', 'tfidf')
    for q in questions:
        start = time.perf_counter()
        use_filter = q.get('filter') if strategy == 'hybrid' else None
        inferred_filter, filter_reason = infer_chapter_filter(q['question'], use_filter) if strategy == 'hybrid' else (None, None)
        try:
            results = index.search(q['question'], top_k=top_k, filters=inferred_filter) if strategy == 'hybrid' else index.search(q['question'], top_k=top_k)
        except TypeError:
            results = index.search(q['question'], top_k=top_k)
        context, ids = pack_context(results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        answerable = q.get('answerable', True) is not False and bool(q.get('gold'))
        gold = q.get('gold', [])
        marked = [{**c, 'hit': hit(c, gold) if answerable else False} for c in results]
        rr = 0.0
        if answerable:
            for idx, c in enumerate(marked, start=1):
                if c['hit']:
                    rr = 1.0 / idx
                    break
        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'answerable': answerable,
            'strategy': strategy,
            'top_k': top_k,
            'budget_chars': budget,
            'applied_filter': inferred_filter,
            'filter_reason': filter_reason,
            'gold': gold,
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']) if answerable else None,
            'hit3': any(c['hit'] for c in marked[:3]) if answerable else None,
            'context_hit': any(c['hit'] and c['id'] in ids for c in marked) if answerable else None,
            'rr': rr if answerable else None,
        })
    return traces


def summarize_traces(strategy, split, traces, index_seconds=0.0):
    ans = [t for t in traces if t.get('answerable', True) and t.get('hit1') is not None]
    n = len(ans)
    return {
        'strategy': strategy,
        'split': split,
        'n': n,
        'total_queries': len(traces),
        'unanswerable_queries': len(traces) - n,
        'hit1': sum(bool(t['hit1']) for t in ans),
        'hit1_rate': (sum(bool(t['hit1']) for t in ans) / n) if n else 0.0,
        'hit3': sum(bool(t['hit3']) for t in ans),
        'hit3_rate': (sum(bool(t['hit3']) for t in ans) / n) if n else 0.0,
        'context_hit': sum(bool(t['context_hit']) for t in ans),
        'context_hit_rate': (sum(bool(t['context_hit']) for t in ans) / n) if n else 0.0,
        'mrr': statistics.mean(t.get('rr', 0.0) for t in ans) if n else 0.0,
        'avg_context_chars': statistics.mean(t['context_chars'] for t in traces) if traces else 0.0,
        'avg_ms': statistics.mean(t['elapsed_ms'] for t in traces) if traces else 0.0,
        'median_ms': statistics.median(t['elapsed_ms'] for t in traces) if traces else 0.0,
        'index_seconds': index_seconds,
        'service_cost_usd': 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Chạy retrieval baseline trên một split rõ ràng')
    parser.add_argument('--chunks', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--strategy', default='tfidf', choices=('tfidf', 'bm25', 'dense', 'hybrid'))
    parser.add_argument('--chroma-dir', default=Path('chroma_db'), type=Path)
    parser.add_argument('--collection', default='hp_strat_b')
    parser.add_argument('--output', required=True, type=Path, help='Thư mục output; file cùng tên sẽ bị ghi đè')
    args = parser.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    # Do not inspect held-out test evidence during dev work.
    selected = [q for q in questions if q.get('split') == args.split]
    validate(chunks, selected)
    started = time.perf_counter()
    if args.strategy == 'tfidf':
        index = TfidfIndex(chunks)
    elif args.strategy == 'bm25':
        index = Bm25Index(chunks)
    elif args.strategy == 'dense':
        index = DenseIndex(chunks, chroma_dir=args.chroma_dir, collection_name=args.collection)
    elif args.strategy == 'hybrid':
        dense = DenseIndex(chunks, chroma_dir=args.chroma_dir, collection_name=args.collection)
        lexical = Bm25Index(chunks)
        index = HybridRrfIndex(dense, lexical)
    index_seconds = time.perf_counter() - started
    traces = evaluate(index, selected)
    summary = summarize_traces(args.strategy, args.split, traces, index_seconds=index_seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (('traces.json', traces), ('summary.json', summary)):
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
