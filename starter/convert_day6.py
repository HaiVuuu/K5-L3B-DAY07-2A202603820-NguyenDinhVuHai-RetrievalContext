"""Convert an exported Day 6 chunk/benchmark JSON array or ChromaDB collection to Day 7 JSON contract."""
import argparse
import json
import re
from pathlib import Path

UNANSWERABLE_QUESTIONS = [
    {
        'id': 'q21',
        'split': 'dev',
        'question': 'What form did Harry Potter’s Patronus take when he drove away the Dementors by the lake?',
        'answerable': False,
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Dementors and the Patronus charm appear in Book 3 (Prisoner of Azkaban), not in Book 1 Chapters 1–4.',
        'gold': [],
    },
    {
        'id': 'q22',
        'split': 'dev',
        'question': 'Who secretly placed Harry Potter’s name into the Goblet of Fire for the Triwizard Tournament?',
        'answerable': False,
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'The Goblet of Fire and Triwizard Tournament occur in Book 4, outside Book 1 Chapters 1–4.',
        'gold': [],
    },
    {
        'id': 'q23',
        'split': 'test',
        'question': 'How did Harry Potter destroy Tom Riddle’s diary inside the Chamber of Secrets?',
        'answerable': False,
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'The Chamber of Secrets and Tom Riddle’s diary appear in Book 2, not in Book 1 Chapters 1–4.',
        'gold': [],
    },
    {
        'id': 'q24',
        'split': 'test',
        'question': 'Which Horcrux did Ron Weasley destroy with the Sword of Gryffindor in the Forest of Dean?',
        'answerable': False,
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Horcruxes and the Forest of Dean events occur in Book 7 (Deathly Hallows), not in Book 1 Chapters 1–4.',
        'gold': [],
    },
]


def convert_chunk(c):
    meta = dict(c.get('metadata', {}))
    source_id = c.get('source_id') or c.get('document_id') or meta.pop('document_id', None)
    source = c.get('source') or meta.pop('source', None)
    page = c.get('page') if c.get('page') is not None else meta.pop('page', None)
    for k in ('document_id', 'source', 'page'):
        meta.pop(k, None)
    return {
        'id': c.get('id', c.get('chunk_id')),
        'source_id': source_id,
        'source': source,
        'page': page,
        'text': c.get('text'),
        'metadata': meta,
    }


def _resolve_evidence(evidence, source_id, page, chunks):
    if not chunks or not evidence:
        return evidence
    page_chunks = [c for c in chunks if c.get('source_id') == source_id and c.get('page') == page]
    if any(evidence in c.get('text', '') for c in page_chunks):
        return evidence
    words = evidence.split()
    for length in range(len(words), 5, -1):
        for start in range(len(words) - length + 1):
            pat = r'\s+'.join(re.escape(w) for w in words[start:start + length])
            for c in page_chunks:
                m = re.search(pat, c.get('text', ''), flags=re.IGNORECASE)
                if m:
                    return m.group(0)
    return evidence


def convert_question(q, chunks=None):
    if q.get('answerable') is False:
        result = {
            'id': q.get('id'),
            'split': q.get('split'),
            'question': q.get('question'),
            'answerable': False,
            'expected_behavior': q.get('expected_behavior', 'search_and_abstain'),
            'absence_reason': q.get('absence_reason', ''),
            'gold': [],
        }
        if q.get('filter') is not None:
            result['filter'] = q['filter']
        return result

    gold = q.get('gold')
    if gold is None:
        ev = _resolve_evidence(q.get('evidence_quote'), q.get('gold_source_id'), q.get('gold_page'), chunks)
        gold = [{'source_id': q.get('gold_source_id'), 'page': q.get('gold_page'), 'evidence': ev}]
    else:
        gold = [
            {**g, 'evidence': _resolve_evidence(g.get('evidence'), g.get('source_id'), g.get('page'), chunks)}
            for g in gold
        ]
    result = {
        'id': q.get('id'),
        'split': q.get('split'),
        'question': q.get('question'),
        'answerable': q.get('answerable', True),
        'expected_behavior': q.get('expected_behavior', 'search_and_answer'),
        'gold': gold,
        'expected_answer': q.get('expected_answer'),
    }
    if q.get('filter') is not None:
        result['filter'] = q['filter']
    return result


def load_chunks_from_chroma(chroma_dir: Path, collection_name: str = 'hp_strat_b'):
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_collection(collection_name)
    res = col.get(include=['documents', 'metadatas'])
    raw = [
        {'id': cid, 'text': doc, 'metadata': meta}
        for cid, doc, meta in zip(res['ids'], res['documents'], res['metadatas'])
    ]
    raw.sort(key=lambda x: (x['metadata'].get('page', 0), x['id']))
    return raw


def main():
    parser = argparse.ArgumentParser(description='Chuyển JSON export hoặc ChromaDB Day 6 sang schema Day 7')
    parser.add_argument('--chunks', type=Path, help='Đường dẫn JSON chunks Day 6')
    parser.add_argument('--chroma-dir', type=Path, help='Đường dẫn thư mục chroma_db Day 6')
    parser.add_argument('--collection', default='hp_strat_b', help='Tên collection ChromaDB')
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--add-unanswerable', action='store_true', help='Thêm câu hỏi không có đáp án cho Day 7')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    if args.chunks and args.chunks.exists():
        raw_chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    elif args.chroma_dir and args.chroma_dir.exists():
        raw_chunks = load_chunks_from_chroma(args.chroma_dir, args.collection)
    else:
        raise ValueError('Cần cung cấp --chunks hoặc --chroma-dir hợp lệ')

    raw_questions = json.loads(args.questions.read_text(encoding='utf-8'))
    if not isinstance(raw_chunks, list) or not isinstance(raw_questions, list):
        raise ValueError('Hai file đầu vào phải là JSON arrays')

    chunks = [convert_chunk(c) for c in raw_chunks]
    questions = [convert_question(q, chunks=chunks) for q in raw_questions]
    if args.add_unanswerable:
        existing_ids = {q['id'] for q in questions}
        for uq in UNANSWERABLE_QUESTIONS:
            if uq['id'] not in existing_ids:
                questions.append(convert_question(uq, chunks=chunks))

    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (('chunks.json', chunks), ('questions.json', questions)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Đã chuyển đổi; kiểm tra output và chạy baseline trên dev để kiểm chứng.')


if __name__ == '__main__':
    main()
