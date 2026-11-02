"""Benchmark annotated retrieval judgments: Hit@1, Hit@5, and MRR@5.

Usage:
    python eval_retrieval.py --labels evaluation/labels.csv

Annotate expected_mp4_paths with one or more exact paths (separate with |).
Rows with blank expected paths are not evaluated. No invented 'ground truth'.
"""
import argparse
import csv
from pathlib import Path

from file_index_catalog import load_file_index
from context_reranker import DEFAULT_RERANKER, build_reranker
from semantic_matcher import (DEFAULT_MODEL, build_encoder, catalog_records,
                              chunks_from_text, load_slide_evidence, prepare_index,
                              HybridMatcher)


def evaluate_rows(rows, search_fn):
    tested = []
    for row in rows:
        query = row.get('query', '').strip()
        expected = {p.strip().replace('\\', '/') for p in row.get('expected_mp4_paths', '').split('|') if p.strip()}
        if not query or not expected:
            continue
        results = search_fn(query)
        found = [str(x['id']).replace('\\', '/') for x in results[:5]]
        rank = next((i+1 for i, path in enumerate(found) if path in expected), None)
        tested.append({'query': query, 'expected_paths': '|'.join(sorted(expected)),
                       'hit_at_1': int(rank == 1), 'hit_at_5': int(rank is not None),
                       'reciprocal_rank_at_5': round(1/rank, 4) if rank else 0,
                       'top_1': found[0] if found else '', 'rank': rank or ''})
    return tested


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', default='evaluation/labels.csv')
    ap.add_argument('--workbook', default='bootcamp_step1_file_index.xlsx')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--index', default='index')
    ap.add_argument('--threshold', type=float, default=0.42)
    ap.add_argument('--top-k', type=int, default=15)
    ap.add_argument('--out', default='evaluation/results.csv')
    ap.add_argument('--context-rerank', action='store_true',
                    help='Apply the same local cross-encoder stage as the app.')
    ap.add_argument('--reranker', default=DEFAULT_RERANKER)
    ap.add_argument('--context-threshold', type=float, default=.45)
    ap.add_argument('--max-context-pairs', type=int, default=320)
    ap.add_argument('--pdf-weight', type=float, default=.35,
                    help='Use verified PDF teaching chunks if a valid index exists.')
    args = ap.parse_args()
    with open(args.labels, newline='', encoding='utf-8-sig') as stream:
        labels = list(csv.DictReader(stream))
    eligible = [row for row in labels if row.get('query', '').strip() and row.get('expected_mp4_paths', '').strip()]
    if not eligible:
        print(f'0 labeled examples in {args.labels}. Fill expected_mp4_paths with real MP4 paths first.')
        return
    inventory = load_file_index(args.workbook)
    records = catalog_records(inventory)
    model = build_encoder(args.model)
    index = prepare_index(records, cache_dir=args.index, model_name=args.model,
                          encoder=model, slide_evidence=load_slide_evidence(Path(args.index)/'verified_video_slides.json'))
    matcher = HybridMatcher(index, model)
    reranker = build_reranker(args.reranker) if args.context_rerank else None
    pdf = None
    if args.pdf_weight > 0:
        try:
            from pdf_evidence import PDFChunkEvidence
            pdf = PDFChunkEvidence(args.index, args.model, [r['id'] for r in records])
            print(f'Using {len(pdf.chunks)} verified Bootcamp PDF teaching chunks.')
        except (FileNotFoundError, ValueError, OSError) as exc:
            print('No compatible verified PDF index; evaluating title-only:', exc)
    def search(query):
        return matcher.search(chunks_from_text(query), limit=5, top_k=args.top_k,
                              threshold=args.threshold, pdf_evidence=pdf,
                              pdf_weight=args.pdf_weight, reranker=reranker,
                              context_min_score=args.context_threshold,
                              max_context_pairs=args.max_context_pairs)
    tested = evaluate_rows(labels, search)
    n = len(tested)
    print(f'Examples labeled: {n} / {len(labels)}')
    print(f"Top-1 hit rate: {sum(r['hit_at_1'] for r in tested)/n:.1%}")
    print(f"Top-5 hit rate: {sum(r['hit_at_5'] for r in tested)/n:.1%}")
    print(f"MRR@5: {sum(r['reciprocal_rank_at_5'] for r in tested)/n:.3f}")
    p = Path(args.out)
    p.parent.mkdir(exist_ok=True, parents=True)
    with p.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tested[0]))
        writer.writeheader()
        writer.writerows(tested)
    print('Detailed results:', p)


if __name__ == '__main__':
    main()
