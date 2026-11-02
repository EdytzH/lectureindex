"""Optional local contextual passage reranking after fast title/PDF retrieval.

This module does not decide what MP4s exist or assign PDF pages. It receives only
previously retrieved real MP4s and *verified* PDF witnesses. It compares full
lecture/PDF passages using a cross-encoder and aggregates near-duplicate lecture
chunks as one distinct piece of evidence.

Cross-encoder sigmoid scores are ranking heuristics, not medical probabilities.
"""
from __future__ import annotations

import re
from collections import defaultdict, Counter

import numpy as np

DEFAULT_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RERANKER_MODELS = [DEFAULT_RERANKER, 'BAAI/bge-reranker-base']


def build_reranker(name=DEFAULT_RERANKER):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, max_length=384)


def _context_scores(model, pairs, batch_size=16):
    """Use raw single-label logits explicitly, then sigmoid exactly once."""
    if not pairs:
        return []
    try:
        from torch import nn
        raw = model.predict(pairs, batch_size=batch_size, show_progress_bar=False,
                            activation_fn=nn.Identity())
    except (TypeError, ImportError):
        # Injected test scorers / older sentence-transformers may not accept
        # activation_fn. Real model deployments above always use raw logits.
        raw = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    values = np.asarray(raw, dtype=float).reshape(-1)
    if len(values) != len(pairs):
        raise ValueError('Contextual reranker returned a different number of scores than pairs.')
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))).tolist()


def concept_groups(chunks, encoder, model_name):
    """Near-duplicate *lecture* concepts, not just repeated keyword counts.

    Merge only very close semantic AND lexical duplicates. Two slides about
    vertebral segmentation vs. vertebral ossification should remain distinct.
    This conservative method is intended to prevent repeated identical slides
    or overlapping text windows from multiplying a video's evidence.
    """
    from semantic_matcher import encode, tokens
    if not chunks:
        return []
    vec = encode(encoder, [c['text'] for c in chunks], model_name=model_name, is_query=True)
    words = [set(tokens(c['text'])) for c in chunks]
    normalized = [re.sub(r'\s+', ' ', c['text'].lower()).strip() for c in chunks]
    leaders = []
    groups = []
    for i, chunk in enumerate(chunks):
        assigned = None
        for group_id, leader in enumerate(leaders):
            a, b = words[i], words[leader]
            overlap = len(a & b) / max(1, len(a | b))
            similar = float(vec[i] @ vec[leader]) if vec.size else 0.0
            if normalized[i] == normalized[leader] or (similar >= .91 and overlap >= .76):
                assigned = group_id
                break
        if assigned is None:
            assigned = len(leaders)
            leaders.append(i)
        groups.append(assigned)
    return groups


def contextual_rank(*, candidates, chunks, encoder, model_name, reranker,
                    min_context=.45, max_pairs=320, max_results=20):
    """Rerank candidate MP4s by their paired lecture and PDF passage evidence.

    `candidates` are already retrieved and contain evidence rows with `chunk_id`,
    `pdf_alternatives` and real MP4 metadata. A title-only match has no PDF
    witness, so it is evaluated against full subject/topic/title metadata.

    Pairs are round-robin distributed across lecture chunks so early slides do
    not consume the entire computational budget. A PDF passage earns credit
    only for its own verified MP4 ID. Results below `min_context` are dropped.
    """
    if not candidates or not chunks:
        return []
    if not 0 <= min_context <= 1:
        raise ValueError('Context cutoff must be in the range 0..1.')
    max_pairs = max(1, int(max_pairs))
    groups = concept_groups(chunks, encoder, model_name)
    # Each item: (candidate_index, evidence_index, provenance, page witness)
    by_chunk = defaultdict(list)
    for ci, candidate in enumerate(candidates):
        for ei, ev in enumerate(candidate['evidence']):
            q = ev.get('chunk_id')
            if q is None or q < 0 or q >= len(chunks):
                continue
            # Title comparison is a real option even if no PDF was indexed.
            by_chunk[q].append((ci, ei, float(ev['score'])))
    for q in by_chunk:
        by_chunk[q].sort(key=lambda x:-x[2])
    slots = []
    # Breadth-first midpoint order spreads even a small comparison budget across
    # the entire lecture, instead of selecting only the first N slides.
    from collections import deque
    spans = deque([(0, len(chunks)-1)])
    chunk_order = []
    while spans:
        lo, hi = spans.popleft()
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        chunk_order.append(mid)
        spans.append((lo, mid-1))
        spans.append((mid+1, hi))
    # Round-robin level order maintains fair opportunity for every lecture page.
    maximum_candidates = max((len(x) for x in by_chunk.values()), default=0)
    for rank in range(min(maximum_candidates, 8)):
        for q in chunk_order:
            if len(slots) >= max_pairs:
                break
            rows = by_chunk.get(q, [])
            if rank >= len(rows):
                continue
            ci, ei, _ = rows[rank]
            ev = candidates[ci]['evidence'][ei]
            options = []
            if ev.get('pdf_supported'):
                for p in ev.get('pdf_alternatives') or []:
                    if p.get('text'):
                        options.append(('verified PDF passage', p))
            if not options:
                options.append(('MP4 title / folder', None))
            else:
                # Metadata is still useful if the PDF covers a different
                # section of the lecture, but cannot impersonate page evidence.
                options.append(('MP4 title / folder', None))
            for source, witness in options:
                if len(slots) >= max_pairs:
                    break
                slots.append((q,ci,ei,source,witness))
        if len(slots) >= max_pairs:
            break

    pairs_by_video = Counter(slot[1] for slot in slots)
    pairs = []
    for q,ci,ei,source,witness in slots:
        candidate = candidates[ci]
        right = (witness['text'] if witness else
                 f"{candidate['title']}. Section: {candidate['topic']}. Subject: {candidate['subject']}.")
        pairs.append((chunks[q]['text'][:1300], right[:1300]))
    measured = _context_scores(reranker,pairs)
    # Keep exactly one best teaching-page witness per lecture chunk/video.
    best_by_pair = {}
    pdf_by_pair = {}
    for slot, score in zip(slots,measured):
        q,ci,ei,source,witness = slot
        key = (ci,q)
        # Prefer actual PDF content on a true score tie.
        item = {
            'context_score': float(score), 'context_type':source,
            'pdf_id':witness.get('pdf_id','') if witness else '',
            'pdf_page':witness.get('page') if witness else None,
            'pdf_excerpt':witness.get('text','')[:850] if witness else '',
            'chunk_id':q, 'evidence_index':ei}
        old = best_by_pair.get(key)
        if old is None or (score,source=='verified PDF passage') > (old['context_score'],old['context_type']=='verified PDF passage'):
            best_by_pair[key] = item
        if source == 'verified PDF passage':
            previous = pdf_by_pair.get(key)
            if previous is None or score > previous['context_score']:
                pdf_by_pair[key] = item

    reranked = []
    for ci,row in enumerate(candidates):
        supported=[]
        # Count the best supported match per distinct teaching concept, not every
        # repeated phrase or overlapping lecture chunk.
        per_concept={}
        for (idx,q), item in best_by_pair.items():
            if idx != ci or item['context_score'] < min_context:
                continue
            group = groups[q]
            if group not in per_concept or item['context_score'] > per_concept[group]['context_score']:
                per_concept[group] = item
        supported = sorted(per_concept.values(),key=lambda x:-x['context_score'])
        if not supported:
            continue
        head=supported[:5]
        # Ranking emphasizes genuine paired relevance; independent concepts add
        # some breadth but repeated words cannot multiply the score.
        top=head[0]['context_score']
        avg=float(np.mean([x['context_score'] for x in head]))
        distinct=len(supported)
        aggregate=.68*top+.27*avg+.05*min(1.,max(0,distinct-1)/4)
        updated_evidence=[]
        for item in head:
            ev=dict(row['evidence'][item['evidence_index']])
            ev['context_score']=round(item['context_score'],4)
            ev['context_type']=item['context_type']
            ev['concept_group']=groups[item['chunk_id']]
            ev['context_pdf_id']=item['pdf_id']
            ev['context_pdf_page']=item['pdf_page']
            ev['context_pdf_excerpt']=item['pdf_excerpt']
            updated_evidence.append(ev)
        # A title may outrank a PDF excerpt for the same lecture page. Keep
        # separately accepted, verified PDF witnesses visible in diagnostics.
        supported_query_ids = {item['chunk_id'] for item in supported}
        pdf_evidence=[item for (idx,q),item in pdf_by_pair.items()
                      if idx==ci and q in supported_query_ids and item['context_score']>=min_context]
        best_pdf=max(pdf_evidence,key=lambda item:item['context_score'],default=None)
        reranked.append({**row,
            'initial_retrieval_score':row['score'],
            'score':round(float(aggregate),4),
            'context_score':round(float(top),4),
            'context_mean':round(float(avg),4),
            'distinct_concepts':distinct,
            'context_pairs_tested':pairs_by_video[ci],
            'context_method':'local cross-encoder',
            'evidence':updated_evidence,
            'pdf_context_score':round(best_pdf['context_score'],4) if best_pdf else None,
            'pdf_context_page':best_pdf['pdf_page'] if best_pdf else None,
            'pdf_context_file':best_pdf['pdf_id'] if best_pdf else '',
            'pdf_match_source':('Context-supported verified PDF passage' if best_pdf else
                                'Indexed, but no PDF passage passed context check' if row.get('pdf_indexed') else
                                'No indexed PDF teaching pages'),
            'pdf_match_page':best_pdf['pdf_page'] if best_pdf else None,
            'pdf_match_file':best_pdf['pdf_id'] if best_pdf else '',
            'pdf_match_excerpt':best_pdf['pdf_excerpt'] if best_pdf else '',
            'score_type':'contextual rank (not probability or coverage)'})
    reranked.sort(key=lambda r:(r['score'],r['distinct_concepts'],r['context_score']),reverse=True)
    return reranked[:max_results]
