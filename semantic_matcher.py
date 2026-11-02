"""Catalog-first hybrid retrieval. Only actual inventory MP4s can be results.

No medical-condition-specific if statements, fabricated video entries, or inferred PDF
page boundaries. Title and context vectors are reusable. Verified handout text can be
attached later using the same video IDs without changing the ranking architecture.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from beta_catalog import records_for_search, folder_candidates

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MODELS = [DEFAULT_MODEL, 'BAAI/bge-small-en-v1.5']
INDEX_VERSION = 2
TOKEN = re.compile(r"[a-z0-9]+(?:['’\-][a-z0-9]+)*", re.IGNORECASE)
STOPWORDS = set(ENGLISH_STOP_WORDS) | {"step", "yield", "lecture", "video", "videos", "topic", "topics"}
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
BULLET = re.compile(r'^\s*(?:[-*•▪]+\s+|\d{1,3}[.)]\s+)')


def tokens(value):
    """Consistent tokenizer for lexical retrieval; no hand-coded disease synonyms."""
    value = re.sub(r'^\s*\d+\s*[.)-]\s*', '', str(value).lower())
    return [m.group().replace('’', "'") for m in TOKEN.finditer(value)
            if m.group().lower() not in STOPWORDS and not m.group().isdigit()]


def strip_number(value):
    return re.sub(r'^\s*\d+\s*[.)-]\s*', '', str(value)).strip()


def catalog_records(inventory):
    """Stable full-path IDs, original folder metadata, and future slide placeholders."""
    result = []
    for video in records_for_search(inventory):
        pdfs = folder_candidates(video, inventory)
        result.append({
            'id': video['path'].replace('\\', '/'),
            'title': video['title'],
            'subject': video['subject'],
            'topic': video['topic'],
            'course_section': f"{video['subject']} / {video['topic']}",
            'video_url': video['path'],  # inventory filesystem path, NOT a public URL
            'mp4_path': video['path'],
            'pdf_candidates': [p['path'] for p in pdfs],
            'slide_text': '',
            'title_embedding': None,  # stored in title_vectors.npy at this row index
            'context_embedding': None,  # stored in context_vectors.npy
            'slide_embedding': None,  # stored in slide_vectors.npy once verified
        })
    return result


def chunks_from_text(raw, source='Pasted text', *, max_words=72, max_chars=600):
    """Sentence/group chunks; each is a separate retrieval query.

    Newline bullets are independent, paragraphs are split into sentences, and very
    long sentences are split into overlapping word windows. Never use the complete
    100-slide lecture as a single query.
    """
    units = []
    for line in re.split(r'\n\s*\n', str(raw).strip()):
        for item in line.splitlines():
            text = BULLET.sub('', item).strip()
            if not text or text.startswith('#'):
                continue
            sentences = SENTENCE_SPLIT.split(text)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                words = sentence.split()
                if len(words) <= max_words and len(sentence) <= max_chars:
                    units.append(sentence)
                else:
                    step = max(10, max_words - 12)
                    for pos in range(0, len(words), step):
                        piece = ' '.join(words[pos:pos + max_words])
                        if piece:
                            units.append(piece)
                        if pos + max_words >= len(words):
                            break
    seen = set()
    out = []
    for text in units:
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            out.append({'source': source, 'page': None, 'text': text})
    return out


def chunks_from_slides(slides, source, *, max_words=72):
    """Preserve coherent slide content rather than embedding each visual line.

    PDF text extraction commonly inserts newlines after every text box. Feeding
    those individual lines through the paragraph parser generated isolated
    queries like 'at 5 weeks' that falsely favored unrelated video titles.
    Join each slide's lines first; split only unusually long slides into
    overlapping windows, and preserve the original lecture page number.
    """
    result = []
    for slide in slides:
        content = ' '.join(str(slide.get('text') or '').split())
        if len(tokens(content)) < 3:
            continue
        words = content.split()
        if len(words) <= max_words:
            result.append({'source': source, 'page': int(slide['number']), 'text': content})
            continue
        step = max(10, max_words - 12)
        for pos in range(0,len(words),step):
            piece = ' '.join(words[pos:pos + max_words])
            if len(tokens(piece)) >= 3:
                result.append({'source':source, 'page':int(slide['number']), 'text':piece})
            if pos + max_words >= len(words):
                break
    return result


def query_text(text, model_name):
    if 'bge-' in model_name.lower():
        return 'Represent this sentence for searching relevant passages: ' + text
    return text


def build_encoder(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode(encoder, texts, *, model_name='', is_query=False):
    texts = [query_text(str(t), model_name) if is_query else str(t) for t in texts]
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = np.asarray(encoder.encode(texts, normalize_embeddings=True,
                                        batch_size=64, show_progress_bar=False), dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    # Keep the code robust for simple injected encoders in automated tests.
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(denom, 1e-12)


def load_slide_evidence(path):
    """User-reviewed per-MP4 evidence only. Never auto-assign PDF content."""
    p = Path(path)
    if not p.exists():
        return {}
    content = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(content, dict):
        raise ValueError('Verified slide text must be a JSON object mapping MP4 paths to text.')
    return {str(k).replace('\\', '/'): str(v) for k, v in content.items() if isinstance(v, str) and v.strip()}


def stable_digest(records, model_name, slide_evidence):
    raw = json.dumps({'version': INDEX_VERSION, 'model': model_name,
                      'videos': [(r['id'], r['title'], r['subject'], r['topic']) for r in records],
                      'slide_evidence': sorted(slide_evidence.items())},
                     ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def _atomic_save_npy(path, arr):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, arr, allow_pickle=False)
    tmp.replace(path)


def prepare_index(records, *, cache_dir, model_name, encoder, slide_evidence=None,
                  force=False):
    """Load reusable vectors if complete and correct; else encode and persist."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    evidence = slide_evidence or {}
    digest = stable_digest(records, model_name, evidence)
    paths = {name: root / name for name in ('videos.json', 'title_vectors.npy', 'context_vectors.npy',
                                           'slide_vectors.npy', 'index_config.json')}
    expected_slide_rows = sum(bool(evidence.get(r['id'])) for r in records)
    meta = None
    if not force and all(path.exists() for path in paths.values()):
        try:
            meta = json.loads(paths['index_config.json'].read_text(encoding='utf-8'))
            if meta.get('digest') == digest:
                titles = np.load(paths['title_vectors.npy'], allow_pickle=False)
                contexts = np.load(paths['context_vectors.npy'], allow_pickle=False)
                slides = np.load(paths['slide_vectors.npy'], allow_pickle=False)
                if (titles.shape[0] == contexts.shape[0] == slides.shape[0] == len(records)
                        and titles.ndim == contexts.ndim == slides.ndim == 2
                        and titles.shape[1] == contexts.shape[1] == slides.shape[1]
                        and meta.get('slide_count') == expected_slide_rows):
                    return {'records': json.loads(paths['videos.json'].read_text(encoding='utf-8')),
                            'title_vectors': titles, 'context_vectors': contexts,
                            'slide_vectors': slides, 'model_name': model_name,
                            'digest': digest, 'from_cache': True}
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            pass

    if not records:
        raise ValueError('No eligible MP4 videos exist in the workbook.')
    title_text = [r['title'] for r in records]
    context_text = [f"{strip_number(r['subject'])}. {strip_number(r['topic'])}. {r['title']}" for r in records]
    title_vectors = encode(encoder, title_text, model_name=model_name)
    context_vectors = encode(encoder, context_text, model_name=model_name)
    slide_vectors = np.zeros_like(title_vectors)
    slide_ids = [i for i, r in enumerate(records) if evidence.get(r['id'])]
    if slide_ids:
        slide_data = [evidence[records[i]['id']] for i in slide_ids]
        slide_vectors[slide_ids] = encode(encoder, slide_data, model_name=model_name)
    enriched = []
    for i, row in enumerate(records):
        has_slide = bool(evidence.get(row['id']))
        enriched.append({**row,
                         'slide_text': evidence.get(row['id'], ''),
                         'title_embedding': i,
                         'context_embedding': i,
                         'slide_embedding': i if has_slide else None})
    _atomic_save_npy(paths['title_vectors.npy'], title_vectors)
    _atomic_save_npy(paths['context_vectors.npy'], context_vectors)
    _atomic_save_npy(paths['slide_vectors.npy'], slide_vectors)
    _atomic_write_json(paths['videos.json'], enriched)
    _atomic_write_json(paths['index_config.json'], {'digest': digest, 'version': INDEX_VERSION,
                                                    'model': model_name, 'video_count': len(records),
                                                    'slide_count': len(slide_ids),
                                                    'title_embedding': 'title_vectors.npy',
                                                    'context_embedding': 'context_vectors.npy',
                                                    'slide_embedding': 'slide_vectors.npy'})
    return {'records': enriched, 'title_vectors': title_vectors,
            'context_vectors': context_vectors, 'slide_vectors': slide_vectors,
            'model_name': model_name, 'digest': digest, 'from_cache': False}


class BM25Index:
    """BM25 over title + folder context. Uses rank_bm25 when available.

    Formula fallback keeps unit tests and the app usable without rank_bm25, but
    rank_bm25 is included in requirements and preferred in deployments.
    """
    def __init__(self, docs):
        self.docs = [tokens(x) for x in docs]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.count = len(self.docs)
        self.df = Counter(t for row in self.docs for t in set(row))
        self.freq = [Counter(row) for row in self.docs]
        self.engine = None
        try:
            from rank_bm25 import BM25Okapi
            self.engine = BM25Okapi(self.docs)
        except ImportError:
            pass

    def scores(self, query):
        words = tokens(query)
        if not words:
            return np.zeros(self.count, dtype=np.float32)
        if self.engine is not None:
            return np.maximum(0, np.asarray(self.engine.get_scores(words), dtype=np.float32))
        # Standard nonnegative-IDF BM25 approximation; vocabulary IDF learned from MP4s.
        total = self.count
        values = np.zeros(total, dtype=np.float32)
        for word in set(words):
            df = self.df.get(word, 0)
            if not df:
                continue
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for i, row in enumerate(self.freq):
                tf = row.get(word, 0)
                if tf:
                    length_factor = 1.2 * (0.25 + 0.75 * len(self.docs[i]) / max(self.avgdl, 1))
                    values[i] += idf * tf * 2.2 / (tf + length_factor)
        return values


class HybridMatcher:
    def __init__(self, index, encoder):
        self.index = index
        self.encoder = encoder
        self.records = index['records']
        # A title gets more lexical weight than its folder, but both are searchable.
        self.bm25 = BM25Index([f"{r['title']} {r['title']} {strip_number(r['topic'])} {strip_number(r['subject'])}"
                                for r in self.records])
        self.vocab = [set(tokens(f"{r['title']} {r['topic']} {r['subject']}")) for r in self.records]
        self.slide_mask = np.array([r['slide_embedding'] is not None for r in self.records])

    def query_scores(self, chunks, *, dense_weight=.70, title_share=.75, slide_weight=.30):
        texts = [c['text'] for c in chunks]
        if not texts:
            count = len(self.records)
            return (np.zeros((0, count), np.float32),) * 3
        queries = encode(self.encoder, texts, model_name=self.index['model_name'], is_query=True)
        title_sim = queries @ self.index['title_vectors'].T
        context_sim = queries @ self.index['context_vectors'].T
        dense = title_share * title_sim + (1 - title_share) * context_sim
        if self.slide_mask.any() and slide_weight > 0:
            slide_sim = queries @ self.index['slide_vectors'].T
            dense[:, self.slide_mask] = ((1 - slide_weight) * dense[:, self.slide_mask]
                                         + slide_weight * slide_sim[:, self.slide_mask])
        # Raw cosine [-1, 1] is retained in evidence. A bounded monotonic transform
        # makes semantic-only paraphrase retrieval feasible when lexical overlap is
        # zero. 0.10 and 0.50 are starting calibration values, NOT probabilities.
        dense_scaled = np.clip((dense - 0.10) / 0.50, 0, 1)
        lex = np.vstack([self.bm25.scores(q) for q in texts])
        # Scale BM25 per query while explicitly tracking whether it had any evidence.
        # Dynamic normalization is for ranking, not a calibrated probability.
        vmax = np.max(lex, axis=1, keepdims=True)
        lex_scaled = np.divide(lex, vmax, out=np.zeros_like(lex), where=vmax > 0)
        mixed = dense_weight * dense_scaled + (1 - dense_weight) * lex_scaled
        return mixed, dense, lex_scaled

    def search(self, chunks, *, limit=20, top_k=15, threshold=.42,
               dense_weight=.70, title_share=.75, slide_weight=.30,
               pdf_evidence=None, pdf_weight=.35, reranker=None,
               context_min_score=.45, max_context_pairs=320):
        if not chunks:
            return []
        mixed, dense, bm25 = self.query_scores(chunks, dense_weight=dense_weight,
                                               title_share=title_share, slide_weight=slide_weight)
        # PDF evidence is an independent retrieval channel, not a fractional
        # multiplier on a possibly weak title match. The old calculation
        # ``(1-w)*title + w*pdf`` at w=.35 could turn excellent PDF=0.9 into
        # 0.315, below the default .42 cutoff, making all PDF evidence invisible.
        #
        # Treat a sufficiently strong verified PDF match as its own candidate.
        # The slider controls the *corroboration* bonus when title AND PDF agree.
        # Never punish a video for having no indexed PDF.
        pdf_scores, pdf_witnesses = None, {}
        pdf_indexed = np.zeros(len(self.records), dtype=bool)
        title_scores = mixed.copy()
        effective_pdf = np.zeros_like(mixed)
        if pdf_evidence is not None and pdf_weight > 0:
            pdf_indexed = np.asarray(pdf_evidence.video_chunk_counts > 0, dtype=bool)
            qvec = encode(self.encoder, [c['text'] for c in chunks],
                          model_name=self.index['model_name'], is_query=True)
            pdf_scores, pdf_witnesses = pdf_evidence.query(
                [c['text'] for c in chunks], qvec, dense_weight=dense_weight,
                witness_limit=2 if reranker is not None else 1)
            # Reject trivial page similarities, including generic shared words.
            # This is a retrieval cutoff, NOT a calibrated medical probability.
            pdf_floor = max(0.50, threshold)
            effective_pdf = np.where(pdf_scores >= pdf_floor, pdf_scores, 0.0)
            independent = np.maximum(mixed, effective_pdf)
            agreement = np.minimum(mixed, effective_pdf)
            mixed = np.clip(independent + pdf_weight * agreement * (1.0-independent), 0.0, 1.0)

        seen = defaultdict(list)
        for i, chunk in enumerate(chunks):
            row = mixed[i]
            take = min(len(self.records), max(1, int(top_k)))
            # Candidate pool comes from BOTH title and verified PDF evidence;
            # PDFs are not allowed to be pruned by title-only top-k selection.
            title_ids = np.argpartition(title_scores[i], -take)[-take:]
            pdf_ids = (np.argpartition(effective_pdf[i], -take)[-take:]
                       if pdf_scores is not None else [])
            inds = sorted(set(map(int,title_ids)) | set(map(int,pdf_ids)),
                          key=lambda j:-float(row[j]))
            for j in inds:
                score = float(row[j])
                if score < threshold:
                    continue
                shared = sorted(set(tokens(chunk['text'])) & self.vocab[j])
                pdf_info = pdf_witnesses.get((i,j), {})
                pdf_score = float(pdf_scores[i,j]) if pdf_scores is not None and pdf_indexed[j] else None
                validated_info = pdf_info if effective_pdf[i,j] > 0 else {}
                seen[j].append({'chunk_id':i, 'source': chunk['source'], 'page': chunk['page'],
                                'text': chunk['text'][:280],
                                'score': round(score,4),
                                'title_score': round(float(title_scores[i,j]),4),
                                'cosine_similarity': round(float(dense[i,j]),4),
                                'bm25_relative': round(float(bm25[i,j]),4),
                                'shared_terms': shared[:12],
                                'pdf_indexed': bool(pdf_indexed[j]),
                                'pdf_page': validated_info.get('page'),
                                'pdf_id': validated_info.get('pdf_id',''),
                                'pdf_excerpt': validated_info.get('text','')[:500],
                                'pdf_alternatives':validated_info.get('alternatives') or
                                    ([{'pdf_id':validated_info.get('pdf_id',''),
                                       'page':validated_info.get('page'),
                                       'text':validated_info.get('text','')}]
                                     if validated_info else []),
                                'pdf_evidence_score': round(pdf_score,4) if pdf_score is not None else None,
                                'pdf_supported': bool(validated_info),
                                'match_source': ('title + PDF' if validated_info and title_scores[i,j]>=threshold
                                                 else 'verified PDF pages' if validated_info
                                                 else 'MP4 title / folder')})
        results = []
        for j, evidence in seen.items():
            ev = sorted(evidence, key=lambda e: -e['score'])
            independent = len({(e['source'], e['page']) if e['page'] is not None else
                               (e['source'], e['text']) for e in evidence})
            best = ev[0]['score']
            mean_top = sum(e['score'] for e in ev[:5]) / min(len(ev), 5)
            # Diverse, independently matching chunks help, but cannot rescue low
            # evidence (all supporting chunks must already pass the cutoff).
            rank = 0.78 * best + 0.22 * mean_top + min(0.06, 0.02 * max(0, independent - 1))
            row = self.records[j]
            # Preserve the strongest *actual* PDF witness independently from
            # the top-eight displayed lecture matches (which may be title-only).
            pdf_ev = max((e for e in evidence if e['pdf_supported']),
                         key=lambda e:e['pdf_evidence_score'], default=None)
            measured_pdf = [e['pdf_evidence_score'] for e in evidence
                            if e['pdf_evidence_score'] is not None]
            results.append({**row, 'score': round(rank,4),
                            'best_chunk_score': best,
                            'pdf_indexed':bool(pdf_indexed[j]),
                            'pdf_match_score':max(measured_pdf,default=None),
                            'pdf_match_page':pdf_ev['pdf_page'] if pdf_ev else None,
                            'pdf_match_file':pdf_ev['pdf_id'] if pdf_ev else '',
                            'pdf_match_excerpt':pdf_ev['pdf_excerpt'] if pdf_ev else '',
                            'pdf_match_source':('PDF evidence' if pdf_ev else
                                               'No matching PDF evidence' if pdf_indexed[j] else
                                               'No indexed PDF teaching pages'),
                            'matching_chunks':len(evidence),
                            'matching_slides':len({(e['source'],e['page']) for e in evidence if e['page'] is not None}),
                            'evidence':ev if reranker is not None else ev[:8],
                            'score_type':'hybrid rank (not coverage or probability)'})
        results.sort(key=lambda r: (r['score'], r['matching_chunks'], r['best_chunk_score']), reverse=True)
        if reranker is not None:
            from context_reranker import contextual_rank
            return contextual_rank(candidates=results, chunks=chunks,
                                   encoder=self.encoder, model_name=self.index['model_name'],
                                   reranker=reranker, min_context=context_min_score,
                                   max_pairs=max_context_pairs, max_results=limit)
        return results[:limit]
