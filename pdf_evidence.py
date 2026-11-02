"""Conservative MP4->PDF association, title-verified page anchors, and optional OCR.

An MP4 path is always the ID. Folder association is not slide verification.
Only exact per-video verified anchors (or reviewed CSV ranges) produce searchable text.
No inferred video IDs, no whole-PDF assignment, no unvalidated bookmark ranges.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from beta_catalog import records_for_search, resolve_pdf_library, folder_candidates
from board_exclusions import is_board_breakdown, page_is_board_breakdown
from file_index_catalog import clean_name
from semantic_matcher import BM25Index, encode, tokens, _atomic_save_npy, _atomic_write_json

SCAN_VERSION = 3
PDF_INDEX_VERSION = 1


def clean_title(value):
    value = clean_name(str(value or ''))
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', value.casefold()).split())


def title_similarity(a, b):
    """Strong title agreement, not generic concept overlap."""
    a, b = clean_title(a), clean_title(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    coverage = len(ta & tb) / max(1, len(tb))
    ratio = SequenceMatcher(None, a, b).ratio()
    if len(tb) >= 2 and len(ta) >= 2 and coverage >= .85 and ratio >= .72:
        return max(.86, ratio)
    if ratio >= .88 and min(len(a), len(b)) >= 8:
        return ratio
    return 0.0


def readable(text):
    """Reject corrupt Unicode cmaps that otherwise pollute embeddings with noise."""
    text = str(text or '')
    letters = [c for c in text if c.isalpha()]
    ascii_letters = sum('a' <= c.lower() <= 'z' for c in letters)
    return len(letters) >= 22 and ascii_letters / len(letters) >= .80 and len(re.findall(r'[A-Za-z]{3,}', text)) >= 5


def ocr_text(page):
    """PyMuPDF/Tesseract, only if user opts in. Never silently invent recovered text."""
    tp = page.get_textpage_ocr(language='eng', dpi=125, full=True)
    return page.get_text(textpage=tp)


def page_lines(page, use_ocr=False):
    native = page.get_text(sort=True)
    if not use_ocr or (readable(native) and not corrupt_heading(native)):
        return native, 'native'
    try:
        recovered = ocr_text(page)
        return (recovered, 'OCR') if readable(recovered) else (native, 'unreadable')
    except Exception as exc:
        return native, 'OCR unavailable: '+str(exc)[:100]



def corrupt_heading(text):
    """A page can contain readable body text but have unreadable PDF-font headings.

    Detect the characteristic mixed-script/corrupt-Unicode headers; do not apply
    this heuristic to the full body of otherwise useful teaching material.
    """
    first = [x.strip() for x in str(text).splitlines() if x.strip()][:5]
    for line in first:
        letters = [c for c in line if c.isalpha()]
        if len(letters) < 7:
            continue
        nonlatin = sum('LATIN' not in unicodedata.name(c, '') for c in letters)
        if nonlatin / len(letters) >= .23:
            return True
    return False


def clean_native_text(text):
    """Remove heavily garbled glyph lines, while retaining readable PDF body text."""
    lines = []
    for line in str(text).splitlines():
        letters = [c for c in line if c.isalpha()]
        if len(letters) >= 6:
            nonlatin = sum('LATIN' not in unicodedata.name(c, '') for c in letters)
            if nonlatin / len(letters) >= .23:
                continue
        lines.append(line)
    return '\n'.join(lines)


def early_heading(text):
    """Short early lines, not arbitrary body mentions or whole table-of-contents pages."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return [l for l in lines[:12] if 3 <= len(l) <= 130]


def is_front_page(text):
    head = ' '.join(text.splitlines()[:12]).lower()
    return bool(re.search(r'\btable of contents\b|\btable of content\b', head))


def check_heading(title, text):
    lines = early_heading(text)
    best = max((title_similarity(line, title) for line in lines), default=0.)
    # Composite titles may be rearranged between MP4 and its cover heading.
    # Require ALL parts of a hyphen-separated title to appear as separate
    # headings on the same cover, not just one generic matching word.
    parts = [x.strip() for x in re.split(r'\s+[-–—]\s+', title) if x.strip()]
    if len(parts) >= 2 and all(any(title_similarity(line, part) >= .90 for line in lines)
                               for part in parts):
        best = max(best, .94)
    return best


def bookmark_heading_support(title, text):
    """Conservative OCR corroboration for a *matching* native PDF bookmark.

    More tolerant of broken OCR line wrapping than title-only detection;
    never called if no matching bookmark with a valid destination exists.
    """
    heading = ' '.join(early_heading(text)[:18])
    title_terms = [t for t in clean_title(title).split() if len(t) >= 3]
    head_terms = set(clean_title(heading).split())
    if not title_terms:
        return 0.0
    matched = len(set(title_terms) & head_terms)
    needed = 1 if len(title_terms) == 1 else max(2, int(len(title_terms)*.75+0.999))
    if matched >= needed:
        return matched/len(set(title_terms))
    return 0.0


def number_of(record):
    match = re.match(r'^\s*(\d{1,3})\s*[.)-]', record.get('filename',''))
    return int(match.group(1)) if match else None


def manual_ranges(path):
    """Only explicit user-reviewed exact path mappings; never applied by filename alone."""
    if not Path(path).exists():
        return {}
    result = {}
    with open(path, encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            if not row.get('MP4 path') or not row.get('PDF path'):
                continue
            try:
                start, end = int(row['Start page']), int(row['End page'])
            except (TypeError, ValueError, KeyError):
                continue
            if start >= 1 and start <= end:
                result[row['MP4 path'].replace('\\', '/')] = {
                    'pdf_id': row['PDF path'], 'start': start, 'end': end}
    return result


def _contents(doc, use_ocr):
    # Read full native text to locate actual title slides; OCR is only on likely anchors
    texts = [p.get_text(sort=True) for p in doc]
    return texts


def scan_document(path, vids, pdf_id, use_ocr=False, overrides=None, anchor_ocr=True):
    import fitz
    overrides = overrides or {}
    with fitz.open(str(path)) as doc:
        n = len(doc)
        toc = [(str(title), int(page)) for _, title, page in doc.get_toc() if 1 <= page <= n]
        texts = _contents(doc, use_ocr)
        original_texts = list(texts)
        toc_pages = {i+1 for i, t in enumerate(texts[:4]) if is_front_page(t)}
        # A page-1 bookmark can be a legitimate lesson divider. Only an
        # actual contents page is ignored; do not blanket-exclude page 1.
        heads = [early_heading(t) for t in texts]
        # Bookmarks are location *candidates*, not proof. Even when their labels
        # differ from MP4 titles (e.g. 'Limbs' vs 'Limb Development'), render
        # each candidate divider with OCR and verify its actual displayed title.
        # Cache OCR once per page rather than once per video.
        bookmark_pages = sorted({pg for _, pg in toc if pg not in toc_pages})
        cover_ocr = {}
        cover_ocr_failed = 0
        cover_ocr_verified = 0
        if anchor_ocr:
            for pg in bookmark_pages:
                # Native headings already provide sufficient evidence; no OCR needed.
                if any(check_heading(v['title'], texts[pg-1]) >= .88 for v in vids):
                    continue
                try:
                    recovered = ocr_text(doc[pg-1])
                except Exception:
                    cover_ocr_failed += 1
                    # Tesseract not installed: one failure predicts the others.
                    break
                if readable(recovered):
                    cover_ocr[pg] = recovered
                    # Count only title labels that truly agree with the OCR page.
                    if any(check_heading(v['title'], recovered) >= .88 for v in vids):
                        cover_ocr_verified += 1
                        texts[pg-1] = recovered
                        heads[pg-1] = early_heading(recovered)
        titles = {v['path'].replace('\\','/'): v['title'] for v in vids}
        rows = {}
        anchors = {}
        for video in vids:
            vid = video['path'].replace('\\','/')
            title = titles[vid]
            if is_board_breakdown(title):
                continue
            manual = overrides.get(vid)
            if manual and manual['pdf_id'] == pdf_id and 1 <= manual['start'] <= manual['end'] <= n:
                anchors[vid] = (manual['start'], manual['end'], 'reviewed manual range', 1.0)
                continue
            # Heading evidence is checked separately from bookmarks. An invalid
            # bookmark (often pointing to page 1) cannot authorize a range.
            book = sorted([(title_similarity(btitle, title), p)
                           for btitle,p in toc if p not in toc_pages and p > 1
                           and title_similarity(btitle, title) >= .85], reverse=True)
            strong = []
            for pi, lines in enumerate(heads, 1):
                if pi in toc_pages:
                    continue
                best = check_heading(title, texts[pi-1])
                if best >= .88:
                    source = 'OCR-confirmed heading' if pi in cover_ocr else 'page heading'
                    strong.append((best, pi, source))
            # Prevent body mentions and large TOCs from self-certifying. If OCR
            # enabled, verify only existing bookmark targets lacking native headings.
            if use_ocr and book:
                for _, pg in book[:3]:
                    if pg in toc_pages or any(p == pg for _,p,_ in strong):
                        continue
                    try:
                        # A readable body can coexist with a corrupt *heading*;
                        # explicitly OCR this bookmark target to verify it.
                        t = ocr_text(doc[pg-1])
                    except Exception:
                        continue
                    if readable(t):
                        s = max(check_heading(title, t), bookmark_heading_support(title, t))
                        if s >= .75:
                            strong.append((s,pg,'OCR title heading'))
                            texts[pg-1] = t
            # Choose first high-confidence title heading, prioritizing matched
            # bookmark destinations when both agree.
            paired = [(s,pg,m) for s,pg,m in strong
                      if any(abs(pg-bpg) <= 1 for _,bpg in book)
                      or (pg in bookmark_pages and pg in cover_ocr)]
            chosen = sorted(paired or strong, key=lambda x: (-x[0], x[1]))
            if chosen:
                s,pg,m = chosen[0]
                anchors[vid] = (pg,pg,'bookmark + '+m if paired else m,round(s,3))
            else:
                rows[vid] = {'status':'PDF associated; page title unverified',
                             'start':None,'end':None,'method':'', 'confidence':0.0}
        # A page that seems to be the start of two different titles is ambiguous.
        by_page = defaultdict(list)
        for vid,(start,end,method,confidence) in anchors.items():
            if method != 'reviewed manual range':
                by_page[start].append(vid)
        for page, ids in by_page.items():
            if len(ids) > 1:
                for vid in ids:
                    anchors.pop(vid)
                    rows[vid] = {'status':'Ambiguous title heading', 'start':None,
                                 'end':None,'method':'','confidence':0.0}
        ordered = sorted(vids, key=lambda v: (number_of(v) is None,number_of(v) or 9999, v['title']))
        # Do not sweep through undetected videos. Expand only when the next
        # *adjacent* video has a confirmed start (a missing video blocks expansion).
        for idx,video in enumerate(ordered):
            vid=video['path'].replace('\\','/')
            if vid not in anchors:
                continue
            start,end,method,confidence=anchors[vid]
            if method != 'reviewed manual range':
                if idx+1 < len(ordered):
                    following = ordered[idx+1]['path'].replace('\\','/')
                    # Subject-level consolidated handouts may list videos from
                    # many *different* folders. Never expand across them.
                    same_topic = video.get('topic') == ordered[idx+1].get('topic')
                    consecutive = (number_of(video) is not None and
                                   number_of(ordered[idx+1]) == number_of(video)+1)
                    if same_topic and consecutive and following in anchors and anchors[following][0] > start:
                        end = anchors[following][0]-1
                else:
                    # The final video can run to the end if no later bookmark
                    # announces an unrepresented section. Stop before known
                    # later sections instead of stealing their material.
                    later = [p for p in bookmark_pages if p > start]
                    end = min(later) - 1 if later else n
                # Otherwise only the evidenced start page is assigned.
            rows[vid] = {'status':'Verified page range' if end > start else 'Verified start page only',
                         'start':start,'end':end,'method':method,'confidence':confidence}
        # Exclude board-style title or reference headings from content, even
        # inside a reviewed range.
        chunks=[]; bad_pages=0; used_ocr=0
        for video in vids:
            vid = video['path'].replace('\\','/')
            entry=rows.get(vid, {})
            if not entry.get('start'):
                continue
            for pg in range(entry['start'], entry['end']+1):
                if pg in toc_pages:
                    continue
                text = texts[pg-1]
                if use_ocr and (not readable(text) or corrupt_heading(text)):
                    text,source=page_lines(doc[pg-1], True)
                    if source == 'OCR': used_ocr += 1
                if not readable(text):
                    bad_pages += 1
                    continue
                if page_is_board_breakdown(text) or is_front_page(text) or re.search(r'^\s*REVIEW\s*\n\s*OUTLINE\b',text,re.I):
                    continue
                # Repeated recap outline pages should not artificially inflate evidence.
                if pg != entry['start'] and original_texts[pg-1].strip() == original_texts[entry['start']-1].strip():
                    continue
                # Remove garbled headings and sources while retaining medical text.
                text = clean_native_text(text)
                lines = [line.strip() for line in text.splitlines()
                         if line.strip() and not line.strip().startswith(('http://','https://'))]
                clean = ' '.join(lines)
                terms = clean.split()
                for i in range(0,len(terms),95):
                    segment=' '.join(terms[i:i+115])
                    if len(segment) > 65 and len(segment.split()) >= 9:
                        chunks.append({'video_id':vid,'pdf_id':pdf_id,'page':pg,
                                       'text':segment[:1800]})
                    if i+115 >= len(terms):break
        return rows, chunks, {'pages':n,'unreadable_pages_in_mapped_ranges':bad_pages,
                              'ocr_pages':used_ocr,'toc_pages':sorted(toc_pages),
                              'anchor_ocr_verified':cover_ocr_verified,
                              'anchor_ocr_failed':cover_ocr_failed}


def tag_library(inventory, *, root, cache_dir, use_ocr=False, anchor_ocr=True, progress=None):
    """Tag folder PDFs, confirm per-video anchors, create cached chunk text.

    Returns manifest. Does not require embedding/model download. When the PDF is
    absent, the manifest still records the inventory candidate accurately.
    """
    path=Path(cache_dir);path.mkdir(parents=True, exist_ok=True)
    videos=records_for_search(inventory)
    pdfs=[r for r in inventory if r['type']=='PDF']
    resolved, discovery_summary, extra_local_files = resolve_pdf_library(inventory, root or None)
    override=manual_ranges(path/'manual_pdf_ranges.csv')
    by_pdf=defaultdict(list)
    manifest={'version':SCAN_VERSION, 'root':str(root), 'use_ocr':bool(use_ocr),'anchor_ocr':bool(anchor_ocr),
              'videos':{},'pdfs':{},'unresolved':[], 'counts':{},
              'discovery':discovery_summary, 'unlisted_local_pdf_files':extra_local_files}
    for video in videos:
        vid=video['path'].replace('\\','/')
        options=folder_candidates(video,inventory)
        candidate = options[0] if len(options) == 1 else None
        candidate_state = resolved.get(candidate['path'],{}) if candidate else {}
        manifest['videos'][vid]={'title':video['title'],'subject':video['subject'],
                                 'topic':video['topic'], 'pdf_candidates':[p['path'] for p in options],
                                 'pdf_id':None,
                                 'local_pdf_path':candidate_state.get('physical',''),
                                 'status': ('PDF found by filename; pages not yet checked'
                                            if candidate_state.get('physical') else
                                            candidate_state.get('status','PDF listed in inventory but unavailable')
                                            if candidate else 'No corresponding PDF in inventory'),
                                 'start':None,'end':None,'method':'','confidence':0.0,
                                 'text_chunks':0}
        if len(options)==1:
            by_pdf[options[0]['path']].append(video)
        elif len(options)>1:
            manifest['videos'][vid]['status']='Multiple PDFs in folder; needs review'
    all_chunks=[]
    scan_cache_file=path/'pdf_scan_cache.json'
    try:
        old_cache=json.loads(scan_cache_file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        old_cache={}
    fresh_cache={}
    for i,pdf in enumerate(pdfs):
        pdfid=pdf['path']
        resolution=resolved.get(pdfid,{})
        physical=resolution.get('physical','')
        info={'subject':pdf['subject'],'topic':pdf['topic'],'filename':pdf['filename'],
              'inventory_path':pdfid,'local_path':physical,
              'lookup_status':resolution.get('status','PDF not accessible'),
              'status':resolution.get('status','PDF not accessible')}
        manifest['pdfs'][pdfid]=info
        vids=by_pdf.get(pdfid,[])
        if progress: progress(i+1,len(pdfs),pdf['filename'])
        if not physical or not vids:
            if not physical:manifest['unresolved'].append(pdfid)
            continue
        try:
            state=Path(physical).stat()
            manual=[(v['path'],override.get(v['path'].replace('\\','/'))) for v in vids]
            fingerprint=hashlib.sha256(json.dumps([SCAN_VERSION,str(Path(physical).resolve()),state.st_size,
                       state.st_mtime_ns,use_ocr,anchor_ocr,manual],sort_keys=True).encode()).hexdigest()
            old=old_cache.get(pdfid,{})
            if fingerprint==old.get('fingerprint'):
                rows, chunks, stats=old['rows'],old['chunks'],old['stats']
            else:
                rows,chunks,stats=scan_document(physical,vids,pdfid,use_ocr,override,anchor_ocr)
            fresh_cache[pdfid]={'fingerprint':fingerprint,'rows':rows,'chunks':chunks,'stats':stats}
            info.update(stats)
            info['status']='PDF opened and analyzed'
            for video in vids:
                vid=video['path'].replace('\\','/')
                item=manifest['videos'][vid]
                item['pdf_id']=pdfid
                item.update(rows.get(vid,{'status':'PDF associated; page title unverified',
                                          'start':None,'end':None,'method':'','confidence':0.0}))
            counter=Counter(c['video_id'] for c in chunks)
            for vid,count in counter.items():manifest['videos'][vid]['text_chunks'] = count
            all_chunks.extend(chunks)
        except Exception as exc:
            info['status']='PDF scan failed: '+str(exc)[:160]
            manifest['unresolved'].append(pdfid)
    manifest['counts']=dict(Counter(v['status'] for v in manifest['videos'].values()))
    _atomic_write_json(path/'pdf_tags.json',manifest)
    _atomic_write_json(path/'pdf_chunks.json',all_chunks)
    _atomic_write_json(scan_cache_file,fresh_cache)
    return manifest


def load_manifest(cache_dir):
    path=Path(cache_dir)/'pdf_tags.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def pdf_chunk_digest(chunks, model):
    content=json.dumps([PDF_INDEX_VERSION,model,[(c['video_id'],c['pdf_id'],c['page'],c['text']) for c in chunks]],
                       ensure_ascii=False).encode()
    return hashlib.sha256(content).hexdigest()


def build_pdf_vectors(cache_dir, encoder, model_name, *, progress=None):
    """Separate per-page-chunk vectors, never re-encode 2,323 titles for PDF changes."""
    root=Path(cache_dir)
    chunks=json.loads((root/'pdf_chunks.json').read_text(encoding='utf-8'))
    digest=pdf_chunk_digest(chunks,model_name)
    config_file=root/'pdf_index_config.json'
    try:
        config=json.loads(config_file.read_text(encoding='utf-8'))
    except (ValueError,OSError):
        config={}
    vec_file=root/'pdf_chunk_vectors.npy'
    if config.get('digest')==digest and vec_file.exists():
        vec=np.load(vec_file,allow_pickle=False)
        if len(vec)==len(chunks):
            return len(chunks),True
    if not chunks:
        if vec_file.exists():vec_file.unlink()
        _atomic_write_json(config_file, {'digest':digest,'model':model_name,'chunks':0})
        return 0,False
    # Batch across PDFs, avoids putting huge PDFs into one model forward pass.
    vectors=[]
    for start in range(0,len(chunks),128):
        if progress:progress(start,len(chunks),'Encoding PDF teaching chunks')
        vectors.append(encode(encoder,[x['text'] for x in chunks[start:start+128]], model_name=model_name))
    _atomic_save_npy(vec_file,np.concatenate(vectors).astype('float32'))
    _atomic_write_json(config_file,{'digest':digest,'model':model_name,'chunks':len(chunks)})
    return len(chunks),False


class PDFChunkEvidence:
    """Vector+BM25 retrieval across verified teaching pages, joined by MP4 ID."""
    def __init__(self, cache_dir, model_name, video_ids):
        root=Path(cache_dir)
        self.chunks=json.loads((root/'pdf_chunks.json').read_text(encoding='utf-8'))
        cfg=json.loads((root/'pdf_index_config.json').read_text(encoding='utf-8'))
        if cfg.get('model')!=model_name or cfg.get('digest')!=pdf_chunk_digest(self.chunks,model_name):
            raise ValueError('PDF evidence vectors are missing or stale for this embedding model. Build PDF evidence index first.')
        if not self.chunks:
            self.vectors=np.zeros((0,0),dtype='float32')
        else:
            self.vectors=np.load(root/'pdf_chunk_vectors.npy',allow_pickle=False)
            if self.vectors.shape[0]!=len(self.chunks):
                raise ValueError('PDF evidence vectors/chunks do not align. Rebuild PDF index.')
        lookup={x.replace('\\','/'):i for i,x in enumerate(video_ids)}
        self.positions=np.array([lookup.get(c['video_id'],-1) for c in self.chunks],dtype=int)
        self.bm25=BM25Index([c['text'] for c in self.chunks]) if self.chunks else None
        self.video_count=len(video_ids)
        # Evidence coverage is recorded independently of the retrieval score.
        # A zero similarity is not the same thing as an unindexed video.
        self.video_chunk_counts=np.bincount(self.positions[self.positions>=0],
                                            minlength=self.video_count)

    def query(self, queries, query_vectors, dense_weight=.70, witness_limit=1):
        """Return per-video best chunk scores and factual cited PDF page witnesses."""
        scores=np.zeros((len(queries),self.video_count),dtype=np.float32)
        witnesses={}
        if not self.chunks:return scores,witnesses
        raw=query_vectors @ self.vectors.T
        scaled=np.clip((raw-.10)/.50,0,1)
        lex=np.vstack([self.bm25.scores(q) for q in queries])
        vmax=np.max(lex,axis=1,keepdims=True)
        lexical=np.divide(lex,vmax,out=np.zeros_like(lex),where=vmax>0)
        hybrid=dense_weight*scaled+(1-dense_weight)*lexical
        # A global best chunk can be shared by multiple videos only when it has
        # separately verified provenance for each of those video IDs.
        by_video=defaultdict(list)
        for j,video_idx in enumerate(self.positions):
            if video_idx>=0:by_video[int(video_idx)].append(j)
        for video_idx,indices in by_video.items():
            sub=hybrid[:,indices]
            best=sub.argmax(axis=1)
            for qi,local in enumerate(best):
                j=indices[int(local)]
                s=float(hybrid[qi,j])
                if s<=0: continue
                scores[qi,video_idx]=s
                c=self.chunks[j]
                row={'pdf_id':c['pdf_id'],'page':c['page'],
                     'text':c['text'][:1000], 'score':round(s,4),
                     'cosine':round(float(raw[qi,j]),4)}
                if witness_limit > 1:
                    # Pick several verified chunks from THIS video, never from a
                    # neighboring lesson. The cross-encoder chooses the best
                    # context match rather than trusting the first lexical hit.
                    top = np.argsort(-sub[qi])[:min(witness_limit,len(indices))]
                    row['alternatives']=[{'pdf_id':self.chunks[indices[int(k)]]['pdf_id'],
                                         'page':self.chunks[indices[int(k)]]['page'],
                                         'text':self.chunks[indices[int(k)]]['text'][:1000],
                                         'retrieval_score':round(float(sub[qi,int(k)]),4)}
                                         for k in top]
                witnesses[(qi,video_idx)]=row
        return scores,witnesses
