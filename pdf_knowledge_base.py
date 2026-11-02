"""Persistent PDF-first evidence library. Every readable page is indexed regardless
of MP4 attribution. Verified links, automatic suggestions and unmatched evidence
are kept separate. No MP4 files or remote services required for indexing.

This is a search index, not a claim that a model has learned the PDF corpus.
A downstream LLM is shown only the relevant retrieved passages for each lecture.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from beta_catalog import records_for_search, resolve_pdf_library, folder_candidates
from file_index_catalog import clean_name
from pdf_evidence import (clean_native_text, corrupt_heading, readable,
                          ocr_text, title_similarity, is_front_page)
from board_exclusions import page_is_board_breakdown
from semantic_matcher import tokens, encode

KB_VERSION = 3


def _connection(location):
    location = Path(location)
    location.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(location), timeout=45)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA busy_timeout=45000')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript('''
      CREATE TABLE IF NOT EXISTS documents(
        pdf_id TEXT PRIMARY KEY, filename TEXT NOT NULL, subject TEXT, topic TEXT,
        local_path TEXT NOT NULL, fingerprint TEXT NOT NULL,
        page_count INTEGER NOT NULL, searchable_pages INTEGER NOT NULL,
        ocr_pages INTEGER NOT NULL, unreadable_pages INTEGER NOT NULL, status TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS passages(
        passage_id TEXT PRIMARY KEY, pdf_id TEXT NOT NULL, page INTEGER NOT NULL,
        part INTEGER NOT NULL, heading TEXT NOT NULL, text TEXT NOT NULL,
        extraction TEXT NOT NULL, video_id TEXT NOT NULL DEFAULT '',
        link_status TEXT NOT NULL DEFAULT 'unmatched',
        candidate_video_ids TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY(pdf_id) REFERENCES documents(pdf_id));
      CREATE INDEX IF NOT EXISTS passage_pdf_page ON passages(pdf_id,page);
      CREATE INDEX IF NOT EXISTS passage_video ON passages(video_id);
      CREATE TABLE IF NOT EXISTS reviews(
        pdf_id TEXT NOT NULL, start_page INTEGER NOT NULL, end_page INTEGER NOT NULL,
        video_id TEXT NOT NULL, PRIMARY KEY(pdf_id,start_page,end_page,video_id));
    ''')
    try:
        conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS passage_search USING fts5(passage_id UNINDEXED, text, heading, tokenize="unicode61")')
        conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS catalog_search USING fts5(video_id UNINDEXED, title, folder, tokenize="unicode61")')
    except sqlite3.OperationalError as exc:
        raise RuntimeError('SQLite FTS5 support is required to search the PDF knowledge base.') from exc
    return conn


def _digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()


def _video_candidates(pdf, inventory, videos):
    """Metadata candidates, not proof of content. Use original virtual identity."""
    if pdf is None:
        return []
    pdf_id = pdf['path']
    return [v for v in videos if pdf_id in {p['path'] for p in folder_candidates(v, inventory)}]


def _verified_from_manifest(manifest, pdf_id, physical):
    ranges = []
    if not manifest:
        return ranges
    for vid, meta in manifest.get('videos', {}).items():
        if meta.get('pdf_id') != pdf_id or not meta.get('start') or not meta.get('end'):
            continue
        try:
            if Path(str(meta.get('local_pdf_path',''))).resolve() != Path(physical).resolve():
                continue
            ranges.append((int(meta['start']), int(meta['end']), vid.replace('\\','/')))
        except (ValueError, OSError):
            continue
    return ranges


def _short_heading(native, toc_title):
    lines=[re.sub(r'\s+', ' ',x).strip() for x in native.splitlines() if x.strip()]
    meaningful=[x for x in lines[:10] if 5 <= len(x) <= 125 and
                len(re.findall(r'[A-Za-z]{3,}', x)) >= 2]
    return (meaningful[0] if meaningful else toc_title or '')[:140]


def _split_text(text, max_chars=850):
    # Page locality is retained even when a single page has long paragraphs.
    clean = re.sub(r'[ \t]+',' ',text).strip()
    clean = re.sub(r'\n{3,}', '\n\n',clean)
    if not clean: return []
    parts=[]
    units = re.split(r'(?<=\.)\s+|\n\s*\n',clean)
    buf=''
    for unit in units:
        unit=unit.strip()
        if not unit:continue
        if len(unit)>max_chars:
            if buf:parts.append(buf);buf=''
            words=unit.split()
            segment=''
            for word in words:
                if len(segment)+len(word)+1>max_chars and segment:
                    parts.append(segment);segment=''
                segment=(segment+' '+word).strip()
            if segment:parts.append(segment)
        elif len(buf)+len(unit)+1>max_chars:
            if buf:parts.append(buf)
            buf=unit
        else:
            buf=(buf+' '+unit).strip()
    if buf:parts.append(buf)
    # Short scraps are only indexed with their neighbors; no standalone noise.
    merged=[]
    for p in parts:
        if merged and len(p)<100 and len(merged[-1])+len(p)<max_chars+130:
            merged[-1] += ' '+p
        else:merged.append(p)
    return [x for x in merged if len(x)>=25 and len(re.findall(r'[A-Za-z]{3,}',x))>=4]


def _suggest_links(heading, toc_title, candidates):
    """Suggest only source-heading/title overlaps in the same virtual folder.

    Weighted by how distinctive each word is AMONG that folder's video names.
    This is a proposal for human review, never proof of video content. No
    medical term, lecture-specific rule, or answer key is embedded.
    """
    if not candidates:return []
    from collections import Counter
    def stem(w):
        w=w.lower()
        if w.endswith('ies') and len(w)>5:return w[:-3]+'y'
        if w.endswith('s') and len(w)>4 and not w.endswith('ss'):return w[:-1]
        return w
    def words(s):return set(stem(w) for w in tokens(s))
    title_words={v['path']:words(v['title']) for v in candidates}
    freq=Counter(w for ws in title_words.values() for w in ws)
    def weight(w):return 0.1+math.log((len(candidates)+1)/(freq.get(w,0)+1))
    matches=[]
    for v in candidates:
        tw=title_words[v['path']]
        best=max(title_similarity(heading,v['title']),title_similarity(toc_title,v['title']))
        for label in (heading,toc_title):
            hw=words(label)
            common=tw & hw
            if not common:continue
            title_fraction=sum(weight(w) for w in common)/max(.01,sum(weight(w) for w in tw))
            heading_fraction=sum(weight(w) for w in common)/max(.01,sum(weight(w) for w in hw))
            score=.73*title_fraction+.27*heading_fraction
            # An extremely common shared word such as 'development' does not
            # suffice by itself when all neighbor videos share it.
            distinctive=max((weight(w) for w in common),default=0.)
            if distinctive>=.40:
                best=max(best,score)
        if best>=.57:
            matches.append((best,v['path'].replace('\\','/')))
    matches.sort(reverse=True)
    return [v for _,v in matches[:3]]


def _iter_documents(inventory, root):
    resolved,summary,extra = resolve_pdf_library(inventory,root)
    found=[]
    for pdf in (r for r in inventory if r['type']=='PDF'):
        path=resolved.get(pdf['path'],{}).get('physical')
        if path:
            found.append((pdf['path'],Path(path),pdf))
    for path in extra:
        physical=Path(path)
        # If a local file is extra/ambiguous, index it, but don't imply the
        # name identifies the authoritative inventory PDF.
        uid='unlisted:'+_digest(str(physical.resolve()))[:20]
        found.append((uid,physical,None))
    return found,summary


def build_knowledge_base(inventory, root, cache_dir, *, use_ocr=False,
                         pdf_manifest=None, progress=None, max_pages_per_pdf=None):
    """Incremental FTS indexing. Reuses unchanged files; leaves reviews intact.

    Every readable passage is preserved, even from unlisted files or without
    video boundaries. OCR is opt-in and failures are counted, never hidden.
    """
    if not str(root).strip():
        raise ValueError('Enter your local PDF folder path before scanning.')
    root=Path(root).expanduser()
    if not root.is_dir():
        raise ValueError('Choose an accessible local PDF folder.')
    cache_dir=Path(cache_dir)
    db_file=cache_dir/'pdf_knowledge.sqlite'
    conn=_connection(db_file)
    videos=records_for_search(inventory)
    valid_video_ids={v['path'].replace('\\','/') for v in videos}
    found,discovery=_iter_documents(inventory,root)
    if not found:raise ValueError('No local PDF files were discovered.')
    # The FTS catalog contains all eligible video filenames/folders regardless
    # of whether a PDF can be linked to the video.
    with conn:
        conn.execute('DELETE FROM catalog_search')
        conn.executemany('INSERT INTO catalog_search(video_id,title,folder) VALUES(?,?,?)',
             ((v['path'].replace('\\','/'),v['title'],v['subject']+' / '+v['topic']) for v in videos))
    current_ids={pid for pid,_,_ in found}
    old_ids={r[0] for r in conn.execute('SELECT pdf_id FROM documents')}
    for missing_id in old_ids-current_ids:
        with conn:
            conn.execute('DELETE FROM passage_search WHERE passage_id IN (SELECT passage_id FROM passages WHERE pdf_id=?)',(missing_id,))
            conn.execute('DELETE FROM passages WHERE pdf_id=?',(missing_id,))
            conn.execute('DELETE FROM documents WHERE pdf_id=?',(missing_id,))
    scanned=0;reused=0;errors=[]
    import fitz
    for ii,(pdf_id,local,pdf) in enumerate(found):
        if progress:progress(ii+1,len(found),local.name)
        try:
            stat=local.stat()
            candidates=_video_candidates(pdf,inventory,videos)
            manual=_verified_from_manifest(pdf_manifest,pdf_id,local)
            reviewed=[tuple(x) for x in conn.execute(
                'SELECT start_page,end_page,video_id FROM reviews WHERE pdf_id=?',(pdf_id,))]
            fp=_digest([KB_VERSION,str(local.resolve()),stat.st_size,stat.st_mtime_ns,
                        bool(use_ocr),max_pages_per_pdf,sorted(manual),sorted(reviewed)])
            old=conn.execute('SELECT fingerprint,status FROM documents WHERE pdf_id=?',(pdf_id,)).fetchone()
            if old and old['fingerprint']==fp and old['status']=='indexed':
                reused+=1;continue
            rows=[];ocr_pages=0;unreadable=0;searchable_pages=0
            with fitz.open(str(local)) as doc:
                n=len(doc)
                upto=min(n,max_pages_per_pdf) if max_pages_per_pdf else n
                toc=sorted([(int(pg),str(title)) for _,title,pg in doc.get_toc() if 1<=pg<=n])
                toc_map=dict(toc)
                toc_idx=0
                current_toc=''
                for idx in range(upto):
                    p=idx+1
                    while toc_idx<len(toc) and toc[toc_idx][0]<=p:
                        current_toc=toc[toc_idx][1]
                        toc_idx+=1
                    native=doc[idx].get_text(sort=True)
                    text=clean_native_text(native)
                    method='native'
                    if use_ocr and (not readable(text) or corrupt_heading(native)):
                        try:
                            recovered=ocr_text(doc[idx])
                            if readable(recovered) and (not readable(text) or
                                len(re.findall(r'[A-Za-z]{3,}',recovered))>len(re.findall(r'[A-Za-z]{3,}',text))*0.7):
                                text=clean_native_text(recovered);method='OCR';ocr_pages+=1
                        except Exception:
                            method='OCR unavailable; native fallback'
                    if not readable(text):
                        unreadable+=1
                        continue
                    excerpt_chunks=_split_text(text)
                    if not excerpt_chunks:
                        unreadable+=1
                        continue
                    searchable_pages+=1
                    heading=_short_heading(text,toc_map.get(p,''))
                    if page_is_board_breakdown(text) or is_front_page(text) or \
                            re.match(r'^\s*REVIEW\s+OUTLINE\b',text,re.I):
                        heading='[REVIEW/CONTENTS] '+heading
                    verified_ids=[v for lo,hi,v in manual if lo<=p<=hi and v in valid_video_ids]
                    review_ids=[v for lo,hi,v in reviewed if lo<=p<=hi and v in valid_video_ids]
                    verified_ids=list(dict.fromkeys(review_ids+verified_ids))
                    suggested=_suggest_links(heading,current_toc,candidates)
                    if len(verified_ids)==1:
                        vid=verified_ids[0]
                        status='reviewed' if vid in review_ids else 'verified'
                    else:
                        vid=''
                        status='ambiguous' if len(verified_ids)>1 else 'suggested' if suggested else 'unmatched'
                    if len(verified_ids)>1:
                        suggested=list(dict.fromkeys(verified_ids+suggested))[:4]
                    # The full page is retained before attribution, never
                    # dropped just because a video couldn't be identified.
                    for part,excerpt in enumerate(excerpt_chunks):
                        passage_id=_digest([pdf_id,p,part])[:32]
                        rows.append((passage_id,pdf_id,p,part,heading,excerpt,method,
                                     vid,status,json.dumps(suggested)))
            with conn:
                conn.execute('DELETE FROM passage_search WHERE passage_id IN (SELECT passage_id FROM passages WHERE pdf_id=?)',(pdf_id,))
                conn.execute('DELETE FROM passages WHERE pdf_id=?',(pdf_id,))
                conn.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                             (pdf_id,local.name,pdf['subject'] if pdf else '',
                              pdf['topic'] if pdf else '',str(local),fp,n,
                              searchable_pages,ocr_pages,unreadable,'indexed')) if old is None else conn.execute(
                             '''UPDATE documents SET filename=?,subject=?,topic=?,local_path=?,fingerprint=?,page_count=?,
                             searchable_pages=?,ocr_pages=?,unreadable_pages=?,status=? WHERE pdf_id=?''',
                             (local.name,pdf['subject'] if pdf else '',pdf['topic'] if pdf else '',
                              str(local),fp,n,searchable_pages,ocr_pages,unreadable,'indexed',pdf_id))
                conn.executemany('INSERT INTO passages VALUES (?,?,?,?,?,?,?,?,?,?)',rows)
                conn.executemany('INSERT INTO passage_search(passage_id,text,heading) VALUES(?,?,?)',
                                 ((r[0],r[5],r[4]) for r in rows))
            scanned+=1
        except Exception as exc:
            errors.append({'pdf':local.name,'error':str(exc)[:230]})
    conn.commit()
    counts={row[0]:row[1] for row in conn.execute('SELECT link_status,COUNT(*) FROM passages GROUP BY link_status')}
    stats={'pdf_files':conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0],
           'pages':conn.execute('SELECT COALESCE(SUM(page_count),0) FROM documents').fetchone()[0],
           'searchable_pages':conn.execute('SELECT COALESCE(SUM(searchable_pages),0) FROM documents').fetchone()[0],
           'passages':sum(counts.values()),'link_status_counts':counts,'scanned':scanned,
           'reused_unchanged':reused,'errors':errors,'discovery':discovery,
           'ocr_pages':conn.execute('SELECT COALESCE(SUM(ocr_pages),0) FROM documents').fetchone()[0]}
    conn.close()
    (cache_dir/'pdf_knowledge_summary.json').write_text(json.dumps(stats,indent=2,default=str),encoding='utf-8')
    return stats


class PDFKnowledgeBase:
    def __init__(self, cache_dir):
        self.cache_dir=Path(cache_dir)
        self.path=self.cache_dir/'pdf_knowledge.sqlite'
        if not self.path.exists():raise FileNotFoundError('Scan the PDF knowledge base first.')
        self.conn=_connection(self.path)
        self._vectors=None
        self._vector_rows=None
        self._vector_model=None

    def close(self):self.conn.close()

    def stats(self):
        docs=self.conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        passages=self.conn.execute('SELECT COUNT(*) FROM passages').fetchone()[0]
        rows={r[0]:r[1] for r in self.conn.execute('SELECT link_status,COUNT(*) FROM passages GROUP BY link_status')}
        linked=self.conn.execute("SELECT COUNT(DISTINCT video_id) FROM passages WHERE video_id != ''").fetchone()[0]
        return {'pdfs':docs,'passages':passages,'linked_mp4s':linked,'status':rows}

    def documents(self):
        return [dict(r) for r in self.conn.execute('SELECT * FROM documents ORDER BY filename')]

    def video_coverage(self, videos):
        counts={r['video_id']:r['n'] for r in self.conn.execute(
            "SELECT video_id, COUNT(*) AS n FROM passages WHERE video_id<>'' "
            'GROUP BY video_id')}
        return [{'MP4 title':v['title'],'Subject':v['subject'],'Video folder':v['topic'],
                 'Linked searchable excerpts':counts.get(v['path'].replace('\\','/'),0),
                 'MP4 inventory path':v['path']} for v in videos]

    def unattributed_excerpts(self, limit=50000):
        sql=("SELECT d.filename AS PDF,p.page AS Page,p.heading AS Heading, "
             "p.text AS Excerpt,p.link_status AS Attribution, "
             "p.candidate_video_ids AS Suggested_MP4_paths,p.pdf_id AS PDF_ID "
             "FROM passages p JOIN documents d ON d.pdf_id=p.pdf_id "
             "WHERE p.link_status NOT IN ('reviewed','verified') "
             "ORDER BY d.filename,p.page,p.part LIMIT ?")
        return [dict(x) for x in self.conn.execute(sql,(limit,))]


    def _rows(self):
        return self.conn.execute('SELECT passage_id,text FROM passages ORDER BY passage_id').fetchall()

    def build_vectors(self, encoder, model_name, progress=None):
        """Optional persisted semantic vectors; no LLM API calls required."""
        rows=self._rows()
        identifiers=[r['passage_id'] for r in rows]
        digest=_digest([KB_VERSION,model_name,identifiers,
                         [hashlib.sha256(r['text'].encode()).hexdigest() for r in rows]])
        cfg_path=self.cache_dir/'pdf_knowledge_vectors.json'
        vec_path=self.cache_dir/'pdf_knowledge_vectors.npy'
        try:cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
        except (OSError,ValueError):cfg={}
        if cfg.get('digest')==digest and vec_path.exists():
            vecs=np.load(vec_path,allow_pickle=False)
            if vecs.shape[0]==len(rows):
                self._vectors=vecs
                self._vector_rows=identifiers
                self._vector_model=model_name
                return {'count':len(rows),'reused':True}
        embeddings=[]
        for start in range(0,len(rows),96):
            if progress:progress(start+1,len(rows),'Embedding PDF excerpts')
            batch=rows[start:start+96]
            if batch:
                embeddings.append(encode(encoder,[r['text'] for r in batch],model_name=model_name))
        vecs=np.concatenate(embeddings).astype('float32') if embeddings else np.zeros((0,0),dtype='float32')
        tmp=vec_path.with_suffix('.tmp.npy')
        np.save(tmp,vecs,allow_pickle=False)
        tmp.replace(vec_path)
        cfg_path.write_text(json.dumps({'digest':digest,'model':model_name,'count':len(rows)}),encoding='utf-8')
        self._vectors=vecs;self._vector_rows=identifiers;self._vector_model=model_name
        return {'count':len(rows),'reused':False}

    def _load_vectors(self, model_name):
        if self._vector_model==model_name:return
        cfg_path=self.cache_dir/'pdf_knowledge_vectors.json'
        vec_path=self.cache_dir/'pdf_knowledge_vectors.npy'
        try:cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
        except (OSError,ValueError):cfg={}
        if cfg.get('model')==model_name and vec_path.exists():
            rows=self._rows()
            digest=_digest([KB_VERSION,model_name,[r['passage_id'] for r in rows],
                     [hashlib.sha256(r['text'].encode()).hexdigest() for r in rows]])
            if cfg.get('digest')==digest:
                vecs=np.load(vec_path,allow_pickle=False)
                if len(vecs)==len(rows):
                    self._vectors=vecs
                    self._vector_rows=[r['passage_id'] for r in rows]
                    self._vector_model=model_name
                    return
        self._vectors=None;self._vector_rows=None;self._vector_model=None

    def search(self, query, limit=12, *, encoder=None, model_name=''):
        """Hybrid retrieve FROM PDFs first, including unlinked excerpts.

        SQLite FTS5 + optional pretrained sentence-embedding similarity.
        Retrieval scores are heuristic, not calibrated confidence values.
        """
        tok=[t for t in tokens(query) if len(t)>1]
        candidates={}
        if tok:
            expr=' OR '.join('"'+t.replace('"','')+'"' for t in dict.fromkeys(tok[:20]))
            try:
                matches=self.conn.execute('''SELECT passage_id,bm25(passage_search) AS rank
                    FROM passage_search WHERE passage_search MATCH ? ORDER BY rank LIMIT ?''',
                    (expr,max(80,limit*8))).fetchall()
                strengths=[max(0.,-float(x['rank'])) for x in matches]
                peak=max(strengths,default=0.)
                for row,score in zip(matches,strengths):
                    candidates[row['passage_id']]=0.45*(score/max(1e-9,peak))
            except sqlite3.OperationalError:
                pass
        if encoder is not None and model_name:
            self._load_vectors(model_name)
            if self._vectors is not None and len(self._vectors):
                q=encode(encoder,[query],model_name=model_name,is_query=True)[0]
                similarities=self._vectors@q
                limit_dense=min(len(similarities),max(90,limit*10))
                top=np.argpartition(-similarities,limit_dense-1)[:limit_dense]
                for idx in top:
                    s=float(similarities[idx])
                    dense_score=max(0.,min(1.,(s-0.12)/0.56))
                    key=self._vector_rows[int(idx)]
                    candidates[key]=min(1.,candidates.get(key,0.)+.65*dense_score)
        ids=sorted(candidates,key=lambda key:-candidates[key])[:max(limit*3,limit)]
        if not ids:return []
        lookup={r['passage_id']:dict(r) for r in self.conn.execute(
            'SELECT p.*,d.filename,d.subject,d.topic FROM passages p JOIN documents d ON d.pdf_id=p.pdf_id '
            'WHERE p.passage_id IN ('+','.join('?' for _ in ids)+')',ids)}
        # Prioritize substantive teaching passages without throwing away
        # review/contents pages; users can still find those if relevant.
        adjusted={pid:sc*(.67 if (lookup.get(pid,{}).get('heading','').startswith('[REVIEW/CONTENTS]')) else 1.)
                  for pid,sc in candidates.items() if pid in lookup}
        ids=sorted(lookup,key=lambda pid:-adjusted.get(pid,0.))
        results=[];seen_pages=Counter();seen_docs=Counter()
        docs_count=self.conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        per_pdf_limit=max(3,(limit+1)//2) if docs_count>=3 else limit
        for pid in ids:
            row=lookup.get(pid)
            if row is None:continue
            pagekey=(row['pdf_id'],row['page'])
            if seen_pages[pagekey]>=2 or seen_docs[row['pdf_id']]>=per_pdf_limit:continue
            seen_pages[pagekey]+=1;seen_docs[row['pdf_id']]+=1
            row['candidate_video_ids']=json.loads(row['candidate_video_ids'])
            row['score']=round(float(adjusted[pid]),4)
            results.append(row)
            if len(results)>=limit:break
        return results

    def search_many(self, queries, limit_per_query=8, *, encoder=None, model_name=''):
        matches=[]
        for index,query in enumerate(queries):
            for hit in self.search(query,limit=limit_per_query,encoder=encoder,model_name=model_name):
                matches.append({**hit,'query_index':index,'lecture_query':query})
        return matches

    def review(self, pdf_id, start_page, end_page, video_id, valid_video_ids):
        """Explicit user-verified attribution. Reversible without PDF rescan."""
        if video_id not in set(valid_video_ids):raise ValueError('Video must be an exact eligible MP4 inventory ID.')
        doc=self.conn.execute('SELECT page_count FROM documents WHERE pdf_id=?',(pdf_id,)).fetchone()
        if not doc:raise ValueError('PDF not found in scanned knowledge base.')
        a,b=int(start_page),int(end_page)
        if not 1<=a<=b<=doc['page_count']:raise ValueError('Invalid document page range.')
        with self.conn:
            # Reject two conflicting reviewer assertions for the same page.
            clash=self.conn.execute('''SELECT 1 FROM reviews WHERE pdf_id=? AND
                  NOT (end_page<? OR start_page>?) AND video_id<>? LIMIT 1''',
                  (pdf_id,a,b,video_id)).fetchone()
            if clash:raise ValueError('Another reviewed MP4 overlaps these pages. Resolve that review first.')
            self.conn.execute('INSERT OR REPLACE INTO reviews VALUES (?,?,?,?)',(pdf_id,a,b,video_id))
            self.conn.execute('''UPDATE passages SET video_id=?,link_status='reviewed'
                                 WHERE pdf_id=? AND page BETWEEN ? AND ?''',(video_id,pdf_id,a,b))
        # Any optional dense vector cache still works; only attribution changed.

    def remove_review(self,pdf_id,start_page,end_page,video_id):
        with self.conn:
            self.conn.execute('DELETE FROM reviews WHERE pdf_id=? AND start_page=? AND end_page=? AND video_id=?',
                              (pdf_id,start_page,end_page,video_id))
            self.conn.execute('''UPDATE passages SET video_id='',link_status='unmatched'
                WHERE pdf_id=? AND page BETWEEN ? AND ? AND video_id=?''',
                (pdf_id,start_page,end_page,video_id))

    def reviews(self):
        return [dict(x) for x in self.conn.execute('SELECT * FROM reviews ORDER BY pdf_id,start_page')]
