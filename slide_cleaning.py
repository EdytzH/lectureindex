"""Extract educational text from PDFs/PPTX, discarding obvious figure credits.

Heuristics intentionally conservative. This does NOT do image understanding or OCR.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from statistics import median

import fitz

_URL = re.compile(r'(https?://|www\.|doi\s*[:/]|10\.\d{4,9}/\S+)', re.I)
_CREDIT = re.compile(
    r'^\s*(?:fig(?:ure)?\.?\s*\d+\s*[:.\-]?\s*(?:source|credit)|'
    r'(?:image|figure|photo|picture|diagram|graphic)\s*(?:credit|source|courtesy|adapted|from|by|:)|'
    r'(?:source|credit|courtesy|adapted)\s*(?:of|from|:)|'
    r'©|copyright\b|all rights reserved)', re.I)
_REFERENCES = re.compile(r'^\s*(references?|bibliography|image\s+credits|photo\s+credits)\s*[:.]?\s*$',re.I)
_CITATION = re.compile(r'(?:\b(?:et\s+al\.?|isbn|pmid|pubmed|accessed\s+on)\b|'
                       r'\(\s*\d{4}\s*\)\s*\.?$)', re.I)
_SMALL_BULLET = re.compile(r'^[\W_]+$', re.UNICODE)


def normalize_line(line: str) -> str:
    line = str(line).replace('\u00ad','').replace('\ufb01','fi').replace('\ufb02','fl')
    return ' '.join(line.split()).strip()


def is_noise_line(line: str) -> bool:
    s = normalize_line(line)
    if not s or _SMALL_BULLET.match(s):
        return True
    if _URL.search(s) or _CREDIT.search(s):
        return True
    if _CITATION.search(s) and len(s.split()) <= 17:
        return True
    if re.fullmatch(r'\s*(?:page|slide)?\s*\d{1,3}\s*',s, re.I):
        return True
    return False


def _dedupe_slide_text(slides):
    # Repeated slide headers/footers have little value for retrieval.
    freq=Counter()
    for s in slides:
        freq.update(set(x.lower() for x in s['lines'] if len(x)<110))
    for s in slides:
        kept=[]; seen=set()
        for t in s['lines']:
            key=t.lower()
            if key in seen: continue
            if freq[key] >= max(3,int(len(slides)*.25)) and len(t.split()) <= 12:
                continue
            seen.add(key); kept.append(t)
        s['text']='\n'.join(kept)
        s['title']=next((x for x in kept if len(x)>3),'')[:90]
        s.pop('lines',None)
    return slides


def extract_pdf_slides(source) -> list[dict]:
    if isinstance(source, bytes):
        doc=fitz.open(stream=source,filetype='pdf')
    else:
        doc=fitz.open(str(source))
    out=[]
    try:
        for pn,page in enumerate(doc,1):
            h=page.rect.height or 1
            entries=[]
            for block in page.get_text('dict').get('blocks',[]):
                for line in block.get('lines',[]):
                    spans=line.get('spans',[])
                    if not spans: continue
                    text=normalize_line(''.join(s.get('text','') for s in spans))
                    if not text: continue
                    y0=min(float(s['bbox'][1]) for s in spans)/h
                    size=max(float(s.get('size',0)) for s in spans)
                    entries.append((y0,size,text))
            center_sizes=[size for y,size,text in entries if .08<=y<=.78 and len(text)>2]
            typical=median(center_sizes) if center_sizes else median([x[1] for x in entries]) if entries else 12
            entries.sort(key=lambda x:x[0])
            lines=[]; stop=False
            for y,size,text in entries:
                if _REFERENCES.fullmatch(text):
                    stop=True
                if stop: continue
                if is_noise_line(text): continue
                # Mostly citation footers and small photo credits; avoid removing
                # a substantive 12pt label on a diagram-heavy slide.
                if y > .79 and size < .76*typical and len(text.split())<24:
                    continue
                lines.append(text)
            out.append({'number':pn,'lines':lines,'text':'','title':'',
                        'image_count':sum(1 for b in page.get_text('dict').get('blocks',[]) if b.get('type')==1)})
    finally:
        doc.close()
    return _dedupe_slide_text(out)


def extract_pptx_slides(data: bytes) -> list[dict]:
    from pptx import Presentation
    prs=Presentation(io.BytesIO(data))
    out=[]
    for pn,slide in enumerate(prs.slides,1):
        lines=[]
        for shape in slide.shapes:
            if not getattr(shape,'has_text_frame',False): continue
            y=float(shape.top)/prs.slide_height if prs.slide_height else 0
            for para in shape.text_frame.paragraphs:
                t=normalize_line(para.text)
                if not t or is_noise_line(t) or _REFERENCES.fullmatch(t):continue
                # A small footer is more likely source metadata than subject text.
                sizes=[int(run.font.size) for run in para.runs if run.font.size]
                if y > .85 and sizes and max(sizes)<12*12700:continue
                lines.append(t)
        out.append({'number':pn,'lines':lines,'text':'','title':'',
                    'image_count':sum(1 for s in slide.shapes if s.shape_type==13)})
    return _dedupe_slide_text(out)


def chunk_text(text: str, chars=1200, overlap=170) -> list[str]:
    words=' '.join(str(text).split()).split()
    if not words: return []
    chunks=[]; current=[]; size=0
    for word in words:
        if current and size+len(word)+1>chars:
            chunks.append(' '.join(current))
            # Retain some prior context, without splitting words in half.
            tail=[];length=0
            for w in reversed(current):
                if length+len(w)+1>overlap:break
                tail.insert(0,w);length+=len(w)+1
            current=tail;size=sum(len(w)+1 for w in current)
        current.append(word);size+=len(word)+1
    if current:
        last=' '.join(current)
        if not chunks or last!=chunks[-1]:chunks.append(last)
    return chunks
