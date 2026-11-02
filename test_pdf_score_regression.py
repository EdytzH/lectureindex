"""Regression: a title-weak but verified teaching-page-strong MP4 must be retrieved.

No external model or actual MP4 files required. This catches the old 35%-PDF
weighted blend that dropped valid PDF hits under the cutoff.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from semantic_matcher import HybridMatcher, prepare_index, chunks_from_text
from pdf_evidence import PDFChunkEvidence, build_pdf_vectors

class FakeEncoder:
    def __init__(self):
        self.vec=HashingVectorizer(n_features=1024, alternate_sign=False, norm='l2', ngram_range=(1,2))
    def encode(self,texts,**kwargs):
        return self.vec.transform(texts).toarray().astype('float32')

videos = [
    {'id':'/subject/embryology/1. Abstract Foundations.mp4', 'title':'Abstract Foundations',
     'subject':'05. Embryology','topic':'Embryonic Period','mp4_path':'/subject/embryology/1. Abstract Foundations.mp4'},
    {'id':'/subject/embryology/2. Limb Development.mp4', 'title':'Limb Development',
     'subject':'05. Embryology','topic':'Embryonic Period','mp4_path':'/subject/embryology/2. Limb Development.mp4'},
    {'id':'/subject/pulmonology/3. Respiratory Basics.mp4', 'title':'Respiratory Basics',
     'subject':'17. Pulmonology','topic':'Respiratory Anatomy','mp4_path':'/subject/pulmonology/3. Respiratory Basics.mp4'},
]
for r in videos:
    r.update(slide_text='',title_embedding=None,context_embedding=None,slide_embedding=None)
q = chunks_from_text('The apical ectodermal ridge causes mesenchymal outgrowth and digital ray formation. The limb bud rotates and ossifies by endochondral bone formation.')
with TemporaryDirectory() as temp:
    cache=Path(temp)
    fake=FakeEncoder()
    index=prepare_index(videos,cache_dir=cache/'titles',model_name='fake-model',encoder=fake)
    matcher=HybridMatcher(index,fake)
    with (cache/'pdf_chunks.json').open('w',encoding='utf-8') as f:
        json.dump([{'video_id':videos[1]['id'], 'pdf_id':'/subject/embryology/0. Embryonic Period.pdf',
                    'page':30,'text':q[0]['text']+' Additional limb information about ectodermal ridge and somites.'}],f)
    build_pdf_vectors(cache,fake,'fake-model')
    pdf=PDFChunkEvidence(cache,'fake-model',[r['id'] for r in videos])
    old=matcher.search(q,threshold=.42,top_k=1,pdf_evidence=None)
    tagged=matcher.search(q,threshold=.42,top_k=1,pdf_evidence=pdf,pdf_weight=.35)
    limb=next((r for r in tagged if r['title']=='Limb Development'),None)
    assert limb, tagged
    assert limb['pdf_indexed'] is True, limb
    assert limb['pdf_match_score'] is not None and limb['pdf_match_score'] > .55,limb
    assert limb['pdf_match_page']==30,limb
    assert limb['pdf_match_source']=='PDF evidence',limb
    assert any(e['match_source'] in ('verified PDF pages','title + PDF') for e in limb['evidence']),limb
    assert all(r['title']!='Limb Development' or r['pdf_match_score'] is None for r in old)
    off=matcher.search(q,threshold=.42,top_k=1,pdf_evidence=pdf,pdf_weight=0)
    assert all(r['pdf_match_score'] is None for r in off)
    print('PASS: PDF-only evidence crosses cutoff and remains independently visible even at PDF weight=.35')
    print('Limb score:',limb['score'],'PDF:',limb['pdf_match_score'],'page:',limb['pdf_match_page'])
