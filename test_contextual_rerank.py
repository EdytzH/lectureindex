"""No downloaded model: regression for contextual reranking and evidence provenance.

This tests control flow, correct MP4/page attribution, rejection, fair sampling
and duplicate-concept aggregation. It does NOT establish medical model accuracy.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from semantic_matcher import HybridMatcher, prepare_index
from pdf_evidence import PDFChunkEvidence, build_pdf_vectors
from context_reranker import contextual_rank, concept_groups

class FakeEncoder:
    def __init__(self):
        self.vec=HashingVectorizer(n_features=1024,alternate_sign=False,norm='l2',ngram_range=(1,2))
    def encode(self,texts,**kw):return self.vec.transform(texts).toarray().astype('float32')

class ContextOracle:
    """A controlled test double, NOT rules used in production matching."""
    def __init__(self):self.pairs=[]
    def predict(self, pairs, **kwargs):
        self.pairs += pairs
        values=[]
        for left,right in pairs:
            match = ('limb buds rotate' in left.lower() and
                     'upper limbs rotate laterally' in right.lower())
            poor = 'respiratory bronchi' in right.lower() or 'pulmonary' in right.lower()
            values.append(6 if match else -8 if poor else -4)
        return np.asarray(values,dtype=np.float32)

VIDEOS=[
 {'id':'/embryology/2. Limb Development.mp4', 'title':'Limb Development',
  'subject':'05. Embryology','topic':'Embryonic Period','mp4_path':'/embryology/2. Limb Development.mp4'},
 {'id':'/pulmonology/1. Embryologic Defects.mp4', 'title':'Embryologic Defects',
  'subject':'17. Pulmonology','topic':'Embryologic Defects','mp4_path':'/pulmonology/1. Embryologic Defects.mp4'},
]
for r in VIDEOS:r.update(slide_text='',title_embedding=None,context_embedding=None,slide_embedding=None)
CHUNKS=[
 {'source':'lecture.pdf','page':8,'text':'The limb buds rotate and form the upper and lower limbs.'},
 {'source':'lecture.pdf','page':9,'text':'The limb buds rotate and form the upper and lower limbs.'},
 {'source':'lecture.pdf','page':18,'text':'Vertebral ossification begins in the axial skeleton.'}
]
ENCODER=FakeEncoder()
cluster=concept_groups(CHUNKS,ENCODER,'fake')
assert cluster[0]==cluster[1] and cluster[0]!=cluster[2], cluster
with TemporaryDirectory() as temp:
    folder=Path(temp)
    ix=prepare_index(VIDEOS,cache_dir=folder/'titles',model_name='fake',encoder=ENCODER)
    matcher=HybridMatcher(ix,ENCODER)
    content=[
        {'video_id':VIDEOS[0]['id'],'pdf_id':'/embryology/0. Embryonic Period.pdf',
         'page':34,'text':'The upper limbs rotate laterally, whereas lower limbs rotate medially. Limb buds rotate.'},
        {'video_id':VIDEOS[0]['id'],'pdf_id':'/embryology/0. Embryonic Period.pdf',
         'page':31,'text':'Limb development. Finger formation from digital rays and apical ectodermal ridge.'},
        {'video_id':VIDEOS[1]['id'],'pdf_id':'/pulmonology/Respiratory.pdf',
         'page':7,'text':'Development of respiratory bronchi, trachea, and lung branching.'},
    ]
    (folder/'pdf_chunks.json').write_text(json.dumps(content),encoding='utf-8')
    build_pdf_vectors(folder,ENCODER,'fake')
    pdf=PDFChunkEvidence(folder,'fake',[r['id'] for r in VIDEOS])
    qvec=ENCODER.encode([c['text'] for c in CHUNKS])
    s,w=pdf.query([c['text'] for c in CHUNKS],qvec,witness_limit=2)
    assert len(w[(0,0)]['alternatives'])==2 and all(x['pdf_id']=='/embryology/0. Embryonic Period.pdf' for x in w[(0,0)]['alternatives'])
    # Force both videos into the stage-1 candidate pool to test actual rejection.
    # The production function must still use provenance from the PDF engine.
    fake_score=ContextOracle()
    result=matcher.search(CHUNKS,limit=10,top_k=2,threshold=.05,
                          pdf_evidence=pdf,pdf_weight=.35,
                          reranker=fake_score,context_min_score=.50,max_context_pairs=100)
    assert [x['title'] for x in result]==['Limb Development'],result
    top=result[0]
    assert top['distinct_concepts']==1,top
    assert top['pdf_context_page']==34,top
    assert top['pdf_context_file']=='/embryology/0. Embryonic Period.pdf',top
    assert top['context_method']=='local cross-encoder',top
    assert any('upper limbs rotate laterally' in x[1].lower() for x in fake_score.pairs)
    assert all('respiratory' not in e.get('context_pdf_excerpt','').lower() for e in top['evidence'])
    # Original title/BM25 matcher remains available when cross-encoder is off.
    old=matcher.search(CHUNKS,limit=10,top_k=2,threshold=.05,pdf_evidence=pdf)
    assert old and all('context_score' not in r for r in old)
    # A real MP4 without a verified PDF must still be recommendable on title and
    # folder metadata when the contextual model supports the paired text.
    class TitleOracle:
        def predict(self, pairs, **kwargs):
            return np.array([7. if ('Limb Development' in right and
                                      'limb development' in left.lower()) else -7.
                             for left,right in pairs], dtype='float32')
    title_only=matcher.search(
        [{'source':'topics','page':None,'text':'The lecture explains limb development.'}],
        threshold=.01,top_k=2,reranker=TitleOracle(),context_min_score=.5)
    assert title_only and title_only[0]['title']=='Limb Development',title_only
    assert title_only[0]['pdf_context_page'] is None,title_only
    assert title_only[0]['pdf_match_source']=='No indexed PDF teaching pages',title_only
    # An intentionally impossible cutoff must give zero matches, never force a
    # video just because there are a fixed number of result slots.
    rejected=matcher.search(CHUNKS,threshold=.05,top_k=2,pdf_evidence=pdf,
                            reranker=ContextOracle(),context_min_score=.9999)
    assert rejected==[],rejected
    print('PASS: context reranking, correct PDF provenance, duplicated-chunk suppression, title-only fallback, and no forced matches')
