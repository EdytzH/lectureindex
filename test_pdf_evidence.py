"""Run with python test_pdf_evidence.py; uses synthetic handout and optional real sample."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from file_index_catalog import load_file_index
from semantic_matcher import catalog_records, chunks_from_text, prepare_index, HybridMatcher, encode
from pdf_evidence import (tag_library, build_pdf_vectors, PDFChunkEvidence,
                          load_manifest, readable)


class FakeEncoder:
    def __init__(self):
        self.vectorizer = HashingVectorizer(n_features=512, alternate_sign=False,
                                           norm='l2', ngram_range=(1,2))
    def encode(self, texts, **kwargs):
        return self.vectorizer.transform(texts).toarray().astype('float32')

root = Path(__file__).resolve().parent
inventory = load_file_index(root / 'bootcamp_step1_file_index.xlsx')
subset = [r for r in inventory if r['subject']=='06. Gross Anatomy' and r['topic']=='05. Hand']
video_records = catalog_records(subset)
assert len(video_records)==10
first=video_records[0]
second=video_records[1]
assert first['title']=='Carpals, Metacarpals and Phalanges'
assert second['title']=='Wrist Joint'
assert all(r['pdf_candidates']==[subset[0]['path']] for r in video_records)

with TemporaryDirectory() as td:
    td=Path(td)
    library=td/'Bootcamp Step 1'
    folder=library/'06. Gross Anatomy'/'05. Hand'
    folder.mkdir(parents=True)
    pdf=folder/'0. Hand.pdf'
    with fitz.open() as doc:
        for text in [
            'TABLE OF CONTENTS\n1. Carpals, Metacarpals and Phalanges\n2. Wrist Joint\n3. Carpal Tunnel',
            'Carpals, Metacarpals and Phalanges\nMetacarpal and phalanx anatomy. Proximal middle distal metacarpals phalanges wrist joints. Carpal structures and bony features.',
            'Metacarpal anatomy\nThe five metacarpal bones articulate with phalanges. Metacarpal joints and anatomy are shown here.',
            'Wrist Joint\nArticular structures of the wrist joint including ligaments, bones, cartilage, and flexion extension movements.',
            'Wrist joint function\nJoint articulation permits flexion and extension. Ligaments and joint capsule contribute to stability.',
            'GENERAL REVIEW\nSummarizing concepts unrelated to any specific confirmed video.'
        ]:
            page=doc.new_page()
            page.insert_text((60,85),text,fontsize=12)
        doc.set_toc([[1,'TABLE OF CONTENTS',1],
                     [2,'Carpals, Metacarpals and Phalanges',2],
                     [2,'Wrist Joint',4],
                     [2,'The Carpal Tunnel - Osteology & Contents',1]])
        doc.save(pdf)
    cache=td/'index'
    manifest=tag_library(subset,root=library,cache_dir=cache)
    first_tag=manifest['videos'][first['id']]
    second_tag=manifest['videos'][second['id']]
    assert (first_tag['start'],first_tag['end'])==(2,3),first_tag
    assert (second_tag['start'],second_tag['end'])==(4,4),second_tag
    assert first_tag['text_chunks']>=1 and second_tag['text_chunks']>=1
    tunnel=next(v for v in manifest['videos'].values() if 'Carpal Tunnel' in v['title'])
    assert not tunnel['start'],tunnel # broken bookmark on page 1 never verifies
    assert all('Board-style Question Breakdown' not in v['title'] for v in manifest['videos'].values())
    saved=load_manifest(cache)
    assert saved['counts']==manifest['counts']
    reloaded=tag_library(subset,root=library,cache_dir=cache)
    assert reloaded['counts']==manifest['counts']
    fake=FakeEncoder()
    count,from_cache=build_pdf_vectors(cache,fake,'fake-model')
    assert count>=2 and not from_cache
    same_count,from_cache=build_pdf_vectors(cache,fake,'fake-model')
    assert from_cache and same_count==count
    index=prepare_index(video_records,cache_dir=td/'title',model_name='fake-model',encoder=fake)
    matcher=HybridMatcher(index,fake)
    pdf_index=PDFChunkEvidence(cache,'fake-model',[r['id'] for r in index['records']])
    q=chunks_from_text('The metacarpal and phalanx bones connect to joints of the hand.')
    cosine=encode(fake,[q[0]['text']],model_name='fake-model',is_query=True)
    scores,evidence=pdf_index.query([q[0]['text']],cosine)
    assert scores.shape==(1,10)
    assert evidence.get((0,0),{}).get('page') in (2,3),evidence
    tagged=matcher.search(q, threshold=.1, pdf_evidence=pdf_index,pdf_weight=.60)
    assert tagged and any(any(e.get('pdf_page') for e in r['evidence']) for r in tagged),tagged
    untagged=matcher.search(q,threshold=.1,pdf_evidence=None)
    assert untagged
    # Exact manual ranges are accepted only for this PDF ID; after reviewed
    # correction, the third video can contribute additional verified chunks.
    tunnel_rec=next(v for v in video_records if 'Carpal Tunnel' in v['title'])
    with open(cache/'manual_pdf_ranges.csv','w',newline='',encoding='utf-8') as out:
        w=csv.writer(out);w.writerow(['MP4 path','PDF path','Start page','End page'])
        w.writerow([tunnel_rec['id'],subset[0]['path'],5,5])
    corrected=tag_library(subset,root=library,cache_dir=cache)
    assert corrected['videos'][tunnel_rec['id']]['start']==5
    assert corrected['videos'][tunnel_rec['id']]['method']=='reviewed manual range'
print('PASS: folder identity, bad bookmark rejection, verified pages, missing-title safety, PDF vectors, search evidence, caching and manual correction')
