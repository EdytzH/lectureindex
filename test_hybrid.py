"""No network, no pretrained model required: pipeline + schema regression tests."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from file_index_catalog import load_file_index
from semantic_matcher import (catalog_records, chunks_from_text, chunks_from_slides,
                              prepare_index, HybridMatcher, load_slide_evidence)
from eval_retrieval import evaluate_rows


class FakeEncoder:
    """Deterministic word-count vectors for testing plumbing, NOT semantic quality."""
    def __init__(self):
        self.vectorizer = HashingVectorizer(n_features=512, alternate_sign=False,
                                            norm='l2', ngram_range=(1, 2))

    def encode(self, texts, **kw):
        return self.vectorizer.transform(texts).toarray().astype('float32')


root = Path(__file__).resolve().parent
inventory = load_file_index(root/'bootcamp_step1_file_index.xlsx')
records = catalog_records(inventory)
assert len([r for r in inventory if r['type']=='MP4']) == 2477
assert len(records) == 2323, len(records)
assert len(set(r['id'].lower() for r in records)) == len(records)
assert all('board-style question breakdown' not in r['title'].lower() for r in records)
hand = next(r for r in records if r['title'] == 'Carpals, Metacarpals and Phalanges'
            and '05. Hand' in r['topic'])
assert len(hand['pdf_candidates']) == 1 and hand['pdf_candidates'][0].endswith('Hand.pdf')
assert '05. Hand' in hand['course_section']

paragraph = ('The highest-yield Step 1 concepts from this lecture are the embryologic origins, '
             'timing, rotation, ossification, vertebral segmentation, and limb defects. '
             'We also review limb bud rotation and malformations.')
chunks = chunks_from_text(paragraph)
assert len(chunks) == 2
assert len(chunks_from_text('Crohn disease\n- Large intestine\n2. Limb rotation')) == 3
assert all(len(chunk['text']) < 650 for chunk in chunks)
slides = chunks_from_slides([{'number': 7, 'text': paragraph}, {'number': 8, 'text': 'Carpal tunnel median nerve.'}], 'lecture.pdf')
assert [c['page'] for c in slides] == [7, 8]

encoder = FakeEncoder()
with TemporaryDirectory() as temp:
    index = prepare_index(records, cache_dir=temp, model_name='fake-model', encoder=encoder)
    assert not index['from_cache']
    assert index['title_vectors'].shape[0] == 2323
    assert index['slide_vectors'].shape == index['title_vectors'].shape
    assert all(item['slide_embedding'] is None for item in index['records'])
    cached = prepare_index(records, cache_dir=temp, model_name='fake-model', encoder=encoder)
    assert cached['from_cache']
    assert cached['digest'] == index['digest']
    matcher = HybridMatcher(cached, encoder)
    crohn = matcher.search(chunks_from_text('We will discuss Crohn Disease in the large intestine.'),
                           threshold=.30, limit=20)
    assert crohn and crohn[0]['title'] == 'Crohn Disease', [x['title'] for x in crohn[:10]]
    assert crohn[0]['id'].endswith('Crohn Disease.mp4')
    assert crohn[0]['evidence'][0]['cosine_similarity'] >= 0
    assert matcher.search(chunks_from_text(''), threshold=.48) == []
    findings = matcher.search(slides, threshold=.25, limit=30)
    assert any(r['matching_slides'] >= 1 for r in findings)
    # Verified slide evidence can be added for *one exact MP4 ID* without schema changes.
    evidence = {hand['id']: 'Verified content about carpals and phalanges anatomy.'}
    newindex = prepare_index(records, cache_dir=temp, model_name='fake-model',
                             encoder=encoder, slide_evidence=evidence)
    assert not newindex['from_cache'] and newindex['digest'] != index['digest']
    assert sum(x['slide_embedding'] is not None for x in newindex['records']) == 1
    assert newindex['records'][next(i for i,r in enumerate(records) if r['id']==hand['id'])]['slide_text']
    assert np.any(newindex['slide_vectors'])
    checked = json.loads((Path(temp)/'videos.json').read_text(encoding='utf-8'))
    assert len(checked) == len(records)
    for key in ('id','title','subject','topic','course_section','video_url','slide_text',
                'title_embedding','context_embedding','slide_embedding'):
        assert key in checked[0], key

sample = evaluate_rows([{'query': 'crohn', 'expected_mp4_paths': crohn[0]['id']}],
                       lambda q: crohn)
assert sample[0]['hit_at_1'] == sample[0]['hit_at_5'] == 1
assert not evaluate_rows([{'query': 'crohn', 'expected_mp4_paths': ''}], lambda q: crohn)
print('PASS: 2,323 real MP4s, title-first hybrid retrieval, paragraph+slide chunking, persistent vectors, verified slide slot, and evaluation logic')
print('Top match for Crohn disease:', crohn[0]['title'])
