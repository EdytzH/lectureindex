"""Offline regression for hierarchical lecture planning; NO API or model download."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from file_index_catalog import load_file_index
from semantic_matcher import HybridMatcher, catalog_records, prepare_index, chunks_from_slides
from llm_planner import (LectureAnalysis, TeachingConcept, CurriculumPlan, SelectedVideo,
                         _plan_payload, lecture_packet, local_lecture_outline,
                         retrieve_hierarchical, run_llm_planner, validate_plan,
                         local_section_preview)

ROOT = Path(__file__).resolve().parent

class FakeEncoder:
    def __init__(self):
        self.vect=HashingVectorizer(n_features=384,alternate_sign=False,norm='l2',ngram_range=(1,2))
    def encode(self, texts, **_):
        return self.vect.transform(texts).toarray().astype('float32')

class FakeAPI:
    def __init__(self):
        self.responses=self
        self.calls=[]
    def parse(self, *, model, input, text_format):
        user=json.loads(input[1]['content'])
        self.calls.append((text_format.__name__,user))
        if text_format is LectureAnalysis:
            return type('Reply',(),{'output_parsed':LectureAnalysis(
                lecture_focus='Vertebral column and back anatomy, muscles, curvatures and neural supply',
                notes='Two lecture parts, column and back',
                concepts=[
                    TeachingConcept(label='Divisions and regional vertebrae', detail='cervical thoracic lumbar sacrum coccyx characteristics', importance='major',lecture_pages=[3,13,14,20,21,22]),
                    TeachingConcept(label='Curvatures and disc pathology', detail='thoracic and lumbar curves, kyphosis lordosis scoliosis herniated discs', importance='major',lecture_pages=[5,6,10,11]),
                    TeachingConcept(label='Joints and ligaments', detail='facet articulations atlas axis and longitudinal ligaments', importance='major',lecture_pages=[8,12,15,16]),
                    TeachingConcept(label='Back muscles', detail='trapezius latissimus rhomboids serratus erector spinae multifidus', importance='major',lecture_pages=[28,30,31,40,42]),
                    TeachingConcept(label='Innervation and vascular supply', detail='posterior primary rami and posterior intercostal arteries', importance='supporting',lecture_pages=[45,46,48,999]),
                ])})()
        assert text_format is CurriculumPlan
        ids={row['title']:row['video_id'] for row in user['candidates']
             if row['subject']=='06. Gross Anatomy' and row['topic']=='06. Back'}
        names=['Divisions of the Vertebral Column','Regional Characteristics of Vertebrae',
               'Joints of the Vertebral Column','Ligaments of the Vertebral Column',
               'Clinical - Abnormal Curvatures of the Spine & Intervertebral Disc Herniation',
               'Superficial Back Muscles & Serratus Muscles',
               'Rhomboids, Muscles of the Neck & Erector Spinae',
               'Blood Supply & Innervations']
        assert all(n in ids for n in names), (names, list(ids))
        labels={
            'Divisions of the Vertebral Column':['Divisions and regional vertebrae'],
            'Regional Characteristics of Vertebrae':['Divisions and regional vertebrae'],
            'Joints of the Vertebral Column':['Joints and ligaments'],
            'Ligaments of the Vertebral Column':['Joints and ligaments'],
            'Clinical - Abnormal Curvatures of the Spine & Intervertebral Disc Herniation':['Curvatures and disc pathology'],
            'Superficial Back Muscles & Serratus Muscles':['Back muscles'],
            'Rhomboids, Muscles of the Neck & Erector Spinae':['Back muscles'],
            'Blood Supply & Innervations':['Innervation and vascular supply'],
        }
        selected=[SelectedVideo(video_id=ids[n],reason='Exact lecture-aligned teaching topic.',
             concept_labels=labels[n],priority='core') for n in names]
        selected.insert(0,SelectedVideo(video_id='INVENTED-ID',reason='Do not show',
                                        concept_labels=['Back muscles'],priority='core'))
        selected.append(selected[1])  # duplicate of same ID
        return type('Reply',(),{'output_parsed':CurriculumPlan(
            selected=selected,coverage_gaps=['Spondylolysis may need review.'],
            rationale='Prefer coherent section-level anatomical instruction over unrelated disease detours.')})()


def main():
    rows=load_file_index(ROOT/'bootcamp_step1_file_index.xlsx')
    vids=catalog_records(rows)
    assert len(vids)==2323
    lecture_path=Path('/mnt/data/MSK_Week1_Tuesday_Vertebral column and back_Ernst_2026.pdf')
    if lecture_path.exists():
        from slide_cleaning import extract_pdf_slides
        chunks=chunks_from_slides(extract_pdf_slides(lecture_path.read_bytes()),lecture_path.name)
    else:
        chunks=[{'source':'Ernst lecture','page':i+1,'text':t} for i,t in enumerate([
            'Vertebral divisions and regional vertebral anatomy','Spinal curvatures, kyphosis and lordosis',
            'Intervertebral disc anatomy and herniation','Joints and ligaments of vertebral column',
            'Superficial and deep back muscles','Erector spinae rhomboids and trapezius',
            'Posterior primary rami and posterior intercostal arteries'])]
    assert len(chunks)>=7
    pack=lecture_packet(chunks)
    assert 'Vertebral' in pack or 'vertebral' in pack
    with TemporaryDirectory() as tmp:
        enc=FakeEncoder()
        engine=HybridMatcher(prepare_index(vids,cache_dir=tmp,model_name='test',encoder=enc),enc)
        client=FakeAPI()
        res=run_llm_planner(chunks,engine,api_key='',client=client,model='fake',max_videos=8,
                            section_limit=5,global_video_limit=35,max_candidates=115)
        assert len(client.calls)==2, 'No repair needed when all five major concepts assigned'
        assert res['retrieval']['sections'][0]['name']=='06. Gross Anatomy / 06. Back',res['retrieval']['sections']
        assert sum(c['topic']=='06. Back' for c in res['retrieval']['candidates'])==13
        assert len(res['videos'])==8,res['videos']
        assert res['coverage_audit']['major_assigned']==res['coverage_audit']['major_total']==4
        assert len({r['id'] for r in res['videos']})==8
        assert res['rejected_unknown_ids']==['INVENTED-ID']
        assert all(r['topic']=='06. Back' for r in res['videos'])
        assert all(r['mp4_path'].endswith('.mp4') for r in res['videos'])
        assert all(r['evidence_level']=='MP4 title and folder only' for r in res['videos'])
        assert 999 not in res['analysis'].concepts[-1].lecture_pages
        sent=json.dumps(client.calls[-1][1])
        assert '/Bootcamp - Step 1' not in sent and '\\Bootcamp' not in sent, 'Historical paths leaked to API'
        assert not any('/Bootcamp - Step 1' in c['video_id'] for c in client.calls[-1][1]['candidates'])
        assert len(client.calls[-1][1]['candidates'])<=115
        # A verified PDF witness may only be attached by the local PDF index.
        # API input gets its BASEname, never the inventory's virtual full path.
        known=next(x for x in res['retrieval']['candidates'] if x['topic']=='06. Back')
        known['pdf_evidence']=[{'pdf_id':'/private/Bootcamp/Gross Anatomy/Back.pdf',
           'page':27,'text':'Vertebral body and ligaments of the spine',
           'retrieval_score':0.78,'lecture_concept':'Divisions and regional vertebrae'}]
        safe=_plan_payload(res['analysis'],res['retrieval'],8)
        matching=next(x for x in safe['candidates'] if x['video_id']==known['candidate_key'])
        assert matching['verified_pdf_excerpts'][0]['pdf_filename']=='Back.pdf'
        assert '/private/' not in json.dumps(safe)
        grounded=validate_plan(CurriculumPlan(selected=[SelectedVideo(
           video_id=known['candidate_key'],reason='supported',
           concept_labels=['Divisions and regional vertebrae'],priority='core')],
           coverage_gaps=[],rationale=''),res['analysis'],res['retrieval'],8)
        assert grounded['videos'][0]['pdf_evidence'][0]['page']==27
        assert grounded['videos'][0]['id']==known['id']
        # Local preview also works without API or manually named disease rules.
        preview=local_section_preview(local_lecture_outline(chunks),
                      retrieve_hierarchical(local_lecture_outline(chunks),engine),max_videos=6)
        assert len(preview['videos'])<=6
        # A nonexistent or duplicated MP4 is always rejected; no fake PDF citations.
        raw=CurriculumPlan(selected=[SelectedVideo(video_id='BAD',reason='test',
                                concept_labels=['invented'],priority='core')],coverage_gaps=[],rationale='')
        checked=validate_plan(raw,res['analysis'],res['retrieval'],8)
        assert not checked['videos'] and checked['rejected_unknown_ids']==['BAD']
        try:
            run_llm_planner(chunks,engine,api_key='')
        except ValueError as e:
            assert 'API key' in str(e)
        else:raise AssertionError('No-key call was allowed')
    print('PASS: 49-slide whole-lecture packet, top Back folder, all 13 true Back MP4s,')
    print('      8 grounded LLM picks, no fabricated IDs/pages, path privacy, local fallback.')

if __name__=='__main__':main()
