"""Coverage-constrained planner regression; all inference is a deterministic fake.

Checks the reported real failure: spine-only candidates initially selected while
back muscles/neurovasculature are available in the same actual inventory folder.
No medical vocabulary is hardcoded in production code.
"""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from file_index_catalog import load_file_index
from semantic_matcher import HybridMatcher, catalog_records, prepare_index
from llm_planner import (LectureAnalysis, TeachingConcept, CurriculumPlan, SelectedVideo,
                         run_llm_planner, _plan_payload, validate_plan)
from puter_workflow import (plan_request, validated_shortlist, repair_request)
from test_llm_planner import FakeEncoder

ROOT=Path(__file__).resolve().parent


class SpineOnlyThenCoverage:
    def __init__(self):
        self.responses=self
        self.calls=[]

    def parse(self, *, model, input, text_format):
        payload=json.loads(input[1]['content'])
        self.calls.append((text_format.__name__,payload,input[0]['content']))
        if text_format is LectureAnalysis:
            return type('R',(),{'output_parsed':LectureAnalysis(
                lecture_focus='Vertebral anatomy, back muscles and neurovascular supply',
                notes='Two substantial halves: spine followed by back muscles',
                concepts=[
                    TeachingConcept(label='Vertebral divisions',detail='Regions and spinal curvatures',
                        importance='major',lecture_pages=[4,5,6]),
                    TeachingConcept(label='Vertebral regions',detail='Cervical thoracic lumbar sacrum coccyx',
                        importance='major',lecture_pages=[7,13,20,21,22,25]),
                    TeachingConcept(label='Vertebral joints and ligaments',
                        detail='Facet joints, atlas axis and vertebral ligaments',
                        importance='major',lecture_pages=[8,12,15,16,17]),
                    TeachingConcept(label='Disc herniation and abnormal curvatures',
                        detail='Kyphosis lordosis scoliosis annulus nucleus disc herniation',
                        importance='major',lecture_pages=[5,6,9,10,11]),
                    TeachingConcept(label='Superficial back muscles',
                        detail='Trapezius latissimus serratus',
                        importance='major',lecture_pages=[28,29,30,31,32,33,34,38]),
                    TeachingConcept(label='Intermediate and deep back muscles',
                        detail='Rhomboids levator scapulae erector spinae multifidus',
                        importance='major',lecture_pages=[35,36,40,41,42,43,44]),
                    TeachingConcept(label='Back blood supply and innervation',
                        detail='Posterior rami intercostal arteries cutaneous innervation',
                        importance='major',lecture_pages=[37,45,46,47,48]),
                    TeachingConcept(label='Sacrum and coccyx detail',detail='Foramina sacral hiatus',
                        importance='supporting',lecture_pages=[22,23,24,25]),
                    TeachingConcept(label='Back muscle functions',detail='Extension rotation lateral flexion',
                        importance='supporting',lecture_pages=[29,32,42,44]),
                ])})()
        assert text_format is CurriculumPlan
        # Each recommendation must use an actual opaque ID, even in a mock.
        ids={r['title']:r['video_id'] for r in payload['candidates']
             if r['subject']=='06. Gross Anatomy' and r['topic']=='06. Back'}
        mapping={
            'Divisions of the Vertebral Column':['Vertebral divisions'],
            'Regional Characteristics of Vertebrae':['Vertebral regions'],
            'Joints of the Vertebral Column':['Vertebral joints and ligaments'],
            'Ligaments of the Vertebral Column':['Vertebral joints and ligaments'],
            'Clinical - Abnormal Curvatures of the Spine & Intervertebral Disc Herniation':['Disc herniation and abnormal curvatures'],
            'Superficial Back Muscles & Serratus Muscles':['Superficial back muscles'],
            'Rhomboids, Muscles of the Neck & Erector Spinae':['Intermediate and deep back muscles'],
            'Blood Supply & Innervations':['Back blood supply and innervation'],
            'Sacrum & Coccyx':['Sacrum and coccyx detail'],
            'Muscle Functions':['Back muscle functions'],
        }
        assert all(name in ids for name in mapping),list(ids)
        assert 'complete_folder_video_lists' in payload
        assert 'concept_search_hints' in payload
        folder=next(x for x in payload['complete_folder_video_lists']
                    if x['folder']=='06. Gross Anatomy / 06. Back')
        assert folder['complete'] and len(folder['videos'])==13
        if len([x for x in self.calls if x[0]=='CurriculumPlan'])==1:
            # Simulate actual reported LLM mistake: 4 spine videos, no muscles.
            choices=list(mapping)[:5:2]+[list(mapping)[3]]
        else:
            assert payload['coverage_audit']['missing_major_concepts'],payload
            choices=list(mapping)
        selected=[SelectedVideo(video_id=ids[n], reason='Teaches '+n+' in source lecture.',
                   concept_labels=mapping[n],priority='core' if j<8 else 'optional')
                  for j,n in enumerate(choices)]
        return type('R',(),{'output_parsed':CurriculumPlan(
            selected=selected,coverage_gaps=[],rationale='Select an inclusive but compact curriculum.')})()


def main():
    rows=load_file_index(ROOT/'bootcamp_step1_file_index.xlsx')
    vids=catalog_records(rows)
    from slide_cleaning import extract_pdf_slides
    from semantic_matcher import chunks_from_slides
    filename=Path('/mnt/data/MSK_Week1_Tuesday_Vertebral column and back_Ernst_2026.pdf')
    chunks=(chunks_from_slides(extract_pdf_slides(filename.read_bytes()),filename.name)
            if filename.exists() else
            [{'page':i+1,'source':'lecture','text':x} for i,x in enumerate([
                'Spine vertebra regions curvatures joints ligaments','Disc herniation kyphosis scoliosis',
                'Trapezius rhomboids latissimus serratus','Erector spinae and back nerves arteries'])])
    with TemporaryDirectory() as d:
        enc=FakeEncoder()
        engine=HybridMatcher(prepare_index(vids,cache_dir=d,model_name='test',encoder=enc),enc)
        client=SpineOnlyThenCoverage()
        result=run_llm_planner(chunks,engine,api_key='',client=client,model='fake',
                               max_videos=10,max_candidates=115)
        assert len(client.calls)==3,[(x[0],list(x[1])) for x in client.calls]
        assert result['coverage_audit']['major_total']==7
        assert result['coverage_audit']['major_assigned']==7
        assert len(result['videos'])==10
        assert sum(x['priority']=='optional' for x in result['videos'])==2
        assert result['coverage_audit']['missing_major_concepts']==[]
        assert 'Superficial Back Muscles & Serratus Muscles' in [x['title'] for x in result['videos']]
        assert 'Blood Supply & Innervations' in [x['title'] for x in result['videos']]
        assert len({x['id'] for x in result['videos']})==10
        assert len({x['candidate_key'] for x in result['retrieval']['candidates']})==len(result['retrieval']['candidates'])
        # Puter path must produce a repair request only when the audit exposes missing concepts.
        analysis=result['analysis']; retrieval=result['retrieval']; pack=_plan_payload(analysis,retrieval,10)
        first_choices=list(SpineOnlyThenCoverage().calls) # fresh object, empty (only used below)
        ids={r['title']:r['video_id'] for r in pack['candidates']}
        raw={'selected':[{'video_id':ids['Divisions of the Vertebral Column'],
             'reason':'Spine regions','priority':'core','concept_labels':['Vertebral divisions']}],
             'coverage_gaps':'No specific gaps identified', 'rationale':'incomplete'}
        first=validated_shortlist(json.dumps(raw),analysis,retrieval,10)
        assert first['coverage_audit']['major_assigned']==1
        assert len(first['coverage_audit']['missing_major_concepts'])==6
        assert not any('No specific gaps' in s for s in first['coverage_gaps'])
        repair=repair_request(analysis,retrieval,first,10)
        assert 'coverage_audit' in repair['payload']
        assert repair['payload']['previous_selection'][0]['video_id']==ids['Divisions of the Vertebral Column']
        assert '/Bootcamp' not in json.dumps(repair['payload'])
        # Unknown model recommendation remains excluded even in repair path.
        corrupted=CurriculumPlan(selected=[SelectedVideo(video_id='FAKE',reason='Fake',
              concept_labels=['Back blood supply and innervation'],priority='core')],
              coverage_gaps=[],rationale='')
        checked=validate_plan(corrupted,analysis,retrieval,10)
        assert not checked['videos'] and checked['coverage_audit']['needs_revision']
    print('PASS: 4 spine-only videos fail coverage; targeted revision selects 8 core + 2 optional,')
    print('      includes back muscles and neurovascular, preserves real-ID whitelist/Puter privacy.')


if __name__=='__main__': main()
