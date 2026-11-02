"""Final planner regression: model claims cannot author the playlist.

Synthetic tests intentionally use NONmedical titles so no Bootcamp-specific
concept/video mapping can be embedded in the production matching algorithm.
The real-inventory test uses an automatic token-matching fake LLM, not a model
or a manually supplied answer key. No external services are contacted.
"""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from coverage_optimizer import (CoverageMatrix, ConceptAssessment, CoverageLink,
                                optimize_shortlist, validate_matrix)
from llm_planner import LectureAnalysis, TeachingConcept, run_llm_planner, coverage_request
from puter_workflow import matrix_request, validated_matrix
from semantic_matcher import HybridMatcher, catalog_records, prepare_index, tokens
from file_index_catalog import load_file_index
from test_llm_planner import FakeEncoder


def sample_retrieval():
    names=['Overview of Solar Panels','Solar Panel Wiring','Solar Panel Wiring - Advanced',
           'Wind Turbine Design','Wind Turbine Maintenance','Turbine Fracture Case',
           'River Water Filtration','River Water Filtration Safety','Overview of Turbine Fractures']
    rows=[]
    for i,name in enumerate(names):
        rows.append({'candidate_key':str(i),'id':f'catalog/folder/{i}.mp4','title':name,
                     'subject':'Renewable Engineering', 'topic':'Systems',
                     'mp4_path':f'catalog/folder/{i}.mp4','retrieval_score':.65,
                     'matched_concepts':[],'pdf_evidence':[],'pdf_indexed':False,'origin':'full topic folder'})
    return {'sections':[],'candidates':rows,'queries':[]}


def sample_analysis():
    return LectureAnalysis(lecture_focus='Technology, design, and safety', notes='',concepts=[
        TeachingConcept(label='Solar panel wiring',detail='Wiring circuits and controllers',importance='major',lecture_pages=list(range(1,12))),
        TeachingConcept(label='Wind turbine design',detail='Turbine blade design',importance='major',lecture_pages=list(range(12,24))),
        TeachingConcept(label='River water filtration',detail='Filtration mechanisms',importance='major',lecture_pages=list(range(24,36))),
        TeachingConcept(label='Fracture case study',detail='Case-specific secondary examples',importance='supporting',lecture_pages=[36])])


def synthetic_test():
    a=sample_analysis();r=sample_retrieval()
    # Duplicate solar-videos, unrelated fracture claims and made-up source quotes.
    matrix=CoverageMatrix(assessments=[
        ConceptAssessment(concept_label='Solar panel wiring',matches=[
            CoverageLink(video_id='1',fit='direct',support='title',evidence='Solar Panel Wiring',focus='wiring basics'),
            CoverageLink(video_id='2',fit='direct',support='title',evidence='Solar Panel Wiring',focus='advanced wiring')]),
        ConceptAssessment(concept_label='Wind turbine design',matches=[
            CoverageLink(video_id='3',fit='direct',support='title',evidence='Wind Turbine Design',focus='turbine design'),
            CoverageLink(video_id='5',fit='direct',support='title',evidence='Turbine Fracture Case',focus='turbine design')]),
        ConceptAssessment(concept_label='River water filtration',matches=[
            CoverageLink(video_id='6',fit='direct',support='title',evidence='River Water Filtration',focus='filter materials'),
            CoverageLink(video_id='7',fit='direct',support='title',evidence='Imaginary quote',focus='filter safety'),
            CoverageLink(video_id='999',fit='direct',support='title',evidence='Imaginary video',focus='invented'),
            CoverageLink(video_id='5',fit='direct',support='title',evidence='Imaginary quote',focus='unrelated claim')]),
        ConceptAssessment(concept_label='Fracture case study',matches=[
            CoverageLink(video_id='5',fit='direct',support='title',evidence='Turbine Fracture Case',focus='case study')]),
    ],summary='Organize foundational systems before one-off cases.')
    edges,bad=validate_matrix(matrix,a,r)
    assert len(bad)>=2,bad
    assert any(e.get('video_id')=='999' for e in bad)
    selected=optimize_shortlist(a,r,matrix,max_videos=4)
    titles=[v['title'] for v in selected['videos']]
    assert 'Solar Panel Wiring' in titles, titles
    assert 'Wind Turbine Design' in titles, titles
    assert 'River Water Filtration' in titles, titles
    assert 'Solar Panel Wiring - Advanced' not in titles, titles # redundant in a 4-item budget
    assert selected['coverage_audit']['major_assigned']==3,selected['coverage_audit']
    assert all(v['id'].endswith('.mp4') for v in selected['videos'])
    assert selected['rejected_unknown_ids']==['999']
    # Browser JSON flows through the identical locally validated path.
    packet=matrix_request(a,r,4)
    assert 'historical' not in json.dumps(packet['payload'])
    other=validated_matrix(matrix.model_dump_json(),a,r,4)
    assert [v['id'] for v in other['videos']]==[v['id'] for v in selected['videos']]
    return titles


class GenericMatrixAPI:
    """Deterministic word-overlap FAKE; deliberately not a curated answer key."""
    def __init__(self):self.responses=self;self.calls=[]
    def parse(self,*,model,input,text_format):
        payload=json.loads(input[1]['content']);self.calls.append((text_format.__name__,payload))
        if text_format is LectureAnalysis:
            return type('R',(),{'output_parsed':LectureAnalysis(
                lecture_focus='Vertebral structure and back anatomy',notes='',concepts=[
                TeachingConcept(label='Vertebral divisions',detail='vertebrae divisions curvature',importance='major',lecture_pages=[4,5]),
                TeachingConcept(label='Joints and vertebral ligaments',detail='joints ligaments vertebral column',importance='major',lecture_pages=[8,12]),
                TeachingConcept(label='Superficial back muscles',detail='back trapezius serratus muscles',importance='major',lecture_pages=[29,30,31,38]),
                TeachingConcept(label='Deep back muscles',detail='rhomboids erector spinae deep muscles back',importance='major',lecture_pages=[35,40,42]),
                TeachingConcept(label='Back blood supply and innervation',detail='blood supply innervations back',importance='major',lecture_pages=[45,46,48]),
            ])})()
        assert text_format is CoverageMatrix,text_format
        allrows=payload['candidates']
        assert 'complete_folder_video_lists' in payload
        assessments=[]
        for c in payload['lecture_concepts']:
            need=set(tokens(c['label']+' '+c['detail']))
            # Fake output purely from shared terms, not a medically knowledgeable oracle.
            available=[]
            for r in allrows:
                have=set(tokens(r['title']))
                overlap=len(need & have)
                if overlap>0:
                    available.append((overlap, r))
            available.sort(key=lambda x:(-x[0],x[1]['title']))
            matches=[CoverageLink(video_id=r['video_id'],fit='direct',support='title',
                                  evidence=r['title'],focus='Title-overlap synthetic test')
                     for _,r in available[:4]]
            assessments.append(ConceptAssessment(concept_label=c['label'],matches=matches))
        return type('R',(),{'output_parsed':CoverageMatrix(assessments=assessments,
             summary='Only a deterministic test of the two-call data path.')})()


def end_to_end_test():
    root=Path(__file__).parent
    rows=load_file_index(root/'bootcamp_step1_file_index.xlsx')
    inventory=catalog_records(rows)
    source=Path('/mnt/data/MSK_Week1_Tuesday_Vertebral column and back_Ernst_2026.pdf')
    if source.exists():
        from slide_cleaning import extract_pdf_slides
        from semantic_matcher import chunks_from_slides
        chunks=chunks_from_slides(extract_pdf_slides(source.read_bytes()),source.name)
    else:
        chunks=[{'source':'example lecture','page':p,'text':txt} for p,txt in enumerate([
            'Vertebral divisions and joints, ligaments',
            'Superficial back muscles and trapezius serratus',
            'Rhomboids, deep erector spinae back muscles',
            'Innervation blood supply of the back'],1)]
    with TemporaryDirectory() as temp:
        enc=FakeEncoder()
        engine=HybridMatcher(prepare_index(inventory,cache_dir=temp,model_name='test',encoder=enc),enc)
        fake=GenericMatrixAPI()
        report=run_llm_planner(chunks,engine,api_key='',client=fake,model='fake',max_videos=10)
        assert [c[0] for c in fake.calls]==['LectureAnalysis','CoverageMatrix']
        assert report['optimizer_method'].startswith('ID-')
        assert report['retrieval']['sections'][0]['name']=='06. Gross Anatomy / 06. Back'
        assert sum(x['topic']=='06. Back' for x in report['retrieval']['candidates'])==13
        assert len({v['id'] for v in report['videos']})==len(report['videos'])
        assert all(v['id'] in {v['id'] for v in inventory} for v in report['videos'])
        assert 'Bootcamp' not in json.dumps(fake.calls[-1][1]) or '/Bootcamp - Step 1' not in json.dumps(fake.calls[-1][1])
        return report


if __name__=='__main__':
    selected=synthetic_test()
    output=end_to_end_test()
    print('PASS: source anchors, rejection of invented citations/IDs, marginal coverage,')
    print('      two-stage Puter validation, real 2,323-video inventory, stable MP4 IDs.')
    print('Synthetic selections:',selected)
    print('Real-inventory smoke selections:',[v['title'] for v in output['videos']])
    print('NOTE: Real-inventory smoke uses naive word overlap, NOT a live LLM benchmark.')
