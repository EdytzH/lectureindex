"""Integration checks for source-first PDF search, attribution safety and persistence.
No medical answer keys or video-to-concept lookup tables in production logic.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import fitz
from pdf_knowledge_base import build_knowledge_base,PDFKnowledgeBase
from file_index_catalog import load_file_index
from test_llm_planner import FakeEncoder
from semantic_matcher import catalog_records,HybridMatcher,prepare_index
from llm_planner import LectureAnalysis,TeachingConcept,retrieve_hierarchical,_plan_payload


def test_source_first():
    with TemporaryDirectory() as t:
        root=Path(t);folder=root/'pdfs';folder.mkdir();cache=root/'ix'
        doc=fitz.open()
        pages=[
            'Alpha Mechanics - Module A  Rotating articulated joints allow movement between rigid structural segments and limit the range of motion.',
            'Beta Control - Module B  Superficial motor actuators produce external motion while deep stabilizing actuators preserve posture and control.',
            'Gamma Review  Special failure patterns occur when one segment is damaged and the electrical signals are interrupted.'
        ]
        for text in pages:
            pg=doc.new_page();pg.insert_textbox((30,30,570,320),text,fontsize=10)
        doc.set_toc([[1,'Alpha Mechanics',1],[1,'Beta Control',2],[1,'Gamma Review',3]])
        doc.save(folder/'Modules.pdf');doc.close()
        inv=[{'type':'PDF','path':'Engineering/Workshop/Modules.pdf','filename':'Modules.pdf','subject':'Engineering','topic':'Workshop'},
             {'type':'MP4','path':'Engineering/Workshop/1. Alpha Mechanics.mp4','filename':'1. Alpha Mechanics.mp4','subject':'Engineering','topic':'Workshop'},
             {'type':'MP4','path':'Engineering/Workshop/2. Beta Control.mp4','filename':'2. Beta Control.mp4','subject':'Engineering','topic':'Workshop'}]
        first=build_knowledge_base(inv,folder,cache,use_ocr=False)
        assert first['passages']>=3,first
        assert sum(first['link_status_counts'].values())==first['passages']
        assert first['link_status_counts'].get('verified',0)==0
        kb=PDFKnowledgeBase(cache)
        hit=kb.search('deep stabilizing actuators preserve posture')[0]
        assert hit['page']==2 and hit['link_status'] in {'suggested','unmatched'},hit
        assert not hit['video_id'] # No inferred video proof.
        assert kb.stats()['linked_mp4s']==0
        vec=kb.build_vectors(FakeEncoder(),'fake')
        assert vec['count']==first['passages']
        results=kb.search('superficial motor actuators',encoder=FakeEncoder(),model_name='fake')
        assert any(h['page']==2 for h in results)
        actual_id=inv[2]['path']
        kb.review(inv[0]['path'],2,2,actual_id,[r['path'] for r in inv if r['type']=='MP4'])
        hit=kb.search('deep stabilizing actuators preserve posture')[0]
        assert hit['video_id']==actual_id and hit['link_status']=='reviewed',hit
        try:
            kb.review(inv[0]['path'],2,2,'fabricated.mp4',[r['path'] for r in inv if r['type']=='MP4'])
        except ValueError: pass
        else:raise AssertionError('Invalid video was accepted')
        kb.close()
        repeated=build_knowledge_base(inv,folder,cache,use_ocr=False)
        # A newly reviewed range changes attribution and therefore refreshes
        # the document mapping once; only the following identical scan is cached.
        assert repeated['scanned']==1,repeated
        reused=build_knowledge_base(inv,folder,cache,use_ocr=False)
        assert reused['reused_unchanged']==1 and reused['scanned']==0,reused
        kb=PDFKnowledgeBase(cache)
        assert kb.stats()['linked_mp4s']==1
        assert kb.reviews()[0]['video_id']==actual_id
        kb.close()
        # A rescan can also import externally verified ranges, separate from
        # heading guesses, if the existing verified-section index is present.
        (folder/'Modules.pdf').touch()
        m={'videos':{inv[1]['path']:{'pdf_id':inv[0]['path'],
                    'start':1,'end':1,'local_pdf_path':str(folder/'Modules.pdf')}}}
        new=build_knowledge_base(inv,folder,cache,pdf_manifest=m)
        assert new['scanned']==1
        kb=PDFKnowledgeBase(cache)
        assert kb.search('rotating articulated joints')[0]['link_status']=='verified'
        assert kb.search('deep stabilizing actuators')[0]['link_status']=='reviewed'
        kb.close()
        return 'synthetic source-first scan, no fabricated proof, reviewed/verified links, saved vectors and repeat scan'


def test_live_available_pdf():
    path=Path('/mnt/data/0. Embryonic Period - Weeks 3-8.pdf')
    if not path.exists():return 'real PDF unavailable in this test environment'
    inv=load_file_index(Path(__file__).parent/'bootcamp_step1_file_index.xlsx')
    with TemporaryDirectory() as t:
        folder=Path(t)/'pdfs';folder.mkdir()
        (folder/path.name).symlink_to(path)
        summary=build_knowledge_base(inv,folder,Path(t)/'ix')
        assert summary['passages']>=40,summary
        kb=PDFKnowledgeBase(Path(t)/'ix')
        hits=kb.search('upper limb buds rotate dorsally during development',limit=20)
        assert any(h['filename']==path.name and 27<=h['page']<=41 for h in hits),hits
        suggested=[h for h in hits if h['candidate_video_ids']]
        assert suggested, hits
        assert all(h['video_id']=='' for h in hits) # Only suggestions until reviewed
        kb.close()
        return 'real 83-page PDF: section text retrievable before video mapping'


def main():
    print('PASS:',test_source_first())
    print('PASS:',test_live_available_pdf())

if __name__=='__main__':main()


def test_concept_retrieval_attribution():
    """Only reviewed/verified PDF passages may credit a candidate video."""
    with TemporaryDirectory() as t:
        root=Path(t);folder=root/'pdf';folder.mkdir();cache=root/'idx'
        doc=fitz.open()
        for sentence in [
            'Energy Conversion Assemblies: Photovoltaic modules generate electrical energy under sunlight illumination from nearby solar cells.',
            'Pump Valve Connections: The intake valves operate using fluid pressure across a tubular hydraulic coupling and a balancing chamber.'
        ]:
            page=doc.new_page();page.insert_textbox((30,30,580,240),sentence,fontsize=10)
        doc.save(folder/'Engineering.pdf');doc.close()
        inv=[{'type':'PDF','path':'Course/Mechanical/Engineering.pdf','filename':'Engineering.pdf','subject':'Course','topic':'Mechanical'},
             {'type':'MP4','path':'Course/Mechanical/1. Solar Energy Conversion.mp4','filename':'1. Solar Energy Conversion.mp4','subject':'Course','topic':'Mechanical'},
             {'type':'MP4','path':'Course/Mechanical/2. Hydraulic Pump Maintenance.mp4','filename':'2. Hydraulic Pump Maintenance.mp4','subject':'Course','topic':'Mechanical'}]
        build_knowledge_base(inv,folder,cache)
        kb=PDFKnowledgeBase(cache)
        encoder=FakeEncoder()
        rows=catalog_records(inv)
        engine=HybridMatcher(prepare_index(rows,cache_dir=root/'title_ix',model_name='fake',encoder=encoder),encoder)
        a=LectureAnalysis(lecture_focus='Mechanical systems',notes='',concepts=[
            TeachingConcept(label='Solar energy conversion',detail='photovoltaic modules generate electrical energy',importance='major',lecture_pages=[1]),
            TeachingConcept(label='Hydraulic valve connections',detail='fluid pressure and intake valve connections',importance='major',lecture_pages=[2])])
        result=retrieve_hierarchical(a,engine,knowledge_base=kb,section_limit=1,global_video_limit=2,max_candidates=6)
        assert result['pdf_first_index_active']
        assert result['unlinked_pdf_excerpts'],result
        assert not any(r['pdf_evidence'] for r in result['candidates']),result
        kb.review(inv[0]['path'],1,1,inv[1]['path'],[inv[1]['path'],inv[2]['path']])
        linked=retrieve_hierarchical(a,engine,knowledge_base=kb,section_limit=1,global_video_limit=2,max_candidates=6)
        candidate=next(r for r in linked['candidates'] if r['id']==inv[1]['path'])
        assert candidate['pdf_evidence'],candidate
        assert candidate['pdf_evidence'][0]['page']==1
        assert any(h['page']==2 for h in linked['unlinked_pdf_excerpts'])
        packet=_plan_payload(a,linked,6)
        assert packet['candidates']
        assert packet['unattributed_pdf_excerpts_for_gap_detection']
        kb.close()
    return 'LLM retrieval: unlinked PDF excerpts surface but cannot falsely support MP4; reviewed excerpts can'

if __name__=='__main__':
    print('PASS:',test_concept_retrieval_attribution())
