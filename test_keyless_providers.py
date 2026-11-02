"""Pure offline regression tests: no Puter sign-in, no API call, no Ollama install."""
import json
from unittest.mock import patch
from io import BytesIO

from puter_workflow import (analysis_request, plan_request, parse_json_response,
                            validated_lecture, validated_shortlist)
from llm_providers import OllamaResponses
from llm_planner import LectureAnalysis, TeachingConcept


def main():
    chunks=[{'source':'back_lecture.pdf','page':3,'text':'Divisions and regional anatomy of the cervical and lumbar spine.'},
            {'source':'back_lecture.pdf','page':12,'text':'Anterior and posterior longitudinal ligaments and ligamentum flavum.'}]
    req=analysis_request(chunks)
    assert 'page 3' in req['payload']['lecture_text_by_page']
    raw={'lecture_focus':'Back anatomy','concepts':[{'label':'Vertebral structure',
       'detail':'Spine parts and regions','importance':'major','lecture_pages':[3,999]},
       {'label':'Ligaments','detail':'Vertebral ligaments', 'importance':'major','lecture_pages':[12]}],
       'notes':'Spinal anatomy'}
    parsed=validated_lecture('```json\n'+json.dumps(raw)+'\n```',chunks)
    assert parsed.concepts[0].lecture_pages==[3]
    assert len(parsed.concepts)==2
    record={'candidate_key':'allowed123','id':'mp4-real-id','title':'Ligaments of the Vertebral Column',
      'subject':'06. Gross Anatomy','topic':'06. Back','mp4_path':'C:\\historical\\x.mp4',
      'retrieval_score':.7,'matched_concepts':[{'label':'Ligaments','score':.8,'importance':'major'}],
      'pdf_evidence':[{'pdf_id':'06. Gross Anatomy/Back.pdf','page':27,'text':'vertebral ligaments',
                       'retrieval_score':.76,'lecture_concept':'Ligaments'}],
      'pdf_indexed':True,'origin':'full topic folder'}
    retrieval={'sections':[], 'candidates':[record], 'queries':['Ligaments']}
    task=plan_request(parsed,retrieval,4)
    assert 'historical' not in json.dumps(task['payload'])
    assert task['payload']['candidates'][0]['video_id']=='allowed123'
    answer={'selected':[{'video_id':'FAKE','reason':'Nope','concept_labels':['Ligaments'],'priority':'core'},
                        {'video_id':'allowed123','reason':'Actual teaching match',
                         'concept_labels':['Ligaments'],'priority':'core'}],
            'coverage_gaps':[],'rationale':'Select coherent anatomy'}
    result=validated_shortlist(json.dumps(answer),parsed,retrieval,4)
    assert len(result['videos'])==1
    assert result['rejected_unknown_ids']==['FAKE']
    assert result['videos'][0]['mp4_path']=='C:\\historical\\x.mp4'
    assert result['videos'][0]['pdf_evidence'][0]['page']==27
    assert result['videos'][0]['lecture_pages']==[12]
    # Real Puter reply: the model returns a single sentence, not ["..."]
    # for coverage_gaps. The shortlist must still validate and preserve real IDs.
    malformed=dict(answer, coverage_gaps='No specific gaps identified; the selected videos collectively cover major concepts.')
    fixed=validated_shortlist(json.dumps(malformed), parsed, retrieval, 4)
    assert len(fixed['videos']) == 1
    assert fixed['videos'][0]['id'] == 'mp4-real-id'
    assert all('No specific gaps' not in gap for gap in fixed['coverage_gaps'])
    assert any('Vertebral structure' in gap for gap in fixed['coverage_gaps']),fixed['coverage_gaps']
    assert fixed['rejected_unknown_ids'] == ['FAKE']
    for gap_value, expected in [(None, []), ('', []), ([], []), (['Disc pathology not covered'], ['Disc pathology not covered'])]:
        clean=validated_shortlist(json.dumps(dict(answer, coverage_gaps=gap_value)), parsed, retrieval, 4)
        assert clean['coverage_gaps'][:len(expected)] == expected, clean['coverage_gaps']
        assert any('Vertebral structure' in x for x in clean['coverage_gaps']),clean['coverage_gaps']
    # Still reject malformed objects instead of fabricating video records.
    try:
        validated_shortlist(json.dumps(dict(answer, coverage_gaps={'not': 'a list'})), parsed, retrieval, 4)
    except ValueError:
        pass
    else:
        raise AssertionError('Unsupported gap object silently accepted')
    try: validated_shortlist('not json',parsed,retrieval,4)
    except ValueError: pass
    else: raise AssertionError('invalid model response accepted')

    class DummyContext:
        def __init__(self, blob):self.blob=blob
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps({'message':{'content':self.blob}}).encode()
    captured=[]
    def mock_urllib(request,timeout=None):
        payload=json.loads(request.data)
        captured.append(payload)
        assert request.full_url=='http://127.0.0.1:11434/api/chat'
        assert payload['format']['title']=='LectureAnalysis'
        return DummyContext(json.dumps(raw))
    adapter=OllamaResponses()
    with patch('urllib.request.urlopen',side_effect=mock_urllib):
        got=adapter.responses.parse(model='test-model', input=[{'role':'system','content':'Test'},
                           {'role':'user','content':'A lecture'}], text_format=LectureAnalysis)
    assert got.output_parsed.lecture_focus=='Back anatomy'
    assert captured[0]['model']=='test-model'
    try:OllamaResponses('https://some-server.example.org')
    except ValueError:pass
    else:raise AssertionError('non-local Ollama endpoint accepted')
    print('PASS: Puter request packets, strict candidate whitelist, no historical paths,')
    print('      JSON validation, page sanitization and offline Ollama provider adapter')

if __name__=='__main__':main()
