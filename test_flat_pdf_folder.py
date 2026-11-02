"""Regression: workbook virtual folders + locally flattened PDFs, no MP4 files."""
from pathlib import Path
from tempfile import TemporaryDirectory
import fitz

from beta_catalog import folder_candidates, records_for_search, resolve_pdf_library
from file_index_catalog import load_file_index
from pdf_evidence import tag_library

here = Path(__file__).resolve().parent
records = load_file_index(here/'bootcamp_step1_file_index.xlsx')
all_videos = records_for_search(records)
assert len(all_videos) == 2323
assert sum(r['type']=='PDF' for r in records) == 94
hand = next(v for v in all_videos if v['title']=='Carpals, Metacarpals and Phalanges')
assert hand['subject']=='06. Gross Anatomy' and hand['topic']=='05. Hand'
hand_pdf = folder_candidates(hand,records)
assert len(hand_pdf)==1 and hand_pdf[0]['filename']=='0. Hand.pdf'

with TemporaryDirectory() as t:
    flat=Path(t)/'all_the_pdfs'
    flat.mkdir()
    # PDF is FLAT: no original Gross Anatomy/Hand directory, and no MP4s anywhere.
    out=flat/'0. Hand.pdf'
    with fitz.open() as doc:
        for text in [
            'TABLE OF CONTENTS\n1. Carpals, Metacarpals and Phalanges\n2. Wrist Joint',
            'Carpals, Metacarpals and Phalanges\nCarpal bones and metacarpal bones with proximal middle and distal phalanges. The limb contains joints and supporting muscles.',
            'Wrist Joint\nWrist movement, carpal articulations, hand anatomy and bones. Ligaments and the proximal carpal row are important.',
        ]:
            pg=doc.new_page()
            pg.insert_text((50,80), text)
        doc.set_toc([[1,'TABLE OF CONTENTS',1],
                     [2,'Carpals, Metacarpals and Phalanges',2],
                     [2,'Wrist Joint',3]])
        doc.save(out)
    with fitz.open() as doc:
        doc.new_page().insert_text((50,90),'Biochemistry protein enzyme metabolism biochemical reactions of living organisms')
        doc.save(flat/'Biochemistry.pdf')
    resolved, summary, other = resolve_pdf_library(records,flat)
    assert summary['local_pdf_files']==2,summary
    assert summary['matched']==2 and summary['ambiguous']==0,summary
    assert summary['missing']==92,summary
    assert resolved[hand_pdf[0]['path']]['physical']==str(out)
    assert 'flat folder' in resolved[hand_pdf[0]['path']]['status']
    assert not other
    manifest = tag_library(records,root=flat,cache_dir=Path(t)/'cached')
    mapped=manifest['videos'][hand['path']]
    assert mapped['pdf_id']==hand_pdf[0]['path'],mapped
    assert mapped['local_pdf_path']==str(out),mapped
    assert mapped['start']==2,mapped
    assert mapped['text_chunks']>=1,mapped
    assert manifest['discovery']['matched']==2
    assert sum(bool(v['pdf_candidates']) for v in manifest['videos'].values())==2266
    assert len(list(flat.rglob('*.mp4')))==0

    # Exact name has precedence if both '0. Hand.pdf' and 'Hand.pdf' are present.
    (flat/'Hand.pdf').touch()
    resolved,summary,extra=resolve_pdf_library(records,flat)
    assert resolved[hand_pdf[0]['path']]['physical']==str(out)
    assert str(flat/'Hand.pdf') in extra
    (flat/'Hand.pdf').unlink()

    # A unique name without the optional index prefix is still accepted.
    out.rename(flat/'Hand.pdf')
    resolved,summary,_=resolve_pdf_library(records,flat)
    assert resolved[hand_pdf[0]['path']]['physical']==str(flat/'Hand.pdf')
    assert 'filename alias' in resolved[hand_pdf[0]['path']]['status']

    # Two alias candidates with no exact inventory name are ambiguous, not guessed.
    (flat/'00. Hand.pdf').touch()
    resolved,summary,_=resolve_pdf_library(records,flat)
    assert resolved[hand_pdf[0]['path']]['physical']==''
    assert summary['ambiguous']==1,summary
    (flat/'00. Hand.pdf').unlink()

    # Paths in Excel are logical only; never open them without an explicit local root.
    no_root,totals,_=resolve_pdf_library(records,None)
    assert not no_root[hand_pdf[0]['path']]['physical']
    assert totals['matched']==0

    # Detect collision if two logical PDFs map to one flat physical PDF.
    fake_other={**hand_pdf[0],'subject':'Other Subject','topic':'Other Topic',
                'path':'/Other Subject/Other Topic/0. Hand.pdf'}
    collision,summary,_=resolve_pdf_library([hand_pdf[0],fake_other],flat)
    assert summary['ambiguous']==2,summary
    assert all(not x['physical'] for x in collision.values())

print('PASS: flat-folder discovery, exact and alias names, collision safety, MP4-free tagging and page evidence')
