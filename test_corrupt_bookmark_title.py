"""Regression: a valid PDF divider can have a mismatched bookmark label and garbled native font.

The OCR function is mocked, so this test requires neither Tesseract nor the 66-MB example PDF.
Run: python test_corrupt_bookmark_title.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import fitz
import pdf_evidence

pdf_id = '/Library/Embryology/Embryonic Period/Embryonic Period.pdf'
folder = '/Library/Embryology/Embryonic Period'
vids = [
    {'path': f'{folder}/1. Face Development.mp4', 'filename':'1. Face Development.mp4', 'title':'Face Development', 'topic':'Embryonic Period'},
    {'path': f'{folder}/2. Limb Development.mp4', 'filename':'2. Limb Development.mp4', 'title':'Limb Development', 'topic':'Embryonic Period'},
    {'path': f'{folder}/3. Skin Development.mp4', 'filename':'3. Skin Development.mp4', 'title':'Skin Development', 'topic':'Embryonic Period'},
]
with TemporaryDirectory() as tmp:
    pdf = Path(tmp)/'Embryonic Period.pdf'
    with fitz.open() as doc:
        content = [
            'TABLE OF CONTENTS\n1. Face\n2. Limbs\n3. Skin',
            'Face Development\nNeural crest cells participate in craniofacial embryonic development. Nasal processes fuse and face structures differentiate.',
            'M?* %XG 2ULHQWDWLRQ\n%URHQ BRB?\nThe limb bud forms during the embryonic period. Skeletal primordia become cartilage and bone.',
            'Limb buds develop digital rays. The apical ectodermal ridge controls outgrowth and bone formation.',
            'Skin Development\nThe epidermis and dermis develop in embryonic life, including hair and integumentary appendages.',
        ]
        for t in content:
            p = doc.new_page()
            p.insert_text((45,80), t, fontsize=11)
        doc.set_toc([[1,'Contents',1], [2,'Face',2], [2,'Limbs',3], [2,'Skin',5]])
        doc.save(pdf)

    original = pdf_evidence.ocr_text
    pdf_evidence.ocr_text = lambda page: (
        'REVIEW\nOUTLINE\nLimb Development\nTopics in this Video\n'
        'Limb bud orientation, digital rays, skeletal formation, and limb rotation.'
        if page.number == 2 else page.get_text()
    )
    try:
        rows, chunks, stats = pdf_evidence.scan_document(pdf,vids,pdf_id,anchor_ocr=True)
    finally:
        pdf_evidence.ocr_text = original
    limb = rows[vids[1]['path']]
    assert (limb['start'],limb['end']) == (3,4), (limb, stats)
    assert 'OCR' in limb['method'], limb
    limb_chunks = [c for c in chunks if c['video_id'] == vids[1]['path']]
    assert limb_chunks and any(c['page'] == 4 for c in limb_chunks), limb_chunks
    assert not any(c['page'] == 1 for c in chunks), 'Contents page must not be searchable'
    print('PASS: mismatched bookmark name + unreadable title -> OCR divider evidence -> verified limb pages')
