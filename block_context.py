"""Optional curriculum block preference for video relevance ranking.

This is a soft rank adjustment, never a filter. It uses video titles and PDF
filenames (source provenance), not noisy page text or embeddings.
"""
import re

BLOCK_TERMS = {
    'Musculoskeletal': ('musculoskeletal', 'msk', 'orthopedic', 'orthopaedic', 'bone', 'joint', 'skeletal', 'muscle', 'limb', 'vertebral', 'spine', 'rheumatology'),
    'Cardiovascular': ('cardiovascular', 'cardiology', 'cardiac', 'heart', 'vascular', 'circulation', 'ecg', 'ekg'),
    'Pulmonary / Respiratory': ('pulmonary', 'respiratory', 'lung', 'lungs', 'airway', 'pleural'),
    'Gastrointestinal': ('gastroenterology', 'gasteroenterology', 'gastrointestinal', 'digestive', 'hepatology', 'hepatic', 'liver', 'pancreas', 'biliary', 'intestine'),
    'Renal / Urinary': ('nephrology', 'renal', 'kidney', 'urinary', 'nephron', 'ureter'),
    'Neurology': ('neurology', 'neuroanatomy', 'neuroscience', 'brain', 'cerebral', 'ventricle', 'subcortical', 'spinal cord'),
    'Dermatology': ('dermatology', 'dermatologic', 'skin', 'cutaneous', 'rash', 'epidermis'),
    'Endocrine': ('endocrine', 'endocrinology', 'thyroid', 'pituitary', 'adrenal', 'hormone'),
    'Hematology / Oncology': ('hematology', 'haematology', 'oncology', 'blood', 'leukemia', 'lymphoma', 'cancer'),
    'Reproductive': ('reproductive', 'obstetric', 'gynecology', 'gynaecology', 'pregnancy', 'uterus', 'testis'),
    'Immunology': ('immunology', 'immune', 'allergy', 'hypersensitivity'),
    'Infectious Disease': ('infectious', 'infection', 'microbiology', 'bacteria', 'virus', 'fungal', 'parasite'),
    'Embryology': ('embryology', 'embryonic', 'developmental', 'pharyngeal', 'fetal'),
}


def block_affinity(block: str, file: str, title: str) -> float:
    """0..1 preference, with a small title signal and stronger file-category signal."""
    if block not in BLOCK_TERMS:
        return 0.0
    terms = BLOCK_TERMS[block]
    def matches(s):
        s = ' ' + re.sub(r'[^a-z0-9]+', ' ', s.lower()) + ' '
        return sum(bool(re.search(r'\b' + re.escape(term) + r'\b', s)) for term in terms)
    name_hits, title_hits = matches(file), matches(title)
    return min(1.0, (0.85 if name_hits else 0) + (0.55 if title_hits else 0))


def apply_block_preference(candidates: list[dict], block: str, boost: float = 0.12) -> list[dict]:
    """Re-score candidates without changing evidence, inclusion, or chunk matching."""
    adjusted = []
    for candidate in candidates:
        affinity = block_affinity(block, candidate.get('file', ''), candidate.get('title', ''))
        adjusted.append({**candidate, 'base_score': candidate['score'],
                         'block_affinity': affinity,
                         'score': round(candidate['score'] * (1 + boost * affinity), 4)})
    return sorted(adjusted, key=lambda c: (c['score'], c.get('support', 0)), reverse=True)
