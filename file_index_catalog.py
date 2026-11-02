"""Authoritative MP4/PDF inventory and folder-aware PDF resolution."""
from pathlib import Path, PureWindowsPath
import re
import pandas as pd


def norm(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).split())


def clean_name(s):
    return re.sub(r'^\s*\d+\s*[.)-]\s*', '', str(s)).strip()


def path_parts(value):
    """Handle Windows and POSIX paths regardless of OS."""
    return tuple(x.casefold() for x in str(value).replace('\\', '/').split('/') if x and x != '.')


def folder_key(r):
    # Subject/topic from the curated inventory is the primary identity.
    return norm(r.get('subject','')), norm(r.get('topic',''))


def same_file(a, b):
    return norm(clean_name(Path(a).stem)) == norm(clean_name(Path(b).stem))


def load_file_index(path):
    df = pd.read_excel(path, sheet_name='Files', dtype=str).fillna('')
    required = {'Subject', 'Topic', 'Filename', 'Type', 'Full path'}
    if not required.issubset(df.columns):
        raise ValueError('File index needs Files sheet columns: ' + ', '.join(sorted(required)))
    records=[]
    for row in df.to_dict('records'):
        typ=str(row['Type']).strip().upper().lstrip('.')
        file=str(row['Filename']).strip()
        if typ not in {'PDF','MP4'} or not file or Path(file).suffix.lower() != '.'+typ.lower():
            continue
        records.append({'subject':str(row['Subject']).strip(), 'topic':str(row['Topic']).strip(),
                        'filename':file,'type':typ,'path':str(row['Full path']).strip()})
    return records


def choose_pdf_record(pdf_path, records, root=None):
    """Match by relative folder path first, then unique filename (allow '0. ' prefix)."""
    pdf_path=Path(pdf_path)
    relparts=path_parts(pdf_path.relative_to(root) if root and pdf_path.is_relative_to(root) else pdf_path)
    pdfs=[r for r in records if r['type']=='PDF']
    candidates=[r for r in pdfs if same_file(r['filename'],pdf_path.name)]
    if not candidates: return None
    if len(candidates)==1: return candidates[0]
    # Nested relative paths in bootcamp_pdfs reflect inventory's subject/topic tree.
    hits=[]
    for r in candidates:
        parts=path_parts(r['path'])
        if len(relparts)>=2 and len(parts)>=len(relparts) and parts[-len(relparts):-1]==relparts[:-1]:
            hits.append(r)
    if len(hits)==1:return hits[0]
    if len(relparts)>=3:
        hits=[r for r in candidates if norm(r['subject'])==norm(relparts[-3]) and norm(r['topic'])==norm(relparts[-2])]
        if len(hits)==1:return hits[0]
    return None


def choose_index_titles(pdf_path, records, root=None):
    pdf=choose_pdf_record(pdf_path,records,root)
    if not pdf:return None,None,[]
    subject,topic=pdf['subject'],pdf['topic']
    direct=not topic or topic.lower().startswith('(files directly')
    items=[r for r in records if r['type']=='MP4' and r['subject']==subject and (direct or r['topic']==topic)]
    def ordering(r):
        m=re.match(r'^\s*(\d+)\s*[.)-]',r['filename'])
        return (norm(r['topic']), int(m.group(1)) if m else 99999, r['filename'].lower())
    items.sort(key=ordering)
    seen=set();titles=[]
    for r in items:
        key=path_parts(r['path'])
        if key in seen:continue
        seen.add(key)
        titles.append(clean_name(Path(r['filename']).stem))
    return titles, f'file index / {subject}' + ('' if direct else f' / {topic}'),list(dict.fromkeys(r['topic'] for r in items))


def discover_pdfs(records, library_root):
    """Prefer existing absolute inventory paths; otherwise search local configured root.

    Returns (list of (physical_path, stable_id), unresolved PDF inventory rows).
    The stable ID always carries subject/topic identity to prevent basename collisions.
    """
    root=Path(library_root).expanduser()
    local=list(root.rglob('*.pdf')) if root.exists() else []
    used=set(); found=[]; unresolved=[]
    for r in (x for x in records if x['type']=='PDF'):
        original=Path(r['path']).expanduser()
        candidates=[]
        if original.is_file(): candidates=[original]
        else:
            candidates=[p for p in local if same_file(p.name,r['filename'])]
            if len(candidates)>1:
                preferred=[p for p in candidates if norm(p.parent.name)==norm(r['topic']) and
                           (norm(p.parent.parent.name)==norm(r['subject']) or not r['subject'])]
                candidates=preferred or candidates
            if len(candidates)>1:
                # A flat folder cannot resolve duplicate PDF titles across topics.
                preferred=[p for p in candidates if path_parts(r['path'])[-3:-1] == path_parts(p)[-3:-1]]
                candidates=preferred or candidates
        if len(candidates)!=1:
            unresolved.append({**r,'reason':'Not found locally' if not candidates else 'Ambiguous local PDF filenames'})
            continue
        physical=candidates[0]
        stable='/'.join((r['subject'],r['topic'],r['filename']))
        if stable.casefold() not in used:
            found.append((physical,stable,r))
            used.add(stable.casefold())
    # Do not silently discard PDFs not included in the inventory.
    return found,unresolved
