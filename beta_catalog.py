"""Authoritative virtual MP4/PDF inventory and physical *flat PDF folder* resolver.

The workbook path describes where a file belonged in Bootcamp, NOT where it
must exist on this computer. Nothing attempts to open an MP4. We link virtual
MP4s to virtual PDFs using the workbook subject/topic columns, then resolve
only the PDF filenames against the user's local PDF collection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from board_exclusions import is_board_breakdown
from file_index_catalog import clean_name, norm


def records_for_search(records):
    """Return eligible distinct logical MP4 paths, excluding board breakdowns."""
    seen, out = set(), []
    for row in records:
        if row['type'] != 'MP4':
            continue
        identity = row['path'].replace('\\', '/').casefold().strip()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        title = clean_name(Path(row['filename']).stem)
        if is_board_breakdown(title) or is_board_breakdown(row['topic']):
            continue
        out.append({**row, 'title': title, 'video_id': identity})
    return out


def folder_candidates(video, records):
    """Logical subject/topic membership from Excel, NOT actual disk folders."""
    pdfs = [r for r in records if r['type'] == 'PDF' and r['subject'] == video['subject']]
    direct = [r for r in pdfs if r['topic'] == video['topic']]
    if direct:
        return direct
    # Larger consolidated PDF stored logically in the subject folder.
    return [r for r in pdfs if not r['topic'] or r['topic'].lower().startswith('(files directly')]


def _exact_key(name):
    return Path(str(name).replace('\\', '/')).name.casefold().strip()


def _alias_key(name):
    """Permit optional '0. ' numeric prefixes and harmless punctuation changes."""
    return norm(clean_name(Path(str(name).replace('\\', '/')).stem))


def _folder_agrees(physical, pdf_row):
    """If a nested PDF library is provided, use its relative folder for tie-breaking."""
    subject, topic = norm(pdf_row['subject']), norm(pdf_row['topic'])
    direct_subject = not topic or topic.startswith('files directly')
    parent = norm(physical.parent.name)
    if direct_subject:
        return parent == subject
    return parent == topic and norm(physical.parent.parent.name) == subject


def resolve_pdf_library(records, pdf_root=None):
    """Locate inventory PDF *filenames* in one local folder (nested also supported).

    Each result is keyed by the virtual PDF path recorded in Excel. The MP4 paths
    are never opened. The local folder takes precedence over historical paths.
    Ambiguous lookups are not guessed or quietly assigned to several lessons.

    Returns (resolution dict, discovery summary, unlisted local filenames).
    """
    pdfs = [r for r in records if r['type'] == 'PDF']
    root = Path(str(pdf_root)).expanduser() if pdf_root and str(pdf_root).strip() else None
    if root is None:
        state = 'Choose your local folder containing the PDF handouts'
    elif not root.is_dir():
        state = 'Local PDF folder does not exist or is not accessible'
    else:
        state = 'Ready'

    files = []
    if state == 'Ready':
        # A flat folder is the normal case; keep nested paths as a convenience.
        files = sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.casefold() == '.pdf'),
                       key=lambda p: str(p).casefold())
    by_exact = defaultdict(list)
    by_alias = defaultdict(list)
    for p in files:
        by_exact[_exact_key(p.name)].append(p)
        by_alias[_alias_key(p.name)].append(p)

    resolution = {}
    for row in pdfs:
        virtual_id = row['path']  # authoritative *logical* identity
        if state != 'Ready':
            resolution[virtual_id] = {'status':state, 'physical':'', 'match_type':''}
            continue
        exact = by_exact.get(_exact_key(row['filename']), [])
        alias = by_alias.get(_alias_key(row['filename']), [])
        candidates, kind = (exact, 'exact filename') if exact else (alias, 'filename alias')
        # A file placed directly in the selected flat folder takes priority.
        # For nested duplicates, the workbook's virtual folder can safely break
        # a tie only when exactly one physical directory agrees with it.
        if len(candidates) > 1:
            flat_hits = [p for p in candidates if p.parent == root]
            if len(flat_hits) == 1:
                candidates = flat_hits
            else:
                expected = [p for p in candidates if _folder_agrees(p, row)]
                if len(expected) == 1:
                    candidates = expected
        if len(candidates) == 1:
            physical = candidates[0]
            layout = 'flat folder' if physical.parent == root else 'nested folder'
            resolution[virtual_id] = {
                'status':f'Matched {kind} ({layout})',
                'physical':str(physical), 'match_type':kind,
            }
        elif candidates:
            resolution[virtual_id] = {
                'status':'Ambiguous local PDF filename', 'physical':'',
                'match_type':'', 'alternatives':[str(p) for p in candidates],
            }
        else:
            resolution[virtual_id] = {
                'status':'PDF filename not found in selected folder',
                'physical':'', 'match_type':'',
            }

    # If separate inventory PDF records both resolve to one physical file, the
    # workbook cannot tell us which logical handout it represents. Refuse to
    # silently assign identical content to unrelated lessons.
    reverse = defaultdict(list)
    for virtual_id, info in resolution.items():
        if info['physical']:
            reverse[str(Path(info['physical']).resolve()).casefold()].append(virtual_id)
    for ids in reverse.values():
        if len(ids) > 1:
            for virtual_id in ids:
                resolution[virtual_id] = {
                    'status':'Ambiguous inventory filenames: one local file matches multiple PDFs',
                    'physical':'', 'match_type':'', 'alternatives':ids,
                }

    used = {str(Path(v['physical']).resolve()).casefold() for v in resolution.values() if v['physical']}
    extra = [str(p) for p in files if str(p.resolve()).casefold() not in used]
    counts = Counter()
    for value in resolution.values():
        if value['physical']:
            counts['matched'] += 1
            if value['match_type'] == 'filename alias':
                counts['alias_matches'] += 1
        elif value['status'].startswith('Ambiguous'):
            counts['ambiguous'] += 1
        else:
            counts['missing'] += 1
    summary = {
        'folder_status':state, 'local_pdf_files':len(files),
        'inventory_pdf_entries':len(pdfs), 'matched':counts['matched'],
        'alias_matches':counts['alias_matches'], 'ambiguous':counts['ambiguous'],
        'missing':counts['missing'], 'unlisted_local_files':len(extra),
    }
    return resolution, summary, extra


def resolve_pdfs(records, pdf_root=None):
    """Compatibility entrypoint for the PDF tagging scanner."""
    return resolve_pdf_library(records, pdf_root)[0]
