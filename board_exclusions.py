"""Consistent exclusion of board-style question breakdown videos and their PDF pages."""
import re

BOARD_TITLE = re.compile(r'\bboard[\s\-–—]*style\s+question\s+breakdown\b|\bboard\s+breakdown\b', re.I)

def is_board_breakdown(title):
    return bool(BOARD_TITLE.search(str(title or '')))

def excluded_pages(detections):
    """Conservatively exclude page ranges explicitly assigned to board breakdowns."""
    return {file: {pg for seg in info.get('segments', [])
                   if is_board_breakdown(seg.get('title', ''))
                   for pg in range(int(seg['start']), int(seg['end']) + 1)}
            for file, info in detections.items()}

def page_is_board_breakdown(text):
    """Ignore a page only when board breakdown is its actual heading, not an incidental mention."""
    head = [x.strip() for x in str(text).splitlines() if x.strip()][:7]
    return any(is_board_breakdown(x) for x in head)
