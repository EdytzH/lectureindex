# PDF-first Bootcamp knowledge base (new)

## Why this is a different architecture

The older index first tried to prove that PDF pages belonged to an MP4. It threw away readable pages that lacked a verified video boundary. Thus, a small number of `pdf_chunks` could be mistaken for a small amount of available teaching content.

The new **PDF-first source index** always extracts and stores readable passages **before** attempting to link them to MP4s. It is a persistent retrieval-augmented generation (RAG) knowledge base, not model training and not an LLM with permanent memory of all the files. An LLM sees only the passages retrieved for a particular lecture. The complete PDF corpus stays on your computer unless a retrieved excerpt is sent to a selected remote provider (such as Puter).

## Data stored in `index/` (keep this folder!)

- `pdf_knowledge.sqlite`: persistent SQLite database containing all searchable PDF passages, page numbers, document names, PDF original virtual paths, FTS5 keyword search, and source attribution. Also contains original Excel subject/topic/video-folder catalog entries and user-reviewed page-to-video links.
- `pdf_knowledge_summary.json`: audit of source pages, extracted passages, OCR and unreadable pages.
- `pdf_knowledge_vectors.npy` and `pdf_knowledge_vectors.json`: optional semantic vectors of **all readable** PDF passages, regardless of their MP4 attribution, generated using your chosen sentence-transformer. These files are reused for future lectures.
- Your existing `pdf_tags.json`, `pdf_chunks.json`, `pdf_chunk_vectors.npy`, `manual_pdf_ranges.csv`, and MP4 embeddings are **left unchanged**. The new index can use the old verified PDF section boundaries where present.

The workbook `bootcamp_step1_file_index.xlsx` remains the single authoritative MP4/PDF inventory. The historical MP4 paths are identifiers, not required local files. Local PDFs can all be stored in one flat folder.

## Getting started

1. Copy the updated program files into your working application folder; **do not replace/delete your existing `index/`**.
2. Install dependencies and run Streamlit:
   ```powershell
   py -m pip install -r requirements.txt
   py -m streamlit run app.py
   ```
3. In **PDF folder diagnostic**, at the top find **PDF-first knowledge base — scan EVERY teaching page**. Enter the directory that contains your entire PDF collection. This is a new control, independent of the legacy PDF tagging field below it.
4. Leave **Build local semantic embeddings after scanning** checked and click **Build / update PDF-first knowledge base**. This opens **every available PDF** and stores useful text even when no MP4 mapping is found. Semantic encoding may take time for the whole library, but it is done once for the chosen local model and cached. PDFs that have not changed are not extracted again. Re-encoding is triggered when the text collection/model changes.
5. If the **PDF-by-PDF extraction audit** shows unreadable pages, enable OCR and rescan. OCR requires local Tesseract; it can significantly increase processing time. The app reports errors instead of silently inventing text.
6. Use **Search the entire PDF corpus and manually resolve uncertain links** to view real excerpts and identify the correct MP4 (if any). Select an excerpt, choose its actual Excel MP4, optionally extend the page range, and click **Confirm reviewed**. Reviewed links persist in SQLite and become legitimate verified evidence in future searches.
7. Under **Match lecture**, select the **LLM-assisted curriculum shortlist** using **Puter.js** or **Ollama** (or OpenAI if you have an API key). Upload a new lecture. The app identifies the lecture's major concepts, runs PDF-first retrieval and folder-aware MP4 discovery, and asks the LLM to assess evidence. The Python optimizer builds the final shortlist.

## What the statuses mean

| Status | Meaning | Can certify a specific video? |
|---|---|---|
| `verified` | Existing PDF tagging found a valid source-page range for this exact MP4. | Yes, as PDF evidence; not a claim that the entire MP4 was watched. |
| `reviewed` | A user manually confirmed this exact source page/range corresponds to this real inventory MP4. | Yes, as reviewed PDF evidence. |
| `suggested` | PDF table-of-contents/section heading resembles a real MP4 title in the appropriate historical folder. | **No.** Inspect it first. |
| `unmatched` | Source passage has no confident section/title proposal. | **No.** It remains searchable and reviewable. |
| `ambiguous` | Multiple existing verified/reviewed links conflict. | **No.** Review before attribution. |

All statuses remain visible. PDF search can find `suggested`/`unmatched` results and the lecture planner shows these in **PDF excerpts matching the lecture but NOT attributed to an MP4**. It never silently turns them into verified video evidence.

## Useful audits and controls

- **PDF-by-PDF extraction audit:** total pages, pages with readable text, OCR pages and pages with no searchable text.
- **Download complete MP4 PDF-evidence coverage audit:** every real eligible MP4, including those with no assigned excerpts, with the linked excerpt count.
- **Download unmatched / suggested PDF excerpts for review:** real source text with PDF filename, PDF page and any proposed MP4 identities.
- **Review an excerpt:** save a user-confirmed MP4 page attribution for future lecture matching.
- **Full-text search** works without local embedding vectors. With vectors built, queries use both semantic similarity and text matching.
- **Legacy PDF tagging** remains available below the new UI and is not required before PDF-first indexing.

## Important limitations

- A file with scanned or corrupt PDF text may need Tesseract OCR; the number of missing/unreadable pages is reported, not guessed. A readable page is not automatically proof of a video mapping.
- A passage may originate from a review question, outline, or clinical vignette. Those pages are **retained** and demoted for retrieval when recognizable. The system cannot guarantee every clinical vignette is detected.
- Existing source-video page boundaries can be incomplete or incorrect; inspect source provenance before accepting crucial matches. A title match is **only a suggested mapping**, not a truth label.
- Retrieval via embeddings is approximate and clinical relevance still requires human validation. The model is not trained on your PDF corpus; the local search index supplies a bounded set of excerpts with citations for each new lecture.
- The new knowledge base avoids repeatedly uploading/re-prompting the *entire* PDF library, but Puter or another hosted provider still needs an analysis and evidence-assessment request for each **new lecture**. Puter account usage/allowance policies apply independently of OpenAI API billing.
- If you move/reorganize your PDF collection, point the app at the new folder and rebuild the source index. When the old tagging path no longer matches, those old verified ranges will not be reused.
- Only the user can review a suggested PDF-to-MP4 page association. The app never fabricates missing MP4s, unverified page ranges, or actual MP4 content.

## Verification performed for this update

This environment had seven (not all 94) physical Bootcamp PDFs. The source-first scanner indexed all seven: **886 PDF pages, 825 readable pages, 1,455 searchable excerpts** without OCR. The remainder of the pages were not indexed as teaching excerpts because text was unreadable/insufficient; enable OCR on your machine where needed. A test query retrieved actual Limb Development PDF text from the 27–41 region and proposed the appropriate real MP4 without falsely declaring the link verified. Synthetic tests additionally verified persistent manual review, paragraph search, vector caching, missing-video rejection, and LLM candidate evidence propagation. Live Puter authentication and the full 94-PDF scan cannot be tested in this environment.
