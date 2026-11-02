# New: PDF-first persistent searchable knowledge base

**Start with [PDF_FIRST_KNOWLEDGE_BASE.md](PDF_FIRST_KNOWLEDGE_BASE.md).**

The application can now index ALL readable pages from the flat collection of Bootcamp PDFs **before** trying to assign them to individual MP4s. Unattributed excerpts stay searchable, visible and available for manual review; exact original video paths remain the only valid MP4 identities. The concept-based LLM planner retrieves relevant local teaching passages and PDF provenance on demand rather than repeatedly sending the entire PDF library to the model. The old verified index remains available for backwards compatibility.

---

# FINAL REDESIGN — evidence matrix + local curriculum optimizer (V3)

**Primary mode:** LLM-assisted curriculum shortlist uses two LLM requests: (1) atomize the
entire lecture into source-page-grounded teaching concepts; (2) assess which actual
MP4s teach EACH concept, returning exact quotations from genuine MP4 titles or
locally verified PDF passages. The LLM **does not choose the playlist**.

`coverage_optimizer.py` validates IDs and quotations, applies corpus-derived title
specificity, supplements omissions conservatively from the best matching
Excel topic folder (marked `title heuristic`), and uses weighted marginal coverage
to select a nonredundant set. Direct/partial scores are **heuristics, not true
probabilities or proof of video coverage**. It prefers distinct major concepts,
not a repeated title match. It flags concepts not sufficiently corroborated.

**No Bootcamp lecture or named video is hardcoded in the planner.** The code reads
the 2,323 eligible historical MP4 identifiers from the bundled inventory and can
use existing verified PDF evidence. The MP4 files are not required on disk.

**Use without OpenAI billing:** choose `Puter.js (browser, no OpenAI key)` and
click the first button to analyze the lecture and the second to assess teaching
evidence. This requires a Puter sign-in and may use the Puter account allowance;
it is not unlimited free cloud inference. Alternatively run a local Ollama model.

**Install:** extract this full folder in place of application code while KEEPING
your current `index/` folder and the flat PDF collection. Then:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Default settings: 3 complete high-ranked sections, 25 extra candidates, up to 90
MP4 candidates, 10 total recommendations. Raise the candidate budget only if a
relevant folder is shown as incomplete. The app includes a candidate audit, a
per-concept source-validated matrix and a rejected-evidence audit.

**Testing:** Run `python test_final_coverage_matrix.py` for nonmedical synthetic
and real-inventory tests. All legacy tests remain runnable. These tests use
controlled/fake model responses: **no live Puter inference or real-model
shortlist accuracy has been certified**. Test the Ernst lecture and compare the
major concept audit as the live acceptance check. Title-only videos should be
shown as provisional, not advertised as watched/verified.

---

**Latest update — coverage-first curriculum planning:** Start with `COVERAGE_FIRST_PLANNER.md`. The LLM now reviews complete Bootcamp folders, covers major lecture objectives, separates core/optional lessons, audits omitted concept groups and can request one revision. Default recommendation count is 10. Puter uses 2 calls normally, 3 if the shortlist needs repair. No OpenAI API key is required with Puter or local Ollama.



## No-OpenAI-key update (Puter.js and Ollama)

The app now defaults to the **Puter.js browser provider**, and offers **Ollama** as a fully local alternative. The previous direct OpenAI API option is retained for anyone with a key. **See `PUTER_AND_LOCAL_SETUP.md` for the current installation and usage instructions.** Older API-key-only steps below apply only when choosing the OpenAI API provider. Puter usage is metered to the signed-in user's allowance; extra usage may incur Puter charges.

> **NEW (October 29):** See [`LLM_PLANNER_README.md`](LLM_PLANNER_README.md) for the LLM-assisted whole-lecture shortlist and hierarchical search setup. The older matching methods remain available.

# Bootcamp contextual passage matcher — semantic/BM25 + verified PDF evidence

## NEW: Contextual cross-encoder reranking

The **Match lecture** sidebar now has **Rerank lecture ↔ Bootcamp passage pairs** enabled by default. The original fast embedding/BM25 search identifies real MP4 candidates. A **local cross-encoder** then reads both texts *together*:

- **With verified PDF pages**: the lecture text and up to two retrieved teaching excerpts from **that exact MP4's approved page range** are compared. The model selects the most contextually relevant excerpt and keeps its real Bootcamp page reference.
- **Without verified PDF pages**: the lecture text is compared against the video's actual title plus subject/topic folder metadata. There is no invented video description or borrowed page from another video.
- Results with cross-encoder scores below the **Minimum contextual pair score** are excluded. A long lecture is split into coherent slide chunks; near-identical/repeated chunks are deduplicated so repetition of terms like *vertebral column* does not automatically multiply evidence. Distinct content about ossification and segmentation remains eligible to count separately.
- Recommendations display contextual score, distinct supporting concepts, literal lecture excerpts, the Bootcamp excerpt and page **if verified**, and the original retrieval (embedding/BM25) values for debugging. Both are *heuristics*, **not verified probabilities of overlap, and not an automated assessment of actual medical coverage**.

**Local reranker choices:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (default, lighter) or `BAAI/bge-reranker-base` (larger/slower). Like the existing sentence-embedding models, they need an initial internet download, then run locally. This is a separate model download; selecting a new one does **not** rebuild MP4 or PDF embeddings. If model installation or downloading fails, the app reports an error; you can uncheck contextual reranking to use the previous hybrid scorer unchanged.

**Runtime:** cross-encoders are costlier than cosine similarity. The `Maximum passage comparisons per lecture` slider (default 320) limits the work and spreads the budget over lecture pages. The cutoff (default .45, after sigmoid of raw model logits) is only an initial tunable heuristic. Use **Evaluation & index** and your human-labeled examples to adjust it; a general-purpose reranker can still make medical mistakes. PDF tagging and OCR workflows, the flat local PDF folder, and the authoritative 2,323-video MP4 catalog are otherwise unchanged.

**No index rebuild needed when upgrading** if you preserve your `index/` folder. Reranking operates on retrieved passages at search time, so there are no new `.npy` vector caches to generate. For a fresh installation, first index the PDFs from your flat folder as before.

**Testing:** `py test_contextual_rerank.py` checks correct PDF-page provenance, rejecting an explicitly irrelevant synthetic passage, duplicate concept grouping, and title-only fallback using a stand-in scorer. It is a pipeline regression, **not** a real-model medical accuracy benchmark. All previous tests still run.

**Offline evaluation (after labeling):**

```powershell
py eval_retrieval.py --context-rerank --context-threshold 0.45 --max-context-pairs 320
```

This uses your existing human-labeled `evaluation/labels.csv`, and the verified PDF index if it is present. Compare with `py eval_retrieval.py` (same underlying embedding model, without contextual reranking). Top-1 / top-5 hit rates and MRR are computed from the real labeled paths; no expected matches are invented.

---

# Bootcamp semantic / BM25 + verified PDF evidence beta

A title-first architecture that uses **the real MP4 paths as the only authoritative video list**, now with a separate, conservative **PDF tagging and teaching-page search index**. Missing or corrupt PDF text never removes an MP4 from title search.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\Ed\Downloads\BootcampSemanticBeta
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Extract this ZIP into its own folder. **Do not overwrite your old project or copy its `index` folder.** The included `bootcamp_step1_file_index.xlsx` is the uploaded MP4/PDF inventory. Start by pasting a lecture paragraph, then press **Find matching MP4 videos**. On first use, `sentence-transformers` downloads the selected model (internet required if model not previously cached). Subsequent matching reuses local model files and persistent `.npy` embeddings. You can also run `py build_index.py` to precompute the index from the terminal before opening Streamlit; the console prints each stage.

## Retrieval

1. Load the workbook's real MP4 paths; filter Board-style Question Breakdown videos; no PDF required.
2. Build title and subject/topic/title vectors once using `sentence-transformers/all-MiniLM-L6-v2` (or BGE-small). Store `index/title_vectors.npy`, `context_vectors.npy`, `slide_vectors.npy`, `videos.json`, `index_config.json`. Regenerate automatically only when the inventory, model, or verified slide text changes.
3. Extract lecture text into small sentences/windows, or by PDF/PPTX slide and sentence/window. Keep lecture slide numbers.
4. For each query chunk, calculate cosine similarity plus a BM25 score over title and folder metadata. Default blend: **70% dense / 30% BM25**. Within semantic retrieval, default: **75% title / 25% subject/topic context**. The slider adjusts these weights; BM25 is relative to the highest BM25 candidate for each query and is not a probability.
5. Select the top-K videos per chunk with a minimum hybrid score. Aggregate matching chunks to rank each *existing* MP4 path. Show supporting lecture excerpts, matching words, cosine, and BM25 evidence.
6. No fixed quota of videos; zero results is valid when the threshold is high. Scores are **relative ranking heuristics, not validated medical coverage**. A short title can still produce false matches; use the evaluation harness to tune before trusting recommendations.

## Record schema and future slide integration

Every row in `index/videos.json` contains:

```
id, title, subject, topic, course_section, video_url, mp4_path,
pdf_candidates, slide_text, title_embedding, context_embedding, slide_embedding
```

`video_url` currently contains the **local inventory MP4 path**, not a web streaming URL. The fields `title_embedding`, `context_embedding`, and `slide_embedding` point to row offsets in matching `.npy` arrays; until a video has reviewed slide content, `slide_text=''` and `slide_embedding=null`.

**Optional legacy reviewed summary slot (still supported):** In addition to the new independent PDF page-level index, you can provide `index/verified_video_slides.json`:

```json
{
  "/Bootcamp - Step 1 (August 2025)/05. Embryology/3. Embryonic Period - Weeks 3-8/3. Limb Development.mp4": "[Text extracted from confirmed Limb Development teaching slides]"
}
```

Only reviewed text **verified to belong to the particular MP4** should be attached, never entire unlabeled PDFs. The legacy summary slot produces one vector per MP4; the new PDF page-level index is better for long teaching text because it stores multiple excerpt vectors and page references. The app still retrieves videos with no PDF evidence. The example above is illustrative and is **not** included as data.

## PDF tagging & search — FLAT local PDF folder (updated)

**Your real setup is supported:** You do **not** need the actual MP4 video files or the old nested Bootcamp directory tree on your computer. The paths in `bootcamp_step1_file_index.xlsx` are **logical identities and metadata**, not live disk paths. For example, a workbook row can describe `06. Gross Anatomy / 05. Hand / Carpals, Metacarpals and Phalanges.mp4` even though no such MP4 exists locally.

Put all actual handouts into **one ordinary folder**, such as:

```text
C:\Users\Ed\Downloads\Bootcamp PDFs\
    0. Hand.pdf
    0. Anterior Arm.pdf
    Biochemistry.pdf
    ... other handouts ...
```

Open the **PDF folder diagnostic** tab. In **Folder containing all your PDF handouts (flat folder)**, paste your own folder's full path, such as `C:\Users\Ed\Downloads\Bootcamp PDFs`. Alternatively set:

```powershell
$env:BOOTCAMP_PDF_ROOT = 'C:\Users\Ed\Downloads\Bootcamp PDFs'
py -m streamlit run app.py
```

### How lookup and tagging work

1. **MP4s only come from the workbook.** The original MP4 paths supply each video's ID, name, subject, and topic. They are **not** opened; no local video files are needed.
2. **MP4-to-PDF associations are virtual.** An MP4 in the `06. Gross Anatomy / 05. Hand` logical folder is associated with the PDF inventory row in the same logical folder (`0. Hand.pdf`), even if your physical PDF files are not nested at all. Subject-wide consolidated handouts remain fallbacks for their subjects.
3. **Find each local PDF by filename.** The resolver scans your chosen folder, prefers exact case-insensitive PDF filenames, then permits an unambiguous numeric-prefix alias (e.g. local `Hand.pdf` matching `0. Hand.pdf` in Excel). It also supports nested PDF folders for convenience, but does **not** require them. The original spreadsheet's `/Bootcamp ...` or drive paths are never treated as live locations.
4. **Preview *before* scanning.** The PDF tab shows how many of the 94 inventory PDF filenames are located, missing, ambiguous, or unlisted in your folder, with exact local file paths. If none match, the scan button is disabled so you don't waste time indexing nothing. Different PDFs with colliding names are **not guessed**.
5. **Then validate actual video pages.** Click **Tag PDFs and build verified teaching-page index**. The app looks for video title headings in the matched PDF; bookmarks supply leads, not proof. Only confirmed pages (or personally reviewed manual page ranges) become PDF evidence. On corrupted pages, enable OCR (Tesseract required) to try to recover text. The app does not distribute one handout's text to every video in its folder.
6. **Search titles and verified PDF evidence together.** `index/pdf_tags.json` records the virtual MP4↔PDF association, PDF filename lookup, verification status, and local PDF location. `index/pdf_chunks.json` contains teaching text only from verified video pages. `index/pdf_chunk_vectors.npy` is an independent semantic index. Missing/unreadable PDFs never remove an MP4 from title search, and PDF updates do not require rebuilding the title vectors.

### Manual page correction

Use **Optional: manually review or correct individual video page ranges** in the PDF tab. Enter *exact workbook path strings* for the `MP4 path` and `PDF path` columns, **not** your actual flattened local PDF path. The local PDF location is resolved separately by filename. Verify start/end pages manually before saving. Corrections live in `index/manual_pdf_ranges.csv`; make a backup before reinstalling.

### Statuses and safety

- **Matched exact filename / Matched filename alias** refers only to *physical PDF discovery*.
- **PDF found by filename; pages not yet checked** and **PDF associated; page title unverified** mean the handout may belong to the video, but no individual pages have been proved.
- **Verified start page only** or **Verified page range** means the PDF scanner found corroborating title evidence, or a user reviewed the range.
- **Ambiguous local PDF filename** means the app deliberately refuses to guess, and will not attach that document's content to videos.
- Board-style Question Breakdown MP4s remain excluded. Review/board-breakdown pages are not used as teaching evidence.

**Testing:** `py test_hybrid.py`, `py test_pdf_evidence.py`, and **`py test_flat_pdf_folder.py`**. The new flat-folder test builds two synthetic PDFs in a single folder, with no MP4 files present, and verifies filename discovery, subject/topic mapping, page verification, aliases, ambiguity handling, caching, and lack of dependence on historical on-disk paths. Your entire 94-PDF collection is not available in this environment; local discovery counts depend on the folder you select.

## Evaluation (strongly recommended)

`evaluation/labels.csv` contains 25 **unlabeled** lecture descriptions. Fill `expected_mp4_paths` for each from the authoritative master MP4 list (use `|` to separate multiple acceptable paths). Then run:

```powershell
py eval_retrieval.py --labels evaluation/labels.csv
```

This reports Hit@1, Hit@5, MRR@5, and writes `evaluation/results.csv`. **The 25 rows are starter prompts, not hand-validated ground truth**. Use 20–30 actual lecture samples with human-reviewed expected videos before judging model/weight changes. For fast self-tests without downloaded embedding weights, run `python test_hybrid.py`.

## Known limitations

- First-time model setup requires downloading model weights, and embedding ~2.3k titles on CPU can take time (but is cached thereafter).
- Embedding-based relevance is approximate; threshold values vary by model and lecture phrasing, so benchmark rather than treating scores as probabilities.
- No automatic visual understanding of lecture diagrams. Optional PyMuPDF/Tesseract OCR handles some corrupted Bootcamp text, but does not interpret diagrams. Bad native text is excluded and uncertain page ranges are explicitly flagged.
- No cross-encoder reranking or large hardcoded synonym lists. The underlying engine can later add them as an independent stage, without changing video IDs or per-video slide evidence fields.


## October 2026 PDF evidence fix: Embryonic Period handout

A false zero-PDF-match occurred even when the workbook correctly associated
`3. Limb Development.mp4` with `0. Embryonic Period - Weeks 3-8.pdf`.
That file has a `7. Limbs` bookmark at page 27; its displayed outline says
`Limb Development`, but PyMuPDF's native text representation of the heading is
corrupted. The previous scanner required a very strong *bookmark-title* match
before attempting OCR, so it created zero video text chunks.

The updated scanner treats bookmark locations only as **candidate dividers** and,
when enabled, OCRs those cover pages once to validate the *visible* MP4 title.
This is a general title-verification algorithm, not a hand-coded exception for
Limb Development. It can match titles appearing in separate lines (e.g.,
`Pharyngeal Arches` and `Face & Pharynx`) and protects against invalid bookmark
destinations. Identical repeated outline pages are not indexed twice.

On the uploaded example PDF, the fix verified the actual sections as follows:

| MP4 title | Verified Bootcamp PDF pages |
|---|---|
| Pharyngeal Arches - Face & Pharynx | 1–13 |
| Face Development | 14–26 |
| Limb Development | **27–41** |
| Skin Development | 42–55 |
| Eye Development | 56–71 |
| Ear Development | 72–83 |

For Limb Development, 14 readable teaching-text chunks were indexed from
pages 28–40. Page 27 is a cover outline; page 41 repeats it and is ignored.
The native text is still partially corrupted in some remaining pages. The
additional *OCR teaching pages* checkbox can improve their quality.

The lecture PDF loader was also fixed: visual line breaks no longer turn into
separate search queries. The uploaded 29-page Ahmed lecture now yields **40
coherent, page-numbered chunks**, down from 285 isolated lines.

### Using the fix

1. Replace the previous app files with this version. Keep your existing `index/`
   directory if it contains manually reviewed ranges; the app will reuse them.
2. Install the external **Tesseract OCR** program if it is not already installed.
   `pip install pymupdf` alone does **not** install Tesseract. Make sure the
   installed OCR engine and English language data are accessible to PyMuPDF.
3. Open **PDF folder diagnostic**, enter the folder containing all your PDFs,
   and leave **OCR PDF section-title pages when headings are corrupted** ON.
4. Click **Tag PDFs and build verified teaching-page index** again. This is
   required even if you already scanned the same PDFs with the earlier app.
   You do **not** need to rebuild MP4 title embeddings.
5. The **Match lecture** tab reports if PDF assistance is actually active,
   including the number of searchable PDF chunks and the model used.

If automatic OCR is unavailable, you can use the existing **manually reviewed
page range** editor with the exact MP4 path and exact inventory PDF path, then
run tagging again. That is a factual manually reviewed mapping, not an
inferred range. All MP4s remain searchable by title without PDF evidence.

Run `python test_corrupt_bookmark_title.py` for a deterministic OCR-mocked
regression without requiring Tesseract or the original Bootcamp PDF.

## Fix: PDF match score stayed zero despite verified pages

This update corrects two separate issues in the earlier search code:

1. **Prematurely suppressing PDF-only matches.** A candidate with poor title
   similarity and excellent page similarity was scored as
   `0.65 * title + 0.35 * PDF` and could fall below the result cutoff. Now
   **verified PDF pages independently qualify a video for retrieval**, even
   when its short MP4 title does not match the lecture wording. When both
   title and verified PDF pages support a match, the sidebar's **PDF
   corroboration strength** adds a small agreement bonus. Setting the slider
   to zero disables PDF evidence. This does not invent new video titles.
2. **Conflating unavailable evidence with a zero measurement.** The results
   now display `PDF score` separately and distinguish `Not indexed`,
   `No matching PDF evidence`, and `PDF evidence`. Each recommended video has
   a PDF page and excerpt when verified teaching-page content supports it;
   the diagnostics also show the actual verified page range and chunk count.

### Verify that the user's PDF index is actually active

In **Match lecture**, open **PDF evidence health check — inspect a specific
video**, and enter `Limb Development`. Check:

- `Tag status` = verified range and `Verified pages` = `27–41`.
- `Indexed chunks` greater than zero; this particular source yielded 14
  teaching chunks in our sample scan with section-title OCR enabled.
- `PDF vectors` count greater than zero, with a model that **exactly matches**
  the `Local embedding model` selected in the sidebar.
- `PDF corroboration strength` above zero, and PDF search shown as active.

If any of those conditions fails, open **PDF folder diagnostic** and rerun
**Tag PDFs and build verified teaching-page index** with section-heading OCR
selected; optionally enable teaching-page OCR to recover more corrupted body
text. An MP4 title-embedding rebuild is **not** required. The previous `index`
folder can be retained and reused; the patch does not change the saved
`pdf_tags.json`, `pdf_chunks.json`, or vector schema.

### Tests

Run `py test_pdf_score_regression.py`, `py test_pdf_evidence.py`,
`py test_corrupt_bookmark_title.py`, `py test_hybrid.py`, and
`py test_flat_pdf_folder.py`. The new test uses a deterministic offline encoder
and demonstrates PDF-only evidence surviving the threshold with `top_k=1` and
PDF corroboration strength `0.35`. The actual pretrained-model quality still
requires your locally cached MiniLM/BGE weights and human-checked examples.
