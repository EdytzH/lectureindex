# Bootcamp LLM-assisted curriculum planner — update

This release builds on the **October 29 contextual-reranker beta** rather than replacing its authoritative MP4 inventory, PDF folder scanner, verified page ranges, vectors, or saved index.

## What is different?

There are now three recommendation methods under **Match lecture → Recommendation method**:

1. **LLM-assisted curriculum shortlist** (default): two OpenAI Responses API calls, plus one optional repair call if the major-concept audit finds omissions. The first identifies the major concepts and supporting topics of the lecture as a whole. The existing *local* semantic/BM25 and verified-PDF indexes then retrieve relevant **virtual Bootcamp subject/topic folders**, retain complete folder video lists when the candidate budget permits, and also include global search hits. The second API call selects a small, complementary list of *actual* MP4s from that candidate pool. The app cross-checks every output ID against the candidates and the Excel inventory.
2. **Local section-first preview (no API)**: demonstrates the section-oriented retrieval without any remote call. This is an automated retrieval preview, **not** equivalent to an LLM-selected curriculum.
3. **Legacy hybrid + contextual reranker**: the preceding matching engine is left available unchanged for side-by-side comparison.

The output includes the whole-lecture concept list, retrieved topic folders (with a `complete` flag), every candidate sent to the selector, a compact ranked video shortlist, lecture page references, verified PDF passage witnesses where present, and suggested coverage gaps. CSV exports are included for shortlisted and candidate videos.


## No-OpenAI-key update (Puter.js and Ollama)

The app now defaults to the **Puter.js browser provider**, and offers **Ollama** as a fully local alternative. The previous direct OpenAI API option is retained for anyone with a key. **See `PUTER_AND_LOCAL_SETUP.md` for the current installation and usage instructions.** Older API-key-only steps below apply only when choosing the OpenAI API provider. Puter usage is metered to the signed-in user's allowance; extra usage may incur Puter charges.


## Install and run on Windows

Extract the ZIP **over your current application code**, retaining your own existing `index/` folder and PDF handouts. The ZIP does not include any local `index/` content.

```powershell
cd "C:\Users\Ed\Downloads\BootcampxCNU"
py -m pip install -r requirements.txt
$env:OPENAI_API_KEY="your-api-key"
py -m streamlit run app.py
```

Alternatively, do **not** set the environment variable and instead paste the key into the masked **OpenAI API key** input in the Match lecture tab. You may also supply `OPENAI_API_KEY` using Streamlit secrets. Never put a real key into code, Excel, or any file you intend to share. An API account with permission to use your chosen model and available credit is needed. ChatGPT subscriptions are separate from API billing.

The initial local embedding model may need to download. Subsequent title vectors should load from the existing disk index. **No MP4 files are required on your PC**; paths in Excel are authoritative identifiers. Your **PDFs can stay together in one flat local folder**. PDF tagging is optional for title-only planning, but add verified PDF page evidence for deeper teaching-content selection.

### Recommended first test

- Select **LLM-assisted curriculum shortlist** and `gpt-6.1-terra` (changeable).
- Upload `MSK_Week1_Tuesday_Vertebral column and back_Ernst_2026.pdf`.
- Set **Maximum total recommendations** to **8**; **Complete sections** 5; **Extra global candidates** 35; **Maximum candidate pool** 115.
- Click **Build lecture video shortlist**.
- Inspect **Whole-lecture analysis** and **Retrieved Bootcamp topic folders**. The expected most relevant section is `06. Gross Anatomy / 06. Back`, which has 13 actual MP4 entries.
- Open **Candidate audit**. Its 13 Back entries should be present and offered to the LLM even if other folders have isolated high similarity scores.
- Judge the final shortlist against your human-reviewed answer. Nothing automatically guarantees identical results; this is the principal evaluation goal.

## Source integrity and privacy

- The complete MP4 list is still defined by `bootcamp_step1_file_index.xlsx`, with Board-style Question Breakdown videos excluded from recommendations.
- The API sees a bounded, evenly sampled set of lecture text with source page markers; no files are uploaded to OpenAI by this app. It sees a compact set of candidate titles, subject/topic fields, **opaque** candidate IDs (not historical file paths), and the *existing scanner's verified* PDF text excerpts when available.
- The actual PDF page and MP4 inventory path are attached by local Python from the index, **never invented by the model**. Output video IDs not in the candidate set are dropped and counted.
- An indexed/verified passage is source evidence for relevance, **not proof that the corresponding MP4 teaches every concept**. The application cannot play or inspect the MP4s.
- The masked API key is used for the API client only. It is not included in cached vectors, output CSVs, or the saved index. There is no automatic fallback from a failed API request to a misleading “LLM-generated” result; select **Local section-first preview** explicitly if desired.
- Each LLM match uses **two or three remote API requests**, depending on coverage repair, and consumes billable input/output tokens; there is no fixed cost guarantee. Respect school policies before transmitting lecture text or PDF excerpts to a third-party API.

## Architecture

- `file_index_catalog.py` + `beta_catalog.py`: authoritative video IDs, virtual folders, flat-folder PDF filename discovery.
- `semantic_matcher.py`: existing title + folder embeddings and BM25, reused from the saved `index/`.
- `pdf_evidence.py`: optional title-verified per-MP4 PDF teaching pages, cached vectors and local excerpt provenance.
- `llm_planner.py` **NEW**: page-aware lecture packet → API concepts → local hierarchical retrieval → API shortlist → hard validation and source audit.
- `app.py`: adds recommendation-method selector, API key and model controls, and explanation/candidate diagnostics.
- `test_llm_planner.py` **NEW**: fully offline mocked API and deterministic embedding regression against the real 49-slide Ernst lecture. These tests do **not** establish live API shortlist accuracy.

## Test offline

```powershell
py test_llm_planner.py
py test_hybrid.py
py test_flat_pdf_folder.py
py test_pdf_evidence.py
py test_corrupt_bookmark_title.py
py test_pdf_score_regression.py
py test_contextual_rerank.py
```

**Important known limit:** The actual OpenAI API cannot be exercised without your own key and network access. The unit test mocks the LLM outputs, checks that the correct 13 Back MP4s are in the candidates, and verifies the validation and data path. Assess the actual generated shortlist before considering it clinically/educationally accurate.
