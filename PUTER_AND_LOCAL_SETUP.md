# Bootcamp Curriculum Planner — no OpenAI API key required

## Option A (default): Puter.js in your browser

This app adds a Streamlit custom frontend (`puter_component/index.html`) which calls the official **Puter.js v2** SDK from `https://js.puter.com/v2/`. **Puter authentication remains in the browser**. The Python app does not request or retain Puter authentication tokens or require an OpenAI API key.

Important: **Puter is not unlimited free for the end user.** Every Puter account has an allowance; if it is exceeded, Puter may require the account holder to pay for additional usage. Check the allowance/billing screen **before sending large lectures**. Puter is a separate service, with its own privacy terms and availability. A popup- and JavaScript-enabled browser plus a live internet connection are required.

1. Extract this ZIP over the application's code, **preserving `index/`**, your local PDF folder, and any `manual_video_ranges.csv` / manual corrections.
2. `py -m pip install -r requirements.txt`
3. `py -m streamlit run app.py`
4. On **Match lecture**, upload a lecture PDF or paste topics.
5. Under **Recommendation method**, choose **LLM-assisted curriculum shortlist**; set **LLM provider** to **Puter.js (browser, no OpenAI key)**.
6. Click **Build lecture video shortlist**. A Puter button appears; click **Sign in / analyze lecture** and complete the Puter sign-in popup (if not already signed in).
7. Once your lecture analysis returns, the app will retrieve entire relevant MP4 topic folders locally. Click **Select videos with Puter** for the second call. If the result fails the major-concept coverage audit, click **Revise missing-topic coverage with Puter** for one optional repair call. The app will choose the version that assigns more major concepts and show any remaining gaps.
8. Review the validated shortlist, candidate audit, lecture concept analysis, and PDF evidence. If a model invents an ID, Python rejects it.

If the component says **Unsupported Origin**, ensure you opened the Streamlit app through `http://localhost:8501`, not `file://`, and that the browser permits cookies/popups for the Puter sign-in. Some browser privacy extensions or network restrictions may interfere with third-party scripts; in that case use Ollama instead. The Puter frontend is experimental; browser sign-in could not be exercised in the offline build environment.

Suggested Puter models available in the selector: `gpt-4o-mini` (default), `gpt-5.4-nano`, `gpt-6-luna`, `gpt-6.1-terra`. Availability and account allowance depend on Puter. The app makes **two separately clicked calls** in the usual case. If the deterministic coverage audit finds missing major lecture concepts, a **third, separately clicked repair call** is offered. Puter usage is subject to account allowances. They use a bounded lecture text sample and a compact candidate packet, not the entire PDFs or MP4 files.

## Option B: Ollama, entirely on your computer

This option is fully local for lecture interpretation and shortlist selection, **with no API key, Puter login, or cloud inference usage**. It requires enough CPU/RAM or a suitable GPU and may be slower or less accurate than a cloud model.

1. Install [Ollama](https://ollama.com/download) for Windows.
2. Open PowerShell: `ollama pull qwen3:8b` (a smaller model is an option if your machine is resource constrained).
3. Make sure Ollama is running; its local endpoint is `http://localhost:11434`.
4. In the app choose **Ollama (fully local, no key)** and enter the installed model name (`qwen3:8b` by default).
5. Click **Build lecture video shortlist**. The app will issue two local requests with JSON-schema validation, plus one additional local selection request if its coverage audit detects omitted major concepts. No separate frontend buttons are needed.

The app deliberately only connects to Ollama on `localhost:11434` to reduce the chance of sending lecture data to an unintended remote server. The MP4 title/PDF embeddings may still download from Hugging Face initially; those embeddings then run locally. Ollama model downloads also require an initial internet connection.

## Option C: Direct OpenAI API (unchanged)

Select **OpenAI API (optional key)** if you later obtain an API key with billing. This is optional; neither Puter nor Ollama requires it.

## What did not change

- Authoritative MP4 IDs and historical folder structures come **only** from `bootcamp_step1_file_index.xlsx`.
- No actual MP4 files are needed; the PDFs may reside in one flat local folder.
- Existing verified PDF-page evidence and embedding caches in `index/` are reused. No retag or rebuild is required just because the LLM provider changed.
- The LLM is **not** allowed to invent video titles or attach invented page numbers; all selected IDs and page evidence are checked against local catalog/index data.
- **Local section-first preview** and **Legacy hybrid + contextual reranker** remain available and make no cloud LLM calls.

## Validation limits

All static/Python offline regression tests are included. The browser-to-Puter authentication and live hosted model behavior were **not** end-to-end tested in this environment, and the Ollama provider was tested with a stubbed local HTTP response rather than a downloaded local model. Model recommendations must be checked against real lectures.

### Puter `coverage_gaps` formatting fix

Some Puter model replies use a sentence for `coverage_gaps` even though the
requested schema specifies a list of strings. This version normalizes a single
sentence to a one-element list (and `null` to an empty list) before validation.
It still rejects unknown MP4 IDs and unverified PDF references.
The instruction sent to Puter also explicitly asks for `[]` when no gaps are
identified. No embedding/PDF index rebuild is necessary for this update.
