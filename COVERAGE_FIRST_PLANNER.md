# Coverage-first Bootcamp curriculum planner (October 2026)

## Why this update exists

The previous LLM planner could return four vertebral-column videos for a long lecture whose second half covered back musculature, even though the actual **06. Gross Anatomy / 06. Back** folder contained the relevant MP4s. This was a **selection objective** failure, not necessarily a missing spreadsheet record. A highly related single clinical topic could win over several major teaching blocks. The previous prompt said to be compact and could pick fewer than the limit without checking completeness.

## What the updated LLM is asked to do

1. Reconstruct the lecture **from start to finish**. For a long mixed-topic lecture, create 7–12 specific concepts from *all* substantive sections, with the lecture's actual page numbers, and distinguish major teaching objectives from supporting clinical asides. A compact list of each page's opening text accompanies the lecture packet so a late section is less likely to be ignored.
2. Retrieve locally using semantic embeddings, BM25 and optional verified PDF snippets. Retain complete relevant logical Excel folders, without needing real MP4 files on disk.
3. Give the LLM a **complete relevant folder list** and a **per-concept candidate guide**, as well as the full 115-video candidate pool. These are discovery hints, not proof of topic coverage.
4. Select a **minimal but sufficiently complete** video collection by comparing specific teaching goals, not just similarity scores. Favor foundational teaching for anatomy-heavy lectures; do not overemphasize a one-slide clinical condition. A substantial lecture may need 8 core videos and up to 2 useful optional additions.
5. Return exact candidate IDs with `core` or `optional`, concept labels drawn from the lecture analysis and an explicit instructional reason. All video titles and locations are resolved from the authoritative spreadsheet, never invented by the model.
6. Independently audit how many **major concepts were assigned** to a selected video. The audit separately labels a chosen concept as locally corroborated by retrieval/PDF evidence or simply **LLM claim only**. These are **not verified coverage probabilities** and the app cannot know a video teaches every claimed detail unless its indexed handout supports that assertion.
7. If a major concept has **no selected video**, ask the LLM to revise the *entire* shortlist once with the missing goals displayed explicitly. Keep the revised list only when it assigns more major concepts. Do not auto-insert a speculative MP4 and do not silently claim coverage if none exists.

## Installation

Extract the updated ZIP over your existing application code, preserving the `index/` folder and your flat folder of PDF handouts. Run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Go to **Match lecture → LLM-assisted curriculum shortlist** and select **Puter.js** (no OpenAI key) or **Ollama** (local) as usual. The default **Maximum total recommendations** is now **10** so the model can propose 8 core and up to 2 optional lessons when helpful; it does not have to force all ten. If you prefer 8, change the slider.

With Puter, the normal flow requires two clicks: **analyze lecture**, then **select videos**. When the local coverage audit finds missing major objectives, a third button appears: **Revise missing-topic coverage with Puter**. That request is optional but uses Puter account allowance, and it sends the same curated lecture/candidate data plus an audit of omissions. Ollama/OpenAI providers can perform a repair call automatically.

The app now displays **Concept-by-concept coverage audit**, with lecture pages, claimed videos, and whether the claim has local retrieval/PDF corroboration. Core and optional videos appear in separate tables. CSV exports still contain the historical Excel MP4 IDs and lecture page associations.

## Regression tested

`test_coverage_curriculum.py` simulates the exact class of failure: a mock LLM initially selects only four vertebral-column lessons from the real Gross Anatomy → Back candidate set. The missing back-muscle and neurovascular concepts trigger a repair call. The mock revision returns eight core lessons and two optional additions, and all ten IDs are checked against the actual catalog. Existing eight regression suites also pass.

**Limits:** The test uses fake LLM outputs and a deterministic test encoder; it does not prove Puter / `gpt-4o-mini` will produce the same shortlist in a live run. The app never watches videos. A selected MP4 without verified PDF pages is supported by its title and folder, not confirmed lesson-by-lesson teaching content. The Puter browser sign-in and real hosted inference still require live testing on your computer. The approach is generic and has no hardcoded anatomy/diagnosis synonym tables or fixed expected video titles.

## Unmodified data

Your authoritative `bootcamp_step1_file_index.xlsx`, eligible MP4 IDs, PDF mapping and `index/` cache format are unchanged. No retagging or title vector rebuild is needed for this update.
