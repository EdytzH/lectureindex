# Evidence-first curriculum planning — implementation and limitations

## Why the earlier approach failed

The old final selection call could omit half the lecture while claiming its own
concept labels proved coverage. A few high semantic matches for disc disease
crowded out dedicated joints, muscles and neurovascular teaching. A local audit
counted the same model's claims rather than independently testing evidence.

## New source-to-output contract

1. Each lecture concept has an exact label, major/supporting importance and
   source PDF slide numbers. Lecture analysis sees a page-by-page overview.
2. Hierarchical retrieval retains complete promising Excel topic folders and
   extra global candidates. The real MP4 inventory controls all possible IDs.
3. The LLM returns up to four video candidates per concept, with a `direct` or
   `partial` assessment and an exact quote from either a candidate's real title
   or a verified PDF teaching excerpt.
4. Local validation rejects unknown video IDs and fabricated PDF/text quotes.
   Paraphrased title quotes can be repaired only using the real catalog title,
   and only when independent corpus-specific lexical alignment is strong.
5. A conservative title-only fallback can surface one better-named video from
   the most coherent retrieved topic folders when a small LLM omitted it. Such
   a link is explicitly labeled **title heuristic**, not PDF proof.
6. Marginal-coverage optimization chooses the video contributing the most
   **new** major concept coverage. Multiple videos assigned to the same concept
   do not add their scores together, avoiding duplicate clinical selections.
   Optional videos are only added after all major concepts meet a conservative
   evidence threshold. Results are ordered by source lecture page.
7. The app exposes the accepted concept-to-video matrix, rejected unsupported
   model statements, candidate inventory and uncovered lecture concepts.

## What isn't claimed

- No MP4 video was watched; file paths are historical identities.
- A title-level inference doesn't prove the video's actual teaching content.
- PDF evidence exists only where source headings and page ranges were verified.
- Reranking scores and concept-strength thresholds aren't calibrated
  probabilities; stronger validation requires human-graded positive/negative
  examples and an actual Puter/Ollama evaluation on the device.
- No live Puter authentication or remote model inference was available in
  this build environment. The API flow was exercised by offline fake providers.

## Evaluation

Use `MSK_Week1_Tuesday_Vertebral column and back_Ernst_2026.pdf` as a first
manual benchmark. Compare coverage of the lecture's anatomy and musculature
sections independently, not only overlap against a particular exact list.
The reference lecture is NOT embedded in production matching logic.

Read the per-concept matrix before trusting titles. If a concept is marked as
uncovered, inspect the candidate-audit folder first. Treat phantom/wrong
conditions as errors, not merely lower-scored recommendations.
