# Week 9 Mini Assignment Rubric — Probing an Agent

Total: 100 points.

## A. Progressive component map — 10 points

- `9–10`: Correctly names the notebook object/cell for at least eight of nine layers, each with one
  concrete failure it can introduce.
- `5–8`: Most layers correct but one or two are vague or generic ("the code handles it").
- `0–4`: Confuses layers (e.g., calls the evaluator the same thing as the tool registry).

## B. Missing-capability probe (P1) — 20 points

- `18–20`: Cites at least three trace IDs across both conversations, correctly classifies
  Conversation B's outcome against the taxonomy, and names the actual mechanism (e.g., the
  `last_weather` state leak, or an honest capability-gap admission).
- `10–17`: Finds the gap but classification is imprecise, or cites fewer than three trace IDs.
- `0–9`: Calls the outcome "wrong" without naming a taxonomy category or mechanism.

## C. Full-capability probe (P2) — 15 points

- `13–15`: Clearly contrasts Conversation B's evidence backing here vs. section B, not just wording.
- `7–12`: Notes the contrast but only compares surface text.
- `0–6`: Treats P2 as unrelated to P1 rather than as its contrast case.

## D. Reflection and evaluation (P3) — 15 points

- `13–15`: Correctly identifies whether the reflection-only FINAL OUTPUT was an honest non-answer or
  an unverified claim, includes both eval results, and explains why neither proves prose fidelity.
- `7–12`: Correct but one component (eval result, or the "why" explanation) is missing or generic.
- `0–6`: Treats a passing eval as proof the final answer is correct.

## E. Your own probe (P4) — 20 points

- `18–20`: Genuinely different condition from P1-P3, correctly classified against the taxonomy, with
  an explicit statement of what it reveals that the required probes did not.
- `10–17`: A real variation but classification is weak or the "what's new" claim is thin.
- `0–9`: A cosmetic reword of an existing probe (e.g., same query, different city, no new mechanism).

## F. Planning, handoff, and regression — 10 points

- `9–10`: Traces the plan validation and reviewer verdict precisely; regression scenario has
  concrete inputs, expected observation, and layer.
- `5–8`: Mostly correct but regression is generic ("add more tests").
- `0–4`: Misreads validation/reviewer output, or omits a regression scenario.

## G. Acceptance and disclosure — 10 points

- `9–10`: Two separate, bounded verdicts (chatbot answer vs. unsupervised trust) each grounded in
  specific trace evidence from B/C/D/E; disclosure names specific AI help, manual checks, and one
  rejected/bounded suggestion.
- `5–8`: Verdicts present but one is unbounded, or disclosure is generic.
- `0–4`: Blanket "accept" for unsupervised use with no reference to a found failure mode; missing or
  indistinguishable disclosure.

## Automatic caps

- A submission that treats a passing component eval as proof of a correct final answer is capped at
  75.
- A submission whose section E "own probe" is a trivial reword of P1-P3 (same mechanism, no new
  finding) loses full credit for section E regardless of write-up quality.
- A ledger/probe row citing a trace ID with no accompanying screenshot receives no credit for that
  row.
- Section H (feedback) is required for the submission to count as complete, but is not scored on
  content.
