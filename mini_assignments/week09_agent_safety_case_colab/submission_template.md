# Week 9 — Probing an Agent: What's Actually Happening Under the Hood

Student: `NAME`
Tutorial group: `GROUP`
Notebook run date: `DATE`
Backend: OpenRouter (`nvidia/nemotron-3-nano-30b-a3b:free`) — model identifier(s) actually returned: `MODEL_ID(S)`

> Export this document to a single PDF before submitting. Every lettered section below must include
> at least one screenshot of the actual notebook cell input/output you are citing. A cited trace ID
> with no accompanying screenshot earns no evidence credit for that row.

## A. Progressive component map — 10 points

| Layer | Notebook object / cell | One failure it can introduce | Screenshot ref. |
| --- | --- | --- | --- |
| Base LLM | | | |
| Agent policy | | | |
| Environment | | | |
| Tools | | | |
| Runtime | | | |
| Memory / trace | | | |
| Evaluator | | | |
| Planner / reviewer | | | |
| Human gate | | | |

## B. Missing-capability probe (P1, section 6.1) — 20 points

**Conversation A trace IDs and what they show:**

`WRITE HERE`

**Conversation B trace IDs and what they show:**

`WRITE HERE`

**Taxonomy classification of Conversation B's outcome:**

`WRITE HERE`

**What the missing "remember" capability caused to go wrong — or be honestly admitted:**

`WRITE HERE`

## C. Full-capability probe (P2, section 6.2) — 15 points

**Conversation B trace IDs here vs. section B's Conversation B:**

`WRITE HERE`

**What changed, and why (same-looking question, different evidence backing):**

`WRITE HERE`

## D. Reflection and evaluation (P3, section 7) — 15 points

**Reflection-only FINAL OUTPUT (trace ID) and whether it was honest about the lack of evidence:**

`WRITE HERE`

**Evidence-grounded FINAL OUTPUT (trace ID) and the real API reading it used:**

`WRITE HERE`

**Component-eval result and end-to-end `answer_matches_tool_data` result:**

`WRITE HERE`

**Why neither a critic nor a passing component check proves the final prose is faithful to the data:**

`WRITE HERE`

## E. Your own probe (P4) — 20 points

**Condition you changed (must not be a probe already shown in B/C/D):**

`WRITE HERE`

**Trace IDs and what they show:**

`WRITE HERE`

**Taxonomy classification:**

`WRITE HERE`

**What this tells you that B-D did not already show:**

`WRITE HERE`

## F. Planning, handoff, and regression — 10 points

**Plan validation outcome (trace ID):**

`WRITE HERE`

**Fact-check reviewer's verdict (trace ID):**

`WRITE HERE`

**Regression scenario:**

- Inputs / preconditions:
- Expected observation:
- Verification layer:
- Why this would catch a future recurrence:

## G. Acceptance and disclosure — 10 points

**Verdict on the chatbot's answer to the user:** `accept` / `accept with revisions` / `reject`

**Verdict on trusting this agent to act unsupervised on your behalf:** `accept` / `accept with revisions` / `reject`

**Trace evidence (at least two IDs):**

`WRITE HERE`

**AI-agent usage disclosure:**

- AI tools used (or "none"):
- What they helped draft or find:
- What I manually checked against the trace:
- One suggestion I rejected or narrowed:

## H. Feedback on this mini-assignment — required, not scored on content

- What was confusing, underspecified, or took longer than it should have:
- Where the notebook's behavior got in the way of demonstrating what I understood:
- One concrete change I would make, and why:
