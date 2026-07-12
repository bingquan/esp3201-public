# Week 9 Mini Assignment — Probing an Agent: What's Actually Happening Under the Hood

> **Course throughline — _confident ≠ correct_ (9 / 10).** An agent's plan,
> patch, reflection, or single passing test is a candidate. It becomes an
> engineering result only when the evidence supports the exact claim made.

## Format

Format B/C hybrid (run the agent notebook yourself, then classify what actually happened against a
taxonomy of agent failure modes).

This replaces the earlier fixed-trace audit. The audit skill is the same — separate supported
behavior from plausible-looking behavior — but you generate the trace yourself, by running a real,
pinned model against a real (if small) tool environment, rather than reading someone else's
recorded run. That is harder to fake and closer to what an engineer actually does when deciding
whether to trust an agent.

## Assignment Intent

You will run `notebooks/week09_agent_colab-v3.ipynb` yourself, end to end, on a real model
(`nvidia/nemotron-3-nano-30b-a3b:free` via OpenRouter) and a real, free, keyless weather API
(Open-Meteo). The notebook progressively exposes the layers of an agentic system using a weather
chatbot as the running example. Your job is to probe how it actually behaves — not how you'd expect
it to behave — under specific conditions, and to classify what you find.

The target skills are:

- distinguishing a base LLM response from an agent interacting with a real environment
- probing an agent's behavior under a missing capability, contrasted with the same task when the
  capability is present
- separating reflection-only self-correction from evidence-grounded correction
- classifying an observed failure against a taxonomy of agent failure modes, rather than just
  noting "something went wrong"
- judging whether an agent's final answer is actually supported by its own trace

## Required Probes (Run All Four)

Everyone runs the same notebook end to end. Four of its sections are the required probes:

| Probe | Section | What it stresses |
| --- | --- | --- |
| P1 — Missing capability | 6.1 (two conversations) | capability gap: does the agent honestly admit a gap, silently omit it, or overclaim? |
| P2 — Full capability | 6.2 (two conversations) | contrast: does memory actually work once the tool is present? |
| P3 — Reflection vs. evidence | 7 | does self-correction without evidence produce an honest non-answer, or a confident but ungrounded one? |
| P4 — Your own variation | — | change one condition not shown in P1-P3 (a different city, an ambiguous phrasing, or a deliberately malformed request) and report what your own run actually produced |

P4 must be genuinely different from the class demo — do not just re-run Tokyo/Singapore with the
same wording. Report only what your own run actually produced, never what you'd expect a model to
produce.

## Agent Failure-Mode Taxonomy

Classify every finding as one of:

- **Capability gap, honest** — the agent correctly declines or omits a task it cannot do.
- **Capability gap, overclaim** — the agent implies success on a capability it does not have.
- **Silent state leak** — a tool's side effect exposes information across a capability boundary
  that should not be visible.
- **Malformed reply** — the model's raw output is not valid JSON and gets caught as a recorded
  error, not a crash.
- **Template/placeholder echo** — the model copies instruction text verbatim instead of writing a
  real answer.
- **Reasoning leakage** — chain-of-thought text leaks into what should be a clean final answer.
- **Unverified claim** — the final answer states something the trace's own evidence does not
  actually support.

A finding that does not fit any category is still valid evidence — name it and explain why the
taxonomy misses it.

## Deliverable

Submit one PDF report (export a copy of `week09_agent_colab-v3_run.json` alongside it — do not
submit only the JSON, and do not submit a raw `.ipynb`). Use `submission_template.md`. Every
lettered section requires at least one screenshot of your own run — a quoted trace ID with no
accompanying screenshot earns no evidence credit for that row.

### A. Progressive component map — 10 points

For each layer — base LLM, agent policy, environment (`ChatSession`), tools, runtime, memory/trace,
evaluator, planner/reviewer, human gate — name the notebook object or cell that implements it and
one failure it can introduce.

### B. Missing-capability probe (P1) — 20 points

Run section 6.1's two conversations. Cite at least three trace IDs across both conversations.
Classify Conversation B's behavior against the taxonomy, and explain what the missing "remember"
capability caused to go wrong — or be honestly admitted.

### C. Full-capability probe (P2) — 15 points

Run section 6.2's two conversations. Contrast Conversation B's outcome here against section B's
Conversation B: same-looking question, different evidence backing.

### D. Reflection and evaluation (P3) — 15 points

Compare reflection-only and evidence-grounded revision (section 7). Include one component-eval
result and the end-to-end `answer_matches_tool_data` result (section 8). Explain why neither a
critic nor a passing component check establishes that the final prose is faithful to the retrieved
data.

### E. Your own probe (P4) — 20 points

Run a condition not shown in B/C/D. Cite trace IDs, classify the outcome against the taxonomy, and
explain what it tells you that B-D did not already show.

### F. Planning, handoff, and regression — 10 points

Audit the plan validation and the fact-check reviewer's verdict (sections 9-10). Propose one
regression scenario with input state, expected observation, layer, and remaining limit.

### G. Acceptance and disclosure — 10 points

Issue accept / accept-with-revisions / reject for the **chatbot's answer to the user** and
separately for **trusting this agent to act unsupervised on your behalf**. Record the model
identifier actually returned, run variability, any empty/malformed replies, AI assistance, manual
checks, and one generated suggestion you rejected or bounded.

**Variability rule:** grading is based on trace-grounded interpretation, not whether the model chose
an instructor-preferred sequence. A malformed JSON action, an empty/null model reply, an invalid
tool request, a premature final answer, or a rate-limit event is valid evidence if documented,
screenshotted, and analyzed.

### H. Feedback on this mini-assignment — required, not scored on content

In a few sentences each: what was confusing or underspecified; where the notebook's behavior got in
the way of demonstrating what you understood rather than testing it; and one concrete change you
would make and why. Be specific — cite a cell ID, section letter, or trace ID wherever you can.

## What strong work looks like

Strong submissions distinguish a model's candidate answer from a verified one, name the exact
mechanism behind a failure rather than "the agent made a mistake," and use the taxonomy to compare
findings across probes rather than describing each probe in isolation.

See `grading_rubric.md` for the full rubric.
