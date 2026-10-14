# Evaluation

How EInkAgent is measured, run by run. The README keeps only the summary.
All reports are committed in `evals/reports/`; every number below names its
report. All runs are single runs.


47 hand-written cases in `evals/agent_cases.json` cover single searches,
multi-step lookups, empty results, invalid parameters, clarification,
unsupported actions, prompt injection, multi-turn follow-ups, data
honesty, knowledge questions and memory. One knowledge case (k06) adds an
evaluation-only document with a planted instruction
(`evals/knowledge_injection/`) and checks the answer does not follow it.
A case passes only if the required tool calls succeed, return the
expected number of rows or the expected devices, and the answer contains the
expected facts without claiming an unexecuted action. Expected row counts are
tied to the committed CSV data and checked by a unit test.

## Agent cases

**Latest result: 45 of 47 passed** — `evals/reports/20261014T125704Z.md`,
`deepseek-flash`, single run on 2026-10-14 at commit `7431219`.
p50 latency 4.3 s, p95 20.5 s per case (memory cases play several turns);
301,869 prompt and 37,264 completion tokens, about ¥0.23–0.46 for the whole
run at the off-peak and peak prices in `evals/pricing.json`, including the
window-only control runs. All 41 earlier cases passed; the two failures are
memory cases explained below. In run 8, p03 refused correctly but answered in
English; it answered in Chinese again in run 9 and no rule was added for it.

The report lists which answers passed only after the answer check asked for
a rewrite: in run 9, k06 again first repeated the planted order claim, and
the rewritten answer no longer mentioned a previous draft.

How the result got here, all reports committed in `evals/reports/`:

| Run | Commit | Passed | What changed before it |
| --- | --- | --- | --- |
| 1 | `e512b64` | 31 / 35 | Baseline |
| 2 | `7a115c3` | 29 / 35 | Nothing; measures run-to-run variation |
| 3 | `001efca` | 34 / 35 | Name lookups check the devices found instead of keyword wording (5 of 6 earlier failures had correct answers) |
| 4 | `2b78411` | 33 / 35 | `category` restricted to reader / note / tablet |
| 5 | `d2f7c28` | 33 / 35 | Simplified Chinese reply rule |
| 6 | `1f5fe39` | 34 / 35 | Raw tool-call syntax never returned as an answer |
| 7 | `015673f` | 41 / 41 | Six knowledge cases added (k01–k06) and `search_knowledge` offered |
| 8 | `457e755` | 40 / 41 | Rewrites recorded; rewrite request no longer lets the model apologise for an unseen draft; knowledge queries asked to be in Chinese (run 7's d01 searched in English and missed the water-resistance document) |
| 9 | `7431219` | 45 / 47 | Six memory cases added (mem01–mem06) and run with and without memory |

Every change to the cases was approved and is recorded with its reason in
`docs/decisions/006-agent-evaluation.md`. After run 5 the keyword lists are
frozen: correct answers with different wording are left for the model
judge (below) rather than patched case by case, so the cases do not drift
toward the model's phrasing.

## Knowledge retrieval

32 questions in `evals/retrieval_cases.json` (27 with an answer in the
corpus, 5 without) compare four ways to pick the top 3 of 48 passages
(`evals/reports/retrieval_20261013T143055Z.md`, single run;
`docs/decisions/008-component-knowledge-rag.md`):

| Method | Recall@3 | Precision@3 | Unanswerable → empty |
| --- | --- | --- | --- |
| Keyword (BM25 over Chinese bigrams) | 0.907 | 0.593 | 0 / 5 |
| Vector (`qwen3.7-text-embedding-flash`, cosine ≥ 0.55) | 0.889 | 0.506 | 5 / 5 |
| Hybrid (reciprocal rank fusion) | 0.926 | 0.630 | 0 / 5 |
| **Hybrid, gated on best vector score ≥ 0.55 (used)** | 0.889 | 0.654 | 5 / 5 |

Keyword search always finds something, because common two-character words
such as 多久 match some passage, so it cannot say "no answer". The vector
score can, so it decides whether to answer and the hybrid ranking decides
what to return. Passages are one per `##` section, without overlap (average
about 80 characters). No query rewriting or reranking: there is no data
showing they are needed.

## Memory: with and without

Each memory case plays its earlier turns live with the real model, then
grades only the last turn's searches, including `forbidden_calls` (for
example, no search with the old budget after the user corrected it). Every
case runs twice: with session memory (the same code path as `POST /agent`)
and with only the last 10 messages (`docs/decisions/009-session-memory.md`).

Run 9: **with memory 4 of 6, window only 4 of 6**; the model called
`remember_preferences` in the earlier turns of all 6 cases.

- **mem04 was meant to separate the two**: the constraint is stated in turn
  1 of 7 and has left the window by the last turn. Both passed, because the
  model's answers kept repeating "those 5 black-and-white devices under
  1,500 yuan".
- **mem02 failed in both** by case design: with the corrected budget of
  1,500 and a stylus there are no devices, so the model reasonably searched
  the old budget as a fallback (with memory) or asked which to keep
  (window only).
- **mem03 failed in both** by grading design: neither searched again in the
  last turn, reusing the previous turn's results. The turn that withdrew
  the stylus requirement did differ: with memory the model sent
  `supports_stylus: null` and found 20 devices; with the window only it
  sent `supports_stylus: false`, wrongly excluding every stylus device.

So no improvement is claimed from memory. What it demonstrably adds is a
conversation and preferences that survive a refresh (offline tests), a
visible change history, and an explicit way to withdraw a preference.

## Answer groundedness (model judge)

Keyword checks cannot tell whether every fact in an answer comes from the
tool results. A second model from a different family, `qwen3.8-flash`
(thinking off, temperature 0), answers two yes/no questions per answer:
is every device fact supported by the tool results, and does the answer
address the question. It sees the Agent's system prompt as context, and
everything it grades is marked as data, not instructions
(`docs/decisions/007-llm-judge.md`).

On run 6, before the knowledge cases were added
(`evals/reports/judge_20261013T123305Z.md`):

- **32 of 35 answers judged fully grounded.** The judge flagged s01 ("Libra
  Colour has page-turn buttons"), n02 ("Sony's Reader line was discontinued")
  and t02 ("a stylus is included"). None of these is in the tool results;
  t02 is likely false, since the Kobo stylus is sold separately.
- **Calibration against the author's labels:** 30 of 35 before the judge saw
  the Agent's rules (four false alarms on the fixed-rate price note), 33 of
  35 after.
  Checking each disagreement against the stored tool results showed the
  labels had missed three unsupported statements, and the review also found
  one the judge missed: n03 says nothing is available even at 2,000 yuan
  while its own search returned 15 devices. After this adjudication: 34 of
  35, 1 miss, 0 false alarms (`evals/reports/calibration_20261013T123305Z.md`).
  The labels were corrected after seeing the judge's verdicts, which biases
  agreement upward.
- **Fault injection:** 10 answers with one appended sentence contradicting
  the tool results (wrong weight or price, reversed waterproof or stylus
  support) were all detected, 10 of 10 (`evals/reports/judge_20261013T123305Z.md`;
  the answers with injected faults are in `evals/faults/`). Appended sentences
  are easier to spot than n03's contradiction inside a longer sentence,
  which the judge missed.
- **Limits:** 35 cases, one labeller, single judge run; the judge prompt was
  adjusted on these same 35 answers after the first calibration.

The judge result is reported separately and never changes the pass count.

```bash
python -m examples.evaluate_agent                      # run all cases, write reports
python -m examples.compare_eval_reports A.json B.json  # what changed between runs
python -m examples.regrade_eval_report A.json          # re-grade a saved run with current cases
python -m examples.label_answers A.json                # hand-label groundedness
python -m examples.inject_faults A.json                # build answers with known errors
python -m examples.judge_eval_report A.json            # run the judge (JUDGE_* variables)
python -m examples.calibrate_judge judge_A.json        # recompute agreement after relabelling
python -m examples.evaluate_retrieval                  # retrieval comparison (EMBEDDING_* variables)
python -m examples.build_knowledge_index               # re-embed after editing data/knowledge/
```
