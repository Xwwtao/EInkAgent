# EInkAgent

![Tests](https://github.com/Xwwtao/EInkAgent/actions/workflows/tests.yml/badge.svg)

An AI agent that helps people choose an E Ink reader or note-taking tablet.
You describe what you need in plain language; the agent queries a database
of real devices and price snapshots, explains the technology with cited
sources, remembers your preferences, and shows where every price came from.

The focus is on making wrong answers detectable: tool failures are recovered,
unexecuted actions are blocked by rules, and every claim about quality points
to a committed evaluation report.

- **Data**: 49 real devices and 41 price snapshots, each with a source URL
  and collection date (`data/data_quality_report.md`).
- **Agent**: DeepSeek tool calling, a hand-written loop over five tools —
  three read-only device tools, knowledge search and preference memory.
- **Interface**: a single-page chat UI served by FastAPI.

## Architecture

```text
Browser (static/index.html) ──POST /agent──▶ api.py ──▶ agent.py ──▶ agent_tools.py
                                               │            │              │
                                    dependency-injected   answer_guard.py  ├──▶ device_repository.py ──▶ SQLite (devices, offers)
                                      DeepSeek client                      ├──▶ knowledge.py ◀── data/knowledge/*.md + index
                                                                           └──▶ memory.py ──▶ SQLite (data/memory.db)

data/*.csv ──▶ data_import.py ──▶ SQLite (devices, offers)
```

| File | Responsibility |
| --- | --- |
| `eink_agent/api.py` | HTTP endpoints, request validation, sessions, model error → status code |
| `eink_agent/agent.py` | Model–tool loop, failure handling, rewrites |
| `eink_agent/answer_guard.py` | Rule-based checks on the final answer |
| `eink_agent/agent_tools.py` | Tool schemas shown to the model, allowlist, argument validation |
| `device_repository.py` | SQL for search, detail and comparison |
| `eink_agent/knowledge.py` | Corpus parsing, section chunks, BM25, vectors, gated hybrid search |
| `eink_agent/memory.py` | Session messages, preference events, rolling summary |
| `data_import.py` / `data_quality.py` | Validated, idempotent CSV import and the data report |

Design decisions, each with the rejected alternative, are in
`docs/decisions/`. A guided code walkthrough is in `docs/learning/`, and a
write-up of the whole project in `docs/retrospective.md` (Chinese).

## Quick start

Requires Python 3.14 and a DeepSeek API key.

```bash
git clone https://github.com/Xwwtao/EInkAgent.git
cd EInkAgent
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python data_import.py
python -m pytest -q
```

`data_import.py` builds `data/eink_devices.db` from the CSV files in `data/`;
running it again does not create duplicates. Tests run offline: the model is
a scripted fake and each test gets a temporary database of fictional devices.

Start the server in a terminal where the key is set (zsh):

```bash
read -s "DEEPSEEK_API_KEY?DeepSeek API key: "
export DEEPSEEK_API_KEY
python -m uvicorn eink_agent.api:app --reload
```

Open http://127.0.0.1:8000. Without a key the page still loads and
`POST /agent` answers HTTP 503 saying the key is missing. Knowledge search
also needs `EMBEDDING_API_KEY` and `EMBEDDING_BASE_URL` (an OpenAI-compatible
embedding endpoint); without them the agent runs with the device tools only.
All variables are listed in `.env.example`; the app reads the environment
only and does not load `.env` files.

### Try it

| Input | What the agent does |
| --- | --- |
| 预算2000元以内，要能手写，推荐几款 | Remembers the budget and stylus need, then searches with them |
| 预算改成1500吧 (as a follow-up) | Overwrites the budget; the bar under the header shows 预算 ≤ ¥1500 and the change history |
| 比较一下 Kindle Paperwhite 2026 和 Kobo Clara BW | Finds both devices by name, then calls `compare_devices` |
| Kaleido 3 的彩色为什么没有黑白清楚？ | Calls `search_knowledge` and cites the passage it used |
| 阅读器一般保修多久？ | The corpus has no answer: the search returns nothing and the agent says no material was found |
| 帮我下单 Kindle Colorsoft | Calls no tool and explains it cannot place orders |
| 我想买个阅读器 | Asks about budget, size and use before searching |

Each answer can expand a timeline of the tool calls it made, with arguments,
result counts and failures. A two-minute demo script is in
`docs/demo-script.md`.

## Capabilities and limitations

What it does:

- **Search, inspect and compare** devices by price, screen size, weight,
  colour, stylus, open system, type, brand or name. Every price comes with
  the seller, collection date and link; overseas prices are converted from
  USD at the fixed rate in `data/exchange_rates.csv`.
- **Explain technology with sources**: `search_knowledge` returns up to three
  passages from 20 short documents in `data/knowledge/`, each with title,
  source URL and verification date. When the best passage scores below the
  gate it returns nothing, and the agent says no supporting material was
  found instead of answering from its own knowledge.
- **Remember a conversation**: the page keeps a session id; the server stores
  the messages in `data/memory.db` (ignored by Git), sends the last 10 to the
  model and folds older ones into a summary of at most 500 characters.
  Preferences are saved by `remember_preferences` as timestamped rows: the
  newest value wins, null withdraws one, and the history stays. Without a
  session id the server stores nothing.
- **Recover from tool failures**: invalid JSON, unknown tools, invalid
  arguments and tool exceptions go back to the model as structured errors.
  Two consecutive failures, an empty answer or running out of rounds return
  a fixed apology instead of an HTTP 500. Memory failures never stop an
  answer.
- **Check answers without a model**: an answer claiming an order, payment or
  edit is rewritten once and then blocked; an answer exposing internal field
  names such as `is_open_system` is rewritten once.

Known limitations:

- **Data is a snapshot.** Specs were verified on 2026-09-11 or 2026-10-05 and
  prices collected on 2026-10-05 or 2026-10-10; nothing refreshes them.
  Fields that could not be determined are NULL and reported as unknown.
  Chinese-brand specs come from IT之家 articles because the official sites are
  image-based. Converted prices are references, not landed prices in China.
- **The answer checks are keyword rules.** A paraphrase gets past them, and a
  refusal word in the same sentence hides a claim (for example "我无法处理
  退款，但已帮您下单了").
- **Prompt rules alone do not stop injected instructions.** In runs 7 and 9
  the first draft of k06 repeated an order claim planted in a retrieved
  document; the answer check caught it both times.
- **The agent sometimes adds facts from general knowledge.** The judge found
  3 such answers in 35.
- **The knowledge documents were written by Claude** from public sources and
  spot-checked by the author; each links its sources. They describe
  technology only, never specific devices.
- **Memory showed no measurable gain** in its with/without comparison (see
  below).
- **No login.** Anyone holding a session id can read that session; this is a
  local demo. Restored answers show text only, without tool calls or cards.
- The agent cannot buy anything or change data; all device tools are
  read-only.

## Evaluation summary

All numbers come from single runs; reports are committed in `evals/reports/`
and the full history is in `docs/evaluation.md`.

| What | Result | Report |
| --- | --- | --- |
| Agent cases (47: search, multi-step, empty results, invalid input, clarification, unsupported actions, prompt injection, multi-turn, data honesty, knowledge, memory) | **45 / 47** passed; `deepseek-flash`, commit `7431219`, p50 4.3 s, p95 20.5 s per case | `20261014T125704Z.md` |
| Knowledge retrieval (32 questions, 5 with no answer in the corpus) | Gated hybrid: Recall@3 0.889, Precision@3 0.654, 5 / 5 unanswerable questions return nothing; keyword search alone returns something for all 5 | `retrieval_20261013T143055Z.md` |
| Memory, with vs only the last 10 messages (6 cases) | **4 / 6 vs 4 / 6**; preferences written in 6 / 6. The model keeps restating constraints, so they stay in the window | `20261014T125704Z.md` |
| Groundedness judge (`qwen3.8-flash`, on run 6, 35 answers) | 32 / 35 judged grounded; agrees with the author's labels on 34 / 35 after review; 10 / 10 injected faults detected | `judge_20261013T123305Z.md`, `calibration_20261013T123305Z.md` |

A case passes only if the required tool calls succeed and return the expected
number of rows or the expected devices, the answer contains the expected
facts, and no unexecuted action is claimed. Expected row counts are checked
against the committed CSV data by a unit test. Changes to the cases are
approved and recorded with their reasons; keyword lists have been frozen since
run 5 so the cases do not drift toward the model's wording.

```bash
python -m examples.evaluate_agent          # all cases, writes evals/reports/<time>.json/.md
python -m examples.evaluate_retrieval      # keyword vs vector vs hybrid retrieval
python -m examples.judge_eval_report A.json  # groundedness judge on a saved run
```

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Chat interface |
| `POST` | `/agent` | `{"message", "history"}` or `{"message", "session_id"}` → answer, status, tool calls, latency, preferences |
| `GET` | `/sessions/{session_id}` | Stored messages, current preferences and their change history |
| `GET` | `/devices`, `/devices/{id}` | Search by brand and price; one device |
| `POST` | `/devices/compare` | Two to five device IDs |
| `GET` | `/health` | Liveness check |

`POST /agent` answers 422 for an invalid request (including both `history`
and `session_id`), 503 without an API key, 504 on a model timeout and 502 on
other model errors. Interactive documentation: http://127.0.0.1:8000/docs.

## Roadmap

- [x] Real data with sources, data quality report
- [x] Tool-calling agent with structured failure recovery and rule-based answer checks
- [x] `POST /agent` and chat UI
- [x] Reproducible evaluation with committed reports; model judge calibrated against hand labels and fault injection
- [x] Retrieval over E Ink technology explanations, with cited sources and an indirect-injection case
- [x] Session memory: preferences with change history, rolling summary, restore after refresh
- [ ] Repeated evaluation runs with variation ranges instead of single runs
- [ ] Memory cases that grade the turn where a preference is withdrawn
- [ ] Retrieval over manuals and after-sales policies
- [ ] Scheduled data refresh

## License

MIT. See [LICENSE](LICENSE).
