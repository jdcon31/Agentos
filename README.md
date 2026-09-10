# AgentOS

**An excerpt from a private production system.** AgentOS runs content generation for a
small social media agency. This repo contains the orchestration spine and the model
routing policy — enough to show how the system is put together, without the parts that
are operationally specific to the business. What's missing is listed at the bottom.

## What the system does

An operator submits a task in plain language ("three carousels on X"). An orchestrator
agent plans it into steps, dispatches specialist agents to do the work, and streams
progress back to the browser while it runs. Everything is persisted, so the operator can
close the tab and come back.

## Architecture

```
HTTP POST /tasks
      │
      ▼
FastAPI (main.py) ──── 202 + task_id ────► client opens /ws/{task_id}
      │                                            ▲
      │ run_in_executor                            │ log lines
      ▼                                            │
CEOAgent (agents/ceo.py) ──────────────────────────┘
      │  1. recall similar past tasks   (local: embeddings)
      │  2. plan into steps             (local: 8B model)
      │  3. escalate if the plan won't parse   (frontier)
      │  4. dispatch, in parallel when there's more than one step
      ▼
specialist agents ──► SQLite (database.py)
```

**Agents run on threads, not the event loop.** They're synchronous and IO-bound — model
calls, image rendering — so they'd block everything if they ran inline. That's why the
log callback they're handed hops back onto the main loop via `run_coroutine_threadsafe`
before touching a WebSocket.

**A late WebSocket connection replays history before attaching to the live stream**, so
reconnecting mid-run doesn't drop you into the middle of a sentence.

**Failure is contained at the step level.** One agent raising doesn't fail the task; the
error lands in that step's result and the rest keep going.

## Model routing (`routing.py`)

Two tiers of inference, split on a cost argument rather than a capability one.

Most calls in a content pipeline are **bounded** — plan this task into steps, embed this
text, classify this row, reshape this JSON. Small output space, and the caller can check
whether the result is right. A local 8B model handles those, for free, on the host
machine.

The calls that are genuinely **open-ended** — writing copy that has to be good, reading
an image well enough to reproduce its layout — have no cheap verification and no
substitute for quality. Those go to the frontier tier and are worth paying for.

```python
select(TaskClass.PLAN).tier             # Tier.LOCAL     — llama3.1:8b
select(TaskClass.COPY_GENERATION).tier  # Tier.FRONTIER  — claude-haiku-4-5
select(TaskClass.LAYOUT_REPRODUCTION)   # Tier.FRONTIER  — claude-sonnet-4-6
```

Vision work gets the stronger frontier model; image-to-layout is where the cheap vision
models fall over.

The escalation path matters as much as the routing. A local plan that doesn't parse is
retried one tier up rather than failed — a paid call is cheaper than a dead task. If that
also fails, a regex fallback builds a single-step plan from the task text, on the theory
that a plan the operator can see and correct beats an error page. Frontier routes don't
escalate; there's nothing above them.

The two clients (`tools/ollama.py`, `tools/claude_api.py`) expose the same call shape
deliberately, so routing selects a module instead of branching at every call site.

## Layout

| Path | What's in it |
|---|---|
| `main.py` | FastAPI app: task lifecycle, background dispatch, WebSocket log streaming |
| `routing.py` | The two-tier model routing policy and its escalation rule |
| `agents/base.py` | `BaseAgent` — run bookkeeping and log capture for every agent |
| `agents/ceo.py` | Orchestrator: recall, plan, escalate, dispatch |
| `tools/ollama.py` | Local inference client (streaming, embeddings) |
| `tools/claude_api.py` | Frontier client, with the retry policy for transient API failures |
| `database.py` | SQLite schema and accessors for tasks and agent runs |

```
pip install -r requirements.txt
cp .env.example .env      # nothing here is required to run the tests
pytest                    # 41 tests, no network and no models needed
uvicorn main:app --reload
```

## What this excerpt leaves out

The running system is larger. Not included here:

- **The specialist agents** — content generation, image remix, vision analysis. The
  registry in `agents/__init__.py` ships empty, so the orchestrator's unknown-agent path
  is the one that executes.
- **All prompt text.** The planner's system and user prompts are placeholders. The
  planner's *contract* — task plus retrieved context in, `{"plan_summary", "steps"}` JSON
  out — is what the code depends on, and that's intact.
- **Client-specific material** — per-client brand configs, voice playbooks, style
  libraries, and the retrieval corpus behind them.
- **Roughly forty HTTP routes** covering client management, asset and style libraries,
  training data, scheduling and OAuth publishing. The task lifecycle is here; the rest
  hangs off it.
- **Additional database tables** for the approval inbox, asset library, and scheduled
  posts.

One note on honesty: in the private system the routing policy is applied at each call
site in the agent layer. It's consolidated into `routing.py` here so the excerpt shows
the rule rather than forty scattered applications of it. The policy itself is unchanged.

## Stack

Python 3.11, FastAPI, Uvicorn, SQLite, Ollama (`llama3.1:8b`, `nomic-embed-text`),
Anthropic API.
