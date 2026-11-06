# AgentFlow

**A multi-agent research assistant built on LangGraph**, with schema-validated tool calling,
retry and error recovery, end-to-end tracing (Langfuse) and trajectory evaluation.
It uses DeepSeek through its OpenAI-compatible API, and it also runs fully offline with a
deterministic mock LLM, so you can clone it, test it and demo it without an API key.

> Built by [Jessie Borras](https://jessiebdev.xyz/services/ai-business-automation) as part of a
> series on production-grade **AI business automation**.
> [Read the case study ↓](#case-study-from-demo-agent-to-production-workflow)

```
START ─▶ planner ─▶ researcher ─▶ writer ─▶ critic ──approved──▶ finalize ─▶ END
                     │  ▲            ▲          │
             tools   ▼  │            └─revise───┘  (bounded by AGENTFLOW_MAX_REVISIONS)
      web_search · http_request · run_python
```

## What it does

| Agent | Job | Output |
|---|---|---|
| **Planner** | Breaks the question into 2–5 research steps | `Plan` (Pydantic-validated JSON) |
| **Researcher** | ReAct-style tool loop: search, call APIs, run code | Bullet notes with `[source: URL]` |
| **Writer** | Writes a cited markdown answer from the notes only | Draft with `## Sources` |
| **Critic** | Scores grounding, citations, completeness | `Critique` → approve or send back |

### Tools with proper schemas
Each tool declares a Pydantic input model that is converted into an OpenAI/DeepSeek
function-calling JSON Schema (`agentflow tools` prints them). Arguments are validated before
anything runs; invalid arguments are returned to the model with a hint so it can self-correct.

* `web_search`: Tavily if `TAVILY_API_KEY` is set, otherwise DuckDuckGo, or an offline corpus in mock mode.
* `http_request`: GET/POST to public APIs, with an SSRF guard (blocks loopback and private IPs), an optional host allowlist, 4xx/5xx classification and truncated responses.
* `run_python`: short calculations in a separate `python -I` process. An AST policy blocks imports outside a stdlib allowlist, file and process builtins, and dunder access. It also enforces a timeout, an empty environment, a scratch directory and CPU/memory rlimits. *This is defence in depth, not a hard sandbox: for untrusted multi-tenant use, run it in a container or microVM (Docker, gVisor, E2B).*

### Error recovery and retries
| Layer | Mechanism |
|---|---|
| LLM call | Exponential backoff + jitter on timeouts, 429s and 5xx (`TransientError`); 4xx fail fast |
| Structured output | Invalid JSON or schema mismatches are fed back to the model for repair (up to 2 times) |
| Tool call | Retries transient errors only, never permanent ones; a per-tool **circuit breaker** disables a tool after 3 consecutive failures and hides it from the model |
| Node | LangGraph `RetryPolicy` re-runs a whole node on transient errors |
| Graceful degradation | Planner falls back to a generic plan, researcher salvages partial tool output when the budget runs out, and a broken critic doesn't block delivery |
| Budgets | `AGENTFLOW_MAX_TOOL_CALLS`, `AGENTFLOW_MAX_REVISIONS`, graph `recursion_limit` |

### Tracing
Every run, node, LLM generation, tool call, retry, recovery and score is a span with a parent,
timing, input and output. Spans are always saved locally to `runs/<run_id>.json` (the *trajectory*).
If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, the same tree is mirrored to Langfuse
as nested `agent` / `generation` / `tool` observations, with critic and trajectory scores attached
to the trace. Tracing failures never break a run.

### Trajectory evaluation
`agentflow eval` runs a dataset (`evals/dataset.json`) and scores **how** the agent worked, not just what it answered:

* expected-tool recall and order (subsequence match)
* tool success rate, recovered vs. unrecovered errors, redundant calls
* step budget adherence (tool and LLM calls)
* citation count, required-content hits, critic score
* optional LLM-as-judge score on the full trajectory

Scores are combined into an `overall` score from 0 to 1 and pushed to Langfuse when it's enabled.

## Quick start

```bash
git clone <this repo> agentflow && cd agentflow
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Works immediately in mock mode (no key, no network):
agentflow run "How does LangGraph help build controllable multi-agent workflows?"
agentflow eval --mock
pytest -q
```

### Use DeepSeek
```bash
cp .env.example .env
# edit .env → DEEPSEEK_API_KEY=sk-...
agentflow run "What are the trade-offs of LangGraph vs. a hand-rolled agent loop?"
agentflow eval --out eval-report.json
```
`DEEPSEEK_MODEL` defaults to `deepseek-chat`. Any OpenAI-compatible endpoint works by changing
`DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`.

### Python API
```python
from agentflow import AgentFlow, Settings

flow = AgentFlow(Settings.from_env())
result = flow.run("Summarise current best practice for agent observability")
print(result.answer, result.critique, result.tool_log, result.trajectory_path)
```

## Project layout
```
src/agentflow/
  graph.py        LangGraph StateGraph, RetryPolicy, AgentFlow entry point
  agents.py       planner / researcher / writer / critic nodes + recovery paths
  llm.py          DeepSeek client, MockLLM, traced + structured (self-repairing) calls
  tools/          base registry (schemas, validation, retry, circuit breaker) + 3 tools
  tracing.py      span tree, local trajectories, Langfuse mirroring
  evaluation.py   trajectory metrics, LLM judge, dataset runner
  retry.py        backoff + jitter helper (tenacity)
  cli.py          `agentflow run | eval | tools`
evals/dataset.json
tests/            29 tests, all offline: retries, repair, self-correction, breaker, revision loop, eval
```

---

## Case study: from demo agent to production workflow

**Client problem (typical):** teams can prototype an "AI agent" in an afternoon, but the
prototype breaks in production. It calls tools with malformed arguments, stalls on a flaky API,
loops forever, and nobody can explain *why* it produced a given answer. Those are the questions
I get most often from businesses exploring
[AI business automation](https://jessiebdev.xyz/services/ai-business-automation).

**What I built:** AgentFlow, a reference multi-agent workflow that treats reliability and
observability as first-class features:

1. **Separation of concerns.** Four specialised agents (plan, research, write, critique) on an
   explicit LangGraph state machine. Each step is inspectable and testable, and the
   critique-and-revise loop is bounded.
2. **Contracts at every boundary.** Tools and agent outputs are Pydantic schemas. Bad model
   output gets repaired, not crashed on, and bad tool arguments come back to the model as
   actionable errors.
3. **Failure as a normal path.** Errors are classified as transient or permanent. Transient
   ones get backoff with jitter, a failing tool trips a circuit breaker, and every agent has a
   degraded-mode fallback, so the user still gets an answer.
4. **Every decision is traceable.** A span tree per run (locally and in Langfuse) shows each
   prompt, tool call, retry and score.
5. **Measured, not vibed.** Trajectory evaluation scores tool choice, ordering, success rate,
   error recovery, budget use and grounding. Regressions show up in CI, before users see them.

**Results in this repo:** 29 automated tests cover each failure mode (injected LLM outages,
malformed JSON, invalid tool arguments, rejected drafts, runaway loops). The offline evaluation
suite completes 100% of cases with full expected-tool recall. The same graph moves from the mock
model to DeepSeek by setting one environment variable. Its OpenAI-compatible API keeps cost low
and avoids vendor lock-in.

**Where this pattern applies:** research and reporting assistants, customer-support triage with
tool access (CRM, order and ticket APIs), lead enrichment, and back-office workflow automation:
anywhere an LLM has to take reliable, auditable actions.

👉 Want something like this for your business? See
**[AI Business Automation services at jessiebdev.xyz](https://jessiebdev.xyz/services/ai-business-automation)**.

---

## License
MIT
