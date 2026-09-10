"""
agents/ceo.py
Orchestrator agent — plans a task and dispatches specialist agents.

Flow, in order:
  1. Retrieve similar past tasks from vector memory (local embeddings).
  2. Plan: turn the free-form task into a structured list of steps. This runs
     on the LOCAL model — the output space is small and the result is checkable
     (it either parses into the expected shape or it doesn't).
  3. Fall back if the plan doesn't parse: first escalate to the frontier tier,
     then, if that also fails, degrade to a single-step plan extracted from the
     task text with regex. A malformed plan should never fail a task outright.
  4. Dispatch: one step runs inline, several run in a bounded thread pool.

EXCERPT: the specialist agents this dispatches to, and the prompt text used for
planning, are not part of this repo. `_PLAN_SYSTEM` and `_PLAN_PROMPT` are
placeholders; the dispatch table resolves to a stub registry. The control flow
and the routing decisions are the parts worth reading here.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import database
from agents.base import BaseAgent
from routing import TaskClass, escalate, select

MAX_PARALLEL_STEPS = 5

# Prompt text lives with the private system. The planner's contract is what
# matters structurally: it is handed the task plus retrieved context, and must
# return JSON of the shape {"plan_summary": str, "steps": [{"agent", "params"}]}.
_PLAN_SYSTEM = "<planning system prompt — not included in this excerpt>"
_PLAN_PROMPT = "<planning prompt template — not included in this excerpt>"


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerating code fences."""
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```", "", text).strip()
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        text = match.group(0)
    return json.loads(text)


def _fallback_plan(task: str, client: str) -> dict:
    """Last-resort plan built from the task text alone, with no model call.

    Cheap to be wrong here: a single-step plan that the operator can see and
    correct beats an error page.
    """
    params = {"topic": task, "client": client}
    slide_match = re.search(r"\b(\d+)\s+slide", task, re.I)
    if slide_match:
        params["slide_count"] = int(slide_match.group(1))
    return {
        "plan_summary": f"Single-step plan for: {task}",
        "steps": [{"agent": "content_generator", "params": params}],
    }


class CEOAgent(BaseAgent):
    """Plans a task, then dispatches the steps it produced."""

    name = "ceo"

    def _run(self, task: str, client: str = "default") -> dict:
        self.log(f"Task received: {task!r}")

        self.log("Searching memory for similar past tasks...")
        context = self._recall(task)
        self.log(f"Found {len(context)} relevant past task(s).")

        plan = self._plan(task, client, context)
        self.log(f"Plan: {plan['plan_summary']}")

        database.update_task(self.task_id, status="running")
        results = self._execute(plan, client)
        return {"plan": plan, "results": results}

    # ── Planning ─────────────────────────────────────────────────────────────

    def _recall(self, task: str) -> list:
        """Vector-memory lookup. Embeddings are a bounded task -> local tier."""
        route = select(TaskClass.EMBED)
        self.log(f"Embedding via {route.tier.value}:{route.model}")
        # Retrieval backend is not part of this excerpt.
        return []

    def _plan(self, task: str, client: str, context: list) -> dict:
        """Produce a structured plan, escalating a tier if the local model can't."""
        route = select(TaskClass.PLAN)
        self.log(f"Planning via {route.tier.value}:{route.model}")

        raw = self._complete(route, _PLAN_SYSTEM, _PLAN_PROMPT)
        try:
            return self._validated(_extract_json(raw))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.log(f"Local plan did not parse ({exc}); escalating.")

        route = escalate(route)
        self.log(f"Re-planning via {route.tier.value}:{route.model}")
        try:
            raw = self._complete(route, _PLAN_SYSTEM, _PLAN_PROMPT)
            return self._validated(_extract_json(raw))
        except Exception as exc:
            self.log(f"Escalated plan also failed ({exc}); using fallback.")
            return _fallback_plan(task, client)

    @staticmethod
    def _validated(plan: dict) -> dict:
        """Reject a plan that parsed as JSON but isn't the shape we dispatch on."""
        if not isinstance(plan.get("steps"), list) or not plan["steps"]:
            raise ValueError("plan has no steps")
        for step in plan["steps"]:
            if not isinstance(step, dict) or "agent" not in step:
                raise ValueError("step is missing an agent")
        plan.setdefault("plan_summary", "")
        return plan

    def _complete(self, route, system: str, prompt: str) -> str:
        """Send a completion to whichever tier the route names.

        The two clients expose the same call shape deliberately, so routing is a
        choice of module rather than a branch in every call site.
        """
        if route.is_local:
            from tools import ollama
            return ollama.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": prompt}],
                model=route.model,
            )
        from tools import claude_api
        return claude_api.complete(system, prompt, model=route.model)

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def _execute(self, plan: dict, client: str) -> list:
        steps = plan["steps"]
        for step in steps:
            step.setdefault("params", {}).setdefault("client", client)

        if len(steps) == 1:
            self.log(f"Dispatching -> {steps[0]['agent']}")
            return [self._dispatch(steps[0])]

        workers = min(len(steps), MAX_PARALLEL_STEPS)
        self.log(f"Running {len(steps)} agents across {workers} worker(s)...")
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._dispatch, s): s for s in steps}
            for future in as_completed(futures):
                step = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    # One failed step must not take down the others.
                    self.log(f"Step failed ({step.get('agent')}): {exc}")
                    results.append({"agent": step.get("agent", "unknown"),
                                    "error": str(exc)})
        return results

    def _dispatch(self, step: dict) -> dict:
        """Resolve an agent name to an implementation and run it.

        EXCERPT: the specialist agents are not included. The registry lookup and
        the unknown-agent path are the structure; the implementations are not.
        """
        agent_name = step["agent"]
        params = step.get("params", {})

        from agents import REGISTRY
        implementation = REGISTRY.get(agent_name)
        if implementation is None:
            self.log(f"Unknown agent: {agent_name} — skipping.")
            return {"agent": agent_name, "skipped": True}

        self.log(f"[{params.get('topic', agent_name)}] Starting...")
        output = implementation(task_id=self.task_id, **params)
        return {"agent": agent_name, "output": output}
