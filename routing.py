"""
routing.py
Model routing — which model class handles which kind of work.

The system runs two tiers of inference and the split is a cost decision, not a
capability one. Most calls in a content pipeline are *bounded*: classify this
row, plan which agents to dispatch, embed this text, normalise this JSON. Those
have a small output space and a checkable result, so an 8B model running locally
is both sufficient and free. The calls that are genuinely open-ended — writing
copy that has to be good, reading an image and reproducing its layout — are the
ones worth paying frontier prices for.

Routing on those properties rather than per-call-site keeps the policy in one
place, so changing where the line sits is a single edit.

    >>> select(TaskClass.PLAN).tier
    <Tier.LOCAL: 'local'>
    >>> select(TaskClass.COPY_GENERATION).tier
    <Tier.FRONTIER: 'frontier'>

In the full system this policy is applied at each call site in the agent layer;
it is consolidated here so the excerpt shows the rule rather than its scattered
applications.
"""
from dataclasses import dataclass
from enum import Enum

from config import config


class Tier(str, Enum):
    LOCAL = "local"        # Ollama on the host machine — free, private, slower
    FRONTIER = "frontier"  # Anthropic API — paid per token


class TaskClass(str, Enum):
    # Bounded: small output space, result is checkable by the caller.
    PLAN = "plan"                      # task -> JSON plan of agents to dispatch
    INTENT = "intent"                  # free text -> one of a fixed set of modes
    EMBED = "embed"                    # text -> vector, for retrieval
    NORMALISE = "normalise"            # reshape/merge structured text
    CLASSIFY = "classify"              # keep/drop, categorise

    # Open-ended: quality is the product, no cheap way to verify it.
    COPY_GENERATION = "copy_generation"    # the words that ship
    STYLE_SYNTHESIS = "style_synthesis"    # infer a voice brief from examples

    # Vision: reading an image well enough to act on it.
    IMAGE_ANALYSIS = "image_analysis"
    LAYOUT_REPRODUCTION = "layout_reproduction"  # image -> HTML


_TIERS = {
    TaskClass.PLAN: Tier.LOCAL,
    TaskClass.INTENT: Tier.LOCAL,
    TaskClass.EMBED: Tier.LOCAL,
    TaskClass.NORMALISE: Tier.LOCAL,
    TaskClass.CLASSIFY: Tier.LOCAL,
    TaskClass.COPY_GENERATION: Tier.FRONTIER,
    TaskClass.STYLE_SYNTHESIS: Tier.FRONTIER,
    TaskClass.IMAGE_ANALYSIS: Tier.FRONTIER,
    TaskClass.LAYOUT_REPRODUCTION: Tier.FRONTIER,
}

# Vision work needs a stronger frontier model than text work does; layout
# reproduction in particular is where the cheap vision models fall down.
_VISION_CLASSES = {TaskClass.IMAGE_ANALYSIS, TaskClass.LAYOUT_REPRODUCTION}


@dataclass(frozen=True)
class Route:
    task_class: TaskClass
    tier: Tier
    model: str

    @property
    def is_local(self) -> bool:
        return self.tier is Tier.LOCAL


def select(task_class: TaskClass, cfg: dict | None = None) -> Route:
    """Resolve a task class to the tier and concrete model that should serve it."""
    cfg = config if cfg is None else cfg
    tier = _TIERS[task_class]

    if tier is Tier.LOCAL:
        model = (cfg.get("ollama_embed_model") if task_class is TaskClass.EMBED
                 else cfg.get("ollama_model"))
    else:
        model = (cfg.get("frontier_vision_model") if task_class in _VISION_CLASSES
                 else cfg.get("frontier_model"))

    return Route(task_class=task_class, tier=tier, model=model)


def escalate(route: Route, cfg: dict | None = None) -> Route:
    """Promote a local route to the frontier tier.

    Used as a fallback, not a default: if the local model is unreachable or its
    output fails to parse after a retry, the work is worth a paid call rather
    than a failed task. Frontier routes are returned unchanged — there is no
    tier above them to escalate to.
    """
    if route.tier is Tier.FRONTIER:
        return route
    cfg = config if cfg is None else cfg
    return Route(task_class=route.task_class, tier=Tier.FRONTIER,
                 model=cfg.get("frontier_model"))
