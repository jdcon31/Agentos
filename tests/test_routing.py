"""Tests for the model-routing policy. No network, no models."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from routing import Route, TaskClass, Tier, escalate, select

CFG = {
    "ollama_model": "llama3.1:8b",
    "ollama_embed_model": "nomic-embed-text",
    "frontier_model": "claude-haiku-4-5",
    "frontier_vision_model": "claude-sonnet-4-6",
}


@pytest.mark.parametrize("task_class", [
    TaskClass.PLAN, TaskClass.INTENT, TaskClass.EMBED,
    TaskClass.NORMALISE, TaskClass.CLASSIFY,
])
def test_bounded_work_stays_local(task_class):
    assert select(task_class, CFG).tier is Tier.LOCAL


@pytest.mark.parametrize("task_class", [
    TaskClass.COPY_GENERATION, TaskClass.STYLE_SYNTHESIS,
    TaskClass.IMAGE_ANALYSIS, TaskClass.LAYOUT_REPRODUCTION,
])
def test_open_ended_work_goes_frontier(task_class):
    assert select(task_class, CFG).tier is Tier.FRONTIER


def test_embedding_uses_the_embedding_model_not_the_chat_model():
    assert select(TaskClass.EMBED, CFG).model == "nomic-embed-text"
    assert select(TaskClass.PLAN, CFG).model == "llama3.1:8b"


def test_vision_gets_the_stronger_frontier_model():
    assert select(TaskClass.LAYOUT_REPRODUCTION, CFG).model == "claude-sonnet-4-6"
    assert select(TaskClass.COPY_GENERATION, CFG).model == "claude-haiku-4-5"


def test_every_task_class_is_routed():
    for task_class in TaskClass:
        assert isinstance(select(task_class, CFG), Route)


def test_is_local_flag_matches_tier():
    assert select(TaskClass.PLAN, CFG).is_local is True
    assert select(TaskClass.COPY_GENERATION, CFG).is_local is False


def test_escalate_promotes_a_local_route():
    local = select(TaskClass.PLAN, CFG)
    promoted = escalate(local, CFG)
    assert promoted.tier is Tier.FRONTIER
    assert promoted.model == "claude-haiku-4-5"
    assert promoted.task_class is TaskClass.PLAN


def test_escalate_is_a_noop_on_a_frontier_route():
    frontier = select(TaskClass.COPY_GENERATION, CFG)
    assert escalate(frontier, CFG) is frontier


def test_routes_are_immutable():
    route = select(TaskClass.PLAN, CFG)
    with pytest.raises(Exception):
        route.tier = Tier.FRONTIER
