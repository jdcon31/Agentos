"""Agent registry.

Maps the agent names a plan can reference to their implementations. In the full
system this is populated with the specialist agents (content generation, remix,
vision analysis, publishing). The excerpt ships it empty so the orchestrator's
unknown-agent path is the one that runs.
"""

REGISTRY: dict = {}
