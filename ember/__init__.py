"""Ember — a virtual world for LLM agents.

Quick start:

    from ember import EmberEnvironment, EmberAction

    env = EmberEnvironment()
    obs = env.reset(task="key_and_door", seed=42)
    print(obs.narrative)
    obs = env.step(EmberAction(action="move", direction="north"))

See `examples/run_llm_agent.py` for the LLM-driven loop.
"""

from .env import EmberEnvironment
from .models import EmberAction, EmberMapState, EmberObservation, EmberState
from .tasks import TASKS, Task, build_task

__all__ = [
    "EmberAction",
    "EmberEnvironment",
    "EmberMapState",
    "EmberObservation",
    "EmberState",
    "TASKS",
    "Task",
    "build_task",
]
