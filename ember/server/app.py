"""Minimal FastAPI wrapper for Ember.

Exposes the env over HTTP so the world can be driven from any language
or watched live by a browser dashboard. The LLM agent in
`examples/run_llm_agent.py` does NOT go through HTTP — it imports the
env directly, which keeps the harness simpler. Use this server only
when you want a long-lived process or a streaming UI.

Run:
    pip install ember-world[server]
    uvicorn ember.server.app:app --reload --port 8000

Endpoints:
    GET  /health      → {"status": "ok"}
    POST /reset       → body {"task": "key_and_door", "seed": 42}
    POST /step        → body {"action": "move", "direction": "north"}
    GET  /state       → full server-side state dump (debug)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..env import EmberEnvironment
from ..models import EmberAction
from ..tasks import TASKS

app = FastAPI(title="Ember", description="LLM-agent virtual world")

_DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"

_env: Optional[EmberEnvironment] = None


def _get_env() -> EmberEnvironment:
    global _env
    if _env is None:
        _env = EmberEnvironment(
            max_steps=int(os.environ.get("EMBER_MAX_STEPS", "150")),
            base_seed=int(os.environ.get("EMBER_SEED", "42")),
        )
    return _env


class ResetRequest(BaseModel):
    task: str = Field("escape_basic", description=f"One of: {', '.join(TASKS)}")
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: str = Field(..., description="move | door | pickup | look | wait")
    direction: Optional[str] = None
    target_id: Optional[str] = None
    door_state: Optional[str] = None


@app.get("/")
def index() -> FileResponse:
    """Serve the dashboard. Open this URL in a browser to watch live."""
    return FileResponse(str(_DASHBOARD_HTML))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "ember"}


@app.get("/scene")
def scene() -> Dict[str, Any]:
    """Compact snapshot the dashboard polls every ~250ms.

    Returns enough to draw the world without including the narrative
    (the dashboard renders its own panels for those).
    """
    env = _get_env()
    if env._fire_sim is None:
        raise HTTPException(status_code=409, detail="No active episode. Call /reset first.")
    st = env.state
    visible_set = env._visible_set_for_state(st)
    return {
        "agent": {
            "x": st.agent_x,
            "y": st.agent_y,
            "health": st.agent_health,
            "alive": st.agent_alive,
            "evacuated": st.agent_evacuated,
            "inventory": list(st.inventory),
        },
        "episode": {
            "id": st.episode_id,
            "step": st.step_count,
            "max_steps": st.max_steps,
            "task": st.task_name,
            "goal": st.goal_text,
            "wind_dir": st.wind_dir,
            "feedback": getattr(env, "_last_feedback", ""),
        },
        "grid": {
            "w": st.grid_w,
            "h": st.grid_h,
            "cells": list(st.cell_grid),
            "fire": [round(v, 3) for v in st.fire_grid],
            "smoke": [round(v, 3) for v in st.smoke_grid],
            "visible": [[x, y] for x, y in sorted(visible_set)],
            "exits": list(st.exit_positions),
            "doors": dict(st.door_registry),
            "locked_doors": list(st.locked_door_ids),
            "keys": dict(st.key_positions),
            "npcs": dict(st.npc_positions),
            "npcs_rescued": list(st.npc_rescued),
        },
    }


@app.post("/reset")
def reset(body: ResetRequest) -> Dict[str, Any]:
    if body.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{body.task}'")
    env = _get_env()
    obs = env.reset(task=body.task, seed=body.seed)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
        "metadata": obs.metadata or {},
    }


@app.post("/step")
def step(body: StepRequest) -> Dict[str, Any]:
    env = _get_env()
    if env._fire_sim is None:
        raise HTTPException(status_code=409, detail="No active episode. Call /reset first.")
    obs = env.step(EmberAction(
        action=body.action,
        direction=body.direction,
        target_id=body.target_id,
        door_state=body.door_state,
    ))
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
        "metadata": obs.metadata or {},
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    env = _get_env()
    return env.state.model_dump()
