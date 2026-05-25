"""
Data models for Ember — an LLM-agent virtual world.

Ember places an LLM agent inside a 2D burning building. The agent reads
first-person narrative observations and chooses from a small discrete
action space; the world simulates fire spread, smoke, and damage in
response.

Cell encoding (cell_grid):
  0 = floor
  1 = wall
  2 = door_open
  3 = door_closed
  4 = exit
  5 = obstacle (burned-out or structural)

These types descend from `openenv.core.env_server.types` purely so the
package can drop into an OpenEnv-style HTTP wrapper if desired. The
fields below are the harness contract — what the LLM sees, what the
world receives, and what's logged.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Map state (numerical grid snapshot — for UI / visualization consumers)
# ---------------------------------------------------------------------------


class EmberMapState(BaseModel):
    """Full numerical state of the environment at a single timestep.

    Intended for UI rendering and external tooling. Not filtered by visibility —
    the complete ground-truth grid is included so a renderer can draw the full
    map with fog-of-war applied client-side using `visible_cells`.

    Grid layout: flat row-major lists of length grid_w * grid_h.
    Index formula: i = y * grid_w + x
    """

    grid_w: int = Field(..., description="Grid width in cells")
    grid_h: int = Field(..., description="Grid height in cells")
    template_name: str = Field(..., description="Floor plan template in use")
    episode_id: str = Field(..., description="Unique ID for this episode")
    step_count: int = Field(..., description="Current step number")
    max_steps: int = Field(..., description="Maximum steps allowed this episode")

    cell_grid: List[int] = Field(
        ...,
        description="Cell types: 0=floor 1=wall 2=door_open 3=door_closed 4=exit 5=obstacle",
    )
    fire_grid: List[float] = Field(
        ..., description="Fire intensity per cell, 0.0 (none) → 1.0 (fully burning)"
    )
    smoke_grid: List[float] = Field(
        ..., description="Smoke intensity per cell, 0.0 (clear) → 1.0 (dense)"
    )

    agent_x: int = Field(..., description="Agent column (0-indexed from west)")
    agent_y: int = Field(..., description="Agent row (0-indexed from north)")
    agent_alive: bool = Field(..., description="Whether the agent is alive")
    agent_evacuated: bool = Field(..., description="Whether the agent has escaped")
    agent_health: float = Field(..., description="Agent health 0–100")

    visible_cells: List[List[int]] = Field(
        ..., description="[[x, y], ...] cells visible to the agent this step"
    )
    exit_positions: List[List[int]] = Field(
        ..., description="[[x, y], ...] coordinates of all exit cells"
    )
    door_registry: Dict[str, List[int]] = Field(
        ..., description="door_id → [x, y] position for every door"
    )

    # Task-specific items (key_and_door, rescue) — empty for tasks that don't use them
    key_positions: Dict[str, List[int]] = Field(
        default_factory=dict, description="key_id → [x, y] position"
    )
    locked_door_ids: List[str] = Field(
        default_factory=list, description="Doors that require a key to open"
    )
    inventory: List[str] = Field(
        default_factory=list, description="Item IDs the agent is currently carrying"
    )
    npc_positions: Dict[str, List[int]] = Field(
        default_factory=dict, description="npc_id → [x, y] position"
    )
    npc_rescued: List[str] = Field(
        default_factory=list, description="NPC IDs the agent has reached"
    )

    fire_spread_rate: float = Field(..., description="Probability of fire spreading per step")
    wind_dir: str = Field(..., description="Wind direction affecting fire spread")
    humidity: float = Field(..., description="Humidity level (higher = slower spread)")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class EmberAction(Action):
    """Action the agent may take.

    action:     "move" | "door" | "pickup" | "wait" | "look"
    direction:  "north"|"south"|"east"|"west"  — used by move and look
    target_id:  door ID / key ID / npc ID       — used by door, pickup
    door_state: "open" | "close"                — used by door

    Why this action space?
      - Cardinal moves keep the action grid finite and parseable.
      - `door` is a separate verb (not bundled into move) so an LLM can
        explicitly choose to close a door to slow fire spread.
      - `pickup` handles key-style items without inventing a generic
        "interact" verb that the model has to disambiguate.
      - `look <direction>` is a no-cost scan (time still advances) that
        gives the LLM longer-range information without bloating the
        default observation.
      - `wait` is a legal pass — useful for cache-warming reasoning or
        deliberately letting the simulation tick.
    """

    action: str = Field(..., description="Action verb")
    direction: Optional[str] = Field(None, description="Cardinal direction for move/look")
    target_id: Optional[str] = Field(None, description="Door / key / npc ID for door/pickup")
    door_state: Optional[str] = Field(None, description="'open' or 'close' for door action")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class EmberObservation(Observation):
    """What the agent sees after a step.

    `narrative` is the primary text the LLM reads. Structured fields below
    expose the same information programmatically so non-LLM agents (and
    log analysis tools) don't have to re-parse English.

    Inherited from Observation base: reward (float), done (bool), metadata (dict).
    """

    goal: str = Field(default="", description="One-line goal string injected by the active Task")
    narrative: str = Field(default="", description="First-person narrative for the LLM agent")
    agent_evacuated: bool = Field(default=False, description="Whether agent has reached a safe exit")
    location_label: str = Field(default="", description="Current zone/room label")
    smoke_level: str = Field(default="none", description="none|light|moderate|heavy")
    fire_visible: bool = Field(default=False, description="Whether fire is in agent's sight")
    fire_direction: Optional[str] = Field(default=None, description="Direction of nearest fire")
    agent_health: float = Field(default=100.0, description="Agent health 0–100")
    health_status: str = Field(default="Good", description="Critical|Low|Moderate|Good")
    wind_dir: str = Field(default="CALM", description="Current wind direction")
    visible_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Objects in sight: [{id, type, relative_pos, state}]",
    )
    blocked_exit_ids: List[str] = Field(
        default_factory=list,
        description="Exit IDs currently blocked by fire",
    )
    audible_signals: List[str] = Field(
        default_factory=list,
        description="Sounds the agent can hear",
    )
    inventory: List[str] = Field(
        default_factory=list,
        description="Item IDs the agent is currently carrying",
    )
    elapsed_steps: int = Field(default=0, description="Steps elapsed in episode")
    last_action_feedback: str = Field(
        default="", description="Natural-language result of the previous action"
    )
    available_actions_hint: List[str] = Field(
        default_factory=list,
        description="Suggested action call strings legal in the current situation",
    )
    task_complete: bool = Field(
        default=False, description="Whether the active Task's success condition is met"
    )
    task_failed: bool = Field(
        default=False, description="Whether the active Task has hit an unrecoverable failure"
    )
    map_state: Optional[EmberMapState] = Field(
        default=None,
        description="Full numerical grid snapshot for UI rendering and external tooling",
    )


# ---------------------------------------------------------------------------
# State (server-side ground truth — NOT sent to the agent)
# ---------------------------------------------------------------------------


class EmberState(State):
    """Complete server-side ground truth for one episode."""

    episode_id: Optional[str] = None
    step_count: int = 0

    # --- Map ---
    grid_w: int = 16
    grid_h: int = 16
    template_name: str = ""
    cell_grid: List[int] = Field(default_factory=list)
    fire_grid: List[float] = Field(default_factory=list)
    smoke_grid: List[float] = Field(default_factory=list)
    burn_timers: List[int] = Field(default_factory=list)
    exit_positions: List[List[int]] = Field(default_factory=list)
    door_registry: Dict[str, List[int]] = Field(default_factory=dict)
    zone_map: Dict[str, str] = Field(default_factory=dict)

    # --- Agent ---
    agent_x: int = 0
    agent_y: int = 0
    agent_alive: bool = True
    agent_evacuated: bool = False
    agent_health: float = 100.0

    # --- Episode fire config (randomized per episode) ---
    max_steps: int = 150
    fire_seed: int = 0
    fire_sources_count: int = 2
    fire_spread_rate: float = 0.25
    wind_dir: str = "CALM"
    humidity: float = 0.25

    # --- Task state (populated by the active Task) ---
    task_name: str = "escape_basic"
    goal_text: str = ""
    key_positions: Dict[str, List[int]] = Field(default_factory=dict)
    locked_door_ids: List[str] = Field(default_factory=list)
    inventory: List[str] = Field(default_factory=list)
    npc_positions: Dict[str, List[int]] = Field(default_factory=dict)
    npc_rescued: List[str] = Field(default_factory=list)
