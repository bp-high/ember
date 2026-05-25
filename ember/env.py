"""
EmberEnvironment — single-agent virtual-world harness for LLM agents.

Orchestrates:
  - Floor plan + fire placement (floor_plan.py, fire_sim.py)
  - Task fixtures and goal text (tasks.py)
  - Narrative observation rendering (narrative.py)
  - Composite reward rubrics (rubrics.py)

Public surface (used by examples/run_llm_agent.py and the HTTP server):

    env = EmberEnvironment()
    obs = env.reset(task="escape_basic", seed=42)
    obs = env.step(EmberAction(action="move", direction="north"))

The `task` parameter replaces the old top-level `difficulty` knob —
difficulty is now a property of the task itself. `seed` is preserved so
runs are reproducible.
"""

from __future__ import annotations

import os
import random
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

from .fire_sim import FireSim, FIRE_BURNING, smoke_level_label, WIND_DIRS
from .floor_plan import generate_episode, template_names
from .models import EmberAction, EmberMapState, EmberObservation, EmberState
from .narrative import (
    build_look_result,
    build_narrative_observation,
    compute_visible_cells,
)
from .rubrics import (
    BFS_INF,
    bfs_exit_dist,
    make_episode_end_rubrics,
    make_per_step_rubrics,
    unblocked_exits,
)
from .tasks import Task, build_task

# Cell type constants
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5

_CARDINAL_DELTA = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}

EXIT_FIRE_THRESHOLD = 0.5

DAMAGE_LIGHT_SMOKE = 0.5
DAMAGE_MODERATE_SMOKE = 2.0
DAMAGE_HEAVY_SMOKE = 5.0
DAMAGE_ON_FIRE = 10.0

# Per-task fire profile. We deliberately keep these tame: the prompt
# rewards goal completion, not survival rate. A demo that fails 80% of
# the time because the building burned down before the LLM finished
# thinking isn't a good demo of LLM reasoning.
_TASK_FIRE_PROFILE: Dict[str, Dict[str, Any]] = {
    "escape_basic": {
        "n_sources_range": (1, 1),
        "p_spread_range": (0.10, 0.20),
        "humidity_range": (0.30, 0.50),
        "wind_choices": ["CALM"],
        "max_steps": 150,
    },
    "key_and_door": {
        "n_sources_range": (1, 1),
        "p_spread_range": (0.05, 0.12),
        "humidity_range": (0.40, 0.60),
        "wind_choices": ["CALM"],
        "max_steps": 180,
    },
    "rescue": {
        "n_sources_range": (1, 2),
        "p_spread_range": (0.10, 0.18),
        "humidity_range": (0.30, 0.50),
        "wind_choices": ["CALM"],
        "max_steps": 180,
    },
}


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def _bfs_first_step_toward_exit(
    sx: int,
    sy: int,
    exits: List[List[int]],
    cell_grid: List[int],
    w: int,
    h: int,
) -> Optional[str]:
    if not exits:
        return None
    exit_set = {(ex[0], ex[1]) for ex in exits}
    if (sx, sy) in exit_set:
        return None
    queue: deque = deque([(sx, sy, None)])
    visited = {(sx, sy)}
    moves = ((0, -1, "north"), (0, 1, "south"), (-1, 0, "west"), (1, 0, "east"))
    while queue:
        cx, cy, first_dir = queue.popleft()
        for dx, dy, dir_name in moves:
            nx, ny = cx + dx, cy + dy
            if not _in_bounds(nx, ny, w, h):
                continue
            if (nx, ny) in visited:
                continue
            ct = cell_grid[_idx(nx, ny, w)]
            if ct in (WALL, OBSTACLE):
                continue
            next_first = dir_name if first_dir is None else first_dir
            if (nx, ny) in exit_set:
                return next_first
            visited.add((nx, ny))
            queue.append((nx, ny, next_first))
    return None


class EmberEnvironment(Environment):
    """LLM-facing virtual world harness.

    A reset starts a new episode for the chosen Task; each step executes
    one action, advances the simulation, and returns a typed
    EmberObservation whose `narrative` field is what the LLM reads.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        max_steps: int = 150,
        base_seed: int = 42,
        full_visibility: Optional[bool] = None,
    ):
        super().__init__()
        self.max_steps = int(os.environ.get("EMBER_MAX_STEPS", max_steps))
        self.base_seed = int(os.environ.get("EMBER_SEED", base_seed))
        if full_visibility is None:
            full_visibility = os.environ.get(
                "EMBER_FULL_VISIBILITY", "1"
            ).strip().lower() not in {"0", "false", "no"}
        self.full_visibility = bool(full_visibility)

        self._state: Optional[EmberState] = None
        self._task: Optional[Task] = None
        self._fire_sim: Optional[FireSim] = None
        self._rng: Optional[random.Random] = None
        self._per_step_rubrics = make_per_step_rubrics()
        self._episode_rubrics = make_episode_end_rubrics()
        self._episode_counter = 0
        self._last_feedback = ""

        # Episode-scoped reward tracking
        self._visited_cells: set = set()
        self._min_exit_dist_reached: int = BFS_INF
        self._rewarded_doors: set = set()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        task: str = "escape_basic",
        seed: Optional[int] = None,
        **kwargs,
    ) -> EmberObservation:
        """Start a new episode.

        Args:
            task: One of "escape_basic" | "key_and_door" | "rescue".
                  Selects fixtures (keys / NPCs / locked doors) and the
                  fire profile (spread rate, max_steps).
            seed: Optional integer for reproducibility. If omitted a
                  deterministic seed derived from `base_seed` is used so
                  successive resets produce different episodes.
        """
        fire_seed = seed if seed is not None else (self.base_seed + self._episode_counter * 37)
        self._episode_counter += 1
        self._rng = random.Random(fire_seed)

        self._task = build_task(task)
        profile = _TASK_FIRE_PROFILE.get(task, _TASK_FIRE_PROFILE["escape_basic"])

        n_sources = self._rng.randint(*profile["n_sources_range"])
        p_spread = round(self._rng.uniform(*profile["p_spread_range"]), 3)
        humidity = round(self._rng.uniform(*profile["humidity_range"]), 3)
        wind_dir = self._rng.choice(profile["wind_choices"])
        max_steps = profile["max_steps"]

        # Pick template by seed (not call order) so (task, seed) is reproducible.
        # Tasks may restrict the template pool (e.g. key_and_door skips
        # open_plan because it has no chokepoint doors).
        templates = template_names()
        whitelist = getattr(self._task, "template_whitelist", None) or []
        pool = [t for t in templates if t in whitelist] if whitelist else templates
        if not pool:
            pool = templates
        template_name = pool[fire_seed % len(pool)]
        floor_plan, _, _, agent_start = generate_episode(
            template_name, npc_count=0, seed=fire_seed
        )

        w, h = floor_plan.w, floor_plan.h
        n_cells = w * h
        fire_grid = [0.0] * n_cells
        smoke_grid = [0.0] * n_cells
        burn_timers = [0] * n_cells

        floor_cells = [
            (x, y) for y in range(h) for x in range(w)
            if floor_plan.cell_grid[_idx(x, y, w)] == FLOOR
        ]
        candidates = [
            pos for pos in floor_cells
            if all(
                _manhattan(pos[0], pos[1], ex[0], ex[1]) >= floor_plan.fire_min_exit_dist
                for ex in floor_plan.exit_positions
            )
            and _manhattan(pos[0], pos[1], agent_start[0], agent_start[1]) >= 4
        ]
        if not candidates:
            candidates = [
                pos for pos in floor_cells
                if pos != agent_start
                and pos not in [(e[0], e[1]) for e in floor_plan.exit_positions]
            ]
        n_sources = min(n_sources, len(candidates))
        self._rng.shuffle(candidates)
        for fx, fy in candidates[:n_sources]:
            fire_grid[_idx(fx, fy, w)] = 0.1

        door_registry: Dict[str, List[int]] = {}
        for j, (dx, dy) in enumerate(floor_plan.door_positions):
            door_registry[f"door_{j + 1}"] = [dx, dy]

        self._state = EmberState.model_construct(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            grid_w=w,
            grid_h=h,
            template_name=template_name,
            cell_grid=floor_plan.cell_grid,
            fire_grid=fire_grid,
            smoke_grid=smoke_grid,
            burn_timers=burn_timers,
            exit_positions=[[ex[0], ex[1]] for ex in floor_plan.exit_positions],
            door_registry=door_registry,
            zone_map=floor_plan.zone_map,
            agent_x=agent_start[0],
            agent_y=agent_start[1],
            agent_alive=True,
            agent_evacuated=False,
            agent_health=100.0,
            max_steps=max_steps,
            fire_seed=fire_seed,
            fire_sources_count=n_sources,
            fire_spread_rate=p_spread,
            wind_dir=wind_dir,
            humidity=humidity,
            task_name=task,
            goal_text="",
            key_positions={},
            locked_door_ids=[],
            inventory=[],
            npc_positions={},
            npc_rescued=[],
        )

        # Let the task install its fixtures before we render the first obs.
        self._task.setup(self._state, self._rng)
        self._state.goal_text = self._task.goal_text(self._state)

        self._last_feedback = "Episode started. Read the goal and assess your surroundings."

        self._visited_cells = {(self._state.agent_x, self._state.agent_y)}
        self._min_exit_dist_reached = BFS_INF
        self._rewarded_doors = set()

        self._fire_sim = FireSim(
            w=w, h=h, rng=self._rng,
            p_spread=p_spread,
            wind_dir=wind_dir,
            humidity=humidity,
            fuel_map=floor_plan.fuel_map,
            ventilation_map=floor_plan.ventilation_map,
        )

        return self._build_observation(done=False, reward=0.0)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: EmberAction, **kwargs) -> EmberObservation:
        st = self._state
        if st is None:
            raise RuntimeError("Call reset() before step().")

        prev_agent_x = st.agent_x
        prev_agent_y = st.agent_y

        feedback = self._execute_action(action, st)
        self._last_feedback = feedback

        # Walking onto an NPC's cell rescues them.
        for npc_id, (nx, ny) in list(st.npc_positions.items()):
            if (st.agent_x, st.agent_y) == (nx, ny) and npc_id not in st.npc_rescued:
                st.npc_rescued.append(npc_id)
                self._last_feedback = (
                    f"You reach {npc_id} and help them up. They follow you out."
                )

        # Self-evacuation check.
        if st.agent_alive and not st.agent_evacuated:
            agent_cell = st.cell_grid[_idx(st.agent_x, st.agent_y, st.grid_w)]
            if agent_cell == EXIT:
                fire_at_exit = st.fire_grid[_idx(st.agent_x, st.agent_y, st.grid_w)]
                if fire_at_exit < EXIT_FIRE_THRESHOLD:
                    st.agent_evacuated = True
                    self._last_feedback = "You step through the exit and escape the building!"
                else:
                    self._last_feedback = (
                        "The exit is engulfed in flames — you can't get through!"
                    )

        # Advance fire sim and apply health damage.
        self._fire_sim.step(st.cell_grid, st.fire_grid, st.smoke_grid, st.burn_timers)

        health_damage = 0.0
        if st.agent_alive and not st.agent_evacuated:
            ai = _idx(st.agent_x, st.agent_y, st.grid_w)
            smoke_label = smoke_level_label(st.smoke_grid[ai])
            if smoke_label == "heavy":
                health_damage += DAMAGE_HEAVY_SMOKE
            elif smoke_label == "moderate":
                health_damage += DAMAGE_MODERATE_SMOKE
            elif smoke_label == "light":
                health_damage += DAMAGE_LIGHT_SMOKE
            if st.fire_grid[ai] >= FIRE_BURNING:
                health_damage += DAMAGE_ON_FIRE
            st.agent_health = max(0.0, st.agent_health - health_damage)
            if st.agent_health <= 0:
                st.agent_alive = False
                self._last_feedback = "You collapse — overwhelmed by fire and smoke."

        st.step_count += 1

        is_new_cell = (st.agent_x, st.agent_y) not in self._visited_cells
        self._visited_cells.add((st.agent_x, st.agent_y))

        exits_reachable = unblocked_exits(st.exit_positions, st.fire_grid, st.grid_w)
        exits = exits_reachable if exits_reachable else st.exit_positions
        cur_dist = bfs_exit_dist(
            st.agent_x, st.agent_y, exits, st.cell_grid, st.grid_w, st.grid_h
        )
        if cur_dist < self._min_exit_dist_reached:
            self._min_exit_dist_reached = cur_dist

        # Refresh goal text (it includes inventory/sub-goal progress).
        st.goal_text = self._task.goal_text(st) if self._task else ""

        task_result = self._task.evaluate(st) if self._task else None
        # The harness treats task completion as `done=True` so the LLM
        # loop exits when the *named goal* is met, not only when the
        # agent physically leaves the building.
        done = (
            (task_result.complete if task_result else False)
            or (task_result.failed if task_result else False)
            or self._terminal(st)
        )

        reward = self._compute_reward(
            action=action.action,
            target_id=action.target_id,
            door_state=action.door_state,
            prev_agent_x=prev_agent_x,
            prev_agent_y=prev_agent_y,
            health_damage=health_damage,
            is_new_cell=is_new_cell,
            st=st,
            done=done,
        )

        return self._build_observation(done=done, reward=reward, task_result=task_result)

    @property
    def state(self) -> EmberState:
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state.")
        return self._state

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: EmberAction, st: EmberState) -> str:
        act = (action.action or "").strip().lower()
        if act == "move":
            return self._action_move(action, st)
        if act == "door":
            return self._action_door(action, st)
        if act == "pickup":
            return self._action_pickup(action, st)
        if act == "look":
            return self._action_look(action, st)
        if act == "wait":
            return "You wait and listen to the building."
        return f"Unknown action '{act}'. Nothing happened."

    def _action_move(self, action: EmberAction, st: EmberState) -> str:
        direction = (action.direction or "").lower()
        delta = _CARDINAL_DELTA.get(direction)
        if delta is None:
            return f"Invalid direction '{direction}'."

        nx, ny = st.agent_x + delta[0], st.agent_y + delta[1]
        if not _in_bounds(nx, ny, st.grid_w, st.grid_h):
            return "You walk into the outer wall — blocked."
        ct = st.cell_grid[_idx(nx, ny, st.grid_w)]
        if ct in (WALL, OBSTACLE):
            return "Blocked by wall or debris."
        if ct == DOOR_CLOSED:
            return f"The door to the {direction} is closed. Open it first."

        st.agent_x = nx
        st.agent_y = ny

        suffix = ""
        smoke = st.smoke_grid[_idx(nx, ny, st.grid_w)]
        fire = st.fire_grid[_idx(nx, ny, st.grid_w)]
        if smoke > 0.5:
            suffix = " The smoke is thick here."
        if fire > 0.1:
            suffix += " You feel intense heat."
        # Step-on-key shortcut: if the agent walks onto a key cell, surface
        # it in the feedback but require an explicit pickup to claim it.
        for kid, (kx, ky) in st.key_positions.items():
            if (nx, ny) == (kx, ky) and kid not in st.inventory:
                suffix += f" A {kid} glints on the floor here — try pickup(target_id='{kid}')."
        return f"You move {direction}.{suffix}"

    def _action_look(self, action: EmberAction, st: EmberState) -> str:
        direction = (action.direction or "").strip().lower()
        if not direction:
            return "look requires a direction: north, south, east, or west."
        return build_look_result(
            direction=direction,
            agent_x=st.agent_x,
            agent_y=st.agent_y,
            cell_grid=st.cell_grid,
            fire_grid=st.fire_grid,
            smoke_grid=st.smoke_grid,
            zone_map=st.zone_map,
            door_registry=st.door_registry,
            w=st.grid_w,
            h=st.grid_h,
        )

    def _action_door(self, action: EmberAction, st: EmberState) -> str:
        target_id = action.target_id
        door_state = (action.door_state or "").strip().lower()
        if not target_id:
            return "door requires a target_id (door ID)."
        if door_state not in ("open", "close"):
            return "door requires door_state='open' or door_state='close'."
        if target_id not in st.door_registry:
            return f"Door '{target_id}' not found."

        dx, dy = st.door_registry[target_id]
        if _manhattan(st.agent_x, st.agent_y, dx, dy) > 2:
            return f"Door '{target_id}' is too far away."
        ct = st.cell_grid[_idx(dx, dy, st.grid_w)]
        if ct not in (DOOR_OPEN, DOOR_CLOSED):
            return f"'{target_id}' is not a door."

        if door_state == "close":
            if ct == DOOR_CLOSED:
                return f"Door '{target_id}' is already closed."
            st.cell_grid[_idx(dx, dy, st.grid_w)] = DOOR_CLOSED
            return f"You close door '{target_id}'. It may slow the fire."

        # door_state == "open"
        if ct == DOOR_OPEN:
            return f"Door '{target_id}' is already open."
        if target_id in st.locked_door_ids and "key_1" not in st.inventory:
            return (
                f"Door '{target_id}' is locked. You need a key to open it."
            )
        st.cell_grid[_idx(dx, dy, st.grid_w)] = DOOR_OPEN
        if target_id in st.locked_door_ids:
            return f"You unlock and open door '{target_id}' with key_1."
        return f"You open door '{target_id}'."

    def _action_pickup(self, action: EmberAction, st: EmberState) -> str:
        target_id = action.target_id
        if not target_id:
            return "pickup requires a target_id (item ID)."
        if target_id not in st.key_positions:
            return f"No item '{target_id}' in this world."
        if target_id in st.inventory:
            return f"You already have {target_id}."
        kx, ky = st.key_positions[target_id]
        if (st.agent_x, st.agent_y) != (kx, ky):
            return f"{target_id} is not at your feet — stand on it first."
        st.inventory.append(target_id)
        return f"You pick up {target_id}. It's now in your inventory."

    # ------------------------------------------------------------------
    # Terminal / done
    # ------------------------------------------------------------------

    def _terminal(self, st: EmberState) -> bool:
        if not st.agent_alive:
            return True
        if st.agent_evacuated:
            return True
        if st.step_count >= st.max_steps:
            return True
        return False

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: str,
        target_id: Optional[str],
        door_state: Optional[str],
        prev_agent_x: int,
        prev_agent_y: int,
        health_damage: float,
        is_new_cell: bool,
        st: EmberState,
        done: bool,
    ) -> float:
        kwargs = dict(
            action=action,
            target_id=target_id,
            door_state=door_state,
            prev_agent_x=prev_agent_x,
            prev_agent_y=prev_agent_y,
            agent_x=st.agent_x,
            agent_y=st.agent_y,
            exit_positions=st.exit_positions,
            cell_grid=st.cell_grid,
            fire_grid=st.fire_grid,
            smoke_grid=st.smoke_grid,
            w=st.grid_w,
            h=st.grid_h,
            door_registry=st.door_registry,
            done=done,
            agent_evacuated=st.agent_evacuated,
            agent_alive=st.agent_alive,
            agent_health=st.agent_health,
            health_damage=health_damage,
            remaining_steps=max(0, st.max_steps - st.step_count),
            is_new_cell=is_new_cell,
            min_exit_dist_reached=self._min_exit_dist_reached,
            rewarded_doors=self._rewarded_doors,
            reachable_exit_count=len(
                unblocked_exits(st.exit_positions, st.fire_grid, st.grid_w)
            ),
        )
        total = 0.0
        for rubric in self._per_step_rubrics:
            total += rubric.score(**kwargs)
        if done:
            for rubric in self._episode_rubrics:
                total += rubric.score(**kwargs)
        return round(total, 4)

    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------

    def _visible_set_for_state(self, st: EmberState) -> set:
        if not st.agent_alive or st.agent_evacuated:
            return set()
        if self.full_visibility:
            return {(x, y) for y in range(st.grid_h) for x in range(st.grid_w)}
        return compute_visible_cells(
            st.agent_x, st.agent_y, st.cell_grid, st.smoke_grid, st.grid_w, st.grid_h,
        )

    def _build_observation(
        self,
        done: bool,
        reward: float,
        task_result=None,
    ) -> EmberObservation:
        st = self._state
        assert st is not None

        visible_set = self._visible_set_for_state(st)

        obs_data = build_narrative_observation(
            step_count=st.step_count,
            agent_x=st.agent_x,
            agent_y=st.agent_y,
            agent_alive=st.agent_alive,
            agent_evacuated=st.agent_evacuated,
            agent_health=st.agent_health,
            cell_grid=st.cell_grid,
            fire_grid=st.fire_grid,
            smoke_grid=st.smoke_grid,
            exit_positions=st.exit_positions,
            door_registry=st.door_registry,
            zone_map=st.zone_map,
            last_action_feedback=self._last_feedback,
            wind_dir=st.wind_dir,
            w=st.grid_w,
            h=st.grid_h,
            visible_override=visible_set,
        )

        # Surface task-specific objects in visible_objects + action hints.
        self._inject_task_objects(obs_data, st, visible_set)

        # Build the final narrative with the goal line on top.
        narrative_body = obs_data.get("narrative", "")
        narrative_with_goal = self._compose_narrative_with_goal(st, narrative_body)

        return EmberObservation(
            goal=st.goal_text,
            narrative=narrative_with_goal,
            agent_evacuated=st.agent_evacuated,
            location_label=obs_data.get("location_label", ""),
            smoke_level=obs_data.get("smoke_level", "none"),
            fire_visible=obs_data.get("fire_visible", False),
            fire_direction=obs_data.get("fire_direction"),
            agent_health=st.agent_health,
            health_status=obs_data.get("health_status", "Good"),
            wind_dir=st.wind_dir,
            visible_objects=obs_data.get("visible_objects", []),
            blocked_exit_ids=obs_data.get("blocked_exit_ids", []),
            audible_signals=obs_data.get("audible_signals", []),
            inventory=list(st.inventory),
            elapsed_steps=st.step_count,
            last_action_feedback=self._last_feedback,
            available_actions_hint=obs_data.get("available_actions_hint", []),
            task_complete=bool(task_result.complete) if task_result else False,
            task_failed=bool(task_result.failed) if task_result else False,
            done=done,
            reward=reward,
            metadata=self._build_metadata(st, visible_set),
            map_state=self._build_map_state(st, visible_set),
        )

    def _compose_narrative_with_goal(self, st: EmberState, body: str) -> str:
        """Inject the active task's goal as the first line of the narrative."""
        if not st.goal_text:
            return body
        # We separate goal from atmosphere with a blank line so the LLM
        # can easily section them in its own reasoning.
        return f"{st.goal_text}\n\n{body}"

    def _inject_task_objects(
        self,
        obs_data: Dict[str, Any],
        st: EmberState,
        visible_set: set,
    ) -> None:
        """Add keys / NPCs to `visible_objects` and matching action hints."""
        if not visible_set:
            return
        objs: List[Dict[str, Any]] = list(obs_data.get("visible_objects", []))
        hints: List[str] = list(obs_data.get("available_actions_hint", []))

        for kid, (kx, ky) in st.key_positions.items():
            if kid in st.inventory:
                continue
            if (kx, ky) not in visible_set:
                continue
            rel = self._relative_pos_str(st.agent_x, st.agent_y, kx, ky)
            objs.append({"id": kid, "type": "key", "relative_pos": rel, "state": "on floor"})
            if (st.agent_x, st.agent_y) == (kx, ky):
                hints.insert(0, f"pickup(target_id='{kid}')")

        for nid, (nx, ny) in st.npc_positions.items():
            if nid in st.npc_rescued:
                continue
            if (nx, ny) not in visible_set:
                continue
            rel = self._relative_pos_str(st.agent_x, st.agent_y, nx, ny)
            objs.append({"id": nid, "type": "npc", "relative_pos": rel, "state": "trapped"})

        # Annotate locked doors so the model knows they need a key.
        for obj in objs:
            if obj.get("type") == "door" and obj.get("id") in st.locked_door_ids:
                if "locked" not in obj.get("state", ""):
                    obj["state"] = f"locked, {obj['state']}"

        obs_data["visible_objects"] = objs
        obs_data["available_actions_hint"] = hints

    @staticmethod
    def _relative_pos_str(ax: int, ay: int, tx: int, ty: int) -> str:
        dx, dy = tx - ax, ty - ay
        dist = abs(dx) + abs(dy)
        if dist == 0:
            return "here"
        if abs(dx) >= abs(dy):
            return f"{dist}m {'east' if dx > 0 else 'west'}"
        return f"{dist}m {'south' if dy > 0 else 'north'}"

    def _build_metadata(self, st: EmberState, visible_set: set) -> Dict[str, Any]:
        reachable = unblocked_exits(st.exit_positions, st.fire_grid, st.grid_w)
        exits_for_dist = reachable if reachable else st.exit_positions
        nearest_dist = bfs_exit_dist(
            st.agent_x, st.agent_y, exits_for_dist, st.cell_grid, st.grid_w, st.grid_h,
        )
        nearest_dir = _bfs_first_step_toward_exit(
            st.agent_x, st.agent_y, exits_for_dist, st.cell_grid, st.grid_w, st.grid_h,
        )
        return {
            "task": st.task_name,
            "agent_health": st.agent_health,
            "step": st.step_count,
            "wind_dir": st.wind_dir,
            "fire_spread_rate": st.fire_spread_rate,
            "fire_sources": st.fire_sources_count,
            "humidity": st.humidity,
            "nearest_exit_distance": nearest_dist,
            "nearest_exit_direction": nearest_dir,
            "reachable_exit_count": len(reachable),
            "visible_cell_count": len(visible_set),
            "inventory": list(st.inventory),
            "npc_rescued": list(st.npc_rescued),
        }

    def _build_map_state(self, st: EmberState, visible_set: set) -> EmberMapState:
        if st.agent_alive and not st.agent_evacuated:
            visible_cells = [[x, y] for x, y in sorted(visible_set)]
        else:
            visible_cells = []
        return EmberMapState(
            grid_w=st.grid_w,
            grid_h=st.grid_h,
            template_name=st.template_name,
            episode_id=st.episode_id or "",
            step_count=st.step_count,
            max_steps=st.max_steps,
            cell_grid=list(st.cell_grid),
            fire_grid=list(st.fire_grid),
            smoke_grid=list(st.smoke_grid),
            agent_x=st.agent_x,
            agent_y=st.agent_y,
            agent_alive=st.agent_alive,
            agent_evacuated=st.agent_evacuated,
            agent_health=st.agent_health,
            visible_cells=visible_cells,
            exit_positions=list(st.exit_positions),
            door_registry=dict(st.door_registry),
            key_positions=dict(st.key_positions),
            locked_door_ids=list(st.locked_door_ids),
            inventory=list(st.inventory),
            npc_positions=dict(st.npc_positions),
            npc_rescued=list(st.npc_rescued),
            fire_spread_rate=st.fire_spread_rate,
            wind_dir=st.wind_dir,
            humidity=st.humidity,
        )
