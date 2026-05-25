"""
Tasks — named, verifiable goals layered on top of the Ember world.

Each Task is a thin object that the environment consults at three points:

  1. setup(state, rng)            — runs at reset() after the floor plan and
                                    fire are placed; mutates state to add
                                    keys / locked doors / NPC markers and
                                    decides any task-specific spawn rules.

  2. goal_text(state) -> str      — returns a single line shown at the top of
                                    every observation so the LLM never loses
                                    sight of what it's trying to do.

  3. is_complete(state) -> bool   — true the moment the success condition is
                                    met. The environment also tracks a
                                    separate `task_failed` flag for tasks
                                    that have a soft failure mode distinct
                                    from agent death.

Tasks deliberately do NOT change the action space. They reuse `move`,
`door`, `pickup`, `look`, `wait` and only modulate which side-effects
those actions have on success/failure. This keeps the harness the LLM
sees identical across tasks — only the goal string and the world
fixtures change.

The three implemented tasks span three different shapes of problem:

  escape_basic  — pure navigation under threat
                  (reach any unblocked exit before HP runs out)
  key_and_door  — sub-goal sequencing under threat
                  (find a key, then unlock a door, then escape)
  rescue        — point-of-interest detour under threat
                  (reach an NPC, then escape, both alive)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import EmberState

# Cell type constants — kept in sync with floor_plan.py / fire_sim.py.
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    complete: bool
    failed: bool = False
    detail: str = ""


class Task:
    """Base class — override `setup`, `goal_text`, `evaluate`."""

    name: str = "task"
    # Difficulty is a property of the task — the env no longer takes a
    # top-level difficulty knob. Tasks pick a profile that fits their
    # required pacing (escape needs more steps; key_and_door needs slower
    # fire so the sub-goal sequence is reachable).
    difficulty: str = "easy"
    # Subset of template names the task wants the env to choose from.
    # Empty list means "any template is fine." We use this to skip the
    # `open_plan` map for `key_and_door` — that floor has no chokepoint
    # doors, so locking one is a no-op.
    template_whitelist: List[str] = []

    def setup(self, state: EmberState, rng: random.Random) -> None:
        """Mutate `state` to add task fixtures. Called after fire placement."""

    def goal_text(self, state: EmberState) -> str:
        return ""

    def evaluate(self, state: EmberState) -> TaskResult:
        return TaskResult(complete=False)


# ---------------------------------------------------------------------------
# escape_basic — reach any unblocked exit before HP hits 0
# ---------------------------------------------------------------------------


class EscapeBasic(Task):
    name = "escape_basic"
    difficulty = "easy"

    def goal_text(self, state: EmberState) -> str:
        exits = ", ".join(f"exit_{ex[0]}_{ex[1]}" for ex in state.exit_positions)
        return f"GOAL [escape_basic]: Reach any unblocked exit ({exits}) before your HP runs out."

    def evaluate(self, state: EmberState) -> TaskResult:
        if state.agent_evacuated:
            return TaskResult(complete=True, detail="agent reached exit")
        if not state.agent_alive:
            return TaskResult(complete=False, failed=True, detail="agent died")
        return TaskResult(complete=False)


# ---------------------------------------------------------------------------
# key_and_door — pickup a key, then exit through a locked door
# ---------------------------------------------------------------------------


class KeyAndDoor(Task):
    """Pickup a key, then escape through a door that requires it.

    Setup logic:
      - Find a door that, when closed, makes ALL exits unreachable (a
        global chokepoint); fall back to a partial chokepoint, then to
        any door.
      - Place a key on a floor cell, biased to be roughly between the
        agent and the locked door.

    `open_plan` is excluded — it has no chokepoint doors.
    """

    name = "key_and_door"
    difficulty = "easy"
    template_whitelist = ["small_office", "t_corridor"]

    def setup(self, state: EmberState, rng: random.Random) -> None:
        w, h = state.grid_w, state.grid_h

        # Strategy: pick a door whose closure isolates a room with NO
        # exits inside. Move the agent into that room, lock the door,
        # and drop the key in the same room. Result: the lock is the
        # only way out, so the puzzle is non-trivially required.
        choice = self._find_isolating_door(state)
        if choice is None:
            # Map shape doesn't admit a clean isolation; fall back to a
            # plain lock somewhere on the map (key still needed but the
            # agent may find an alternate route — degrades gracefully).
            self._fallback_setup(state, rng)
            return

        locked_id, room_cells = choice
        state.locked_door_ids.append(locked_id)

        dx, dy = state.door_registry[locked_id]
        state.cell_grid[_idx(dx, dy, w)] = DOOR_CLOSED

        # Move the agent into the isolated room.
        room_cells_list = sorted(room_cells)
        spawn = rng.choice(room_cells_list)
        state.agent_x, state.agent_y = spawn

        # Place key in the same room, ideally not on the spawn cell.
        candidates = [c for c in room_cells_list if c != spawn and state.fire_grid[_idx(c[0], c[1], w)] < 0.1]
        if not candidates:
            candidates = room_cells_list
        kx, ky = rng.choice(candidates)
        state.key_positions["key_1"] = [kx, ky]

    def goal_text(self, state: EmberState) -> str:
        # Show the goal as a small sequence so the LLM can self-monitor
        # progress against the sub-steps.
        steps: List[str] = []
        if state.key_positions and "key_1" not in state.inventory:
            kx, ky = state.key_positions["key_1"]
            steps.append(f"(1) pick up key_1 at [{kx},{ky}]")
        elif "key_1" in state.inventory:
            steps.append("(1) ✓ key_1 in inventory")

        if state.locked_door_ids:
            did = state.locked_door_ids[0]
            dx, dy = state.door_registry[did]
            verb = "open" if "key_1" in state.inventory else "(need key first)"
            steps.append(f"(2) {verb} {did} at [{dx},{dy}]")

        steps.append("(3) reach any exit")
        return "GOAL [key_and_door]: " + "  →  ".join(steps)

    def evaluate(self, state: EmberState) -> TaskResult:
        if state.agent_evacuated:
            return TaskResult(complete=True, detail="agent escaped after unlocking door")
        if not state.agent_alive:
            return TaskResult(complete=False, failed=True, detail="agent died")
        return TaskResult(complete=False)

    # ---- helpers ----

    def _bfs_reachable_exits(
        self, state: EmberState, blocked_door: Optional[str] = None
    ) -> int:
        """Count exits reachable from the agent's position.

        If `blocked_door` is given, treat that door as impassable (a wall)
        for the purpose of the BFS, which is how we test chokepoints.
        """
        from collections import deque

        w, h = state.grid_w, state.grid_h
        blocked_pos: Optional[Tuple[int, int]] = None
        if blocked_door and blocked_door in state.door_registry:
            dx, dy = state.door_registry[blocked_door]
            blocked_pos = (dx, dy)

        exits = {(e[0], e[1]) for e in state.exit_positions}
        start = (state.agent_x, state.agent_y)
        seen = {start}
        queue = deque([start])
        found: set = set()
        while queue:
            x, y = queue.popleft()
            if (x, y) in exits:
                found.add((x, y))
                # Don't return early — we want to count distinct exits.
            for dxd, dyd in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dxd, y + dyd
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) in seen:
                    continue
                if blocked_pos == (nx, ny):
                    continue
                ct = state.cell_grid[_idx(nx, ny, w)]
                if ct in (WALL, OBSTACLE):
                    continue
                seen.add((nx, ny))
                queue.append((nx, ny))
        return len(found)

    def _find_isolating_door(
        self, state: EmberState
    ) -> Optional[Tuple[str, set]]:
        """Find a door whose closure separates the map into a region of
        floor cells containing NO exit. Returns (door_id, region_cells).

        Walks every door and computes the connected component of one of
        its neighbours with the door treated as impassable. If that
        component is exit-free and non-trivial (>=2 cells), the door
        qualifies. The first qualifying door is returned — combined with
        a per-seed RNG this still varies across episodes.
        """
        from collections import deque
        w, h = state.grid_w, state.grid_h
        exits = {(e[0], e[1]) for e in state.exit_positions}

        for door_id, (dx, dy) in state.door_registry.items():
            # For each cell adjacent to the door, BFS treating the door
            # itself as a wall and see what region we land in.
            for ax, ay in ((dx, dy - 1), (dx, dy + 1), (dx - 1, dy), (dx + 1, dy)):
                if not (0 <= ax < w and 0 <= ay < h):
                    continue
                ct = state.cell_grid[_idx(ax, ay, w)]
                if ct in (WALL, OBSTACLE):
                    continue
                region: set = {(ax, ay)}
                queue: deque = deque([(ax, ay)])
                while queue:
                    cx, cy = queue.popleft()
                    for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                        nx, ny = cx + ddx, cy + ddy
                        if not (0 <= nx < w and 0 <= ny < h):
                            continue
                        if (nx, ny) == (dx, dy):
                            continue  # door is locked-closed
                        if (nx, ny) in region:
                            continue
                        nct = state.cell_grid[_idx(nx, ny, w)]
                        if nct in (WALL, OBSTACLE):
                            continue
                        region.add((nx, ny))
                        queue.append((nx, ny))
                if not (region & exits) and len(region) >= 2:
                    return door_id, region
        return None

    def _fallback_setup(self, state: EmberState, rng: random.Random) -> None:
        """Place a key + locked door even when no clean isolation exists.

        Used so `key_and_door` always has something to pick up; the
        agent may still find an alternate path on highly-connected
        maps, which we accept and surface in logs.
        """
        if not state.door_registry:
            return
        locked_id = rng.choice(list(state.door_registry))
        state.locked_door_ids.append(locked_id)
        dx, dy = state.door_registry[locked_id]
        state.cell_grid[_idx(dx, dy, state.grid_w)] = DOOR_CLOSED
        key_pos = self._place_key(state, rng)
        if key_pos is not None:
            state.key_positions["key_1"] = list(key_pos)

    def _find_chokepoint_doors(self, state: EmberState) -> List[str]:
        """Doors whose closure makes ALL exits unreachable (true chokepoints).

        We prefer global chokepoints because they make the lock matter —
        the agent can't route around. If none exist (the building has
        redundant connectivity), the caller falls back to partial
        chokepoints, which reduce the number of reachable exits.
        """
        global_chokes: List[str] = []
        for door_id in state.door_registry:
            if self._bfs_reachable_exits(state, blocked_door=door_id) == 0:
                global_chokes.append(door_id)
        if global_chokes:
            return global_chokes

        # No global chokepoint; settle for partial.
        baseline = self._bfs_reachable_exits(state, blocked_door=None)
        partial: List[str] = []
        for door_id in state.door_registry:
            if self._bfs_reachable_exits(state, blocked_door=door_id) < baseline:
                partial.append(door_id)
        return partial

    def _pick_blocking_door(
        self,
        state: EmberState,
        candidates: List[str],
        rng: random.Random,
    ) -> str:
        """Pick a candidate door — prefer ones near the agent's escape route."""
        # Score each candidate by Manhattan distance from the agent (closer
        # is better — the puzzle should be in front of the agent, not on
        # the opposite side of the building).
        scored: List[Tuple[int, str]] = []
        for did in candidates:
            dx, dy = state.door_registry[did]
            scored.append((_manhattan(state.agent_x, state.agent_y, dx, dy), did))
        scored.sort()
        # Pick uniformly from the closest third — keeps variety without
        # picking a door 12 cells away.
        cutoff = max(1, len(scored) // 3)
        return rng.choice(scored[:cutoff])[1]

    def _place_key(self, state: EmberState, rng: random.Random) -> Optional[Tuple[int, int]]:
        w, h = state.grid_w, state.grid_h
        floor_cells = [
            (x, y)
            for y in range(h)
            for x in range(w)
            if state.cell_grid[_idx(x, y, w)] in (FLOOR, DOOR_OPEN)
            and state.fire_grid[_idx(x, y, w)] < 0.1
            and (x, y) != (state.agent_x, state.agent_y)
            and (x, y) not in {(e[0], e[1]) for e in state.exit_positions}
        ]
        if not floor_cells:
            return None

        # Prefer cells closer to the agent than the locked door is — keeps
        # the puzzle solvable on small maps.
        locked = state.locked_door_ids[0]
        lx, ly = state.door_registry[locked]
        agent_to_door = _manhattan(state.agent_x, state.agent_y, lx, ly)
        near = [
            (x, y) for (x, y) in floor_cells
            if _manhattan(state.agent_x, state.agent_y, x, y) <= max(3, agent_to_door)
        ]
        pool = near or floor_cells
        return rng.choice(pool)


# ---------------------------------------------------------------------------
# rescue — reach an NPC marker, then escape
# ---------------------------------------------------------------------------


class Rescue(Task):
    """Reach a downed NPC, then make it out together.

    Once the agent steps onto the NPC's cell the NPC is marked rescued
    (added to `state.npc_rescued`). After that, the success condition is
    identical to escape_basic.

    For storytelling, the NPC is described as "trapped" in the narrative;
    mechanically it's just a fixed (x, y) on the grid.
    """

    name = "rescue"
    difficulty = "easy"

    def setup(self, state: EmberState, rng: random.Random) -> None:
        npc_pos = self._place_npc(state, rng)
        if npc_pos is None:
            return
        nx, ny = npc_pos
        state.npc_positions["npc_1"] = [nx, ny]

    def goal_text(self, state: EmberState) -> str:
        rescued = "npc_1" in state.npc_rescued
        if not state.npc_positions:
            return "GOAL [rescue → escape_basic fallback]: Reach any exit."
        nx, ny = state.npc_positions["npc_1"]
        if not rescued:
            return (
                f"GOAL [rescue]: (1) reach npc_1 at [{nx},{ny}]  →  "
                f"(2) escape through any exit. Both must happen before HP runs out."
            )
        return "GOAL [rescue]: ✓ npc_1 rescued  →  now reach any exit."

    def evaluate(self, state: EmberState) -> TaskResult:
        if not state.agent_alive:
            return TaskResult(complete=False, failed=True, detail="agent died")
        if state.agent_evacuated:
            if "npc_1" in state.npc_rescued or not state.npc_positions:
                return TaskResult(complete=True, detail="escaped with NPC")
            return TaskResult(
                complete=False, failed=True, detail="escaped without rescuing NPC"
            )
        return TaskResult(complete=False)

    def _place_npc(self, state: EmberState, rng: random.Random) -> Optional[Tuple[int, int]]:
        w, h = state.grid_w, state.grid_h
        floor_cells = [
            (x, y)
            for y in range(h)
            for x in range(w)
            if state.cell_grid[_idx(x, y, w)] in (FLOOR, DOOR_OPEN)
            and state.fire_grid[_idx(x, y, w)] < 0.1
            and (x, y) != (state.agent_x, state.agent_y)
            and (x, y) not in {(e[0], e[1]) for e in state.exit_positions}
        ]
        if not floor_cells:
            return None
        # Put the NPC at least a few cells away from the agent so there's
        # actually a detour to make.
        far = [
            (x, y) for (x, y) in floor_cells
            if _manhattan(state.agent_x, state.agent_y, x, y) >= 4
        ]
        return rng.choice(far or floor_cells)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TASKS: Dict[str, type] = {
    EscapeBasic.name: EscapeBasic,
    KeyAndDoor.name: KeyAndDoor,
    Rescue.name: Rescue,
}


def build_task(name: str) -> Task:
    """Look up a task by name and instantiate it. Raises on unknown name."""
    if name not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{name}'. Known tasks: {known}")
    return TASKS[name]()
