"""
Scripted reference agent — a BFS-planner used to:

  1. Validate that each Task is solvable on a given seed (smoke test).
  2. Produce a reference JSONL trace for the README without needing an
     LLM API key.

This is NOT what the prompt asks for — the LLM harness in
`run_llm_agent.py` is. This script just shows the harness machinery
(reset → observe → act → log) with a known-good policy.

Usage:
    python examples/run_scripted_agent.py \
        --task key_and_door --seed 11 --log examples/traces/key_and_door.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ember import EmberAction, EmberEnvironment, EmberObservation, TASKS  # noqa: E402
from ember.tasks import DOOR_CLOSED, OBSTACLE, WALL  # noqa: E402


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def plan(state, targets: Set[Tuple[int, int]]) -> Optional[List[Tuple[Tuple[int, int], str]]]:
    """BFS path from agent to any target cell. Returns list of (cell, dir)."""
    sx, sy = state.agent_x, state.agent_y
    w, h = state.grid_w, state.grid_h
    if (sx, sy) in targets:
        return []
    seen = {(sx, sy): None}
    queue: deque = deque([(sx, sy)])
    end: Optional[Tuple[int, int]] = None
    while queue:
        x, y = queue.popleft()
        if (x, y) in targets:
            end = (x, y); break
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in seen:
                continue
            ct = state.cell_grid[_idx(nx, ny, w)]
            if ct in (WALL, OBSTACLE):
                continue
            # Closed doors are traversable for planning; we'll open them en route.
            seen[(nx, ny)] = (x, y)
            queue.append((nx, ny))
    if end is None:
        return None
    path: List[Tuple[Tuple[int, int], str]] = []
    cur = end
    while cur != (sx, sy):
        prev = seen[cur]
        dx, dy = cur[0] - prev[0], cur[1] - prev[1]
        dirn = {(0, -1): "north", (0, 1): "south", (-1, 0): "west", (1, 0): "east"}[(dx, dy)]
        path.append((cur, dirn))
        cur = prev
    return list(reversed(path))


def choose_action(state) -> EmberAction:
    # Goal selection: key → door → exit (key_and_door)  or  npc → exit (rescue)
    targets: Set[Tuple[int, int]] = set()
    if state.key_positions and "key_1" in state.key_positions and "key_1" not in state.inventory:
        if (state.agent_x, state.agent_y) == tuple(state.key_positions["key_1"]):
            return EmberAction(action="pickup", target_id="key_1")
        targets = {tuple(state.key_positions["key_1"])}
    elif state.npc_positions and "npc_1" in state.npc_positions and "npc_1" not in state.npc_rescued:
        targets = {tuple(state.npc_positions["npc_1"])}
    else:
        targets = {(e[0], e[1]) for e in state.exit_positions}

    path = plan(state, targets)
    if not path:
        return EmberAction(action="wait")
    (nx, ny), dirn = path[0]
    ct = state.cell_grid[_idx(nx, ny, state.grid_w)]
    if ct == DOOR_CLOSED:
        did = next(
            (k for k, v in state.door_registry.items() if (v[0], v[1]) == (nx, ny)),
            None,
        )
        if did is None:
            return EmberAction(action="wait")
        return EmberAction(action="door", target_id=did, door_state="open")
    return EmberAction(action="move", direction=dirn)


class _HttpDriver:
    """Mirror of run_llm_agent.HttpEnv but exposing `state` for the BFS planner."""

    def __init__(self, base_url: str):
        import requests
        self._r = requests
        self.base_url = base_url.rstrip("/")
        self._state: Optional[SimpleNamespace] = None

    def _refresh_state(self) -> None:
        r = self._r.get(f"{self.base_url}/state", timeout=15)
        r.raise_for_status()
        self._state = SimpleNamespace(**r.json())

    def _to_obs(self, payload):
        obs_data = payload["observation"]
        obs_data.setdefault("metadata", payload.get("metadata", {}))
        return EmberObservation(**obs_data)

    def reset(self, task: str, seed: int) -> EmberObservation:
        r = self._r.post(f"{self.base_url}/reset",
                         json={"task": task, "seed": seed}, timeout=30)
        r.raise_for_status()
        obs = self._to_obs(r.json())
        self._refresh_state()
        return obs

    def step(self, action: EmberAction) -> EmberObservation:
        body = action.model_dump(exclude_none=True)
        body.pop("metadata", None)
        r = self._r.post(f"{self.base_url}/step", json=body, timeout=30)
        r.raise_for_status()
        obs = self._to_obs(r.json())
        self._refresh_state()
        return obs

    @property
    def state(self):
        return self._state


def main():
    ap = argparse.ArgumentParser(description="Scripted (BFS) reference agent")
    ap.add_argument("--task", choices=list(TASKS), default="escape_basic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--log", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--use-http", default="",
                    help="If set, drive the env via the FastAPI server at this URL "
                         "(e.g. http://localhost:8000) so a dashboard can watch.")
    ap.add_argument("--step-delay", type=float, default=0.0,
                    help="Seconds to sleep after each step (slows down for video).")
    args = ap.parse_args()

    env = _HttpDriver(args.use_http) if args.use_http else EmberEnvironment()
    obs = env.reset(task=args.task, seed=args.seed)

    log = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log = open(args.log, "w")
        log.write(json.dumps({
            "event": "reset", "task": args.task, "seed": args.seed,
            "agent": "scripted-bfs", "goal": obs.goal,
            "initial_narrative": obs.narrative,
            "episode_id": getattr(env.state, "episode_id", ""),
            "transport": "http" if args.use_http else "in_process",
        }) + "\n")

    print(f"\n=== Scripted run: task={args.task} seed={args.seed} ===")
    print(obs.goal, "\n")

    for turn in range(args.max_steps):
        action = choose_action(env.state)
        new_obs = env.step(action)
        record = {
            "event": "step", "turn": turn, "elapsed": new_obs.elapsed_steps,
            "observation_to_agent": obs.narrative,
            "agent_reasoning": "BFS toward current goal target",
            "parsed_action": action.model_dump(exclude_none=True),
            "feedback": new_obs.last_action_feedback,
            "reward": new_obs.reward,
            "hp": new_obs.agent_health,
            "inventory": new_obs.inventory,
            "task_complete": new_obs.task_complete,
            "task_failed": new_obs.task_failed,
            "done": new_obs.done,
        }
        if log:
            log.write(json.dumps(record) + "\n")

        if args.verbose:
            first_line = new_obs.narrative.split("\n")[0][:80]
            print(f"[t{turn:02d}] hp={new_obs.agent_health:5.1f} "
                  f"act={action.model_dump(exclude_none=True)} → {new_obs.last_action_feedback}")

        obs = new_obs
        if args.step_delay > 0:
            time.sleep(args.step_delay)
        if obs.done:
            break

    summary = {
        "event": "summary", "task": args.task, "seed": args.seed,
        "turns": turn + 1, "env_steps": obs.elapsed_steps,
        "final_hp": obs.agent_health,
        "task_complete": obs.task_complete, "task_failed": obs.task_failed,
        "agent_evacuated": obs.agent_evacuated, "inventory": obs.inventory,
    }
    if log:
        log.write(json.dumps(summary) + "\n")
        log.close()
    print(f"\n=== {('SUCCESS' if obs.task_complete else 'FAILED')} ===")
    print(json.dumps(summary, indent=2))
    if args.log:
        print(f"Trace: {args.log}")


if __name__ == "__main__":
    main()
