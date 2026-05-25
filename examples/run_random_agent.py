"""
Random baseline agent for Ember.

Picks uniformly from the env's `available_actions_hint` list each step.
Useful as a sanity check (does the env reach `done` on its own?) and as
a floor against which the LLM agent's performance should be obviously
better.

Usage:
    python examples/run_random_agent.py --task escape_basic --episodes 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ember import EmberAction, EmberEnvironment, TASKS  # noqa: E402


def parse_hint(hint: str) -> EmberAction:
    """Map a single action-hint string back to an EmberAction."""
    h = hint.strip()
    if h.startswith("move("):
        m = re.search(r"direction='(\w+)'", h)
        return EmberAction(action="move", direction=m.group(1) if m else "north")
    if h.startswith("door("):
        tid = re.search(r"target_id='([^']+)'", h)
        ds = re.search(r"door_state='(\w+)'", h)
        return EmberAction(
            action="door",
            target_id=tid.group(1) if tid else None,
            door_state=ds.group(1) if ds else "open",
        )
    if h.startswith("pickup("):
        tid = re.search(r"target_id='([^']+)'", h)
        return EmberAction(action="pickup", target_id=tid.group(1) if tid else None)
    if h.startswith("look("):
        m = re.search(r"direction='(\w+)'", h)
        return EmberAction(action="look", direction=m.group(1) if m else "north")
    return EmberAction(action="wait")


def run_episode(env, task: str, seed: int, rng: random.Random) -> dict:
    obs = env.reset(task=task, seed=seed)
    steps = 0
    total_reward = 0.0
    while not obs.done and steps < 200:
        hints = obs.available_actions_hint or ["wait()"]
        action = parse_hint(rng.choice(hints))
        obs = env.step(action)
        total_reward += obs.reward or 0.0
        steps += 1
    return {
        "task": task, "seed": seed, "steps": steps,
        "task_complete": obs.task_complete, "task_failed": obs.task_failed,
        "evacuated": obs.agent_evacuated, "final_hp": obs.agent_health,
        "total_reward": round(total_reward, 3),
    }


def main():
    ap = argparse.ArgumentParser(description="Random baseline for Ember")
    ap.add_argument("--task", choices=list(TASKS), default="escape_basic")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    env = EmberEnvironment()
    results = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        r = run_episode(env, args.task, ep_seed, rng)
        results.append(r)
        print(json.dumps(r))

    completed = sum(r["task_complete"] for r in results)
    print(f"\n{args.task}: {completed}/{len(results)} task_complete | "
          f"mean steps={sum(r['steps'] for r in results) / len(results):.1f}")


if __name__ == "__main__":
    main()
