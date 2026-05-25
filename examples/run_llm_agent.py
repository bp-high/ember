"""
Standalone LLM-agent harness for Ember — Google Gemini.

What this is: the reference reset → observe → ask-LLM → act → loop, with
the full reasoning trace written to a JSONL log so a reviewer can read
the agent's "mind" for the whole episode without re-running anything.

Two modes:
  • In-process (default): import EmberEnvironment directly. Fastest. No
    server needed. Trace is still a JSONL file.
  • HTTP (--use-http URL): drive the env over the FastAPI wrapper at
    `ember/server/app.py`. Use this when you want a live dashboard to
    watch the agent move during a screen recording.

Usage:
    GEMINI_API_KEY=... python examples/run_llm_agent.py \
        --task key_and_door --seed 11 --log run.jsonl

    # With a live dashboard (in a second terminal first):
    #     uvicorn ember.server.app:app --port 8000
    #     open http://localhost:8000
    GEMINI_API_KEY=... python examples/run_llm_agent.py \
        --use-http http://localhost:8000 \
        --task rescue --seed 11 --log run.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the package importable when run from a checkout without `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ember import EmberAction, EmberEnvironment, EmberObservation, TASKS  # noqa: E402

SYSTEM_PROMPT = """You are an embodied agent inside a 2D burning building.
Each turn you receive a first-person narrative and must choose ONE action.

Action grammar (return EXACTLY one JSON object on its own line):
  {"action": "move",   "direction": "north"|"south"|"east"|"west"}
  {"action": "door",   "target_id": "door_3", "door_state": "open"|"close"}
  {"action": "pickup", "target_id": "key_1"}
  {"action": "look",   "direction": "north"|"south"|"east"|"west"}
  {"action": "wait"}

Rules of the world:
  • Fire and smoke spread each step; standing in either drains HP.
  • Closed doors block your movement until you open them.
  • Locked doors need a matching key (use `pickup`, then `door … open`).
  • Exits engulfed by fire are unusable — find another.

You may think briefly before the JSON. Keep reasoning under 4 sentences;
the world advances every step regardless. Output the JSON as the LAST
thing in your reply."""


# ---------------------------------------------------------------------------
# Observation formatting + reply parsing
# ---------------------------------------------------------------------------

def build_user_prompt(obs: EmberObservation) -> str:
    """Format an observation as the user-turn content for the LLM."""
    return (
        f"{obs.narrative}\n\n"
        f"Inventory: {obs.inventory or '[]'}\n"
        f"Step {obs.elapsed_steps} | HP {obs.agent_health:.0f} "
        f"({obs.health_status}) | Reward so far this step: {obs.reward}\n"
        "Choose your next action. End with a single JSON object."
    )


def parse_action(text: str) -> EmberAction:
    """Extract the last JSON object from the model's reply."""
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            payload = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "action" in payload:
            return EmberAction(**payload)
    return EmberAction(action="wait")


def serialize_usage(usage) -> Optional[Dict[str, Any]]:
    """Token-usage extraction tolerant of SDK version drift."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    return {
        k: getattr(usage, k)
        for k in ("prompt_token_count", "candidates_token_count", "total_token_count")
        if hasattr(usage, k)
    } or None


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def call_gemini(client, model: str, messages: List[Dict[str, str]],
                thinking_level: Optional[str] = "HIGH"):
    """Drive Google Gemini.

    Translates the harness's `{"role": "assistant"}` messages to the
    SDK's `Content(role="model")` and passes SYSTEM_PROMPT via
    `system_instruction`. We deliberately do NOT enable any tools —
    the agent should reason from the observation, not the open web.
    """
    from google.genai import types

    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
        )

    cfg_kwargs: Dict[str, Any] = dict(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=512,
    )
    if thinking_level:
        tc = _build_thinking_config(types, thinking_level)
        if tc is not None:
            cfg_kwargs["thinking_config"] = tc

    config = types.GenerateContentConfig(**cfg_kwargs)
    resp = client.models.generate_content(model=model, contents=contents, config=config)
    text = resp.text or ""
    usage = getattr(resp, "usage_metadata", None)
    return text, usage


def _build_thinking_config(types_mod, level: str):
    """ThinkingConfig compatible with both old and new google-genai SDKs."""
    fields = set(getattr(types_mod.ThinkingConfig, "model_fields", {}).keys())
    if "thinking_level" in fields:
        try:
            return types_mod.ThinkingConfig(thinking_level=level)
        except Exception:
            pass
    if "thinking_budget" in fields:
        budget = {"LOW": 512, "MEDIUM": 2048, "HIGH": 8192}.get(level, 2048)
        try:
            return types_mod.ThinkingConfig(thinking_budget=budget)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Env adapters — in-process or HTTP
# ---------------------------------------------------------------------------

class InProcessEnv:
    """Drive EmberEnvironment in this Python process. Fastest path."""

    def __init__(self):
        self._env = EmberEnvironment()

    def reset(self, task: str, seed: int) -> EmberObservation:
        return self._env.reset(task=task, seed=seed)

    def step(self, action: EmberAction) -> EmberObservation:
        return self._env.step(action)

    @property
    def episode_id(self) -> str:
        return self._env.state.episode_id or ""


class HttpEnv:
    """Drive an Ember FastAPI server (so a live dashboard can watch)."""

    def __init__(self, base_url: str):
        import requests
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self._last_episode_id = ""

    def _to_obs(self, payload: Dict[str, Any]) -> EmberObservation:
        obs_data = payload["observation"]
        # The server returns metadata at the top level too; preserve it.
        obs_data.setdefault("metadata", payload.get("metadata", {}))
        return EmberObservation(**obs_data)

    def reset(self, task: str, seed: int) -> EmberObservation:
        r = self._requests.post(
            f"{self.base_url}/reset",
            json={"task": task, "seed": seed},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        self._last_episode_id = (
            payload["observation"].get("map_state", {}).get("episode_id", "")
            if payload["observation"].get("map_state") else ""
        )
        return self._to_obs(payload)

    def step(self, action: EmberAction) -> EmberObservation:
        body = action.model_dump(exclude_none=True)
        # The OpenEnv base Action has a `metadata` field that we don't
        # need to send across the wire.
        body.pop("metadata", None)
        r = self._requests.post(f"{self.base_url}/step", json=body, timeout=30)
        r.raise_for_status()
        return self._to_obs(r.json())

    @property
    def episode_id(self) -> str:
        return self._last_episode_id


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run a Gemini LLM agent in Ember")
    ap.add_argument("--task", choices=list(TASKS), default="escape_basic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="Any Gemini model ID accepted by the Gemini API.")
    ap.add_argument("--thinking", choices=["OFF", "LOW", "MEDIUM", "HIGH"],
                    default="HIGH",
                    help="Thinking budget for Gemini models that support it.")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="Hard cap on LLM turns (separate from env's max_steps).")
    ap.add_argument("--log", default="run.jsonl",
                    help="JSONL path. Each line = one harness step.")
    ap.add_argument("--history", type=int, default=4,
                    help="Number of prior (obs, reply) pairs to keep in chat.")
    ap.add_argument("--use-http", default="",
                    help="If set, drive the env via the FastAPI server at this URL "
                         "(e.g. http://localhost:8000) so a dashboard can watch.")
    ap.add_argument("--step-delay", type=float, default=0.0,
                    help="Seconds to sleep after each step (slows down for video).")
    args = ap.parse_args()

    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    thinking = None if args.thinking == "OFF" else args.thinking

    env = HttpEnv(args.use_http) if args.use_http else InProcessEnv()
    obs = env.reset(task=args.task, seed=args.seed)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")

    log.write(json.dumps({
        "event": "reset", "task": args.task, "seed": args.seed,
        "model": args.model, "goal": obs.goal,
        "initial_narrative": obs.narrative,
        "episode_id": env.episode_id,
        "transport": "http" if args.use_http else "in_process",
    }) + "\n")
    log.flush()

    messages: List[Dict[str, str]] = []
    print(f"\n=== Ember Gemini run: task={args.task} seed={args.seed} model={args.model} ===")
    print(obs.goal, "\n")

    turn = 0
    for turn in range(args.max_steps):
        user_msg = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_msg})

        t0 = time.time()
        reply, usage = call_gemini(client, args.model,
                                   messages[-2 * args.history:],
                                   thinking_level=thinking)
        latency_ms = int((time.time() - t0) * 1000)
        messages.append({"role": "assistant", "content": reply})

        action = parse_action(reply)
        new_obs = env.step(action)

        log.write(json.dumps({
            "event": "step", "turn": turn, "elapsed": new_obs.elapsed_steps,
            "observation_to_llm": user_msg,
            "llm_reply": reply,
            "parsed_action": action.model_dump(exclude_none=True),
            "feedback": new_obs.last_action_feedback,
            "reward": new_obs.reward,
            "hp": new_obs.agent_health,
            "inventory": new_obs.inventory,
            "done": new_obs.done,
            "task_complete": new_obs.task_complete,
            "task_failed": new_obs.task_failed,
            "latency_ms": latency_ms,
            "usage": serialize_usage(usage),
        }) + "\n")
        log.flush()

        print(f"[t{turn:02d}] hp={new_obs.agent_health:5.1f} "
              f"act={action.model_dump(exclude_none=True)} → {new_obs.last_action_feedback}")

        obs = new_obs
        if args.step_delay > 0:
            time.sleep(args.step_delay)
        if obs.done:
            break

    summary = {
        "event": "summary",
        "task": args.task,
        "seed": args.seed,
        "model": args.model,
        "turns": turn + 1,
        "env_steps": obs.elapsed_steps,
        "final_hp": obs.agent_health,
        "task_complete": obs.task_complete,
        "task_failed": obs.task_failed,
        "agent_evacuated": obs.agent_evacuated,
        "inventory": obs.inventory,
    }
    log.write(json.dumps(summary) + "\n")
    log.close()

    print(f"\n=== {('SUCCESS' if obs.task_complete else 'FAILED')} ===")
    print(json.dumps(summary, indent=2))
    print(f"Trace: {log_path}")


if __name__ == "__main__":
    main()
