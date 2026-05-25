# Ember — design notes

A short, honest writeup of how Ember answers the four things the prompt
says it cares about: observation representation, action space,
goal-directed task completion, and harness simplicity.

## Observation: narrative first, structured second

The LLM sees a single string. Every step it's regenerated from server
state by [`ember/narrative.py`](ember/narrative.py). The fields chosen
for inclusion are the ones an agent actually needs to act:

```
GOAL [key_and_door]: (1) pick up key_1 at [3,2]  →  (2) (need key first) door_3 at [7,2]  →  (3) reach any exit

You are in the **west_rooms**. The air is **none**.
Health: ██████████ 100/100 (Good)  |  Wind: calm
No fire directly visible.
Exits visible: exit_0_8 at 5m west, exit_15_8 at 12m east.
Doors: door_3 (locked, closed) at 4m east, door_4 (open) at 3m south.
Last action: Episode started. Read the goal and assess your surroundings.
Available actions: move(direction='north')  pickup(target_id='key_1')  …
```

What's in there and why:

- **Goal line on top.** The task's `goal_text()` is prepended to every
  narrative — not just at reset. Long episodes drop the original goal
  out of an LLM's working memory; rendering it every step is cheap and
  removes a whole class of "the model forgot what it was doing" bugs.
  For multi-step tasks the line includes sub-step progress
  (`✓ key_1 in inventory`, `(need key first)`), giving the LLM a
  cheap "where am I in the plan" signal without prompting tricks.
- **Location label + smoke level + health bar.** Ground state for any
  policy decision. The health bar is rendered ASCII (`██████████ 100/100`)
  because LLMs read those better than a bare integer — they tokenize
  the proportion visually.
- **Visible doors and exits with relative positions** ("5m east"). Not
  absolute (x, y) — LLMs are bad at coordinate geometry but good at
  "go in the direction X." Distances stay in cell units because that's
  also how the action space is denominated; mixing units would invite
  off-by-one reasoning errors.
- **Last action feedback.** One sentence describing what the previous
  action did ("You step through the exit and escape the building!").
  Without this, the model can't tell whether a move actually happened
  — it would just see the new state and have to infer the delta.
- **Available actions hint.** A filtered list of legal action call
  strings for *this* situation: walls aren't in `move()` hints,
  non-adjacent doors aren't in `door()` hints. The LLM can ignore it,
  but in practice it dramatically reduces "tried to walk through wall"
  steps and keeps the agent's reasoning focused on choosing between
  viable options rather than rediscovering legality.

There's also a parallel **structured** observation (`visible_objects`,
`inventory`, `blocked_exit_ids`, `task_complete`, …) returned in the
same `EmberObservation`. Two reasons:

1. The harness shouldn't have to parse English to know when an episode
   ended. `task_complete` is a bool; the LLM loop reads it directly.
2. Non-LLM agents (the random baseline, the BFS reference, a future
   PPO trainer) need the same world but can't use prose. Sharing one
   observation type with both views keeps them in lockstep — when we
   add a field, every agent gets it for free.

### What did not work for observations

- **Raw cell grid in the prompt.** Tried an "ASCII map with `@` for the
  agent, `#` for walls, `F` for fire" view. LLMs sort of read it, but
  spatial reasoning on small ASCII grids is noisy — they'd misjudge
  adjacency more than parse it. Narrative ("door_3 (closed) at 4m
  east") was strictly better.
- **Frame stacking (last N observations concatenated).** Useful for
  PPO's tiny MLP because it has no memory. Useless for LLMs because the
  chat history already carries that — and counterproductive, because
  duplicating prior obs in the *current* message confuses the model
  about which step it's reasoning about. The LLM harness keeps a
  rolling chat of the last `--history` (obs, reply) pairs instead.
- **Visibility/fog-of-war by default.** The env supports flood-fill
  visibility (radius shrinks under smoke), but the LLM harness ships
  with full visibility on (`EMBER_FULL_VISIBILITY=1`). With the small
  16×16 templates, partial visibility forced extra `look` actions that
  burned LLM tokens without producing more interesting episodes. The
  switch is one env var if you want it back on.

## Action space: 5 verbs, structured JSON

```jsonc
{"action": "move",   "direction": "north"|"south"|"east"|"west"}
{"action": "door",   "target_id": "door_3", "door_state": "open"|"close"}
{"action": "pickup", "target_id": "key_1"}
{"action": "look",   "direction": "north"|"south"|"east"|"west"}
{"action": "wait"}
```

Design choices and what each one buys us:

- **Cardinal moves only.** No diagonals, no rotation. The world is a
  4-connected grid; adding more verbs adds disambiguation cost (turn
  vs. move-and-turn?) without adding capability. The smaller the
  action space, the easier the model can be steered by the legal-hints
  list.
- **`door` is its own verb, with explicit `open`/`close`.** Closing a
  door deliberately is a key tactical option (it slows fire spread by
  `DOOR_CLOSED_FIRE_FACTOR = 0.15` in `fire_sim.py`). If we'd folded
  door interaction into `move`, the LLM would have no way to express
  "close the door behind me." The split makes that decision explicit.
- **`pickup`, not generic `interact`.** Adding the *exact* item ID
  (`pickup(target_id='key_1')`) makes the action self-documenting and
  easy to log. A generic `interact("here")` would force the env to
  guess intent and the model to phrase it ambiguously.
- **`look <direction>` is a no-cost scan.** It returns a description
  of up to 5 cells in that direction. Time still ticks — fire spreads
  — so the model trades a step for information. Without `look`, an
  agent's only way to learn about distant cells would be to walk into
  them and risk fire damage.
- **`wait`.** Legal no-op. Sometimes the right move is to stand still
  one step (e.g. wait for fire to burn out of an exit, or just buy
  reasoning time on a longer chain-of-thought).

### Parsing LLM output

The harness uses `re.finditer(r"\{[^{}]*\}", text)` to find the **last**
JSON object in the model's reply. This is deliberate:

- Models love to "think out loud" before emitting structured output.
  We let them — we just grab the final JSON.
- Multi-object replies (some models emit a tool-call-like wrapper first
  and the real action second) work without a special case.
- If no parseable JSON is present, the agent gets a `wait()` for the
  turn. Better than crashing the episode and losing the trace.

We deliberately did *not* use Gemini's function-calling / tool-use
schema. Reason: tool-call schemas differ between SDK versions and
across providers, and coupling the harness to one version means
re-validating every time the SDK ships a breaking change. A plain
JSON object in the message body works identically across models — and
the same parsing strategy survives a future provider swap.

## Tasks: making the goal legible

The prompt's examples — "go to the red cube", "find the key and open
the door" — are named, verifiable, and small. A bare "survive the
fire" goal would not fit that mold, so Ember layers a `Task`
abstraction on top of the world in [`ember/tasks.py`](ember/tasks.py):

```python
class Task:
    name: str
    template_whitelist: list[str]
    def setup(self, state, rng) -> None: ...
    def goal_text(self, state) -> str: ...
    def evaluate(self, state) -> TaskResult: ...
```

A Task does three things, no more:

1. **`setup`** — mutates the state to place keys, lock doors, drop NPC
   markers, and (for `key_and_door`) move the agent into a sealed
   room. Runs at reset after the floor plan and fire are placed.
2. **`goal_text`** — returns the one-liner shown at the top of every
   observation, including sub-step progress.
3. **`evaluate`** — returns `(complete, failed)` based on the current
   state. The env checks this each step and surfaces both flags in
   the observation so the harness loop can exit when the *named goal*
   is met, not only when the agent leaves the building.

Three tasks ship:

- **`escape_basic`** — reach any unblocked exit before HP runs out.
  Tests pure navigation under threat.
- **`key_and_door`** — agent spawns inside a sealed room with the key.
  Setup picks a door whose closure disconnects the agent from all
  exits (an "isolating door"), then drops the agent on the inside.
  The lock matters: there's no alternate path.
- **`rescue`** — an NPC marker is placed elsewhere on the map. Walking
  onto it rescues them. Escaping *without* rescuing is a hard failure
  (`task_failed=True`). Tests detour-then-escape.

### What did not work for tasks

- **Treating "difficulty" as the top-level reset knob.** A first cut
  used `env.reset(difficulty="easy")` to couple fire pacing, map size,
  and (sort of) goal complexity into one parameter. With named tasks,
  fire pacing belongs to the *task* (`key_and_door` needs slow fire so
  the sub-goal sequence is solvable), and "difficulty" as a
  user-facing concept disappears. Reset takes a `task` name now.
- **Finding chokepoint doors by "block-and-recount-exits."** First
  attempt: for each door, treat it as closed and count reachable
  exits. If fewer than baseline, it's a chokepoint. Didn't work — the
  hand-authored templates have redundant connectivity, so most
  building floors had *zero* chokepoint doors. Switched to: spawn the
  agent inside a one-door room with the door treated as closed; this
  guarantees the lock matters by construction. See
  `KeyAndDoor._find_isolating_door` in
  [`ember/tasks.py`](ember/tasks.py).

## What's in the harness, ranked by what a reviewer should look at

In priority order:

1. [`examples/run_llm_agent.py`](examples/run_llm_agent.py) — the
   reset → observe → ask-LLM → act → loop. Drives Google Gemini.
   Writes a JSONL trace per step.
2. [`ember/env.py`](ember/env.py) `EmberEnvironment.step()` — the
   contract between the env and any agent. Action handling, fire tick,
   damage, reward, observation rebuild.
3. [`ember/tasks.py`](ember/tasks.py) — the Task abstraction. Adding a
   fourth task (say, "close all doors adjacent to fire then escape")
   would be ~40 lines here and zero changes anywhere else.
4. [`examples/traces/*.jsonl`](examples/traces/) — saved episodes. Each
   line is one step with the observation, reply, action, feedback, and
   reward. Open one and read the whole episode without running
   anything.

## What I'd add next

- **Streaming the LLM's reply token-by-token** into the dashboard, so a
  recorded run shows the model "thinking" in real time next to the
  agent moving on the grid.
- **A `close_doors_and_escape` task.** Reward closing doors adjacent to
  fire as a sub-goal; tests that LLMs can value-act on the
  fire-suppression mechanic the reward rubrics already model.
- **Per-task seeds tested for solvability.** Right now `key_and_door`
  falls back to a partial-chokepoint setup on templates with no
  isolating door. We surface the fallback in logs, but a curated seed
  list (one per task) would make the reference traces cleaner.
