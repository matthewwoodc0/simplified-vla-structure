# SVLA Learning Guide

A hands-on walkthrough for understanding this repo even if you did not write every
line of code yourself. Work through the sections in order. Each lab asks you to
open real files, change something small, and observe what happens.

**Time estimate:** 4–8 hours if you do every lab carefully. You can stop after
any section and come back later.

---

## Table of Contents

1. [What this project is really about](#1-what-this-project-is-really-about)
2. [Before you start](#2-before-you-start)
3. [The mental model (read this first)](#3-the-mental-model-read-this-first)
4. [Repo map](#4-repo-map)
5. [Major files — high-level tour](#5-major-files--high-level-tour)
6. [Hands-on labs](#6-hands-on-labs)
7. [How to read demo data](#7-how-to-read-demo-data)
8. [Roadmap: where you are and what is next](#8-roadmap-where-you-are-and-what-is-next)
9. [When something breaks](#9-when-something-breaks)
10. [Brief checkpoint quiz](#10-brief-checkpoint-quiz)
11. [Answer key](#11-answer-key)

---

## 1. What this project is really about

This is **not** a finished VLA (Vision-Language-Action model). It is an
**experiment platform** for answering one question:

> If we teach a robot policy the same pickup task, does it learn faster when
> actions are **end-effector deltas** (move hand left 2 cm) or **joint deltas**
> (rotate elbow 0.03 radians)?

A **controller** sits between the policy and the physics simulator. It handles
inverse kinematics (IK): turning "move the hand here" into joint commands.

We build the foundation first so later ML results are trustworthy:

```text
Phase 1  Simulator + controller          DONE
Phase 2  Action-space adapters          DONE
Phase 3  Pickup task + scripted expert  DONE
Phase 4  Recorded demonstrations        DONE (sample scale)
Phase 5  State-based behavioral cloning IMPLEMENTED (first baseline)
Phase 6  Vision                         NOT STARTED
Phase 7  Language / VLA                 NOT STARTED
```

**Key idea:** If you skip straight to "train a VLA," you will not know whether
failures came from the model, the controller, the task, or bad data.

**Important honesty check:** Phase 5 being "implemented" does not mean the
learning problem is solved. The current BC code proves that the repo can load
state observations, train/evaluate joint-delta and EE-delta policies, and roll
them out in MuJoCo. It is still a small nearest-neighbor baseline. The strongest
current result is closer to executable demo replay than robust generalization.
That is useful engineering evidence, not a VLA-ready research result by itself.

---

## 2. Before you start

### Activate the environment

From the repo root:

```bash
source .venv/bin/activate
```

### Baseline health check

Run this once before any lab. All tests should pass.

```bash
pytest -q
```

If tests fail, fix that before changing code. The tests are your safety net.

### Important path note

This folder name contains spaces (`Simplified VLA Structure`). Do **not** run
`.venv/bin/mjpython` directly. Use the shell wrappers:

```bash
bash scripts/run_mujoco_gui.sh
bash scripts/open_raw_mujoco_viewer.sh
```

### PYTHONPATH for one-off commands

Scripts add `src` to the path automatically. For ad hoc Python:

```bash
PYTHONPATH=src .venv/bin/python -c "from svla.sim import ArmSim; print(ArmSim().ee_position)"
```

---

## 3. The mental model (read this first)

Think of the system as a stack. Data flows **down**; measurements flow **up**.

```text
┌──────────────────────────────────────────────────────────────┐
│  Policy (Phase 5 now, richer models later)                   │
│  "Given state observation → output action"                    │
└────────────────────────────┬─────────────────────────────────┘
                             │ action vector
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Action adapters  (`action_spaces.py`)                       │
│  Same motion, two label formats: joint_delta vs ee_delta     │
└────────────────────────────┬─────────────────────────────────┘
                             │ CartesianCommand or joint targets
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Controller  (`controller.py`)                               │
│  IK, clipping, gripper, telemetry                            │
└────────────────────────────┬─────────────────────────────────┘
                             │ joint actuator commands
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  MuJoCo physics  (`sim.py` + `assets/*.xml`)                 │
│  Gravity, contacts, joint limits                             │
└──────────────────────────────────────────────────────────────┘
```

### Vocabulary cheat sheet

| Term | Plain English |
|------|---------------|
| **MuJoCo** | Physics simulator — a fake world for the robot |
| **End effector (EE)** | The gripper / tool at the end of the arm |
| **IK (inverse kinematics)** | Math that converts hand position → joint angles |
| **Controller** | Code that runs IK every timestep and reports errors |
| **Action space** | The format of numbers a policy outputs |
| **Behavioral cloning (BC)** | Watch expert demos, imitate them (supervised learning) |
| **VLA** | Model that takes vision + language and outputs actions |
| **Telemetry** | Logged measurements: target vs actual pose, clipping flags |
| **Clipping** | Command was too big or hit a limit, so it was scaled down |

---

## 4. Repo map

```text
Simplified VLA Structure/
├── assets/                  # MuJoCo robot + scene XML blueprints
│   ├── so101_arm.xml        # Arm definition
│   ├── so101_scene.xml      # Arm in empty scene (reach demos)
│   ├── pickup_scene.xml     # Table + cube pickup scene
│   └── simple_arm.xml       # Older/simple arm (legacy reference)
├── src/svla/                # Core library code
│   ├── controller.py        # Cartesian IK controller
│   ├── sim.py               # Thin MuJoCo wrapper
│   ├── action_spaces.py     # joint_delta vs ee_delta label adapters
│   ├── pickup_task.py       # Pickup environment + scripted expert
│   ├── demo_recorder.py     # Records demos with aligned labels
│   ├── state_bc.py          # Phase 5: nearest-neighbor BC training/eval
│   ├── teleop_*.py          # Keyboard/gamepad GUI teleoperation
│   └── ...
├── scripts/                 # Runnable entry points
│   ├── run_reach_demo.py
│   ├── render_reach_demo.py
│   ├── run_pickup_trials.py
│   ├── record_pickup_demos.py
│   ├── train_state_bc.py
│   ├── train_reach_policy.py
│   └── open_mujoco_gui.py
├── tests/                   # Automated checks (read these as examples!)
├── outputs/                 # Generated videos, JSON logs, trained models
├── README.md                # Project overview + research question
├── AGENTS.md                # Notes for AI assistants working in this repo
└── LEARNING_GUIDE.md        # You are here
```

**Rule of thumb:**

- `assets/` = *what exists in the world*
- `src/svla/` = *how the robot is controlled and evaluated*
- `scripts/` = *things you run from the terminal*
- `tests/` = *proof that pieces still work*
- `outputs/` = *artifacts from running scripts*

---

## 5. Major files — high-level tour

Open these files in your editor and skim the top comments before doing labs.

### `assets/pickup_scene.xml`

MuJoCo scene: arm, table, cube, contacts. No Python logic — just geometry,
joints, actuators, and physics parameters. If the cube falls through the table,
the bug is probably here (or in how contacts are configured).

### `src/svla/controller.py`

The heart of the control stack.

- `CartesianCommand` — what a policy *would* send: xyz delta, rotation delta,
  gripper open amount.
- `ControllerLimits` — max step sizes per tick (prevents huge jumps).
- `CartesianIKController` — reads current EE pose, computes joint targets via
  damped least-squares IK, clips to joint limits, stores `last_telemetry`.

**Read the docstring** on `CartesianIKController` — it explains the separation
between teleop (human sets targets) and IK (controller tracks targets).

### `src/svla/sim.py`

`ArmSim` class: load model, reset arm, `step(command)`, `move_to(xyz)`.
This is the smallest "hello world" interface to the simulator.

### `src/svla/action_spaces.py`

`JointDeltaActionAdapter` and `EndEffectorDeltaActionAdapter` turn a
before/after robot state into training labels. `label_transition_all()` returns
both formats for the same transition — that is what makes the experiment fair.

### `src/svla/pickup_task.py`

The first real manipulation task.

- `PickupTaskEvaluator` — reset, step, observe, measure success.
- Scripted expert trajectory (not learned).
- Success requires: contact → lift off table → hold without dropping.
- `default_trial_specs()` — 36 trials (3 yaw × 3 object poses × 2 approaches
  × 2 repeats).

### `src/svla/demo_recorder.py`

Runs the scripted expert and saves JSON with per-step observations, controller
telemetry, and **both** action label formats.

### `src/svla/state_bc.py`

Phase 5 learning code. Loads demo JSON, trains a simple nearest-neighbor policy
(no PyTorch), rolls it out in the pickup task, compares `joint_delta` vs
`ee_delta`.

Be precise about what this baseline proves:

- It proves the state-BC loop can run end to end.
- It proves both action spaces can be loaded, trained, saved, and evaluated.
- It does **not** prove a learned policy generalizes well yet.
- The exact monotonic settings (`k=1`, `search_window=1`) are mostly a replay
  sanity check. They are valuable, but they are not the same as a robust policy.

### `scripts/*.py`

Thin wrappers — good starting points because they show *how* library code is
wired together without much extra logic.

### `tests/*.py`

Short, readable examples of expected behavior. When confused, find the test
for the module you are studying.

---

## 6. Hands-on labs

Each lab follows the same pattern:

1. **Predict** — write down what you think will happen.
2. **Change** — edit one file.
3. **Run** — execute the command.
4. **Observe** — compare to your prediction.
5. **Revert** — undo the change (or keep a note in a scratch branch).

---

### Lab 0 — Meet the arm (no code changes)

**Goal:** Confirm the simulator loads and the controller reaches nearby targets.

```bash
python scripts/run_reach_demo.py
```

You should see four lines like:

```text
target 1: target=[...] ee=[...] error=0.00XXm
```

**Questions to answer in your notes:**

- What unit is `error` in? (meters)
- Is sub-2 cm error good enough for a smoke test? (yes, for reach — not for precision assembly)

**Optional video:**

```bash
python scripts/render_reach_demo.py
open outputs/reach_demo.mp4
```

---

### Lab 1 — Change where the arm reaches

**File:** `scripts/run_reach_demo.py`

Find `TARGET_DELTAS` near the top. It is a tuple of 3D offsets added to the
current hand position.

**Experiment A — bigger moves:**

Change the first delta from `[-0.04, 0.05, 0.03]` to `[-0.10, 0.12, 0.08]`.

```bash
python scripts/run_reach_demo.py
```

Did error increase? Did any target fail to converge within tolerance?

**Experiment B — tiny moves:**

Change the first delta to `[-0.005, 0.005, 0.005]`.

**Revert** when done.

**What you learned:** The controller works locally. Huge jumps may clip or
stall; tiny moves converge easily.

---

### Lab 2 — See clipping in action

**File:** `tests/test_controller.py`

Read `test_delta_command_is_clipped_and_keeps_state_finite`. It sends a
ridiculous command `[1.0, 1.0, 1.0]` meters in one step.

Run just that test:

```bash
pytest tests/test_controller.py::test_delta_command_is_clipped_and_keeps_state_finite -v
```

**Experiment — break the safety rail (temporarily):**

**File:** `src/svla/controller.py`

Find `ControllerLimits` and change `max_step_xyz` from `0.025` to `0.5`.

Re-run the test. Still clipped? (The test command is still huge.)

Now run:

```bash
python scripts/run_reach_demo.py
```

Watch for wild motion or worse errors.

**Revert `max_step_xyz` to `0.025`.**

**What you learned:** Clipping is intentional. Policies that output large
actions get softened by the controller — and that affects learning.

---

### Lab 3 — Drive the arm yourself (GUI)

**Goal:** Connect human input → teleop → controller → physics.

```bash
bash scripts/run_mujoco_gui.sh
```

Controls (from `src/svla/teleop_inputs.py`):

| Key | Action |
|-----|--------|
| W/S | Forward/back (gripper-local X) |
| A/D | Left/right (gripper-local Y) |
| Q/E | Up/down (gripper-local Z) |
| Mouse drag | Pitch/yaw |
| Space | Toggle gripper |
| 1–4 | Jump to preset reach offsets |
| R | Reset |
| H | Help |

**Experiment:**

1. Press `R` to reset.
2. Press `1` — arm moves to a preset offset.
3. Use W/Q to nudge. Watch whether motion feels smooth or jerky.
4. Press `P` to pause IK tracking. What happens when you press W now?

**Files to peek at after playing:**

- `scripts/open_mujoco_gui.py` — main loop
- `src/svla/teleop_controller.py` — integrates input into targets
- `src/svla/teleop_workspace.py` — reachable workspace box

**What you learned:** Teleop sets *targets*; the controller *tracks* them. That
is the same boundary a learned policy will use.

---

### Lab 4 — Run the pickup benchmark

**Goal:** See the scripted expert pass/fail across 36 conditions.

```bash
python scripts/run_pickup_trials.py --repeats 2
```

Check the summary:

```bash
cat outputs/pickup_trials.summary.json
```

Look for `success_rate`, `failure_categories`, and breakdowns by orientation,
object pose, and approach.

**Experiment — fewer trials (faster):**

```bash
python scripts/run_pickup_trials.py --repeats 1
```

**Optional showcase video:**

```bash
python scripts/render_pickup_showcase.py
open outputs/pickup_showcase.mp4
```

**What you learned:** Pickup success is measurable and mostly works scripted.
Known weak spot: some `yaw_0` + `right` + `high_staged_vertical_pregrasp`
trials lose retention.

---

### Lab 5 — Inspect one recorded demonstration

**Generate fresh demos (small set):**

```bash
python scripts/record_pickup_demos.py
```

**Open in your editor:**

`outputs/scripted_pickup_demos/pickup_demo_01_yaw_-18_center_vertical_pregrasp.json`

This file is large. Do not read it all. Search within the file for:

1. `"format"` — schema version
2. `"phase_summaries"` — approach, grasp, lift, hold phases
3. `"samples"` → first entry → `"labels"` — both `joint_delta` and `ee_delta`
4. `"controller_telemetry"` → `"clipped_joints"` — was IK struggling?

**Compare label sizes:**

- `joint_delta` has **6** numbers (5 arm joints + gripper)
- `ee_delta` has **7** numbers (xyz + rotvec + gripper)

There are two label fields:

- `labels` are the observed transition after physics: "what actually happened
  between this state and the next state."
- `policy_labels` are executable commands derived from the controller target and
  telemetry: "what a policy should output to reproduce this step."

For Phase 5 training, `policy_labels` are the better default. Raw `labels` are
still useful for analysis, but they can include physics lag and contact effects
that are not directly executable commands.

**What you learned:** One trajectory produces two training label formats. That
is the core of the action-space comparison experiment.

---

### Lab 6 — Action adapters in isolation

**File:** `tests/test_action_spaces.py`

Read the two adapter tests. They use fake before/after states (no simulator).

```bash
pytest tests/test_action_spaces.py -v
```

**Experiment — change a label:**

In `test_joint_delta_adapter_labels_transition_with_gripper_command`, change
`after` joint positions and predict the new `joint_delta` values by hand:

```text
delta = after_joints - before_joints
```

Run the test. If your math matches, you understand the adapter.

**What you learned:** Action adapters are just encoding rules. The physics
does not care about them — the *learner* does.

---

### Lab 7 — Phase 5 behavioral cloning (small run)

**Goal:** Train and evaluate the simple nearest-neighbor policies.

```bash
python scripts/train_state_bc.py --demo-repeats 1 --eval-repeats 1 --stride 10
```

This writes to `outputs/state_bc/` by default:

- `joint_delta_training_summary.json`
- `ee_delta_training_summary.json`
- `eval/*_policy_trials.jsonl`

**Read the training summaries.** Compare `sample_count`, `action_size`, and
eval success rates between action spaces.

The default command uses the executable `policy_labels` field. That matters:
the policy should output commands, not merely describe what the simulator did
after contacts and actuator lag.

**Experiment — denser training data:**

```bash
python scripts/train_state_bc.py --demo-repeats 2 --eval-repeats 2 --stride 2
```

Did either action space improve? This is early evidence — not a final paper
result.

Be careful interpreting the results:

- High success with `k=1` and `search_window=1` mostly says "the recorded
  command sequence can be replayed in the same task context."
- Lower success with broader nearest-neighbor search says the state features and
  baseline policy are not yet robust.
- A real next step is a stronger state policy and held-out contexts, not vision.

**What you learned:** The ML loop exists. The research question is now testable,
but the current baseline should be treated as a first pass, not a settled answer.

---

### Lab 8 — Break a test on purpose, then fix it

**Goal:** Practice the debug loop researchers actually use.

**File:** `tests/test_controller.py`

In `test_controller_reaches_nearby_target`, change the assertion from
`error < 0.02` to `error < 0.001` (unrealistically strict).

```bash
pytest tests/test_controller.py::test_controller_reaches_nearby_target -v
```

It should fail. Read the failure output: actual error vs expected bound.

**Fix:** Restore `0.02` and confirm green.

**What you learned:** Tests encode acceptance criteria. Changing them is how
you formalize "good enough."

---

### Lab 9 — Trace one function call chain (read-only)

Pick one command from `scripts/run_pickup_trials.py` and trace:

```text
run_pickup_trials.py
  → PickupTaskEvaluator.run_trial()   (pickup_task.py)
    → scripted_controller_commands()
    → step_controller_command()
      → CartesianIKController.move_toward()   (controller.py)
        → mujoco.mj_step()
```

Draw this on paper. Add boxes for where observations and labels are created.

**What you learned:** Scripts are thin. The real logic lives in `src/svla/`.

---

## 7. How to read demo data

Each demo JSON (`format: svla_pickup_demo_v1`) contains:

| Section | What it is |
|---------|------------|
| `metadata.trial_spec` | Which yaw, object pose, approach strategy |
| `summary` | Trial-level success, failure category, clip counts |
| `phase_summaries` | Per-phase sample counts and errors |
| `samples[]` | Per-timestep training rows |

Each sample row:

| Field | Meaning |
|-------|---------|
| `observation` | Robot state *before* the step (joints, EE pose, gripper) |
| `command` | Controller target pose the scripted expert commanded |
| `labels.joint_delta` | Training label if policy uses joint space |
| `labels.ee_delta` | Training label if policy uses EE space |
| `policy_labels.joint_delta` | Executable joint-delta command for BC replay |
| `policy_labels.ee_delta` | Executable EE-delta command for BC replay |
| `next_observation` | State after physics step |
| `controller_telemetry` | Target vs actual, errors, clip flags |
| `success_metrics` | Contact, lift height, retention counters |

**Pickup success definition** (from `pickup_task.py`):

1. End effector reaches the commanded grasp pose.
2. Gripper contacts the object.
3. Object lifts at least `LIFT_CLEARANCE` (0.018 m) off the support.
4. Object stays lifted through the hold window (`RETENTION_CLEARANCE`).

---

## 8. Roadmap: where you are and what is next

```text
YOU ARE HERE ──► Phase 5 complete enough to run, not complete enough to trust
                 ├── exact replay baseline works
                 ├── broader nearest-neighbor generalization is weak
                 ├── next: stronger state policy
                 └── next: held-out contexts + multiple seeds

AFTER THAT ──► Phase 6: add camera observations (RGB render)
AFTER THAT ──► Phase 7: add language conditioning (real VLA)
```

### What counts as real progress

| Milestone | Evidence |
|-----------|----------|
| Controller works | Reach error < ~2 cm, tests pass |
| Task works | Scripted pickup benchmark mostly succeeds |
| Data is fair | Same trajectories, both label formats |
| Learning loop works | State BC trains, saves, loads, and rolls out |
| Learning is convincing | Policy beats simple replay on held-out trial specs |
| Action-space answer | Same demos/seeds — one format wins repeatably |

### What does NOT count as progress yet

- A reach-policy MP4 looking cool (`train_reach_policy.py` is a toy visualization).
- Calling something a "VLA" before vision and language exist.
- Training on mismatched observations or different demo sets per action space.
- Treating exact replay as proof of robust learning.

---

## 9. When something breaks

Use this checklist before blaming the ML:

```text
1. Physics / scene     → assets/*.xml, object pose, contacts
2. Controller / IK     → clipping, unreachable targets, telemetry
3. Task logic          → success thresholds, scripted phases
4. Labels / adapters   → joint_delta vs ee_delta alignment
5. Policy / training   → only after 1–4 are ruled out
```

**Commands for diagnosis:**

```bash
pytest -q                              # regression check
python scripts/run_reach_demo.py       # controller smoke test
python scripts/run_pickup_trials.py    # task smoke test
```

**Read failure categories** in `pickup_trials.summary.json` — they tell you
which layer likely failed.

---

## 10. Brief checkpoint quiz

Take this after reading the guide once. Keep it short: if you cannot answer
these without looking, revisit the matching section.

1. What is the main research question this repo is trying to test?
2. What does the controller do that a policy should not have to rediscover?
3. What is the difference between `joint_delta` and `ee_delta`?
4. Why does each demo store both `labels` and `policy_labels`?
5. What four things must happen before a pickup counts as a success?
6. What does a `controller_or_ik_failure` tell you to inspect first?
7. Why is exact replay with `k=1` and `search_window=1` not strong evidence of
   generalization?
8. Why should the project still avoid vision/language until state BC is more
   convincing?
9. Which file would you open first to understand the state-BC policy?
10. Which command would you run to verify the full repo still passes tests?

### Self-grade

| Score | Meaning |
|-------|---------|
| 9-10 | You understand the current project boundary |
| 6-8  | Good start; reread the action-space and BC sections |
| 0-5  | Redo Labs 4-7 before moving on |

---

## 11. Answer key

**Stop here if you have not tried the quiz yet.**

<details>
<summary>Click to reveal answers</summary>

1. Whether controller-level actions, especially end-effector deltas, make the
   pickup task easier to learn than raw joint deltas under fair conditions.
2. IK, target tracking, clipping, gripper commands, joint-limit handling, and
   telemetry.
3. `joint_delta` changes arm joints directly; `ee_delta` commands end-effector
   translation/rotation and lets the controller turn that into joint targets.
4. `labels` describe the observed transition after physics; `policy_labels`
   describe executable commands a policy can output during rollout.
5. The gripper must reach the grasp, contact the object, lift it clear of the
   support, and retain it through the hold window.
6. Inspect controller telemetry, clipping, reachability, joint limits, and IK
   target tracking before blaming the learning code.
7. It mostly replays a memorized command sequence in a known task context. That
   verifies plumbing, but it does not show the policy can handle new states.
8. Vision/language would add more failure modes before the state/action/control
   loop is clearly reliable.
9. `src/svla/state_bc.py`
10. `pytest -q`

</details>

---

## Suggested study schedule

| Day | Activity |
|-----|----------|
| 1 | Sections 1–5, Lab 0–1 |
| 2 | Lab 2–4, GUI play session |
| 3 | Lab 5–7, read one demo JSON carefully |
| 4 | Lab 8–9, brief quiz, re-run `pytest` |

After finishing, you should be able to explain to someone else:

1. What question this repo tests.
2. What each layer of the stack does.
3. How to run reach vs pickup vs BC experiments.
4. Where to look when pickup fails.

That is real understanding — even if AI helped write the code.
