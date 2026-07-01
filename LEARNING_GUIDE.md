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
> actions are **tool-axis hand-move deltas** (move hand left 2 cm, tilt slightly)
> or **joint deltas** (rotate elbow 0.03 radians)?

A **controller** sits between the policy and the physics simulator. It handles
inverse kinematics (IK): turning "move the hand here" into joint commands.

We build the foundation first so later ML results are trustworthy:

```text
Phase 1  Simulator + controller          DONE
Phase 2  Action-space adapters          DONE
Phase 3  Pickup task + scripted expert  DONE
Phase 4  Recorded demonstrations        DONE (sample scale)
Phase 5  State-based behavioral cloning DONE (MLP baseline, fair A/B)
Phase 6  Vision                         NOT STARTED (next)
Phase 7  Language / VLA                 NOT STARTED (blocked until multi-behavior)
```

**Key idea:** If you skip straight to "train a VLA," you will not know whether
failures came from the model, the controller, the task, or bad data.

**Important honesty check:** Phase 5 is complete enough to begin vision, but it
does **not** answer the research question yet. The repo can load state
observations, train/evaluate `joint_delta` and `ee_tool_delta` MLP policies, and
roll them out in MuJoCo with categorized failures. Latest held-out eval on the
frozen final split: **61/72 joint (84.7%)** vs **60/72 ee_tool_delta (83.3%)**
— essentially a tie, not evidence that either action space wins. All failures
were gripper/contact; controller/IK failures were zero. That is trustworthy
infrastructure evidence, not a VLA-ready research result by itself.

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
pytest -q //The -q makes it quiet, try -v to see more detailed output
```

You should see **37 passed**. If tests fail, fix that before changing code.
The tests are your safety net.

### Important path note //(THIS IS TO DEAL WITH SPACES)

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
│  Same motion, multiple label formats (joint vs ee_tool_delta)│
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
| **ee_tool_delta** | Five-DOF hand-move action: XYZ + local X/Y tilt; roll handled by controller |
| **policy_labels** | Executable commands a policy should output (preferred for BC training) |

---

## 4. Repo map //This is really helpful to see

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
│   ├── action_spaces.py     # joint_delta, ee_delta, ee_tool_delta adapters
│   ├── pickup_task.py       # Pickup environment + scripted expert
│   ├── demo_recorder.py     # Records demos with aligned labels
│   ├── state_bc.py          # Phase 5: MLP + nearest-neighbor BC training/eval
│   ├── teleop_*.py          # Keyboard/gamepad GUI teleoperation
│   └── ...
├── scripts/                 # Runnable entry points
│   ├── run_reach_demo.py
│   ├── inspect_clipping.py       # Lab 2: policy vs demo clipping numbers
│   ├── render_clipping_demo.py   # Lab 2: clipping comparison MP4
│   ├── render_reach_demo.py
│   ├── render_bc_rollout.py      # Lab 7: learned policy rollout MP4
│   ├── run_pickup_trials.py
│   ├── record_pickup_demos.py
│   ├── train_state_bc.py
│   ├── validate_controller_quality.py
│   ├── validate_action_replay.py
│   ├── train_reach_policy.py
│   └── open_mujoco_gui.py
├── tests/                   # Automated checks (read these as examples!)
│   ├── test_controller_quality.py
│   └── ...
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
- `ControllerLimits` — layered per-tick limits (`max_step_xyz`,
  `max_target_lag_xyz`, joint step/accel caps).
- `CartesianIKController` — reads current EE pose, computes joint targets via
  damped least-squares IK, clips to joint limits, stores `last_telemetry`.
- `apply_delta()` — integrates small hand-move intentions into a persistent
  target; used by the `ee_tool_delta` policy path.
- `tool_axis` orientation mode — policy controls XYZ + local X/Y tilt; roll about
  the gripper Z axis is resolved by deterministic posture bias.

**Read the docstring** on `CartesianIKController` — it explains the separation
between teleop (human sets targets) and IK (controller tracks targets).

### `src/svla/sim.py`

`ArmSim` class: load model, reset arm, `step(command)`, `move_to(xyz)`.
This is the smallest "hello world" interface to the simulator.

### `src/svla/action_spaces.py` //(MUST USE SAME TRAJECTORIES, SO TRANSLATE ONE RECORDING INTO THREE LABELS)

`JointDeltaActionAdapter`, `EndEffectorDeltaActionAdapter`, and
`ToolAxisEndEffectorDeltaActionAdapter` turn a before/after robot state into
training labels. `label_transition_all()` returns all three formats for the same
transition. Phase 5 training compares **`joint_delta`** vs **`ee_tool_delta`**
(the five-DOF tool-axis hand-move format). Full `ee_delta` is kept for analysis
but is not the policy-facing EE path anymore.

### `src/svla/pickup_task.py`

The first real manipulation task.

- `PickupTaskEvaluator` — reset, step, observe, measure success.
- `step_joint_delta_action()` — joint baseline path (no hand-move middle layer).
- `step_ee_tool_delta_action()` — tool-axis hand-move path through the controller.
- Scripted expert trajectory (not learned).
- Success requires: contact → lift off table → hold without dropping.
- `default_trial_specs()` — 36 trials (3 yaw × 3 object poses × 2 approaches
  × 2 repeats).

### `src/svla/demo_recorder.py`

Runs the scripted expert and saves JSON with per-step observations, controller
telemetry, and **both** action label formats.

### `src/svla/state_bc.py`

Phase 5 learning code. Loads demo JSON, trains **MLP behavioral-cloning**
policies (numpy-only, no PyTorch) or a nearest-neighbor sanity baseline, rolls
them out in the pickup task, and compares `joint_delta` vs `ee_tool_delta`.

Key classes:

- `MLPBCPolicy` — learned small neural net; **deterministic** at inference (same
  state in → same action out).
- `NearestNeighborBCPolicy` — wiring/replay check; useful but not the main result.
- `rollout_policy()` — closed-loop eval with telemetry and failure categories.
- `fit_mlp_policy()` — supervised training from demonstration actions.

Be precise about what this baseline proves:

- The state-BC loop runs end to end for both action spaces.
- Learned MLP policies reach ~84% success on the frozen final eval split.
- Joint vs EE is essentially a tie (61/72 vs 60/72); neither has won yet.
- Failures are categorized as gripper/contact, not hidden controller bugs.
- It does **not** prove vision or language will behave the same way.

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
2. **Run** — execute the command.
3. **Watch** — open the MP4 or GUI when the lab provides one (from Lab 2 onward).
4. **Observe** — compare video/numbers to your prediction.
5. **Revert** — undo any code edits (or keep a note in a scratch branch).

**Watching results:** Most labs write MP4s under `outputs/`. On macOS:

```bash
open outputs/clipping_demo.mp4
```

Numbers tell you *that* something happened; video tells you *what* it looked like.

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

Typical errors are **~8–12 mm**. That is normal: `position_tolerance` in
`ControllerLimits` is **0.008 m (8 mm)**.

**Questions to answer in your notes:**

- What unit is `error` in? (meters) //ERROR was in meters
- Is sub-2 cm error good enough for a smoke test? //(yes, for reach — not for precision assembly)
- Which controller API does this script use? (`move_to` → repeated `move_toward`, **not** `apply_delta`)

//Question: Why would the model... be wrong??? Like we gave it perfect data so why would it be off? In a simulated enviornemnt shouldnt it work perfectly

**Optional video:**

```bash
python scripts/render_reach_demo.py
open outputs/reach_demo.mp4
```

The arm looks slow because each frame is many small IK steps plus physics substeps.
That is rate limiting (`max_joint_step`, `max_joint_accel_step`), not broken physics.
//Video showed the robot moving so slowly... why was that lol shouldnt the robot be moving faster?

---

### Lab 1 — Change where the arm reaches

**File:** `scripts/run_reach_demo.py`

Find `TARGET_DELTAS` near the top. It is a tuple of 3D offsets added to the
current hand position to form absolute reach goals.

**Predict:** If you only change the **first** delta, which target lines in the
output should change? (Only target 1.)

**Experiment A — bigger first move:**

Change the first delta from `[-0.04, 0.05, 0.03]` to `[-0.10, 0.12, 0.08]`.

```bash
python scripts/run_reach_demo.py
```

Expect target 1 error to increase (farther goals need more iterations and may
stall before tolerance). Targets 2–4 should be nearly unchanged because their
deltas were not edited.
//ERror increased substantially... interesting it really did go up a ton but then fortarget 3 and 4 it didnt really change just first two
//i felt like the robot moved slightly faster

**Experiment B — tiny first move:**

Change the first delta to `[-0.005, 0.005, 0.005]`.

Expect target 1 error to drop — easy local reach.
//error actually went really low

**Watch after Experiment A** (optional but recommended):

```bash
python scripts/render_reach_demo.py --output outputs/reach_demo_lab1.mp4
open outputs/reach_demo_lab1.mp4
```

You should see the arm crawl toward the first (larger) target. Only the first
segment of motion reflects your edit — targets 2–4 use the original deltas.

**Revert** `TARGET_DELTAS` when done.

**What you learned:** The controller works locally. Huge jumps may clip or
stall; tiny moves converge easily.
//maybe it actually didnt change the speed at all..
//okay wait so since i only modified first target delta... it only changed
//error for the first bit okay that makes sense
//and target  deltas only changed commanded ee goal positions
//made it further away when larger, which meant we needed more timesteps
//this often led to larger final error

`run_reach_demo` does **not** exercise the policy-facing `apply_delta` path or
huge action clipping (that is Lab 2).

---

### Lab 2 — See clipping in action

**Goal:** Understand layered clipping on the **policy path** (`apply_delta`), and
why `run_reach_demo` is the wrong place to look for it.

#### Two paths (read before running anything)

| Path | Used by | Command shape | What limits matter |
|------|---------|---------------|-------------------|
| `move_to` | `run_reach_demo`, scripted pickup | Absolute XYZ goal | Iterates until error ≤ tolerance |
| `apply_delta` | `ArmSim.step`, Lab 2 test | Per-step delta + integrated target | `max_step_xyz`, `max_target_lag_xyz`, joint rate limits |
| `step_ee_delta_action` | Pickup `ee_tool_delta` rollouts | Per-step delta from current EE | `max_step_xyz`, `max_step_rot`, joint rate limits |

The pytest smoke test in `tests/test_controller.py` only checks that a huge
`apply_delta` command gets clipped and state stays finite. **Passing does not
show you what happened.**

#### Step 1 — Read the test

Open `test_delta_command_is_clipped_and_keeps_state_finite`. Note it sends
`[1.0, 1.0, 1.0]` m (norm ≈ 1.73 m) in one step.

```bash
pytest tests/test_controller.py::test_delta_command_is_clipped_and_keeps_state_finite -v
```

//it tells me that test passed but i cant really see what happened...

Green means: clipping fired somewhere, no NaNs. Nothing more.

#### Step 2 — Inspect both paths (default limits)

```bash
python scripts/inspect_clipping.py
```

You should see something like:

```text
Policy path ... target ahead of EE: 0.0250 m
Demo path   ... final tracking error: 0.0080 m
```

**Predict before Step 3:** If you loosen only `max_step_xyz` to `0.5`, will the
policy-path `target ahead of EE` change? (No — `max_target_lag_xyz` is still
0.025.)

#### Step 3 — Loosen one rail

**File:** `src/svla/controller.py` — or use the inspect script flag below (no
file edit required).

Find `ControllerLimits` and change `max_step_xyz` from `0.025` to `0.5`.
// okay i modified
//honestly.. test still passed and it worked fine

```bash
python scripts/inspect_clipping.py --max-step-xyz 0.5
```

Policy path output should be **identical** to Step 2. The command is still
clipped (1.73 m > 0.5 m), and `max_target_lag_xyz=0.025` still caps how far
the tracking target can run ahead of the actual EE each step.

Re-run the pytest. It should still pass (`clipped_translation` remains `True`).

#### Step 4 — Loosen both Cartesian rails

```bash
python scripts/inspect_clipping.py --max-step-xyz 0.5 --max-target-lag-xyz 0.5 --policy-steps 5
```

Now `target ahead of EE` should jump to **~0.5 m**. The EE still moves slowly
(`joint_step_norm` capped by `max_joint_step` / `max_joint_accel_step`). The
**ghost target** races ahead; physical motion does not.

#### Step 5 — Confirm the reach demo is unaffected

```bash
python scripts/run_reach_demo.py
```

You should still see ~8–12 mm errors. Loosening limits on the policy path does
not change a demo that uses small absolute goals via `move_to`.

#### Step 6 — Watch the clipping comparison video

This is the main payoff — you will *see* the ghost target race ahead in clip 2.

```bash
python scripts/render_clipping_demo.py
open outputs/clipping_demo.mp4
```

The MP4 has two back-to-back clips:

| Clip | Limits | What to look for |
|------|--------|------------------|
| 1/2 | Default (0.025 m) | Green mocap ball stays near the arm; slow creep |
| 2/2 | Loose (0.5 m) | Mocap ball pulls far ahead; arm still moves slowly |

The on-screen text shows `target ahead` (mm) and `ee moved` (mm). In clip 2 the
gap between those numbers is the visual version of the telemetry from Steps 2–4.

**Do not leave controller limits modified.** If you edited `ControllerLimits` in
`controller.py` during experiments, restore defaults:

```python
max_step_xyz: float = 0.025
max_target_lag_xyz: float = 0.025
```

**What you learned:** Clipping is intentional. Policies that output large
actions get softened by the controller — and that affects learning. It is layered
and mostly invisible unless you read telemetry on `apply_delta` or watch
`clipping_demo.mp4`. The gap between **commanded intention** and **executed
motion** matters for learning.

---

### Lab 3 — Drive the arm yourself (GUI)

**Goal:** Connect human input → teleop → controller → physics.

**Requires a graphical window** (macOS desktop). If the GUI cannot open, skip
to reading the teleop files listed at the bottom — the architecture is the same.

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
| 1–4 | Jump to preset reach offsets (same family as `TARGET_DELTAS`) |
| R | Reset arm + teleop target |
| P | Pause/resume IK tracking |
| H | Print help in the terminal |

**Experiment:**

1. Press `R` to reset.
2. Press `1` — teleop target jumps to a preset offset; controller tracks it.
3. Hold `W` briefly. Motion should feel smooth (small per-step limits), not instant.
4. Press `P` to pause IK. Press `W` again — the **target** still moves in
   teleop state, but the arm stops tracking until you press `P` again.

**Watch:** The MuJoCo window *is* the video for this lab — you are seeing
teleop → controller → physics live. There is no separate MP4. Focus on:

- Smooth vs jerky motion when holding `W`/`Q`
- The arm freezing while paused (`P`) even as you keep pressing keys
- Preset `1` jumping the target, then the arm catching up

**Predict:** Is teleop integrating deltas into a persistent target (like
`apply_delta`) or teleporting the arm directly? (Integrating targets — same
pattern as policies.)

//it is just moving the target and the model is trying to find it on itself 

**Files to peek at after playing:**

- `scripts/open_mujoco_gui.py` — main loop
- `src/svla/teleop_controller.py` — integrates input into targets
- `src/svla/teleop_workspace.py` — reachable workspace box

**What you learned:** Teleop sets *targets*; the controller *tracks* them. That
is the same boundary a learned `ee_tool_delta` policy will use.

---

### Lab 4 — Run the pickup benchmark

**Goal:** See the scripted expert pass/fail across controlled trial conditions.

**Quick smoke run (~30 s):**

```bash
python scripts/run_pickup_trials.py --repeats 1 --stop-after 6
```

Each printed line shows `success=`, `failure=`, contact/lift/retain flags, and
EE errors. All six should show `success=1 failure=none` on a healthy checkout.

**Full benchmark (~2–3 min):**

```bash
python scripts/run_pickup_trials.py --repeats 2
```

This runs **36 trials** (3 yaw × 3 object poses × 2 approaches × 2 repeats).
Expect `success_rate: 1.0` and `failure_categories.none: 36` in the summary.

Check the summary:

```bash
cat outputs/pickup_trials.summary.json
```

Look for `success_rate`, `failure_categories`, and breakdowns by `by_orientation`,
`by_object_pose`, and `by_approach`.

**Watch — full scripted expert showcase (~1 min to render):**

```bash
python scripts/render_pickup_showcase.py
open outputs/pickup_showcase.mp4
```

Three clips (trials 1, 8, 18) show approach → grasp → close → lift → hold with
on-screen contact/lift/hold metrics. This is the scripted expert with **no ML**.

**Quick watch — single trial only:**

```bash
python scripts/render_pickup_showcase.py --trial-id 1 --output outputs/pickup_demo_01.mp4
open outputs/pickup_demo_01.mp4
```

**What you learned:** Pickup success is measurable and decomposed. A green
benchmark means the scripted expert + controller + scene are aligned. Keep
reading `failure_categories` in future runs — contact retention is
narrow-margin even when everything passes.

---

### Lab 5 — Inspect one recorded demonstration

**Goal:** See how one trajectory becomes training rows with aligned labels.

Demos are **not** checked into `outputs/` by default — generate a small set first
(~30 s):

```bash
python scripts/record_pickup_demos.py --count 1
```

You should see:

```text
demo=pickup_demo_01_yaw_-18_center_vertical_pregrasp.json success=1 ...
```

**Peek without opening the whole file** (it has ~1,800+ sample rows):

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/scripted_pickup_demos/pickup_demo_01_yaw_-18_center_vertical_pregrasp.json")
demo = json.loads(path.read_text())
sample = demo["samples"][0]
print("format:", demo["format"])
print("phases:", [p["phase"] for p in demo["phase_summaries"]])
print("policy_labels.joint_delta len:", len(sample["policy_labels"]["joint_delta"]))
print("policy_labels.ee_tool_delta len:", len(sample["policy_labels"]["ee_tool_delta"]))
tel = sample["controller_telemetry"]
print("telemetry clip flags:", {k: tel[k] for k in tel if "clip" in k or k == "saturated"})
PY
```

**In your editor**, search within the same JSON for:

1. `"format"` — schema version (`svla_pickup_demo_v1`)
2. `"phase_summaries"` — approach, grasp, lift, hold phases
3. `"samples"` → first entry → `"policy_labels"` — `joint_delta` and
   `ee_tool_delta`
4. `"controller_telemetry"` — target vs actual, clipping / saturation fields

**Compare label sizes:**

- `joint_delta` has **6** numbers (5 arm joints + gripper)
- `ee_tool_delta` has **6** numbers (xyz + local X/Y tilt + gripper)
- `ee_delta` has **7** numbers (xyz + full rotvec + gripper) — analysis only

There are two label fields per sample:

- `labels` — observed transition after physics ("what actually happened").
- `policy_labels` — executable commands from controller targets/telemetry
  ("what a policy should output").

For Phase 5 training, `policy_labels` are the default. Raw `labels` can include
physics lag and contact effects that are not directly executable.

**Watch — same trial you just recorded:**

```bash
python scripts/render_pickup_showcase.py --trial-id 1 --output outputs/pickup_demo_01.mp4
open outputs/pickup_demo_01.mp4
```

Play this side-by-side with your JSON peek. When the overlay says `PHASE grasp_align`,
search the JSON for `"phase": "grasp_align"` and read that row's `policy_labels`.
You are connecting **motion on screen** to **training numbers in the file**.

**What you learned:** One trajectory produces multiple label formats from the
same motion. That alignment is what makes the joint vs `ee_tool_delta` comparison fair.

---


//HAVE NOT COMPLETED ANYTHING BELOW THIS 

### Lab 6 — Action adapters in isolation

**File:** `tests/test_action_spaces.py`

These tests use fake before/after states — **no simulator, no clipping**.

```bash
pytest tests/test_action_spaces.py -v
```

**Experiment — predict `joint_delta` by hand:**

In `test_joint_delta_adapter_labels_transition_with_gripper_command`:

```text
before joints = [0.0, 0.1, 0.2, 0.3, 0.4]
after  joints = [0.1, 0.0, 0.25, 0.35, 0.2]
gripper command = 0.25

joint_delta (first 5) = after - before
                      = [0.1, -0.1, 0.05, 0.05, -0.2]
last element          = gripper command = 0.25
```

Run the test. If it passes without edits, your manual math matches the adapter.

**Optional:** Read `test_tool_axis_adapter_omits_local_z_roll` — note that
`ee_tool_delta` drops local Z roll (controller resolves it).

**Watch (no new render):** This lab has no simulator. Re-open the Lab 5 video and
pick one phase transition — the adapter math you did by hand is what produced
the `policy_labels` numbers driving that motion.

```bash
open outputs/pickup_demo_01.mp4
```

**What you learned:** Action adapters are encoding rules only. Physics does not
care which format you train on — the *learner* does.

---

### Lab 7 — Phase 5 behavioral cloning (quick MLP run)

**Goal:** Run the full train → save → rollout loop on a **small budget**.

```bash
python scripts/train_state_bc.py --demo-repeats 1 --eval-repeats 1 --stride 10
```

This takes **~2–3 minutes**. Default flags worth knowing: `--policy-type mlp`,
`--label-source policy_labels`, and both `joint_delta` + `ee_tool_delta` train
automatically.

Writes to `outputs/state_bc/`:

- `models/joint_delta_mlp_bc.npz`
- `models/ee_tool_delta_mlp_bc.npz`
- `eval/*_policy_trials.jsonl`
- `state_bc_summary.json`

**Read `outputs/state_bc/state_bc_summary.json`.** Compare `by_action_space`
success rates and `failure_categories`.

**Expect low success on this quick run** (often ~15–25% overall). That is normal:
`--stride 10` subsamples demos and `--eval-repeats 1` uses a thin eval grid.
You are verifying the **plumbing**, not reproducing the frozen baseline score.

The command uses executable `policy_labels`, not raw `labels`.

**Watch — learned policy vs scripted expert:**

After training finishes, render rollouts for both action spaces (uses the models
you just trained):

```bash
python scripts/render_bc_rollout.py \
  --policy outputs/state_bc/models/joint_delta_mlp_bc.npz \
  --trial-id 1
python scripts/render_bc_rollout.py \
  --policy outputs/state_bc/models/ee_tool_delta_mlp_bc.npz \
  --trial-id 1
open outputs/state_bc/joint_delta_rollout_trial01.mp4
open outputs/state_bc/ee_tool_delta_rollout_trial01.mp4
```

Compare to the scripted expert from Lab 4/5. The quick-budget policy may fail
(`success: false` in the terminal) — that is fine. Watch *how* it fails: missed
grasp, weak contact, drop during lift. Those match `failure_categories` in the
JSON summary.

**Experiment — nearest-neighbor wiring check (~1 min):**

```bash
python scripts/train_state_bc.py --policy-type nearest --k 1 --search-window 1 --demo-repeats 1 --eval-repeats 1
```

High success here mostly verifies demo replay plumbing, not robust generalization.

**Experiment — stricter eval (matches the frozen Phase 5 baseline, ~10+ min):**

```bash
python scripts/train_state_bc.py \
  --output-dir outputs/state_bc_learning_guide \
  --policy-type mlp \
  --train-grid dense \
  --eval-mode final \
  --epochs 300 \
  --hidden-sizes 128 128 \
  --seeds 0 1 2 \
  --joint-action-gain 1.0 \
  --ee-action-gain 1.0
```

Expect on the order of **~84%** success for both action spaces — a tie, not a
winner. Check that failures are `gripper_or_contact_model_failure`, not
controller/IK failures.

**What you learned:** The ML loop exists and the fair A/B comparison runs. A
fast run proves wiring; the stricter run approximates the real baseline. Neither
shows that hand-move actions win yet.

---

### Lab 7b — Controller and action-replay validation

**Goal:** Confirm the middle layer and both action adapters are trustworthy
before blaming the policy.

```bash
python scripts/validate_controller_quality.py
python scripts/validate_action_replay.py --repeats 1
```

The second command takes **~30–60 s** with `--repeats 1` (longer at default
`--repeats 2`). Both should end with `"pass": true` in the printed JSON.

Check:

- `outputs/controller_quality_summary.json` — deterministic repeatability,
  joint-step/accel within thresholds, nearby EE tracking within ~2 cm
- `outputs/action_replay_tool_axis_summary.json` — direct replay successes for
  both `joint_delta` and `ee_tool_delta` (18/18 at `--repeats 2`, 9/9 at
  `--repeats 1`)

**Watch — refresh your mental model:**

Re-watch the scripted expert and (if you did Lab 7) the learned rollout. Replay
passing means the **commands in the demos are executable**; a sloppy learned
policy video means the **model** is wrong, not the controller plumbing.

```bash
open outputs/pickup_demo_01.mp4
open outputs/state_bc/ee_tool_delta_rollout_trial01.mp4
```

**What you learned:** If replay passes but the learned policy fails, suspect
policy/contact behavior first — not broken IK.

---

### Lab 8 — Break a test on purpose, then fix it

**Goal:** Practice the debug loop researchers actually use.

**File:** `tests/test_controller.py`

In `test_controller_reaches_nearby_target`, change the assertion from
`error < 0.02` to `error < 0.001` (unrealistically strict for this arm).

```bash
pytest tests/test_controller.py::test_controller_reaches_nearby_target -v
```

It should fail with something like `0.008 < 0.001`. Read the failure output:
actual error vs your bound. The controller typically lands near **8 mm**, not
sub-millimeter.

**Fix:** Restore `0.02` and confirm green.

**Watch — acceptance criteria in motion:**

```bash
python scripts/render_reach_demo.py --output outputs/reach_demo_acceptance.mp4
open outputs/reach_demo_acceptance.mp4
```

The ~8 mm tracking error you see here is exactly what `error < 0.02` encodes.
Your broken `0.001` assertion was asking for sub-millimeter reach — the video
shows why that is unrealistic for this scaffold.

**What you learned:** Tests encode acceptance criteria. Tightening them is how
you formalize "good enough" vs "broken."

---

### Lab 9 — Trace one function call chain (read-only)

**Goal:** See that scripts are thin wrappers over library code.

#### Chain A — pickup benchmark (`run_pickup_trials.py`)

```text
run_pickup_trials.py
  → PickupTaskEvaluator.run_trial()          (pickup_task.py)
    → _move_to_pose()  [loop per waypoint]
      → CartesianIKController.move_toward()  (controller.py)
        → mujoco.mj_step()
```

`run_trial` inlines the scripted trajectory (approach → grasp → close → lift →
hold). It does **not** call `apply_delta`.

#### Chain B — demo recording + BC labels (`record_pickup_demos.py`)

```text
record_pickup_demos.py
  → PickupDemoRecorder.record_trial()        (demo_recorder.py)
    → scripted_controller_commands()
    → step_controller_command()
      → CartesianIKController.move_toward()
        → mujoco.mj_step()
    → label_transition_all() + policy_labels  (action_spaces.py)
```

#### Chain C — learned policy rollout (`train_state_bc.py` eval)

```text
train_state_bc.py
  → rollout_policy()                         (state_bc.py)
    → step_joint_delta_action()              (joint path)
    → step_ee_tool_delta_action()            (ee_tool_delta path)
         → step_ee_delta_action()           (pickup_task.py)
           → clip delta → move_toward()      (controller.py)
        → mujoco.mj_step()
```

Note: `ArmSim.step()` in Lab 2 calls `apply_delta()`, which also integrates a
persistent target and applies `max_target_lag_xyz`. Pickup rollouts use
`step_ee_delta_action()` — same clipping idea, slightly different target update.

Draw these on paper. Mark where **clipping telemetry** appears (Chains B and C).

**Watch — map clips to chains:**

| Video | Chain |
|-------|-------|
| `outputs/clipping_demo.mp4` | Policy `apply_delta` path (Chain C cousin) |
| `outputs/pickup_demo_01.mp4` | Demo recording (Chain B) |
| `outputs/pickup_trials` headless run | Benchmark (Chain A) — no video, numbers only |
| `outputs/state_bc/*_rollout*.mp4` | Learned policy (Chain C) |

Re-watch one clip and point at the frame where each box in your diagram is active.

**What you learned:** Same robot, three entry points. Learned policies emit
deltas; scripted benchmark/demo code mostly commands absolute poses via
`move_toward`.

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
| `labels.joint_delta` | Observed joint transition after physics |
| `labels.ee_tool_delta` | Observed tool-axis hand-move transition |
| `labels.ee_delta` | Observed full EE transition (analysis only) |
| `policy_labels.joint_delta` | Executable joint-delta command for BC |
| `policy_labels.ee_tool_delta` | Executable tool-axis command for BC (Phase 5 EE path) |
| `policy_labels.ee_delta` | Executable full EE command (not used for Phase 5 training) |
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
YOU ARE HERE ──► Phase 5 DONE — ready for Phase 6 vision
                 ├── controller validation: deterministic, measurable
                 ├── action replay: joint + ee_tool_delta 18/18
                 ├── scripted pickup benchmark: 36/36
                 ├── state MLP BC: joint 61/72 vs ee_tool_delta 60/72 (tie)
                 └── failures: gripper/contact only; controller/IK: 0

NEXT ──► Phase 6: fixed-camera RGB + vision BC (keep state MLP as baseline)
THEN ──► Phase 7: multiple instruction-distinct behaviors + compact VLA
         (language blocked until pick/place variants validate on scripted + BC)
```

### What counts as real progress

| Milestone | Evidence |
|-----------|----------|
| Controller works | Reach error < ~2 cm, quality validation passes |
| Task works | Scripted pickup benchmark 36/36 |
| Data is fair | Same trajectories, aligned `policy_labels` for both action spaces |
| Learning loop works | MLP BC trains, saves, loads, rolls out with failure categories |
| State BC baseline | Held-out final eval with multiple seeds completes |
| Action-space answer | Demo-count curves or repeatable win on held-out eval — **not done yet** |
| Vision answer | Same fair comparison with RGB — Phase 6 |
| Language VLA | Multiple behaviors + instructions — Phase 7 |

### What does NOT count as progress yet

- A reach-policy MP4 looking cool (`train_reach_policy.py` is a toy visualization).
- Calling something a "VLA" before vision and multiple behaviors exist.
- Training on mismatched observations or different demo sets per action space.
- Claiming `ee_delta` results from old bounded-EE experiments (use `ee_tool_delta`).
- A 1-success gap (61 vs 60) as proof that joint actions win.

---

## 9. When something breaks

Use this checklist before blaming the ML:

```text
1. Physics / scene     → assets/*.xml, object pose, contacts
2. Controller / IK     → clipping, unreachable targets, telemetry
3. Task logic          → success thresholds, scripted phases
4. Labels / adapters   → joint_delta vs ee_tool_delta alignment
5. Policy / training   → only after 1–4 are ruled out
```

**Commands for diagnosis:**

```bash
pytest -q                                    # regression check (37 tests)
python scripts/run_reach_demo.py             # controller smoke test
python scripts/run_pickup_trials.py          # task smoke test
python scripts/validate_controller_quality.py
python scripts/validate_action_replay.py
python scripts/train_state_bc.py --eval-mode final  # BC smoke test
```

**Read failure categories** in `pickup_trials.summary.json` — they tell you
which layer likely failed.

---

## 10. Brief checkpoint quiz

Take this after reading the guide once. Keep it short: if you cannot answer
these without looking, revisit the matching section.

1. What is the main research question this repo is trying to test?
2. What does the controller do that a policy should not have to rediscover?
3. What is the difference between `joint_delta` and `ee_tool_delta`?
4. Why does each demo store both `labels` and `policy_labels`?
5. What four things must happen before a pickup counts as a success?
6. What does a `controller_or_ik_failure` tell you to inspect first?
7. Why is exact replay with `k=1` and `search_window=1` not strong evidence of
   generalization?
8. Why is vision the next phase but language still blocked?
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

1. Whether controller-level actions, especially `ee_tool_delta`, make the pickup
   task easier to learn than raw joint deltas under fair conditions.
2. IK, target tracking, clipping, gripper commands, joint-limit handling, and
   telemetry.
3. `joint_delta` changes arm joints directly; `ee_tool_delta` commands small
   hand moves (XYZ + local X/Y tilt) and lets the controller resolve roll and
   joint targets.
4. `labels` describe the observed transition after physics; `policy_labels`
   describe executable commands a policy can output during rollout.
5. The gripper must reach the grasp, contact the object, lift it clear of the
   support, and retain it through the hold window.
6. Inspect controller telemetry, clipping, reachability, joint limits, and IK
   target tracking before blaming the learning code.
7. It mostly replays a memorized command sequence in a known task context. That
   verifies plumbing, but it does not show the policy can handle new states.
8. State BC is trustworthy enough for vision now; language needs multiple
   instruction-distinct behaviors first or it would just relabel one pickup task.
9. `src/svla/state_bc.py` — start with `MLPBCPolicy` and `rollout_policy()`
10. `pytest -q` (expect 37 passed)

</details>

---

## Suggested study schedule

| Day | Activity |
|-----|----------|
| 1 | Sections 1–5, Lab 0–1 |
| 2 | Lab 2–4, GUI play session |
| 3 | Lab 5–7b, read one demo JSON carefully |
| 4 | Lab 8–9, brief quiz, re-run `pytest` |

After finishing, you should be able to explain to someone else:

1. What question this repo tests.
2. What each layer of the stack does.
3. How to run reach vs pickup vs BC experiments.
4. Where to look when pickup fails.

That is real understanding — even if AI helped write the code.
