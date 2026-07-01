# Simplified VLA Structure

This project is a controller-first simulation scaffold for testing whether a small
vision-language-action model can learn manipulation more efficiently when it outputs
controller-level actions instead of raw joint actions.

The point is not to build the VLA first. The point is to build a stable action/control
interface first, then compare learning difficulty across action spaces while keeping the
task, observations, dataset, and evaluation loop as identical as possible.

## Core Research Question

Does a controller-level action space make small VLAs more data-efficient and stable than
predicting low-level joint actions?

The working hypothesis is that a policy should learn faster when its output looks like:

```text
[delta_x, delta_y, delta_z, delta_rotation, gripper_open]
```

instead of:

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

because the controller can absorb inverse kinematics, target clipping, smoothing,
joint-limit handling, velocity limits, and some reachability checks. That does not prove
the higher-level action space is always better. The experiment should find where the
abstraction helps, where it hides information the policy needs, and where a hybrid action
space is more honest.

## Current Status

This repo currently has the first simulation/control smoke test:

- `assets/simple_arm.xml` defines a small six-joint MuJoCo arm with a gripper placeholder.
- `src/svla/controller.py` implements damped-least-squares Cartesian IK with target clipping.
- `src/svla/sim.py` wraps MuJoCo in a small simulation API.
- `src/svla/teleop_inputs.py` maps keyboard, mouse/trackpad, and optional gamepad input to
  gripper-local teleop intent.
- `src/svla/teleop_controller.py` integrates that intent into a clipped Cartesian target.
- `src/svla/teleop_workspace.py` defines the conservative reachable target box used by
  manual teleoperation.
- `src/svla/action_spaces.py` defines aligned joint-delta, full EE-delta, and reduced
  tool-axis EE-delta trajectory labels behind a shared adapter interface.
- `scripts/run_reach_demo.py` runs a headless reaching demo.
- `scripts/render_reach_demo.py` exports a MuJoCo MP4 so the environment is visible.
- `scripts/train_reach_policy.py` trains a tiny numpy-only reach policy and renders it.
- `assets/pickup_scene.xml` adds a small table/object pickup task for controller-only grasp validation.
- `src/svla/pickup_task.py` exposes a reusable pickup task API and runs scripted grasp,
  lift, hold, and failure classification trials.
- `src/svla/demo_recorder.py` records deterministic scripted pickup demos with aligned
  joint-delta, full EE-delta, and reduced tool-axis labels from the same trajectory.
- `src/svla/state_bc.py` implements a small numpy-only state behavioral-cloning baseline
  with grouped nearest-neighbor and phase-aware MLP policies plus closed-loop MuJoCo
  rollout evaluation.
- `scripts/run_pickup_trials.py` runs the multi-bucket pickup evaluator and writes JSONL logs.
- `scripts/record_pickup_demos.py` writes small local scripted demo JSON files and a manifest.
- `scripts/train_state_bc.py` generates scripted pickup demos, trains joint-delta and
  `ee_tool_delta` state BC policies, rolls them out in the pickup task, and writes
  model/eval artifacts.
- `scripts/validate_controller_quality.py` checks deterministic replay, continuity,
  local predictability, smoothness, tracking error, and saturation reporting.
- `scripts/validate_action_replay.py` replays both executable action labels across the
  complete controller benchmark and reports saturation rates.
- `scripts/validate_grasp_geometry.py` verifies that the calibrated grasp-center TCP reaches
  every pickup target without touching or moving the object before closure.
- `tests/` verifies controller telemetry, action adapters, pickup evaluation, the reusable
  task API, demo label alignment, executable policy-label replay, controller quality,
  and BC model loading.

This now includes an initial tabletop pickup benchmark. It is still controller-first:
there is no VLA logic, no images, no language, and no PyTorch in the pickup path.

## How To See It

There are several different things you can run or open. They are not the same feature.

1. Interactive MuJoCo window:

   ```bash
   source .venv/bin/activate
   bash scripts/run_mujoco_gui.sh
   ```

   This opens the live simulator window with tool-frame teleoperation:

   ```text
   W/S = forward/back along the gripper local X axis
   A/D = left/right along the gripper local Y axis
   Q/E = up/down along the gripper local Z axis
   mouse or trackpad drag = pitch/yaw
   I/K/J/L/U/O = keyboard pitch/yaw/roll backup
   Space = toggle gripper open/closed
   1-4 = fixed target offsets, N = random target, R = reset, P = pause, H = help
   ```

   Xbox-style and PlayStation-style gamepads are also detected through pygame/SDL when
   connected, but keyboard input is the baseline path to validate first.

   To open the raw native MuJoCo model viewer with no controller script:

   ```bash
   bash scripts/open_raw_mujoco_viewer.sh
   ```

   Do not run `.venv/bin/mjpython` directly from this project path. Because this folder has
   spaces in its name, the generated `.venv/bin/mjpython` script has a broken shebang.
   The shell scripts above bypass that by running the wrapper through `.venv/bin/python`.

2. Rendered controller demo video:

   ```bash
   python scripts/render_reach_demo.py
   open outputs/reach_demo.mp4
   ```

   This shows the deterministic controller moving the end effector to target markers.

3. Controller-only pickup validation:

   ```bash
   python scripts/run_pickup_trials.py
   ```

   This runs 36 deterministic pickup trials by default:

   - 3 gripper yaw buckets,
   - 3 object start poses,
   - 2 approach strategies,
   - 2 repeats per combination.

   Each JSONL record reports object start pose, commanded grasp pose, gripper orientation,
   final EE pose error, pre-close contact/displacement, closure contact, lift, hold retention,
   and likely failure category. A pickup only counts if the open-gripper approach is clean,
   the object is contacted during closure, lifted clear of the support, and retained through
   the hold window.

4. Scripted pickup demo export:

   ```bash
   python scripts/record_pickup_demos.py
   ```

   This writes deterministic JSON demos to `outputs/scripted_pickup_demos/`. Each demo
   records observations, joint positions, EE pose, gripper commands, controller telemetry,
   contact/lift/retention metrics, and aligned `joint_delta`, full `ee_delta`, and reduced
   `ee_tool_delta` labels for the same controller trajectory.

5. State-based behavioral cloning:

   ```bash
   python scripts/train_state_bc.py
   ```

   This writes artifacts under `outputs/state_bc/` by default:

   - `scripted_pickup_demos/`: deterministic local demos,
   - `models/joint_delta_nearest_neighbor_bc.npz`,
   - `models/ee_tool_delta_nearest_neighbor_bc.npz`,
   - `eval/*_policy_trials.jsonl`,
   - `state_bc_summary.json`.

   The default is an exact monotonic nearest-neighbor BC baseline (`k=1`,
   `search_window=1`). It is useful for verifying that the demo schema, action labels,
   action-space execution, and simulator rollout evaluation are wired correctly. It is not
   evidence of generalization.

6. Rendered toy training rollout:

   ```bash
   python scripts/train_reach_policy.py
   open outputs/trained_reach_policy.mp4
   ```

   This shows a tiny supervised reach policy. It is only a proof that a policy can drive
   the controller interface. It is not real RL and not a VLA.

## What Is Actually Being Built

The end project is an experiment platform for comparing robot action spaces:

- joint-space policy output,
- end-effector controller-level policy output,
- later, hybrid or subgoal-level policy output.

The current code is only the bottom of that stack:

```text
current code = MuJoCo arm + deterministic controller + pickup task + demos + state BC audit
future code  = vision observations + multiple behaviors + language-conditioned VLA inputs
```

If the controller layer is wrong, every later ML result is meaningless. That is why the
first build step is low-level and may look underwhelming: it is validating the interface
the model will eventually act through.

## Extended Timeline

```text
Phase 0 - Project framing
  Goal: define the research question and avoid choosing tools blindly.
  Status: done.
  Output: controller-level action-space hypothesis and MuJoCo-first plan.

Phase 1 - Local simulator and controller scaffold
  Goal: make a robot arm move in simulation through a controller-level interface.
  Status: done.
  Current output:
    - MuJoCo arm XML.
    - Cartesian IK controller.
    - headless reach demo.
    - MP4 render scripts.
    - tiny reach-policy visualization.
  What this proves:
    - the sim loads,
    - the controller moves the end effector,
    - the environment can be rendered,
    - a policy-shaped command can drive the controller.
  What this does not prove:
    - grasping,
    - pick/place,
    - real training advantage,
    - VLA usefulness.

Phase 2 - Make the controller experimentally usable
  Goal: turn the controller from a demo into a measurable API.
  Status: done for the current MuJoCo pickup experiment.
  Current output:
    - action-space adapters,
    - controller telemetry in task/demo records,
    - environment-style pickup reset/step/observe/metrics API,
    - bounded stateless EE intention actions tracked over each control interval,
    - joint-step and joint-acceleration bounds,
    - deterministic damped null-space/posture bias,
    - separate joint-limit, joint-step, and joint-acceleration saturation telemetry,
    - explicit saturation, infeasibility, numerical controller-failure, and failure-reason
      fields,
    - feasible EE labels reconstructed from the bounded joint intention instead of clipped
      absolute pose error,
    - a five-DOF `ee_tool_delta` action: world XYZ plus local X/Y tilt while deterministic
      posture control resolves roll about the gripper's local Z axis,
    - tests for adapter labels, demo alignment, repeatability, continuity,
      unreachable actions, and posture behavior.
  Latest controller-quality artifact:
    - `outputs/controller_quality_grasp_tcp_summary.json`,
    - exact repeat deterministic error: 0 for joints, velocities, EE position, and EE
      quaternion,
    - mean tracking error: 0.00308 m,
    - max executed joint step: 0.00205 rad,
    - max executed joint acceleration step: 0.00027 rad,
    - zero Cartesian, joint-step, joint-acceleration, joint-limit, infeasible, or controller
      failure events on the 160-step validation stream.
  Remaining caution:
    - direct reduced-action replay averages 10.8% total saturation, almost entirely bounded
      joint-step clipping; hard-limit/infeasible rates are 0.06% and remain explicit.
  Exit condition met:
    - the same scripted trajectory can be represented in joint and EE action spaces,
      `outputs/action_replay_grasp_tcp_summary.json` reports 18/18 successful direct replay
      and 18/18 collision-free approaches for both action spaces, and the policy-facing EE
      path is measurable rather than hidden-state-dependent.

Phase 3 - Build the first real manipulation task
  Goal: create a simple table/cube pick-place environment.
  Status: done for pickup, not generalized to arbitrary pick-place.
  Current output:
    - table/object pickup scene,
    - deterministic scripted pickup policy,
    - a calibrated grasp-center TCP instead of a controller site on the fixed-jaw tip,
    - separate pre-close and closure-contact telemetry plus object-displacement reporting,
    - contact, lift, hold-retention, clipping, and failure-category metrics,
    - `outputs/grasp_geometry_summary.json`: 36/36 clean approaches, zero pre-close contact
      steps, and zero measurable pre-close object displacement,
    - `outputs/pickup_trials_grasp_tcp.jsonl`,
    - final 36-trial benchmark at 36/36 successful pickups and 36/36 clean approaches.
  Remaining controller/task weakness:
    - contact retention is still a narrow-margin simulator behavior even though the final
      repeated benchmark passed all buckets; keep the failure categories in future runs.

Phase 4 - Dataset generation
  Goal: collect identical demonstrations for all action-space comparisons.
  Status: sample dataset path done.
  Current output:
    - `scripts/record_pickup_demos.py`,
    - `outputs/scripted_pickup_demos/manifest.json`,
    - `svla_pickup_demo_v2_grasp_tcp` JSON demos with observations, joint-delta labels, full
      EE labels, reduced tool-axis labels, controller telemetry, collision-free approach
      metrics, and success metrics.
  Exit condition met at sample scale:
    - one deterministic trajectory source can export labels for multiple policy heads fairly.

Phase 5 - State-based behavioral cloning
  Goal: answer the action-space question without vision/language noise.
  Status: implemented and strong enough to begin the vision phase.
  Current output:
    - `src/svla/state_bc.py`,
    - `scripts/train_state_bc.py`,
    - executable `policy_labels` in scripted demos,
    - joint-delta and reduced tool-axis `.npz` policy artifacts,
    - rollout JSONL logs with contact, lift, retention, clipping, action magnitude,
      smoothness, nearest-neighbor distance, and failure categories.
  Latest local evidence:
    - full tests: 38 passed.
    - controller pickup benchmark: 36/36 successes and 36/36 collision-free approaches.
    - demo generation: 30/30 dense-grid scripted demonstrations succeeded and include
      aligned executable labels plus controller/contact telemetry.
    - the earlier `outputs/state_bc_bounded_ee_final_test/` result is historical only: its
      EE labels encoded clipped absolute pose error and saturated on most rollout steps,
      so its apparent 69/72 EE advantage is not readiness evidence.
    - final untouched artifact: `outputs/state_bc_grasp_tcp_final/state_bc_summary.json`.
    - final joint-delta: 63/72, 87.5%; per-seed 16/24, 23/24, 24/24.
    - final `ee_tool_delta`: 63/72, 87.5%; per-seed 22/24, 21/24, 20/24.
    - both policies used the same 30 demos, observations, task contexts, seeds, final
      starts, 128x128 MLP architecture, 300 epochs, rollout limit, gain 1.0, and success
      metrics.
    - joint-delta had 9 gripper/contact failures and 72/72 collision-free approaches.
    - reduced EE had 5 pre-close grasp-geometry failures, 4 gripper/contact failures, and
      67/72 collision-free approaches; these are policy interpolation failures because the
      scripted and direct-replay EE trajectories are 100% clean.
    - there were zero controller/IK or numerical controller-failure steps.
    - reduced EE averaged 27.3% total joint saturation, 7.0% hard-limit clipping, and 5.5%
      infeasible steps; joint delta averaged 8.7% hard-limit/infeasible steps.
  What this proves:
    - the state/demo/action-space/evaluation pipeline runs end to end,
    - joint-delta and reduced EE labels are executable in closed-loop MuJoCo rollouts,
    - failures are categorized instead of hidden behind supervised loss,
    - the bounded tool-axis controller is deterministic, locally predictable, smooth, and
      free of nominal validation saturation,
    - state BC reaches equal aggregate closed-loop performance at 63/72 for each action
      space, which is not evidence of an action-space advantage.
  Likely residual EE failure mode:
    - learned reduced-EE outputs leave the demonstrated local intention distribution often
      enough to invoke the bounded joint-velocity path more frequently than joint delta;
      five rollouts also drifted into the object before closure despite clean demonstrations.
      This is a policy-generalization weakness, not a hidden controller failure, and it must
      remain visible during Phase 6.
  What this does not prove:
    - that reduced EE actions are universally better than joint actions,
    - that parity will survive image observations,
    - language conditioning is meaningful on a single pickup task.
  Readiness verdict:
    - ready to begin Phase 6 vision experiments while preserving this corrected state
      baseline and the collision-free-approach metric,
    - not ready for language-conditioned VLA training because only one semantic behavior
      is validated.
  Required language gate:
    - add at least two validated behavior variants with instructions that require
      different actions, such as pick, place-to-left, and place-to-right,
    - verify the scripted controller and state BC baseline on every variant first.

Phase 6 - Vision policy
  Goal: add camera observations after the action-space result is measurable.
  Needed work:
    - add fixed-camera RGB observations to the same deterministic demonstrations,
    - freeze the controller, action definitions, train/final starts, seeds, and success
      metrics from Phase 5,
    - treat any pre-close contact or more than 1 mm of pre-close object displacement as a
      failed rollout,
    - train the same small vision encoder capacity for joint delta and `ee_tool_delta`,
    - retain the state MLP as a non-visual upper-bound/debugging baseline,
    - keep the final split untouched until architecture and gain choices are frozen.
  Exit condition:
    - vision policy reproduces the state-based trend or exposes why it changes.

Phase 7 - Language-conditioned VLA
  Goal: make it a real VLA instead of a generic policy.
  Needed work:
    - add multiple tasks or object-conditioned instructions,
    - add language tokens,
    - train compact multimodal policy,
    - compare whether controller-level actions still help.
  Exit condition:
    - language changes behavior across tasks, not just labels a single behavior.

Phase 8 - Unity / Isaac branches, if justified
  Goal: test portability or scale after the MuJoCo result exists.
  Unity ML-Agents:
    - useful as a game-like learnability benchmark.
    - not the main VLA architecture.
  Isaac Sim:
    - useful later for higher-fidelity simulation or NVIDIA workflows.
    - not the first local path on this Mac.
```

The next concrete build step is Phase 6 RGB observations using the frozen tool-axis
controller and state-BC comparison. Language remains blocked until multiple
instruction-distinct behaviors are validated.

## Local Setup

Use the venv already created in this folder:

```bash
source .venv/bin/activate
```

Run the tests:

```bash
pytest
```

Run a headless reaching demo:

```bash
python scripts/run_reach_demo.py
```

Render the physics environment to video:

```bash
python scripts/render_reach_demo.py
```

Generate small scripted pickup demos:

```bash
python scripts/record_pickup_demos.py
```

Run the pickup benchmark:

```bash
python scripts/run_pickup_trials.py --repeats 2
```

Validate grasp-center calibration and collision-free approaches:

```bash
python scripts/validate_grasp_geometry.py --repeats 2
```

Run controller-quality validation:

```bash
python scripts/validate_controller_quality.py
```

Validate executable action-label replay:

```bash
python scripts/validate_action_replay.py
```

Run a small simulation-training loop and render the trained rollout:

```bash
python scripts/train_state_bc.py
```

Run the stricter held-out state-BC check:

```bash
python scripts/train_state_bc.py --output-dir outputs/state_bc_generalization --eval-mode both
python scripts/train_state_bc.py \
  --output-dir outputs/state_bc_grasp_tcp_final \
  --policy-type mlp \
  --train-grid dense \
  --eval-mode final \
  --epochs 300 \
  --hidden-sizes 128 128 \
  --seeds 0 1 2 \
  --joint-action-gain 1.0 \
  --ee-action-gain 1.0
```

Run a small reaching-only toy training loop and render the trained rollout:

```bash
python scripts/train_reach_policy.py
```

If rebuilding the venv from scratch:

```bash
python -m pip install -r requirements-dev.txt
```

If you want MuJoCo's interactive viewer, run this directly from your terminal rather than
through a sandboxed agent session:

```bash
bash scripts/run_mujoco_gui.sh
```

## Simulator Plan

Start with MuJoCo on the local Mac. It is the right first simulator for this project because
it runs locally, has Python bindings, and is enough to validate controller architecture,
action-space adapters, scripted demonstrations, and small behavioral-cloning experiments.

Do not start with Isaac Sim on this machine. Isaac is useful later for higher-fidelity
robotics workflows, synthetic data, and GPU-heavy environments, but it is not the practical
first local path for an Apple Silicon MacBook Air. Treat Isaac as a later Linux/NVIDIA step
after the controller and experiment design are already working.

Unity ML-Agents is a secondary benchmark option, not the main robotics stack. It can be
useful for testing whether a controller-style continuous action space is learnable in a
game-like environment, but it should not replace the MuJoCo controller work or be mistaken
for a VLA architecture.

## Controller Architecture

The intended control stack is:

```text
policy action, 10-20 Hz
  -> action adapter
  -> workspace and magnitude clipping
  -> Cartesian target integration
  -> IK / joint target generation, 100-250 Hz
  -> joint target clipping and smoothing
  -> MuJoCo position actuators / physics step
```

The controller should be responsible for:

- End-effector target integration.
- Damped-least-squares IK.
- Joint-limit clipping.
- Cartesian step-size limits.
- Gripper target handling.
- Reporting when actions were clipped or unreachable.
- Eventually, velocity/acceleration/jerk limiting and null-space posture control.

The policy should not be asked to rediscover basic IK if the experiment is about whether
controller-level action spaces simplify learning.

## Action Spaces To Compare

The first serious experiment should compare action spaces behind a common adapter API:

1. Joint delta baseline:
   ```text
   [delta_joint_1, ..., delta_joint_6, gripper_open]
   ```

2. End-effector delta:
   ```text
   [delta_x, delta_y, delta_z, delta_rotvec_x, delta_rotvec_y, delta_rotvec_z, gripper_open]
   ```

3. Hybrid / redundancy-aware end-effector action:
   ```text
   [delta_xyz, delta_rotvec, elbow_or_nullspace_preference, gripper_open]
   ```

Subgoal actions such as `pick_pose` and `place_pose` are valuable, but they are a
hierarchical-planning experiment, not just another low-level action representation. Do not
mix that into the first comparison unless the basic delta-action result is already clear.

## Planned Engineering Sequence

1. Stabilize the controller scaffold.
   - Add explicit action adapters.
   - Add controller telemetry logs.
   - Add tests for clipping, joint limits, unreachable targets, and repeatability.

2. Build a minimal task environment.
   - Add a table, cube, and reachable pick/place region.
   - Make scripted pick/place work before training any model.
   - Track contacts, object pose, gripper state, and task success.

3. Generate demonstrations.
   - Use one scripted expert trajectory source.
   - Derive joint-delta and end-effector-delta labels from the same trajectories.
   - Save observations, actions, controller status, and task metadata.

4. Train state-based policies first.
   - Use identical observations across action spaces.
   - Start with MLP behavioral cloning.
   - Compare demo counts such as 25, 50, 100, 250, and 500.
   - Run multiple seeds before making claims.

5. Add vision.
   - Add RGB observations only after the state-based action-space result is measurable.
   - Keep action-space comparisons unchanged.

6. Add language last.
   - Add language only once there are multiple tasks or object-conditioned instructions.
   - A one-task "VLA" usually just hides a weak experiment behind a bigger model label.

## Metrics

Track at least:

- Task success rate.
- Demonstrations required to reach a target success rate.
- Variance across random seeds.
- End-effector tracking error.
- Action clipping frequency.
- IK failures or unreachable commands.
- Joint-limit events.
- Collision/contact failures.
- Action smoothness or jerk.

## Near-Term Acceptance Gates

Before training:

- The controller reaches nearby valid end-effector targets within about 1-2 cm.
- No NaNs in joint state, controls, or observations.
- Commands report clipping instead of silently failing.
- Scripted reaching works headlessly.
- Scripted pick/place works before ML is introduced.

Before claiming the controller-level action space helps:

- The same tasks and demonstrations are used across action spaces.
- Observations are identical across policies.
- Evaluation uses multiple random seeds.
- Failures are categorized as policy failure, controller failure, IK infeasibility, or task design failure.

## Development Notes

This project should stay small and inspectable. Do not add a robotics framework, RL stack,
or VLA model until the controller and dataset loop justify it. The first useful result is a
boring, reproducible controller benchmark, not a flashy model demo.
