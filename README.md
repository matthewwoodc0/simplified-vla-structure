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
- `src/svla/action_spaces.py` defines aligned joint-delta and end-effector-delta
  trajectory labels behind a shared adapter interface.
- `scripts/run_reach_demo.py` runs a headless reaching demo.
- `scripts/render_reach_demo.py` exports a MuJoCo MP4 so the environment is visible.
- `scripts/train_reach_policy.py` trains a tiny numpy-only reach policy and renders it.
- `assets/pickup_scene.xml` adds a small table/object pickup task for controller-only grasp validation.
- `src/svla/pickup_task.py` exposes a reusable pickup task API and runs scripted grasp,
  lift, hold, and failure classification trials.
- `src/svla/demo_recorder.py` records deterministic scripted pickup demos with aligned
  joint-delta and EE-delta labels from the same trajectory.
- `src/svla/state_bc.py` implements a small numpy-only state behavioral-cloning baseline
  with grouped nearest-neighbor and phase-aware MLP policies plus closed-loop MuJoCo
  rollout evaluation.
- `scripts/run_pickup_trials.py` runs the multi-bucket pickup evaluator and writes JSONL logs.
- `scripts/record_pickup_demos.py` writes small local scripted demo JSON files and a manifest.
- `scripts/train_state_bc.py` generates scripted pickup demos, trains joint-delta and
  EE-delta state BC policies, rolls them out in the pickup task, and writes model/eval
  artifacts.
- `scripts/validate_controller_quality.py` checks deterministic replay, continuity,
  local predictability, smoothness, tracking error, and saturation reporting.
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
   final EE pose error, contact, lift, hold retention, and likely failure category. A
   pickup only counts if the object is contacted, lifted clear of the support, and retained
   through the hold window.

4. Scripted pickup demo export:

   ```bash
   python scripts/record_pickup_demos.py
   ```

   This writes deterministic JSON demos to `outputs/scripted_pickup_demos/`. Each demo
   records observations, joint positions, EE pose, gripper commands, controller telemetry,
   contact/lift/retention metrics, and aligned `joint_delta` and `ee_delta` labels for the
   same controller trajectory.

5. State-based behavioral cloning:

   ```bash
   python scripts/train_state_bc.py
   ```

   This writes artifacts under `outputs/state_bc/` by default:

   - `scripted_pickup_demos/`: deterministic local demos,
   - `models/joint_delta_nearest_neighbor_bc.npz`,
   - `models/ee_delta_nearest_neighbor_bc.npz`,
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
future code  = corrected controller action space + vision + language-conditioned VLA inputs
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
    - tests for adapter labels, demo alignment, repeatability, continuity,
      unreachable actions, and posture behavior.
  Latest controller-quality artifact:
    - `outputs/controller_quality_summary.json`,
    - exact repeat deterministic error: 0 for joints, velocities, EE position, and EE
      quaternion,
    - mean tracking error: 0.00308 m,
    - max executed joint step: 0.00247 rad,
    - max executed joint acceleration step: 0.00056 rad,
    - zero Cartesian input clipping and zero joint-limit clipping on the validation stream.
  Remaining caution:
    - the DLS joint-step request still saturates on 92/160 synthetic validation steps,
      but the executed motion is bounded, smooth, deterministic, and explicitly reported.
  Exit condition met:
    - the same scripted trajectory can be represented in joint and EE action spaces,
      direct replay succeeds without hard-limit clipping, and the policy-facing EE path is
      measurable rather than hidden-state-dependent.

Phase 3 - Build the first real manipulation task
  Goal: create a simple table/cube pick-place environment.
  Status: done for pickup, not generalized to arbitrary pick-place.
  Current output:
    - table/object pickup scene,
    - deterministic scripted pickup policy,
    - contact, lift, hold-retention, clipping, and failure-category metrics,
    - `outputs/pickup_trials_controller_quality.jsonl`,
    - final 36-trial benchmark at 36/36 successful pickups.
  Remaining controller/task weakness:
    - contact retention is still a narrow-margin simulator behavior even though the final
      repeated benchmark passed all buckets; keep the failure categories in future runs.

Phase 4 - Dataset generation
  Goal: collect identical demonstrations for all action-space comparisons.
  Status: sample dataset path done.
  Current output:
    - `scripts/record_pickup_demos.py`,
    - `outputs/scripted_pickup_demos/manifest.json`,
    - JSON demos with observations, joint-delta labels, EE-delta labels, controller
      telemetry, and success metrics.
  Exit condition met at sample scale:
    - one deterministic trajectory source can export labels for multiple policy heads fairly.

Phase 5 - State-based behavioral cloning
  Goal: answer the action-space question without vision/language noise.
  Status: implemented, but the corrected EE comparison does not pass the vision gate.
  Current output:
    - `src/svla/state_bc.py`,
    - `scripts/train_state_bc.py`,
    - executable `policy_labels` in scripted demos,
    - joint-delta and EE-delta `.npz` policy artifacts,
    - rollout JSONL logs with contact, lift, retention, clipping, action magnitude,
      smoothness, nearest-neighbor distance, and failure categories.
  Latest local evidence:
    - full tests: 36 passed.
    - controller pickup benchmark: 36/36 successes.
    - demo generation: 30/30 dense-grid scripted demonstrations succeeded and include
      aligned executable labels plus controller/contact telemetry.
    - the earlier `outputs/state_bc_bounded_ee_final_test/` result is historical only: its
      EE labels encoded clipped absolute pose error and saturated on most rollout steps,
      so its apparent 69/72 EE advantage is not readiness evidence.
    - corrected untouched audit artifact:
      `outputs/state_bc_feasible_ee_final_audit/state_bc_summary.json`.
    - final joint-delta: 61/72, 84.7%; per-seed 21/24, 20/24, 20/24.
    - final feasible EE-delta: 42/72, 58.3%; per-seed 14/24, 15/24, 13/24.
    - both policies used the same 30 demos, observations, task contexts, seeds, audit
      starts, rollout limit, gain 1.0, and success metrics.
    - zero numerical controller-failure steps occurred in 144 audit rollouts.
    - EE averaged 779 joint-saturated steps and 39 explicitly infeasible steps per
      rollout; joint delta averaged 130 hard-limit steps per rollout.
  What this proves:
    - the state/demo/action-space/evaluation pipeline runs end to end,
    - joint-delta and EE-delta labels are executable in closed-loop MuJoCo rollouts,
    - failures are categorized instead of hidden behind supervised loss,
    - the bounded controller itself is deterministic and direct feasible-label replay is
      smooth, but learned EE predictions leave the five-DOF arm's feasible local motion
      manifold often enough to trigger saturation and lose pickups,
    - joint delta currently beats corrected EE delta by 19 successes over 72 matched,
      untouched audit rollouts.
  What this does not prove:
    - that the current six-dimensional EE action is a good learning interface for this
      five-joint arm,
    - that adding images would repair the state-policy/controller mismatch,
    - language conditioning is meaningful on a single pickup task.
  Readiness verdict:
    - not ready for Phase 6 vision or language-conditioned VLA training,
    - redesign and validate a lower-dimensional or redundancy-aware controller action,
      then rerun this state-only audit before adding observation complexity.
  Required language gate:
    - add at least two validated behavior variants with instructions that require
      different actions, such as pick, place-to-left, and place-to-right,
    - verify the scripted controller and state BC baseline on every variant first.

Phase 6 - Vision policy
  Goal: add camera observations after the action-space result is measurable.
  Needed work:
    - add RGB rendering,
    - train small CNN/MLP or compact vision encoder policy,
    - keep the same action-space comparison.
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

The next concrete build step remains controller/state-policy work: replace the
overconstrained six-dimensional EE action with a locally feasible lower-dimensional or
hybrid action, validate continuity and saturation again, and rerun the untouched state-BC
comparison. Vision and language remain blocked.

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

Run controller-quality validation:

```bash
python scripts/validate_controller_quality.py
```

Run a small simulation-training loop and render the trained rollout:

```bash
python scripts/train_state_bc.py
```

Run the stricter held-out state-BC check:

```bash
python scripts/train_state_bc.py --output-dir outputs/state_bc_generalization --eval-mode both
python scripts/train_state_bc.py \
  --output-dir outputs/state_bc_feasible_ee_final_audit \
  --policy-type mlp \
  --train-grid dense \
  --eval-mode audit \
  --epochs 160 \
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
