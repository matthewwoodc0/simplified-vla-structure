# Agent Notes

Be critical of weak assumptions in this repo. The user wants a thinking partner, not a
rubber stamp. If a plan mixes controller bugs, simulator limitations, and ML failure modes,
separate them before implementing.

## Project Intent

This repo is for a controller-first Robot VLA experiment. The core question is whether a
small VLA or imitation policy learns more efficiently from controller-level actions than
from low-level joint actions.

Keep the order of operations strict:

1. Controller works.
2. Task environment works.
3. Scripted demonstrations work.
4. State-based behavioral cloning works.
5. Vision is added.
6. Language/VLA complexity is added.

Do not jump straight to VLA training. That would hide whether the controller/action-space
idea is actually working.

## Current Implementation State

Known scaffold:

- `assets/simple_arm.xml`: simple MuJoCo 6-joint arm with gripper placeholder.
- `src/svla/controller.py`: damped-least-squares Cartesian IK controller.
- `src/svla/sim.py`: small MuJoCo wrapper.
- `scripts/open_mujoco_gui.py`: live MuJoCo GUI launcher with keyboard target controls.
- `scripts/run_reach_demo.py`: headless reaching demo plus optional viewer.
- `scripts/render_reach_demo.py`: offscreen MP4 export of the physics environment.
- `scripts/train_reach_policy.py`: numpy-only toy reach-policy training and rollout render.
- `tests/test_controller.py`: basic controller smoke tests.

The current controller is a bring-up scaffold. It reaches nearby relative targets at roughly
1-2 cm error in the current tests/demo. It is not yet validated for table pick/place.

The user explicitly asked for a "YOU ARE HERE" roadmap after feeling the work was drifting.
Keep future work anchored to this sequence:

1. Phase 1 is current: MuJoCo arm, controller, visual smoke tests.
2. Phase 2 next: action-space adapters and telemetry.
3. Phase 3 after that: table/cube manipulation task and scripted pick/place.
4. Phase 4: demonstrations with equivalent joint and EE labels.
5. Phase 5: state-based behavioral cloning comparison.
6. Phase 6: vision.
7. Phase 7: language-conditioned VLA.

Do not present the current toy reach-policy video as meaningful training evidence. It is
only a visualization that a policy-shaped command can drive the controller.

## Commands

Use the existing venv:

```bash
source .venv/bin/activate
pytest
bash scripts/run_mujoco_gui.sh
python scripts/run_reach_demo.py
python scripts/render_reach_demo.py
python scripts/train_reach_policy.py
```

For ad hoc Python commands from the repo without installing the package, use:

```bash
PYTHONPATH=src .venv/bin/python ...
```

The current venv has `mujoco`, `numpy`, `scipy`, and `pytest`. Editable install previously
failed because `setuptools` was not installed and the sandbox had no PyPI network access.
Do not make editable install a required path unless you also fix that dependency issue.
Do not tell the user to run `mjpython` from this venv. `.venv/bin/mjpython` has a broken
shebang because the repo path contains spaces. Use `bash scripts/run_mujoco_gui.sh`, which
runs `.venv/bin/python .venv/bin/mjpython scripts/open_mujoco_gui.py`.
`ffmpeg` is available at `/opt/homebrew/bin/ffmpeg` in the current environment and is used
for MP4 export.

## MuJoCo Gotchas

- MuJoCo Python exposes site orientation through `data.site_xmat`, not `data.site_xquat`.
- Use `scipy.spatial.transform.Rotation` to convert the site matrix to quaternion/rotvec.
- Keep position-only reaching separate from full pose reaching. Full pose IK is stricter and
  can make simple reach tests look broken.
- The controller should report clipping and infeasibility instead of silently masking them.

## Design Constraints

- MuJoCo first on the local Mac.
- Isaac Sim later only when Linux/NVIDIA hardware or cloud GPU is available and justified.
- Unity ML-Agents may be a secondary learnability benchmark, not the main VLA/controller stack.
- Prefer small, inspectable Python modules over a large framework.
- Do not overclaim results from reaching tests. Reaching is only the controller smoke test.

## Next Useful Work

The next implementation should add:

- `src/svla/action_spaces.py` with joint-delta and end-effector-delta adapters.
- Controller telemetry: target pose, actual pose, clipping flags, joint targets, joint states.
- Tests for unreachable targets, joint-limit clipping, and repeatability.
- A simple table/cube scene only after the action adapter boundary is clean.

When evaluating policies later, keep observations, demonstrations, task initialization, and
success metrics identical across action spaces. Otherwise the result will not answer the
research question.
