# Quality Assurance

Assemble the dataset, review it, then convert or publish. The steps below run
in that order: join the recording sessions, look at them, measure what is
measurable, decide the rest, and only then produce joint targets. Every command
takes the dataset and `--robot <name>`; nothing here is specific to a task or an
embodiment.

## 1. Join Sessions Into One Dataset

A task recorded over several sittings lands as several datasets. Join them
before reviewing, so one review covers the whole set:

```bash
handumi dataset merge \
  outputs/datasets/session-a outputs/datasets/session-b \
  --output outputs/datasets/session-all \
  --task "<one wording for the whole set>"
```

Every episode survives untouched, in the order the sources are given; only the
episode numbering and the task table change. Sources are read, never written,
and `--dry-run` prints which output episode range each session claims.

Pass `--task`. Sessions recorded weeks apart carry different wordings for the
same task, and without it the result holds one task per wording, so a
language-conditioned policy sees several tasks where there is one.

Sources that disagree on fps, robot type, chunk size, or a feature's dtype,
shape or video encoding are refused by name before anything is written.
Metadata they state differently -- a per-session workspace pose, the curation
record a curated source carries -- is dropped rather than attributed to episodes
it never described. The comparison is per entry, so the schema keys every source
shares survive.

`meta/handumi_merge.json` maps every output episode back to its source and
records each source's payload fingerprint and every dropped metadata path.
Quality reports do not carry over, because they index source episode numbers:
review the merged dataset rather than reusing them.

## 2. Replay and Inspect

Every command below takes `DATASET` as either a local dataset root or a Hugging
Face repository id. A repository id is fetched on first use and cached, so
`handumi dataset qa your-name/handumi-demo` works with nothing downloaded by
hand; `--revision` selects a branch, tag or commit.

For local recordings, pass the local root as `DATASET`; no dataset is
downloaded:

```bash
JAX_PLATFORMS=cpu handumi replay \
  outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0
```

See [Replay a Local Recording in Simulation](replay_in_sim.md) for the current
OpenArm v1 and TRLC-DK1 commands, calibration semantics, measured IK results,
and Viser mesh troubleshooting.

Choose the target robot explicitly. Piper is a currently available example:

```bash
TARGET_ROBOT=piper
handumi replay \
  your-name/handumi-demo \
  --robot "$TARGET_ROBOT"
```

In Viser, check the bimanual geometry, table alignment, motion continuity, and
unreachable poses. Use `--headless` for automated checks and `--strict-ik` to
fail when IK error exceeds the configured limits.
Add `--hide-trajectories` to show only the robot and scene without the target
and achieved TCP paths.

Table-calibrated datasets preserve recorded bimanual geometry automatically.

## Extract the initial joint distribution

After converting a recording to a robot joint dataset, extract one initial
``observation.state`` per episode into an external YAML profile. Every episode
has the same weight regardless of its duration, and the input dataset is never
modified:

```bash
handumi dataset initial-distribution \
  outputs/datasets/handumi-demo-piper-joints
```

By default this writes
``outputs/initial-distributions/handumi-demo-piper-joints-initial-distribution.yaml``.
Pass ``--output`` only when a different location or filename is useful.

The profile stores the mean, population standard deviation, min/max,
percentiles, median, and the recorded initial state nearest to the mean. Values
are always written in HandUMI's physical joint layout (arm radians and gripper
opening meters), including when the input uses an external LeRobot follower
layout. Use ``--dry-run`` to inspect the dataset and recommendation without
writing, and ``--replace`` to explicitly replace an existing profile. For a
Hub dataset the command downloads only ``meta/`` and ``data/`` under
``outputs/datasets``; videos are not needed to calculate joint statistics.

:::{dropdown} Absolute-table replay and calibration precedence
For an explicit geometry-preserving replay:

```bash
handumi replay your-name/handumi-demo \
  --robot "$TARGET_ROBOT" \
  --retarget-mode absolute-table \
  --deployment-profile sim
```

`absolute-table` applies `robot_from_table` to both TCP trajectories, preserving
their bimanual separation. By default, replay aligns each tool orientation on
the first frame and preserves subsequent wrist rotations. Use
`--absolute-orientation table-absolute` only when the HandUMI and robot TCP
frames were externally calibrated.

Controller-to-TCP calibration is selected in this order:

1. Explicit `--controller-tcp-calibration`.
2. Identity-bound snapshot stored in the dataset.
3. Robot/device calibration from `configs/robots/*.yaml`.
4. Device fallback for legacy data.

Replay prints the calibration source and hash, TCP distances, minimum height,
bimanual separation and workspace bounds, the resolved deployment profile,
table-to-robot transform, and IK errors. `auto` selects the laboratory-local
file from `configs/rig.yaml` when configured and otherwise selects
`configs/calibration/table/sim/<robot>.yaml`; use `sim` in portable QA so a
machine-local rig cannot change the result.
:::

Offline playback of a dataset on physical arms is not currently exposed.
`handumi teleop-real` consumes live HandUMI motion and is not a recorded-dataset
replay command.

## 3. Run Automated Validation

```bash
handumi validate \
  outputs/datasets/handumi-demo --strict
```

The report is written to `meta/handumi_quality.json`. Review rejected episodes
for tracking loss, stale sensors, synchronization errors, frozen poses, motion
jumps, or invalid duration. Rejected episodes are removed during curation, not
during conversion: conversion refuses them so its output cannot disagree with
the dataset it was given.

## 4. Check the Demonstration Direction

A session holds more than demonstrations. Putting the object back between takes
is itself a recorded episode whenever the recorder was left running, and it
carries the same task string as the takes around it. Nothing else in the review
sees this: such an episode is a flawless recording -- sensors healthy, tracking
valid, timestamps regular -- that retargets as well as any other. The defect is
in what it teaches, and a policy trained on it learns to undo the task.

```bash
handumi dataset direction outputs/datasets/handumi-demo
```

The check is told nothing about the task. Each episode's scene change is the
difference between how the workspace camera sees the table at its end and at its
start, and it compares that against the dataset's own median change: an episode
running the other way scores negative, and zero is the boundary. The comparison
weights each pixel by how consistently it changes across the dataset, which is
what keeps the operator from deciding the verdict -- their arm is the largest
moving thing in frame and moves differently every take, while the object is
small and changes the same way every time.

The finding is a **warning**, not a rejection. A dataset may hold both
directions deliberately, and the reference is the majority of this dataset, so
only a reviewer knows which case it is. Watch the flagged episodes before
curating.

It needs a workspace camera and at least three episodes, since two cannot
establish a majority. `--skip-direction` leaves it out of `handumi dataset qa`.

## 5. Inspect Captured Signals

Raw datasets preserve the information needed to validate, recalibrate, or
retarget a capture:

```text
observation.images.left_wrist
observation.images.right_wrist
observation.images.workspace
observation.state                  # controller poses + gripper widths
observation.feetech.*              # ticks, width, time, health
observation.tracking.*             # device poses, validity, aligned time
observation.sync.*                 # shared target and record times
observation.camera.<name>.*        # sample time and health
```

`observation.state[14:16]` stores left/right gripper widths in meters. Tool,
controller mount, calibration hashes, source enablement, and coordinate layout
are stored in metadata. Raw controller poses remain unchanged so the same
capture can be checked against another supported robot.

## 6. Screen the Dataset Against the Target Robot

Validation and analysis grade the *recording*. Neither knows whether a given
robot can follow it: an episode with perfect tracking can still leave the arm
folded into itself, or be the only session recorded with a different wrist
pose. Screen the dataset before its episodes become joint targets:

```bash
JAX_PLATFORMS=cpu handumi dataset screen \
  outputs/datasets/handumi-demo \
  --robot piper
```

Every episode runs through the same solver `handumi convert` uses, so the
numbers are the ones that would be written, not an approximation. Per episode
the report records TCP position and orientation error, the start-pose residual,
self-collision clearance, and tabletop clearance, then grades it:

- `retarget_start_pose_unreachable`, `retarget_position_error` — rejections;
  the robot cannot perform the episode.
- `retarget_rotation_error`, `retarget_self_collision` — warnings for you to
  judge. A brief self-intersection during the retraction after the task is very
  different from one during the grasp, so the finding reports the **share of the
  episode** the arm spends intersecting itself, not just a frame count. Below
  roughly 1% it is a touch at one pose; a double-digit share is a configuration
  the trajectory holds, which no arm can execute and which no policy should be
  shown. Read the share, then watch the replay: the count alone cannot tell a
  387-frame episode with one bad pose from a 90-frame one that is folded through
  itself the whole way.
- `retarget_rotation_outlier` — orientation tracking inconsistent with the rest
  of the dataset. The threshold is a multiple of the dataset median, not a
  fixed ceiling: a second recording session stays well inside any absolute
  limit while still teaching a policy two behaviors for one task.
- `retarget_base_rotation` — a rejection, opt-in through
  `--max-base-rotation-deg`. The report always records how far each arm's first
  joint (the base yaw) swings from home; the limit turns that into a rejection.
  Position is weighted far above orientation, so when a demonstration brings
  the tool to the inner edge of the arm's reach with the wrist pitched down,
  the folded arm runs out of wrist range and the solver keeps the position by
  swinging the base 60 to 110 degrees and rolling the wrist. The rest cost only
  anchors to the previous frame, so nothing pulls the base back. On the Piper,
  `60` sits just above what real teleoperation reaches; episodes that end by
  parking the tool next to the robot base are the ones that trip it.

Tabletop contact is reported as a metric without a finding. Fingertips reaching
the surface during a grasp is the demonstration itself, plus the deliberate
slack of the capsule fit over the finger mesh.

Three more metrics carry no finding but say how the solver and the operators
are doing: the share of frames with any arm joint within 3 degrees of a stop
(`limit (%)` in the report), the largest base swing, and per-arm tool travel
with the resulting idle verdict (`idle`). An idle arm is the hand a
single-arm demonstration rested somewhere; the robot has to hold that pose
for the whole episode, so where operators rest the idle hand is a protocol
matter the report makes visible.

The report lands at `meta/handumi_screening_<robot>.json` and uses the
`handumi validate` quality-report schema, so it feeds the existing pipeline
directly.

Episodes are independent -- the IK warm-starts each frame from the previous one
within an episode, never across them -- so `--jobs N` solves several at once.
`handumi convert` defaults to up to 8 worker processes for episodes without a
matching screening or complete replay cache. The worker count is limited by
the number of uncached episodes and available CPUs; pass `--jobs 1` to solve
sequentially or `--jobs N` to set another maximum. Dataset writing remains
sequential. Measured on a 201-episode dataset, screening went
from 5.7 to 1.5 seconds per episode, saturating around six workers because the
solver already uses about three cores inside one episode.

Separate solves are not guaranteed to produce bit-identical trajectories:
process settings can change floating-point operations and the optimizer's path.
Conversion reuses the trajectory screening graded when a matching cache is
available, and applies the same strict IK checks to newly solved trajectories.

### One command for the whole review

The reviews are separate commands because they answer separate questions, but
running them by hand invites leaving one out. `handumi dataset qa` sequences
them and merges the result:

```bash
JAX_PLATFORMS=cpu handumi dataset qa \
  outputs/datasets/handumi-demo \
  --robot piper
```

That is exactly `validate`, then `dataset direction`, then `screen` once per
`--robot`, then `analyze`.
`analyze` merges **every** findings report the dataset carries — the recording
report and one screening report per embodiment — and labels each finding with
the producer that raised it, so a merged review names the dimension that
objected. Pass `--quality-report` (repeatable) to select reports explicitly.

Each dimension is measured once, by one producer; `analyze` merges; a human
decides; `curate` applies. Conversion executes that decision and does not
re-judge it.

### What needs a reviewer, and what does not

Severity decides who acts, so the reviewer only sees decisions that are
actually theirs:

- **`reject`** — a mechanical failure: corrupt or non-finite state, an episode
  below the minimum duration, an unreachable start pose, a position error the
  robot holds, sensors down past the warm-up window. There is no judgement to
  add, so `handumi dataset curate --exclude-rejected` removes them all without
  anyone retyping indices a report already computed.
- **`warning`** — context decides: a brief self-intersection is one thing
  during the retraction after a task and another during the grasp; an
  orientation outlier may be a second session or legitimate variation. These
  stay opt-in through `--exclude`.

Both compose, and the curation report records which removals were automatic and
which a reviewer chose, so the audit trail still shows who decided what:

```bash
handumi dataset curate <dataset> --output <new_root> \
  --exclude-rejected \
  --exclude 11,19
```

`handumi convert` refuses to run when this report is missing, when episodes it
flagged are still included, or when it is stale. Staleness covers both the
dataset payload and the robot: the report records a fingerprint of the URDF,
the embodiment YAML, and the resolved table calibration, because moving the arm
bases or the table invalidates every joint solution in it while leaving the
recorded episodes untouched.
It refuses on a missing or stale report, and on episodes it rejected that are
still included. Warnings are reported but do not block: a reviewer already
settled those during curation, and blocking would make the override habitual
when that same override switches off the checks that must never be waived.

It also refuses when an episode fails the recording-quality checks, instead of
skipping it: an output that silently holds fewer episodes than the curated
input it was given is worse than a conversion that stops and says why.

Conversion reuses the trajectories screening already solved, from the npz
sidecar beside the report, whenever the solver settings match exactly --
retarget mode, calibrations, orientation policy, frame selection. That halves
the IK work of the pipeline, but the reason it matters more is fidelity: the
solve warm-starts each frame from the previous one, so float32 differences
compound and two runs of the same conversion disagree on a small fraction of
frames. Reusing the screened solve is what makes the joints that get written
the same ones the review graded. `--no-solve-cache` forces a fresh solve.
Override with `--allow-flagged-episodes` when you have made the call
deliberately. Screening is per embodiment: a dataset cleared for one robot says
nothing about another.

## 7. Curate Rejected or Incomplete Data

When a dataset contains rejected or incomplete episodes, create a separate
curated derivative before conversion or publication. Curation copies the
compressed video of the episodes it keeps instead of re-encoding them, which is
both exact and far quicker -- 35 seconds rather than 28 minutes on a
201-episode dataset, with frames identical to the source rather than a
generation older. It falls back to re-encoding when the streams cannot be
copied, which is when an episode does not begin on a keyframe. The analysis and curation
steps are intentionally separate so statistical outliers can be reviewed before
any data is removed. See [Analyze and Curate Datasets](dataset_curation.md).

## 8. Convert and Check Target Motion

### Dataset naming

Names carry what makes a dataset specific, so a directory or Hub listing can be
read without opening anything:

| kind | name | example |
|---|---|---|
| raw capture | `<task>` | `handumi-demo` |
| curated derivative | `<task>-clean` | `handumi-demo-clean` |
| joint angles (canonical) | `<source>-<robot>-joints` | `handumi-demo-clean-piper-joints` |
| LeRobot follower layout | `<source>-<robot_type>` | `handumi-demo-clean-bi_piper_follower` |

Conversion derives the name automatically from the source and the `--robot`
it was given, so the same capture converted for two embodiments produces two
names that cannot be confused with each other or with their source. With
`--output-layout`, the LeRobot `robot_type` names the result instead, because
it says both which robot and which joint vector the dataset holds. `--output`
overrides it when a repository is already named.


Conversion creates a target-specific dataset while preserving the raw source.
`--retarget-mode` defaults to `auto` and is resolved with the exact same rule
`handumi replay` uses, identically for every embodiment: when the source
dataset declares a calibrated table workspace, conversion runs the same
`absolute-table` solver as replay (validating
the selected local or `configs/calibration/table/sim/<embodiment>.yaml`
deployment) for exact qpos parity; otherwise
it falls back to `local-relative`. For Piper, use the validated `--robot piper`
profile to convert the replay result to physical Piper commands:

```bash
JAX_PLATFORMS=cpu handumi convert \
  outputs/datasets/handumi-demo \
  --robot piper \
  --output your-name/handumi-demo-piper
```

The Piper state has 14 physical commands: six replay arm joints in radians
plus one gripper opening in meters per side. Its pairs are
`observation.state[t] = command[t]` and `action[t] = command[t+1]`. The two
mirrored URDF finger joints are reconstructed from the single opening only when
rendering simulation. Other embodiments use the same `--robot <name>` interface;
absolute-table support requires their corresponding
simulation file or lab-local deployment, and an explicit `--retarget-mode` can
override auto detection for any of them. Use `--deployment-profile sim` for a
portable converted dataset and record the resulting deployment metadata.

Replay and validate the converted motion before using it with a robot-specific
integration. See [Add a New Robot Embodiment](../development/new_embodiment.md)
when adding another simulation model or hardware backend.

### Write the LeRobot follower layout directly

The canonical vector above is physical, which is what keeps the data
comparable across embodiments. A LeRobot training and deployment stack,
however, expects the vector its own robot plugin records, and LeRobot leaves
that encoding to each plugin's `MotorNormMode`: a Feetech-based follower
normalizes joints to `-100..100` over its calibrated range, a Damiao-based
follower reports degrees, and a CAN driver may apply its own limits and signs.
`--output-layout <robot_type>` writes that plugin's vector straight from the
raw capture, with no intermediate dataset to keep:

```bash
JAX_PLATFORMS=cpu handumi convert \
  outputs/datasets/handumi-demo-clean \
  --robot piper \
  --output-layout bi_piper_follower
```

The conversion still solves the canonical vector first, in a hidden staging
directory, then rewrites it at the file level (parquet columns, `info.json`,
statistics) and links the videos without re-encoding. The staging directory is removed
afterwards, so one command yields one dataset. `--keep-canonical` also keeps
the canonical dataset under its usual name; it is a debugging aid for reading
angles in radians, not a step of the pipeline, since `replay-joints` and
`replay-real` decode the follower layout themselves.

| `--output-layout` | HandUMI robot | Arm joints | Gripper | Encoding read from |
|---|---|---|---|---|
| `bi_piper_follower` | `piper` | `-100..100` over firmware limits, joints 1, 4 and 6 sign-flipped | `0..100` | XHUMAN `piper_sdk_interface.py` |
| `bi_openarm_follower` | `openarmv1` | degrees from the calibration zero | degrees, `0` closed to `-60` open | LeRobot `openarm_follower.py` |

Each layout lives in `src/handumi/dataset/external_layouts.py` as one
`ExternalJointLayout` with a `JointEncoding` per column, citing the plugin it
was read from, because a dataset only holds the resulting numbers, never the
limits and signs that produced them. `--use-degrees` mirrors LeRobot's
`use_degrees` config for plugins that expose it: arm joints switch to degrees,
the gripper keeps its mode. A plugin without that option refuses the flag, as
LeRobot itself would.

Two things the export does that a plain unit change would not:

- **It clips the solver's overshoot.** The IK solver's soft joint-limit
  constraint settles up to about a millidegree past a limit. A driver with a
  hard accepted range (`bi_piper_follower` rejects anything outside
  `-100..100` instead of clipping, and the inference engine then freezes the
  robot) would refuse those commands. Overshoots within
  `--clip-tolerance-rad` (default `0.002`) are folded back; anything larger is
  reported, and the conversion aborts because the dataset would not deploy.
  Every clipped value is counted in `handumi.export.clipping`.
- **It renames cameras** to what the plugin's recordings use, by default
  `left_wrist -> left`, `workspace -> top`, `right_wrist -> right` for the
  Piper layout, so fine-tuning from a checkpoint trained on the plugin's own
  captures sees the same image keys. `--camera-map old=new,...` changes it.

`handumi dataset export` applies the same rewrite to a canonical dataset that
already exists, and `--compare-with <reference>` checks the result against a
dataset the stack is known to train on: features, names, shapes and
`robot_type` must match; a differing camera resolution is reported but does
not block, since policies resize their inputs.

```bash
JAX_PLATFORMS=cpu handumi dataset export \
  outputs/datasets/handumi-demo-clean-piper-joints \
  --strict --compare-with outputs/datasets/bi_piper_pick_and_place_fruits_mantra
```

What the export cannot change is what `observation.state` means. A follower
recording stores measured feedback, with the servo lag and gravity sag of a
real arm; a HandUMI conversion stores the ideal IK command, because no robot
was in the loop. The metadata says so in `handumi.export.state_semantics`.

### Validate on the physical robot

The final check before training runs the converted dataset on the real arms.
`handumi replay-real` streams the stored joints through the robot's hardware
backend (the one `teleop-real` uses) and compares the measured joints with the
command. It reads canonical and LeRobot-layout datasets alike, so the vector a
policy will emit is what gets validated:

```bash
handumi replay-real \
  outputs/datasets/handumi-demo-clean-bi_piper_follower \
  --robot piper --episode 0 1 2 --dry-run   # limits and speed only
handumi replay-real \
  your-name/handumi-demo-clean-bi_piper_follower \
  --robot piper --episode 0 1 2 --speed 0.5
handumi replay-real \
  your-name/handumi-demo-clean-bi_piper_follower \
  --robot piper --episode 0 1 2 --loop      # repeat until Ctrl+C
```

The dry run checks every frame against the URDF limits and the backend's
joint speed limit, and predicts the tracking error the command stream itself
will cause; frames faster than the backend are reported and, past 5 % of an
episode, refused with a suggested `--speed`. The hardware run homes,
ramps into the first frame over `--approach-seconds`, plays the episodes and
returns home. Each episode gets a PASS or FAIL against `--tolerance-deg` and
`--tolerance-mm` (lag-compensated joint and TCP tracking error), and a log
under `outputs/replay_in_real/<dataset>/` with the commanded and measured
joints. See [Physical Robot Teleoperation](../physical_robots/real_teleoperation.md)
for the safety behavior.

## 9. Publish Accepted Data

Upload only after the replay and validation checks pass:

```bash
hf auth login
huggingface-cli upload your-name/handumi-demo \
  outputs/datasets/handumi-demo --repo-type dataset
```
