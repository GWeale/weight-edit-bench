# Harbor weight-editing tasks

Weight Edit Bench exports self-contained tasks for **Harbor 0.25.0**, using task
schema **1.4** and benchmark protocol **`weight-edit-harbor-v1`**. Each task has an
instruction, Docker build context, reference solution and separate verifier.
The Python APIs remain available for trainers that need individual environment
steps. Harbor supplies terminal-agent trials, trajectories and rewards; this
integration does not train the editor's weights.

## Run the CPU controls

Use Python 3.12+ and Docker. Python 3.13 is used by the task image. Install from
the repository root:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[harbor]'
bash examples/harbor-smoke.sh runs/harbor-smoke
```

The script exports seed 0's adaptation and retention tasks, runs `oracle` and
`nop` serially, and checks their actual verifier outputs. It calls no model API.
The CPU oracle is Adam with learning rate 0.03 for up to 20 edits. It should earn
positive reward on these two sanity tasks; `nop` should earn zero. Those outcomes
establish a working harness, not an RL advantage.

To generate a larger dataset and run a specific task:

```sh
weight-edit-harbor --out runs/harbor-tasks --seeds 0,1,2 --split test
harbor run --path runs/harbor-tasks/cpu-retain-0 --agent oracle --env docker --n-concurrent 1 --n-attempts 1 --max-retries 0 --jobs-dir runs/oracle
```

Use `--split train` for training tasks. Export test and OOD tasks separately;
freeze the policy before evaluating them. Every export requires a fresh output
directory. The export copies package source and its MIT license, so the task can
run without a checkout or host Python package inside the container.

For rootless Docker, set `DOCKER_HOST` to your existing daemon socket and unset a
conflicting `DOCKER_CONTEXT`. CPU task defaults are two CPUs, 1 GiB memory, a
300-second agent limit, a 120-second verifier limit and a 600-second build limit.
Harbor backend support determines which resource settings are enforced. Image
builds have separate resource needs. Run serially on constrained machines.

## What the agent submits

The agent starts with an empty `/app/submission.json`:

```json
{"protocol": "weight-edit-harbor-v1", "actions": []}
```

It can inspect the public observation and replay a candidate with:

```sh
python -m weight_edit_lab.harbor_runtime observe
```

For CPU tasks, append `{"delta": [...]}` with one finite number per weight.
The default student has 25 weights. `{"stop": true}` ends the episode. For
coding tasks, use the documented `inspect`, `probe`, `edit`, `rollback` and
`stop` actions from the [direct-program API](agents.md). The observation contains
the tensor-language reference, supported modules, feedback and remaining budgets.

The final file must be regular UTF-8 JSON, at most 256 KiB, with exactly
`protocol` and `actions`. CPU tasks allow 20 edits and an optional earlier stop;
coding tasks allow three editing actions within 16 total turns. Actions after
termination are invalid. Malformed, missing, over-budget or invalid transcripts
earn zero. Runtime or scoring failures remain trial errors and do not produce a
fabricated zero reward.

The agent can try multiple candidates within Harbor's wall-time and resource
limits. Submitted-transcript counters do **not** include discarded attempts.
This is a terminal-agent evaluation protocol; it does not enforce a global
number of private exploratory probes. Report agent tokens, wall time and
external training costs separately.

## How verification works

Harbor transfers the submitted JSON artifact into a fresh verifier environment.
The verifier loads `/tests/spec.json`, reconstructs the original student, replays
the actions and evaluates the resulting checkpoint. It uses trusted package code
from the image with isolated Python imports. Agent-side changes to libraries,
local weights, specifications or score files do not transfer into the grader.

The task follows Harbor's [separate-verifier contract](https://docs.harborframework.com/tasks/separate-verifier).
Harbor also transfers its standard artifact directory; the grader never imports
code from that directory or treats it as a source of scores. The verifier has no
network access. This design is not a security audit of Docker or Harbor.

The result files under each trial's verifier logs are:

- `reward.json`: exactly `{"reward": number}`, with reward in [0, 1].
- `details.json`: status, retention decision and underlying metrics. Coding tasks
  include baseline, raw candidate and accepted outcomes plus action costs.

Keep the task export, Harbor job configuration, agent trajectory and verifier
results together. Record the source commit/hash, model revision, image digest,
Harbor version and runtime versions. The image uses a versioned base tag; retain
its resolved digest because a tag alone does not guarantee identical rebuilds.

## Terminal reward

For CPU tasks, let `L0` be initial test MSE, `L1` final test MSE, `A1` final
retention MSE and `n` the number of edits. Adaptation reward is:

```text
clip((L0 - L1) / max(L0, 0.05) - 0.001 * n, 0, 1)
```

For retention tasks, replace `L1` by `L1 + A1`. Reward is zero if `A1 > 0.05`.
An unchanged student earns zero. The hard retention threshold applies only to
the Harbor protocol, not the original CPU trainer's return.

For coding tasks, the existing development gate first selects the accepted
checkpoint. Terminal tests then compare it with the starting checkpoint:

```text
clip(coding_function_accuracy_gain + 0.25 * coding_case_accuracy_gain, 0, 1)
```

Reward is zero if coding function or case accuracy declines, any previously
correct retention function is lost, or retention case accuracy declines. Raw
rejected candidates remain in `details.json`. Diagnostic/edit costs are reported
separately and are not subtracted from this terminal reward. This reward differs
from the stepwise Python environment return; do not pool them as one metric.

## Export direct coding tasks

The coding image installs CPU PyTorch and Transformers, downloads a public model
at an immutable revision during build, and runs offline afterward. It needs more
disk, memory and time than the small CPU controls. Only Qwen-compatible module
layouts are supported. Model weights retain their separate license.

Create `runs/harbor-model.json` with a pinned model identity, for example:

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
  "revision": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
}
```

An existing `weight-edit-coding download` manifest also works; host paths are
excluded from the export. No host coding dependencies are needed for export.

```sh
weight-edit-harbor --track coding --model runs/harbor-model.json --seeds 0 --out runs/harbor-coding
harbor run --path runs/harbor-coding --agent nop --env docker --n-concurrent 1 --n-attempts 1 --max-retries 0 --jobs-dir runs/coding-nop
```

Default coding limits are two CPUs, 4 GiB memory and 1,800 seconds each for build,
agent and verifier. The reference `oracle` scales one MLP matrix by 0.99; it is
an edit-execution example with **no guaranteed positive reward**. Coding headroom
and useful edits must be calibrated before an RL study. Export with
`--family-partition evaluation` only after training and tuning are frozen.
Coding episodes always contain support, development and final partitions;
`--split` selects a namespace only for CPU tasks.

## Connect an editor or trainer

Use a supported [Harbor agent](https://docs.harborframework.com/agents/pre-integrated-agents)
or implement its agent interface. The agent receives the task instruction and
terminal, explores permitted observations and writes the submission file. Its
model/provider configuration belongs to Harbor, not to this benchmark. Paid
provider calls require your own account and budget; the smoke commands use none.

An external RL trainer can collect these trajectories and terminal rewards, then
update an open-weight editor and repeat on training tasks. This repository does
not include a distributed trainer for free-form tensor programs. Hosted frozen
agents are evaluation policies; running them does not update their weights.

Public task source, seeds and generators allow final-data reconstruction. The
separate verifier protects grading integrity, but does not make the tasks secret
or contamination-proof. Publish policies, matched frozen-editor controls and
independent evaluation families before claiming learned editing skill. These
tasks alone cannot establish mechanistic understanding or recursive improvement.

## Validation scope

Local tests cover task schema acceptance by Harbor 0.25.0, replay, positive CPU
reference reward, no-edit reward, action/file limits, trusted-spec separation
and infrastructure-error handling. Real Docker acceptance results are recorded
with the release. The coding image requires a separate end-to-end qualification;
the small CPU controls do not establish coding-model compatibility or headroom.

Tasks are distributed from this repository through the exporter. No Harbor
registry entry, hosted leaderboard or hosted rollout service is claimed.
