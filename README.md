# Weight Edit Bench

**Can an agent improve a neural network by editing its weights while preserving what it already does well?**

Weight Edit Bench is a [Harbor](https://harborframework.com/)-based benchmark for
neural weight-editing agents, with a Python environment API for RL trainers.
An editor proposes numerical changes to a separate student network. Harbor runs
the agent, transfers its edit transcript to a fresh verifier, and records a reward
from the student's task performance and retention.

**Status: early research release, v0.2.0.** The direct-edit pipeline works; our first
frozen-editor pilot showed no coding gain. This is not evidence of recursive
self-improvement, mechanistic understanding or an RL advantage.

## Choose a track

| Track | Student | Agent actions | Included training |
| --- | --- | --- | --- |
| CPU numerical editing | 25-parameter tanh network | Full weight-delta vector | Dependency-free REINFORCE over gradient step sizes |
| Direct coding edits | Local Qwen-style causal LM | Inspect tensors, probe activations, author tensor programs, roll back | JSON environment for external trainers; no free-form program-policy trainer |
| Coding recipe baseline | Local Qwen-style causal LM | Bounded LoRA and recovery recipes | Categorical editor-adapter RL |

Direct coding programs cannot call an optimizer or autograd. The recipe baseline
uses gradients and is reported separately. Neither track modifies a hosted
provider's editor weights.

## Quickstart: Harbor

Requires Python 3.12+ and a working Docker daemon. The CPU tasks need no model
download, GPU or API key. Harbor is pinned to 0.25.0.

```sh
git clone https://github.com/GWeale/weight-edit-bench.git
cd weight-edit-bench
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[harbor]'

weight-edit-harbor --out runs/harbor-tasks --seeds 0
harbor run --path runs/harbor-tasks --agent oracle --env docker --n-concurrent 1 --n-attempts 1 --max-retries 0 --jobs-dir runs/harbor-oracle
harbor run --path runs/harbor-tasks --agent nop --env docker --n-concurrent 1 --n-attempts 1 --max-retries 0 --jobs-dir runs/harbor-nop
```

This exports two standard Harbor tasks: adaptation and adaptation with retention.
The CPU `oracle` is a fixed Adam reference; `nop` submits no edits. These are
installation controls, not learned editors. Use a fresh export directory for
each task configuration. [Harbor setup, coding tasks, agent integration and
scoring](docs/public/harbor.md) describe the full contract and verification scope.

## Python API: no container required

Requires Python 3.11+. The core has no third-party runtime dependencies.

```sh
git clone https://github.com/GWeale/weight-edit-bench.git
cd weight-edit-bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

weight-edit-bench episode --agent adam --seed 0 --out runs/example.json
weight-edit-bench verify runs/example.json
weight-edit-bench replay runs/example.json --out runs/replay.html
```

Open `runs/replay.html` directly to inspect exact parameter changes and predictions.
No server or hosted service is needed.

Train a small RL editor, tune the optimizer baselines on validation tasks, and
compare their frozen policies on separate test tasks:

```sh
weight-edit-bench train --episodes 32 --seed 0 --out runs/policy.json
weight-edit-bench tune --seeds 2 --out runs/tuning.json
weight-edit-bench evaluate --seeds 3 --split test --policy runs/policy.json --tuning runs/tuning.json --out runs/smoke-test
```

These small counts are an installation check, not a research result. The starter
learns a step-size policy; it does not learn an unrestricted update direction.
Use multiple training seeds, more evaluation tasks and the OOD split for a study.
The [protocol](docs/protocol.md) defines the reward and baseline tuning procedure.

## Write an editor

```python
from weight_edit_lab import WeightEditEnv

env = WeightEditEnv(mode="retain")
observation, info = env.reset(seed=42, split="train")
while not observation["done"]:
    action = {"delta": [-0.08 * g for g in observation["gradient"]]}
    observation, reward, terminated, truncated, info = env.step(action)
```

Replace the fixed gradient step with your policy. The import name
`weight_edit_lab` is retained for compatibility with the original prototype.

For Gymnasium trainers:

```sh
python -m pip install -e '.[gym]'
```

```python
import gymnasium as gym

env = gym.make("weight_edit_lab.gym_env:WeightEditBench-v0", mode="retain")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(-0.08 * observation["gradient"])
```

The Gymnasium adapter supplies continuous actions and dictionary observations.
Final test metrics are available only through `env.unwrapped.final_metrics()`
after the episode; they are excluded from the trainer's `info` channel.

## Direct edits to a coding model

Install the optional runtime and download a separately licensed public model.
The download command pins its exact revision; reuse that revision for comparisons.

```sh
python -m pip install -e '.[coding]'
weight-edit-coding download --out runs/model.json
weight-edit-coding direct --model runs/model.json --actions examples/direct-actions.json --seed 0 --out runs/direct-smoke
```

This illustrative action file inspects and scales one real tensor. It is a syntax
and execution example, not a proposed useful edit. The run saves both the raw
candidate and accepted checkpoint. Add `--final` only when decisions are frozen
and you intend to open terminal tests. Use `examples/no-edit.json` with the same
seed and settings for the no-edit control.

For interactive agents or RL rollout workers, use
`weight_edit_lab.coding.direct_env.DirectEditEnv` directly. It supports repeated
inspect → edit → feedback turns and an explicit terminal evaluator. Any model or
trainer that can produce the documented JSON actions can be connected. No Astra
account, private agent runtime or provider SDK is required.

[Harbor integration](docs/public/harbor.md) ·
[Agent API and training boundaries](docs/public/agents.md) ·
[Benchmark specification](docs/public/benchmark.md) ·
[Baseline and result guide](docs/public/results.md)

## What makes a result credible

- Compare identical starting students, tasks, decoding settings and access budgets.
- Report raw candidates and accepted checkpoints, including rollbacks and failures.
- Keep coding accuracy, partial-case accuracy, retention and compute separate.
- Freeze policies before final evaluation. Public task generators are not secret tests.
- Test whether an RL-trained editor beats its frozen version and tuned baselines.

The coding suite includes 12 development families, four reserved evaluation
families and eight retention skills. Each coding function must pass all 64 cases.
These are small synthetic tasks with a restricted Python scorer, not a replacement
for broad software-engineering evaluations.

## Reproduce and contribute

```sh
python -m pip install -e '.[coding,gym,dev]'
python -m unittest discover -s tests -v
python -m build
```

Tests use local fixtures and do not download models. Optional tests skip when their
extras are absent. CI is configured for the core, Gymnasium, tiny coding fixtures
and Harbor CPU oracle/no-edit trials. See the Harbor guide for observed validation.

Source, task generators and examples are MIT licensed. Model weights are not
included and keep their own licenses. See [third-party attribution](THIRD_PARTY.md),
[security boundaries](SECURITY.md), [contributing](CONTRIBUTING.md) and
[CITATION.cff](CITATION.cff). Research notes and large local run artifacts are not
part of the public release.
