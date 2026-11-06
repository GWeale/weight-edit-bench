# Agent API

## CPU numerical environment

`WeightEditEnv.reset(seed=..., split=...)` returns `(observation, info)`.
`step({"delta": [numbers]})` returns `(observation, reward, terminated,
truncated, info)`. `{"stop": true}` ends the episode early. Deltas cover all
parameters and are projected to the configured norm and weight bounds. The
[CPU protocol](../protocol.md) specifies access, rewards and data generation.

The optional `WeightEditBench-v0` Gymnasium adapter uses the
[standard reset/step contract](https://gymnasium.farama.org/api/env/).
Its action space is a float64 vector of `3 * hidden + 1` values in [-1000, 1000].
The core applies its normal clipping. Dictionary observations contain weights,
gradient, last delta, support/anchor examples, two losses, progress and retention
weight. A fixed horizon ends episodes; the continuous adapter has no stop action.
It strips terminal test metrics from `info`. Evaluation code can retrieve them
with `env.unwrapped.final_metrics()` after the episode ends.

`reset(seed=42)` seeds a deterministic stream of task seeds. Subsequent resets
without a seed draw new tasks from that stream. For exact pairing with core-API
tasks, use the core API's explicit task seed rather than assuming the Gymnasium
seed is the same task identifier.

## Direct tensor-program environment

```python
import json
from weight_edit_lab.coding.backend import Student
from weight_edit_lab.coding.direct_env import DirectEditEnv

with open("runs/model.json") as handle:
    manifest = json.load(handle)
student = Student(manifest, device="auto", max_new_tokens=512)
env = DirectEditEnv(student)
observation, info = env.reset(seed=0)

# Replace this with a local or remote policy. The benchmark calls no provider.
observation, reward, terminated, truncated, info = env.step({
    "kind": "inspect",
    "code": "W=weight('model.layers.0.mlp.down_proj'); emit('norm',norm(W))",
})
observation, reward, terminated, truncated, info = env.step({"kind": "stop"})
env.save("runs/my-direct-episode")
```

The student must support the module layout used by the backend: `model.layers`
with MLP and attention projections. Qwen2.5-Coder-0.5B-Instruct has been exercised;
arbitrary Hugging Face architectures are not claimed compatible. One environment
owns a student instance; do not share it between concurrent rollout workers.

Actions are JSON objects with exactly these fields:

| Kind | Other fields | Effect |
| --- | --- | --- |
| `inspect` | `code` | Run a read-only tensor program and return bounded emitted values |
| `probe` | `prompt`, `modules` | Record one forward pass at one or two supported linear modules |
| `edit` | `code` | Validate and apply one complete tensor program; evaluate development behavior |
| `rollback` | none | Restore the episode's starting checkpoint; evaluate and charge the action |
| `stop` | none | End adaptation without another edit |

Observations contain support solutions, support generations/grades, aggregate
development metrics, edit history, weight statistics, tensor catalog, remaining
access budgets, last feedback and the tensor-language reference. They omit the
task seed, development case inputs, final examples and final scores. The backend
and evaluator know these values; the API does not defend against hostile code
with access to the same Python process or filesystem.

Programs support tensor reads, cached activations, arithmetic, matrix operations,
reductions, indexing, column replacement, bounded `emit`, and staged `write`.
They cannot import modules, access attributes, call arbitrary Python, open files,
use a network, run gradients or invoke an optimizer. A patch can write at most
four supported tensors; changes above 10% of a pretrained tensor's norm are
rejected, not silently clipped. Dtype rounding is part of the actual edit.

Each probe accepts at most two distinct supported modules and a prompt of at
most 4,000 characters / 512 chat-formatted tokens. It exposes last-prompt-token
vectors, not generated answers. Probe IDs are returned in feedback. Cached
activations can be referenced only while the weights match their checkpoint;
reset clears the cache.

`DirectConfig` defaults to 16 total turns, eight inspection requests and four
probe requests. Every call consumes a turn, including malformed actions. A
failed inspect/probe request consumes its category budget once admitted. Invalid
actions cost 0.001; successful diagnostics cost 0.0001. Executed forward passes
are counted separately from probe requests. Runtime failures propagate instead
of receiving a fabricated task score.

`CodingConfig` defaults to three editing actions/evaluations for this interface,
zero optimizer steps and the algorithmic bridge curriculum. A successful edit or
rollback consumes an editing action; diagnostics use their separate budget.
Exhausting either the editing horizon or turn budget truncates the episode.
`stop` terminates it. Both paths apply the development acceptance gate.

The final transition includes the gate's reward adjustment. Sum the rewards
returned by `step`; do not add that adjustment again. Terminal `info` contains
development acceptance and cost data, never terminal test scores. Calls after
termination fail until reset.

`env.evaluate()` explicitly opens final tests after closure and reports baseline,
raw candidate and accepted checkpoint outcomes. `env.save(path, include_final=True)`
does the same and writes separate candidate/accepted `.safetensors` files and
`episode.json`. Without that flag, saving does not open final tests. Use fresh
paths; completed output is not overwritten. Inherited starting patches are
accepted by `reset(starting_patch=...)`, and rollback restores that starting state.

## Training integration

For a trainable editor, sample actions from its current policy, collect the
returned rewards and log probabilities, and update that policy with your chosen
RL trainer. Start each rollout with a fresh/reset student and use training
families/seeds. Keep terminal evaluation in a separate evaluator. Editor tokens,
trainer updates and student access costs must all be reported.

This release includes a working CPU REINFORCE trainer for step-size selection
and a categorical coding-recipe trainer. It does **not** include a sequence-level
trainer for arbitrary tensor programs, a distributed rollout service or access
to a hosted model's weights. An external frozen agent is an evaluation baseline,
not an RL-trained editor.

Historical file-exchange trial scripts are retained under `examples/` for
compatibility. Their editor names are metadata labels, not working API model
identifiers or provider integrations. The new `DirectEditEnv` is the portable
entry point for program-writing agents.
