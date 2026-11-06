# Experiment protocol

Protocol identifier: `weight-edit-v1`.

## Target and task

The target computes `y = b_out + sum(v_i * tanh(a_i * x + b_i))`. The weight vector contains the eight input weights, eight hidden biases, eight output weights, and one output bias, in that order. The environment exposes their names and values. Gradients are analytic derivatives of mean squared error, checked against finite differences.

Each task generates a fresh initial network and a function `A sin(f x + phase) + slope*x`. Learning inputs lie in `[-2, 2]`. Amplitude ranges from 0.3 to 1.2, phase from -pi to pi, and slope from -0.3 to 0.3. Frequency ranges from 0.5 to 1.5 for train/validation/test and 2.5 to 4 for OOD. OOD measures a specific function-frequency shift, not arbitrary transfer.

The retention target is the initial network's output at inputs in `[-4, -2.5]` or `[2.5, 4]`. It measures preservation of previous behavior, not recall of a previously trained benchmark skill. Task and retention regions do not overlap, but shared parameters couple them.

## Data access

Independent random streams, keyed by protocol, split, task seed, and purpose, generate:

| Partition | New task | Retention | Editor access |
| --- | ---: | ---: | --- |
| Support | 24 | 16 | Examples, loss, and gradients |
| Scoring | 48 | 32 | Reward differences only |
| Final test | 128 | 64 | Terminal metrics only |

The environment does not expose the hidden generating function, seed, split, scoring examples, or final-test examples in observations. The runner logs task data only in terminal audit records. The runner knows the seed and split; these are not secrets against hostile local code. This is an API separation, not an operating-system sandbox. Run untrusted editors in a separate restricted process before using this as an adversarial benchmark.

Reset creates a new episode. `step` before reset or after termination raises an error. Malformed actions fail before mutation. A batch records an agent/runtime failure separately from an outcome and returns a failing CLI exit code if any episode failed.

## Actions and transitions

`{"delta": [...]}` supplies exactly one finite number per parameter. Each proposed component must be within [-1000, 1000]. The environment clips the vector's L2 norm to 0.35, then projects resulting weights into [-8, 8]. Both thresholds are configurable. Logs retain the proposal, actual applied delta, resulting weights, and clipping flag.

`{"stop": true}` terminates voluntarily without another edit or edit cost. A delta action, including a zero vector, consumes one edit. Reaching the default 20-edit horizon returns `truncated=True`. Voluntary stop returns `terminated=True`. The built-in comparisons use the full horizon; custom policies may stop early.

## Reward

Let `J_t = task_scoring_MSE_t + lambda * retention_scoring_MSE_t`, with lambda zero in Adapt and 1 by default in Retain.

`reward_t = (J_(t-1) - J_t) / max(J_0, 0.05) - 0.001`

The fixed episode normalization makes the sum telescope to total improvement minus edit costs. Initial retention error is zero by construction. The reward sets a tradeoff, not a hard preservation guarantee. Raw task and retention errors are always reported independently so a combined reward cannot conceal damage to one objective. Task scales and the retention coefficient define what this benchmark values.

## Training and comparisons

The starter policy is a linear softmax over six gradient learning rates: 0, 0.005, 0.02, 0.08, 0.2, 0.5. Its features are a bias, episode progress, clipped log support error, clipped log gradient norm, and the mode indicator. It uses the same support objective gradient provided to SGD and Adam. It emits a full delta vector and receives the same clipping rules.

REINFORCE uses undiscounted reward-to-go, a lagged per-step exponential baseline, advantage clipping to [-5, 5], and one averaged update per episode. Training alternates Adapt and Retain on training tasks. It never selects updates or checkpoints using final-test metrics. Evaluation uses the final checkpoint's highest-probability action. No claim is made that this starter is the best RL algorithm for the environment.

SGD searches five fixed learning rates; Adam searches five. The validation command chooses a separate rate for each mode by the average combined final metric on validation tasks. Validation terminal examples are therefore part of model selection, not final reporting. Frozen rates are then evaluated on test or OOD tasks.

Every agent/mode pair gets the same starting network and data for each seed and split. Each episode gets a fresh baseline object, including fresh Adam moments and random-editor state. Random edits use the same random directions across modes. Policy training can be repeated with distinct `--seed` values; one run measures only one trained policy.

## Reproducibility and accounting

Manifests contain the protocol, source fingerprint, full environment config, seeds, splits, agent rates, tuning provenance, and learned checkpoint. Atomic episode writes allow resumption. Changing a manifest input rejects resume. Replay verification regenerates the task, applies every action, and checks frames, final metrics, and the edited checkpoint exactly. Exact reproduction assumes the same Python/runtime floating-point behavior; cross-platform bitwise equality is not promised.

Each built-in episode has at most 20 edits and one support and one retention gradient computation per returned observation, including reset. Each initial state or edit triggers one reward-score query. The recording layer also computes diagnostic losses; these are not extra feedback to the agent. Reported wall time starts after reset. It includes editing, episode recording, and terminal evaluation, but excludes initialization (including the initial observation), checkpoint training, tuning, and file I/O. These counters describe access and work categories, not FLOP equality or a speed benchmark.

Bootstrap intervals resample task seeds 1000 times within each mode. Paired comparisons subtract scores for identical tasks before resampling. Intervals are omitted for fewer than two tasks. They describe task variation for the evaluated policies, not uncertainty across policy training seeds. All errors are counted and omitted from outcome means, with missing-pair counts reported.
