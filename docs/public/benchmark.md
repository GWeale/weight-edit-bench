# Benchmark contract

Release: **0.2.0**. Status: **research preview**. Record both the release/commit
and the environment protocol; scores from different protocols are not interchangeable.

The question is whether an editor can improve a student's task performance under
a stated access budget while preserving specified behavior. Successful execution,
nonzero weight changes and passing an acceptance gate are not evidence of useful
learning by themselves.

## Tracks and data

The CPU track uses a scalar-input tanh network and deterministic regression tasks.
Its original protocol is `weight-edit-v1`. Training, validation, test and OOD task
namespaces differ. OOD changes function frequency. See the full
[CPU protocol](../protocol.md) for reward normalization, support/score/test sizes
and optimizer baselines.

The coding task generator is `algorithmic-coding-v2`. It contains 12 development
families, four reserved evaluation families and eight retention skills. Bridge
episodes draw four intermediate families; mixed episodes draw two intermediate
and two stress families; stress episodes draw four stress families. Reserved
evaluation uses coin change, gapped increasing subsequences, balanced signs and
clipped trapped water. This is a specific family shift, not broad coding transfer.

Every episode has four coding functions and four retention functions per scoring
partition. Each function has 64 cases, including edge fixtures and random cases.
Passing means every case passes. Partial-case accuracy is reported separately.
Support, development and terminal problem parameters differ, but some input edge
fixtures are shared. Distinct seeds can repeat semantic contexts; do not count
every case or repeated prompt as an independent statistical observation.

The scorer interprets a restricted Python subset and uses trusted local oracles.
Unsupported syntax, runtime failures and wrong answers remain distinguishable.
These tasks do not measure repository navigation, tool use or general production
software engineering. Calibrate headroom before increasing task difficulty:
ceiling scores and uniformly unsolvable tasks both hide editor differences.

All generators are public. Reserved families are a protocol convention, not
secret data or a contamination guarantee. No hosted leaderboard is supplied.

## Harbor execution

The Harbor task protocol is `weight-edit-harbor-v1`, using Harbor 0.25.0 task
schema 1.4. The [Harbor guide](harbor.md) specifies terminal rewards, JSON
submissions, container boundaries and execution commands. Its final reward is
distinct from the Python environments' stepwise training returns.

The agent can explore in its terminal, then submit a bounded transcript. A fresh
verifier reconstructs the student and replays that transcript using trusted code.
Transcript counters exclude discarded explorations. Report Harbor wall time,
agent tokens and external training costs alongside those counters; equal
submitted edit counts alone do not establish equal compute.

## Direct-program track

The reusable interface protocol is `direct-edit-bench-v1`, separate from the
earlier frozen-editor pilot. It supplies a JSON action contract, a restricted
tensor interpreter and per-episode access limits. It uses no student optimizer.
See the [API](agents.md) for exact actions and defaults.

Let `V(s) = function_accuracy(s) + 0.25 * case_accuracy(s)` by default, and
`J_t = V(coding_t) - V(coding_start)
       - retention_weight * max(0, V(retention_start) - V(retention_t))`.

An edit/rollback rewards `J_t - J_(t-1) - action_cost`; default action cost is
0.001. Stop has no edit cost. Diagnostics and invalid requests have the separate
costs listed in the API. Config changes must accompany results.

On closure, acceptance requires nondecreasing coding function accuracy, no lost
previously correct retention function and nondecreasing coding/retention case
accuracy. A failed gate restores the episode's starting weights and removes the
candidate objective contribution from return. Spent costs remain charged. The
raw candidate and its development results are retained.

Terminal evaluation happens only after decisions end. It reports the starting,
candidate and accepted checkpoints independently. Do not select edits from
terminal scores or conceal regressions behind rollback. For a multi-condition
study, finish every policy decision before opening any condition's terminal tests.

## Baselines and controls

Use no edit and fixed/tuned numerical updates at minimum. Compare a trained editor
with the same frozen editor under identical access and decoding budgets. Separate
gradient-driven LoRA recipes from direct tensor programming; equal edit counts
do not imply equal compute.

`coding.direct_controls.matched_controls` constructs random-sign and other-layer
controls from the raw authored delta. Requested support/norm matching happens
before student-dtype rounding. Report realized norms and changed-entry counts.
Reject an infeasible control rather than silently rescaling it. One control draw
does not establish a causal mechanism; repeated controls and localization tests
are needed for interpretability claims.

## Reporting and reproduction

Record source hash/commit, model ID and immutable revision, dtype, device,
decoding settings, seeds, task protocol, budgets, reward configuration, actions,
failures and raw/accepted outcomes. Record inference/training costs for the editor
separately from student diagnostics, scoring rounds and optimizer steps. Current
access counters are not a FLOP or complete wall-time measure. Direct programs
are validated before scoring, including an additional read-only validation pass.

Freeze training and tuning before test and reserved-family runs. Use paired task
comparisons, independent editor-training seeds and uncertainty at the appropriate
task/family level. The CPU runner provides paired bootstrap summaries over task
seeds; those intervals do not measure variation across policy-training seeds.

Exact replay is checked within the same runtime. Cross-device or cross-library
bitwise equality is not promised. The current task set and limited pilots do not
support claims about AGI, general self-improvement or broad model superiority.
