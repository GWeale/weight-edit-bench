# Baselines and the initial direct-edit result

The initial direct-edit pilot produced **no measured coding gain**. It establishes
that a frozen external editor can inspect tensors, author a numerical patch and
have that patch evaluated and replayed. It does not establish that the editor
understands the changed weights or that RL improves editing.

The pilot used one Qwen2.5-Coder-0.5B-Instruct checkpoint, seed 37, four known
coding families, four retention functions and one authored edit. The external
editor was internally labeled Astra; this repository does not provide that
hosted model or claim a publicly reproducible model identity for it. A sanitized
[reference record](../../benchmarks/reference/direct-v1.json) preserves the student
revision, authored program and measured aggregate results. Private controller
exchanges and downloaded weights are not distributed.

| Condition | Raw coding functions | Raw coding cases | Retention functions | Student optimizer steps |
| --- | ---: | ---: | ---: | ---: |
| No edit | 2/4 | 158/256 | 4/4 | 0 |
| Authored direct patch | 2/4 | 158/256 | 4/4 | 0 |
| Random-sign control | 2/4 | 158/256 | 4/4 | 0 |
| Other-layer control | 2/4 | 158/256 | 4/4 | 0 |
| Fixed LoRA reference | 1/4 | 135/256 | 4/4 | 8 |

The LoRA candidate failed the development gate and rolled back. Its accepted
checkpoint scored 2/4 and 158/256. It is not a compute-matched baseline.
All conditions passed 256/256 retention cases. One sample per control is not a
statistical test of mechanism.

The direct patch changed 2,782,244 scalar entries in one MLP output matrix. Its
rank-one construction in float32 used three activation vectors to steer the local
response while protecting one control direction. Student-dtype rounding changed
the exact update. Two coding program texts changed, but scores did not improve.

A fresh process replayed the successful probes and reproduced the authored
candidate tensor bits, saved candidate/accepted final programs, and rollback.
This verification establishes the numerical effect, not the hypothesized
behavioral meaning.

The original file-exchange interface omitted its two-module probe limit, costing
one rejected request. Recorded cost metadata also mislabeled four requests as
four forwards; only three forwards executed. The preserved reference record
states the correction. The reusable public interface documents the limit and
counts requests separately from forwards. Its diagnostic costs and multi-turn
protocol differ from the original one-patch pilot; do not compare their returns
as the same benchmark setting.

## Run your own comparisons

For CPU experiments, run `train` on training tasks, `tune` on validation tasks,
then `evaluate` on test and OOD tasks with frozen checkpoints. Repeat training
with independent seeds. `report` creates an offline comparison page; `replay`
verifies exact logged numerical edits before rendering an episode.

For coding recipes, `weight-edit-coding pilot --help` exposes the existing
categorical editor-training baseline. It changes local editor-adapter weights
and selects a small recipe catalog. It does not train a free-form tensor-program
policy. The `download` command can prepare separate student and editor manifests.

For direct programs, run the same editor and no-edit policy over matched seeds,
recording each episode through `DirectEditEnv.save`. Start on development
families, calibrate difficulty, freeze the policy, then use
`CodingConfig(family_partition="evaluation", task_suite="algorithmic",
max_optimizer_steps=0)` for reserved-family episodes. Do not tune after seeing
those results and continue calling them held-out tests.

Historical `examples/direct_trial.py` preserves the five-condition file-exchange
experiment for research reuse, but requires a separately implemented controller.
The public JSON environment is the recommended integration surface.
