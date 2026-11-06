"""Offline report for the bounded harder-task benchmark trial."""

import argparse
import html
import json
from pathlib import Path


def render(result):
    headroom = result["training_headroom"]
    trained = result["rl_executed"]
    conditions = list(dict.fromkeys(r["condition"] for r in result["terminal_results"]))
    names = {"no_edit": "No edit", "ordinary_lora": "Ordinary LoRA", "tuned_fixed": "Tuned fixed recipe",
             "behavioral_selector": "Behavioral selector", "frozen_editor": "Frozen editor"}
    scores = {condition: {kind: sum(r[f"{kind}_after"] for r in result["terminal_results"]
        if r["condition"] == condition)/sum(r["condition"] == condition for r in result["terminal_results"])
        for kind in ("coding", "retention")} for condition in conditions}
    rows = []
    for condition in conditions:
        records = [r for r in result["terminal_results"] if r["condition"] == condition]
        count = len(records)*result["manifest"]["config"]["problems_per_split"]
        label = names.get(condition, "RL editor · seed " + condition.rsplit("_", 1)[-1])
        bars = []
        for kind in ("coding", "retention"):
            score = sum(r[f"{kind}_after"] for r in records)/len(records)
            passed = round(score*count)
            bars.append(f'<div class="measure"><span>{kind.capitalize()}</span><div class="track" role="img" '
                f'aria-label="{html.escape(label)} {kind}: {passed} of {count} fully correct"><i class="{kind}" style="width:{score*100:g}%"></i></div>'
                f'<b>{passed}/{count}</b></div>')
        rows.append(f'<section class="row"><strong>{html.escape(label)}</strong><div>{"".join(bars)}</div></section>')
    if trained:
        baseline = max(scores[c]["coding"] for c in ("no_edit", "ordinary_lora", "tuned_fixed", "behavioral_selector"))
        rl_scores = [scores[c]["coding"] for c in conditions if c.startswith("trained_editor_")]
        title = ("RL did not beat the strongest simple baseline on terminal coding accuracy."
                 if max(rl_scores) <= baseline else
                 "An RL editor exceeded the simple baselines on terminal coding accuracy in this small pilot.")
    else:
        title = "The harder-task pilot did not meet the threshold for RL training."
    status = (f'{len(result["editor_training"])} editor-training seeds completed. Each row is reported separately.' if trained else
              'Editor RL was skipped under the frozen rule. Frozen-editor and conventional comparisons still ran.')
    fixed, ceiling = headroom["fixed_accepted_gain"], headroom["hindsight_accepted_gain"]
    scale = max(.25, fixed, ceiling)
    opportunity = ''.join(f'<div class="opportunity"><span>{name}</span><div class="track" role="img" aria-label="{value*100:g} percentage point gain">'
        f'<i class="coding" style="width:{value/scale*100:g}%"></i></div><b>{value*100:.2f} pp</b></div>'
        for name, value in (("Fixed recipe + acceptance", fixed), ("Per-context hindsight", ceiling)))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Harder-task weight-editing benchmark results</title><style>
body{{margin:0;font-family:var(--font-sans,system-ui);color:var(--foreground,#1b2c38)}}main{{max-width:850px;margin:auto}}
p{{font-size:14px;line-height:1.5}}.lead{{font-size:19px;font-weight:650}}.note{{font-size:13px;color:var(--muted-foreground,#546270)}}
.sublead{{font-size:16px;font-weight:600;margin-top:25px}}.row{{display:grid;grid-template-columns:170px minmax(0,1fr);gap:12px;align-items:center;padding:13px 0;border-bottom:1px solid var(--border,#bbc6d0)}}
.row strong{{font-size:14px}}.measure{{display:grid;grid-template-columns:66px minmax(0,1fr) 38px;gap:8px;align-items:center;margin:6px 0;font-size:12px}}
.track{{background:var(--secondary,#e8edf1);border:1px solid var(--border,#bbc6d0);height:20px}}.track i{{display:block;height:20px}}
.coding{{background:var(--chart-1,#1e63c6)}}.retention{{background:var(--chart-2,#087f75)}}b{{font:600 13px var(--font-mono,monospace);text-align:right}}
.opportunity{{display:grid;grid-template-columns:200px minmax(0,1fr) 78px;gap:10px;align-items:center;margin:10px 0;font-size:13px}}
.axis{{display:flex;justify-content:space-between;font-size:12px;color:var(--muted-foreground,#546270);margin:5px 88px 0 210px}}
@media(max-width:500px){{.row{{grid-template-columns:1fr;gap:2px}}.opportunity{{grid-template-columns:minmax(0,1fr) 78px}}.opportunity>span{{grid-column:1/-1}}.axis{{margin-left:0}}.lead{{font-size:18px}}}}
</style></head><body><main>
<p class="lead">{title}</p><p>{status}</p>
<p class="sublead">Frozen choices on terminal task parameters</p>
<p class="note">Qwen 1.5B editor → 0.5B student · two related student states · four shared task families · all 64 cases must pass for a function to count.</p>
{''.join(rows)}
<p class="note">Choices were fixed before validation candidate outcomes and terminal tests. Retention bars measure the accepted models; rejected trials still incur their compute costs. These are small pilot counts, not independent models or broad generalization estimates. The four reserved algorithm families remain unopened.</p>
<p class="sublead">Training gate: useful choice between edits</p>
<p class="note">Mean additional fully correct programs across {headroom['episode_contexts']} training contexts. Both choices use the same development acceptance gate. Hindsight selection is a diagnostic ceiling, not a trained editor.</p>
{opportunity}<div class="axis"><span>0</span><span>{scale*100:g} pp</span></div>
<p>Selection headroom: <b>{headroom['selection_headroom']*100:.2f} percentage points</b>. The frozen gate requires at least 5 points and two editable contexts; {headroom['editable_contexts']} contexts were editable.</p>
<p class="note">{result['cost']['matrix_episodes']} matrix episodes · {result['cost']['optimizer_steps']} student optimizer steps · {result['cost']['invocation_wall_seconds']/60:.1f} minutes of runtime. No API calls from the benchmark. This run tests finite-recipe selection; descendant self-editing and causal interpretation remain untested.</p>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(json.loads(args.summary.read_text())))


if __name__ == "__main__":
    main()
