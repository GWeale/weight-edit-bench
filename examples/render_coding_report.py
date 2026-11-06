"""Render the small coding pilot's observed accuracy tradeoff, with no external assets."""

import argparse
import html
import json
import re
from pathlib import Path


def render(summary):
    names = {"no_edit": "No edit", "frozen_editor": "Frozen editor",
             "trained_editor": "RL-trained editor", "tuned_fixed": "Tuned fixed recipe"}
    rows = summary["evaluation"]
    count = summary["manifest"]["config"]["problems_per_split"]
    groups = {key: [r for r in rows if r["condition"] == key] for key in names}
    bars = []
    for key, label in names.items():
        group = groups[key]
        n = len(group)*count
        if not n:
            raise ValueError("missing comparison condition")
        metrics = []
        for metric, title in (("coding", "Coding"), ("retention", "Retention")):
            passed = sum(r["final"][metric]["accuracy"]*count for r in group)
            value = passed/n*100
            if not 0 <= value <= 100:
                raise ValueError("invalid accuracy")
            metrics.append(f'<div class="measure"><span class="metric-label">{title}</span>'
                f'<div class="track"><div class="bar {metric}" style="width:{value:.4f}%"></div></div>'
                f'<span class="value">{passed:.0f}/{n}</span></div>')
        bars.append(f'<section class="row" aria-label="{label}"><div class="name">{label}</div>'
                    f'<div class="measures">{"".join(metrics)}</div></section>')
    same = all(next(r for r in groups["frozen_editor"] if r["seed"] == row["seed"])["action"] == row["action"]
               for row in groups["trained_editor"])
    finding = ("Frozen and RL-trained editors chose the same recipe on all evaluation contexts. "
               "This run does not show an RL advantage." if same else
               "The editor changed some choices after RL. Compare both coding and retention before judging the change.")
    details = []
    for row in rows:
        details.append('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in
            (row["seed"], names[row["condition"]], row["action"],
             f'{row["final"]["coding"]["accuracy"]:.0%}', f'{row["final"]["retention"]["accuracy"]:.0%}'))+'</tr>')
    def model_name(role):
        model = summary["manifest"][role]["model_id"]
        size = re.search(r"-([\d.]+B)-", model)
        return f"{size.group(1)} Qwen" if size and model.startswith("Qwen/") else model
    algorithmic = summary["manifest"]["config"].get("task_suite","micro") == "algorithmic"
    cases = 64 if algorithmic else 12
    scope = ("Reserved editor-evaluation families" if algorithmic else "Four shared task templates")
    context = (f"{model_name('editor')} editor → {model_name('student')} student · Final results after the selected "
               f"recipe, including any recovery step. Each function must pass all {cases} cases.")
    action_names = {a["label"]: a["name"].replace("_", " ") for a in summary["manifest"]["catalog"]}
    actions_text = ". ".join(f"{label} = {action_names[label]}" for label in sorted({r["action"] for r in rows}))
    n_contexts = len(summary["manifest"]["eval_seeds"])
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Coding and retention after weight editing</title>
<style>
body{margin:0;color:var(--foreground,#162c3a);font-family:var(--font-sans,system-ui)}
main{max-width:850px;margin:auto}p{font-size:14px;line-height:1.5;margin:10px 0}
.finding{font-weight:650;font-size:17px;line-height:1.45}.context,.note{color:var(--muted-foreground,#536571)}
.row{display:grid;grid-template-columns:150px minmax(0,1fr);gap:14px;padding:13px 0;border-bottom:1px solid var(--border,#c8d3dc)}
.name{align-self:center;font-weight:600;font-size:14px}.measure{display:grid;grid-template-columns:65px minmax(0,1fr) 36px;gap:8px;align-items:center;margin:5px 0}
.metric-label{font-size:12px}.track{height:18px;background:var(--secondary,#e9eef2);border-left:1px solid var(--border,#c8d3dc)}
.bar{height:18px}.coding{background:var(--chart-1,#0673b5)}.retention{background:var(--chart-2,#b25b08)}
.value{font:13px var(--font-mono,monospace);text-align:right}.axis{display:flex;justify-content:space-between;margin:8px 44px 12px 237px;font-size:12px;color:var(--muted-foreground,#536571)}
.note{font-size:13px;margin:16px 0}summary{cursor:pointer;font-size:13px;padding:8px 0;line-height:1.4}
summary:focus-visible{outline:3px solid var(--ring,#0673b5)}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px}
td,th{text-align:left;padding:7px 5px;border-bottom:1px solid var(--border,#c8d3dc)}th{font-weight:600}
@media(max-width:450px){.row{grid-template-columns:1fr;gap:4px;padding:12px 0}.axis{margin-left:73px}.finding{font-size:16px}.name{font-size:14px}}
</style></head><body><main><p class="finding">''' + html.escape(finding) + '''</p>
<p class="context">''' + html.escape(context) + '''</p>
<div role="group" aria-label="Fully correct coding and retained-skill functions, by editor">''' + ''.join(bars) + '''</div>
<div class="axis"><span>0%</span><span>Fully correct functions</span><span>100%</span></div>
<p class="note">''' + str(n_contexts) + ''' evaluation contexts. ''' + scope + '''; one student checkpoint and one editor-training seed. These counts describe a feasibility pilot, not broad coding generalization or evidence of AGI.</p>
<details><summary>Inspect the evaluation contexts and selected actions</summary><div class="table-wrap"><table>
<thead><tr><th>Seed</th><th>Condition</th><th>Action</th><th>Coding</th><th>Retention</th></tr></thead><tbody>''' + ''.join(details) + '''</tbody></table></div>
<p class="note">''' + html.escape(actions_text) + '''. The tuned fixed recipe was selected on training contexts only.</p></details>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = render(json.loads(args.summary.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document)
    print(args.out)


if __name__ == "__main__":
    main()
