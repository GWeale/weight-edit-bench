"""Show observed difficulty without hiding unsupported submissions."""

import argparse
import html
import json
from pathlib import Path


def render(summary):
    labels = {"micro":"Original tasks", "bridge":"Intermediate", "stress":"Stress test"}
    rows=[]
    for key,label in labels.items():
        s=summary["cohorts"][key]
        n,correct=s["programs"],s["correct"]
        unsupported=s["failure_categories"].get("unsupported_subset",0)
        rows.append(f'<section class="row"><span class="label">{label}</span>'
            f'<div class="track" role="img" aria-label="{correct} of {n} correct, {unsupported} unsupported">'
            f'<span class="passed" style="width:{100*correct/n:.4f}%"></span>'
            f'<span class="unsupported" style="width:{100*unsupported/n:.4f}%"></span></div>'
            f'<span class="value">{correct}/{n}<small>{100*correct/n:.0f}%</small></span></section>')
    families=''.join('<tr><td>'+html.escape(f.replace('_',' '))+'</td><td>'+str(s['correct'])+'/'+str(s['programs'])+'</td><td>'+
        str(s['failure_categories'].get('unsupported_subset',0))+'</td></tr>' for f,s in summary['families'].items())
    bridge=summary['cohorts']['bridge']
    rejected=bridge['failure_categories'].get('unsupported_subset',0)
    support_note=(f"{rejected} intermediate submissions use unsupported Python features. Among the "
        f"{bridge['programs']-rejected} supported submissions, {bridge['correct']} pass. "
        f"The stress tier has {summary['cohorts']['stress']['failure_categories'].get('unsupported_subset',0)} unsupported submissions.")
    retained=summary['cohorts']['retention']
    scope_note=(f"{len(summary['manifest']['seeds'])} case seeds; prompts repeat across some seeds, so these are not independent task families. "
        f"The {len(summary['manifest']['reserved_families_not_evaluated'])} reserved evaluation families remain unopened. "
        f"Existing-skill anchors passed {retained['correct']}/{retained['programs']} before editing.")
    model=html.escape(summary['manifest']['model']['model_id'].split('/')[-1])
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calibrated coding difficulty</title><style>
body{margin:0;font-family:var(--font-sans,system-ui);color:var(--foreground,#162c3a)}main{max-width:850px;margin:auto}
p{font-size:14px;line-height:1.5}.lead{font-size:18px;font-weight:650}.note{color:var(--muted-foreground,#536571);font-size:13px}
.row{display:grid;grid-template-columns:135px minmax(0,1fr) 58px;gap:12px;align-items:center;padding:16px 0;border-bottom:1px solid var(--border,#c8d3dc)}
.label{font-size:14px;font-weight:600}.track{display:flex;height:26px;background:var(--secondary,#e7ecf0)}.passed{background:var(--chart-1,#096cbc)}
.unsupported{background:var(--warning,#bd7d14);background-image:repeating-linear-gradient(135deg,transparent,transparent 4px,var(--border,#d2b36f) 4px,var(--border,#d2b36f) 6px)}
.value{font:14px var(--font-mono,monospace);text-align:right}.value small{display:block;font-size:12px;color:var(--muted-foreground,#536571);margin-top:4px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;margin:16px 0}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{width:12px;height:12px;display:inline-block}
.other{background:var(--secondary,#e7ecf0);border:1px solid var(--border,#c8d3dc)}
.axis{display:flex;justify-content:space-between;margin:8px 70px 12px 147px;font-size:12px;color:var(--muted-foreground,#536571)}
summary{font-size:13px;cursor:pointer;padding:10px 0}summary:focus-visible{outline:3px solid var(--ring,#096cbc)}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px 4px;text-align:left;border-bottom:1px solid var(--border,#c8d3dc)}
@media(max-width:450px){.row{grid-template-columns:1fr 58px;gap:8px}.label{grid-column:1/-1}.axis{margin-left:0}.lead{font-size:17px}}
</style></head><body><main><p class="lead">Measured difficulty across the three tiers</p>
<p class="note">'''+model+''' unchanged · development calibration · up to '''+str(summary['manifest']['max_new_tokens'])+''' generated tokens. New tasks use 64 executable cases and a fixed function signature.</p>
'''+''.join(rows)+'''<div class="axis"><span>0%</span><span>Correct functions</span><span>100%</span></div>
<div class="legend"><span><i class="swatch passed"></i>Correct</span><span><i class="swatch unsupported"></i>Unsupported submission</span><span><i class="swatch other"></i>Wrong answer / runtime error</span></div>
<p>'''+html.escape(support_note)+'''</p>
<p class="note">'''+html.escape(scope_note)+'''</p>
<details><summary>Inspect individual development families</summary><table><thead><tr><th>Family</th><th>Correct</th><th>Unsupported</th></tr></thead><tbody>'''+families+'''</tbody></table></details>
</main></body></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(render(json.loads(args.summary.read_text())))
    print(args.out)


if __name__ == "__main__":main()
