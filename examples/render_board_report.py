"""Render the measured decision headroom and proposed inheritance test offline."""

import argparse
import json
import math
from pathlib import Path


def render(report):
    s = report["summary"]
    values = [s["fixed_accepted_gain"], s["hindsight_accepted_gain"]]
    headline = ("Choosing a recipe per context added no strict-correctness gain"
                if s["selection_headroom"] <= 1e-12 else "Hindsight edit selection exceeds the best fixed recipe")
    recipe_count = len(report["contexts"][0]["cells"])-1
    scale = max(.25, math.ceil(max(values)/.25)*.25)
    rows = "".join(f'<div class="row"><span>{label}</span><div class="track" role="img" '
        f'aria-label="{value*100:g} percentage point gain"><i class="bar {kind}" style="width:{value/scale*100:g}%"></i></div>'
        f'<strong>{value*100:g} pp</strong></div>'
        for label, value, kind in zip(("Fixed recipe + acceptance gate", "Best recipe per context, with hindsight"), values, ("fixed", "oracle")))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>What a stronger weight-editing benchmark must measure</title><style>
body{{margin:0;color:var(--foreground,#182b38);font-family:var(--font-sans,system-ui)}}main{{max-width:850px;margin:auto}}
p{{font-size:14px;line-height:1.5}}.lead{{font-size:19px;font-weight:650}}.note{{color:var(--muted-foreground,#5b6674);font-size:13px}}
.row{{display:grid;grid-template-columns:245px minmax(0,1fr) 70px;align-items:center;gap:12px;padding:14px 0;border-bottom:1px solid var(--border,#bac5cf);font-size:14px}}
.track{{height:25px;background:var(--secondary,#edf0f3);border:1px solid var(--border,#bac5cf)}}.bar{{display:block;height:25px;background:var(--chart-1,#2368d5)}}
.oracle{{background:var(--warning,#a36a00);background-image:repeating-linear-gradient(135deg,transparent,transparent 5px,var(--background,#fff) 5px,var(--background,#fff) 7px)}}
strong{{font:600 14px var(--font-mono,monospace);text-align:right}}.axis{{display:flex;justify-content:space-between;margin:9px 82px 16px 257px;font-size:12px;color:var(--muted-foreground,#5b6674)}}
.sublead{{font-size:16px;font-weight:600;margin-top:28px}}.lineage{{display:grid;grid-template-columns:1fr 24px 1fr 24px 1fr;align-items:center;text-align:center;gap:6px}}
.state{{border:1px dashed var(--border,#8995a0);padding:12px 6px}}.state b{{font:600 19px var(--font-mono,monospace)}}.state span{{display:block;font-size:12px;line-height:1.4;margin-top:6px}}
@media(max-width:500px){{.row{{grid-template-columns:minmax(0,1fr) 70px;gap:8px}}.row>span{{grid-column:1/-1}}.axis{{margin-left:0}}.lead{{font-size:18px}}}}
</style></head><body><main>
<p class="lead">{headline}</p>
<p class="note">Development reanalysis · {s['episode_contexts']} contexts · {recipe_count} executed recipes · both comparisons reject coding regressions and lost retained programs.</p>
{rows}<div class="axis"><span>0</span><span>{scale*100:g} pp</span></div>
<p>Extra fully correct programs, relative to each starting model. Measured selection headroom: <b>{s['selection_headroom']*100:g} percentage points</b>.</p>
<p class="note">Hindsight is a diagnostic ceiling, not a deployable editor. These are reused development results, not evidence of generalization. The harder suite still needs its own editability measurement.</p>
<p class="sublead">The proposed test for inherited self-editing</p>
<div class="lineage" role="img" aria-label="Checkpoint W0 chooses an update producing W1. W1 must itself choose the next update producing W2. This test has not been executed.">
<div class="state"><b>W₀</b><span>chooses its update</span></div><span aria-hidden="true">→</span>
<div class="state"><b>W₁</b><span>chooses the next update</span></div><span aria-hidden="true">→</span>
<div class="state"><b>W₂</b><span>coding and editing retested</span></div></div>
<p class="note">Each descendant must produce its own next directive. The current runner records target-checkpoint lineage; a descendant has not yet acted as the editor.</p>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(json.loads(args.report.read_text())))


if __name__ == "__main__":
    main()
