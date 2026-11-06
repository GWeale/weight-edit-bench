"""Prepare checks against the recorded direct pilot for the isolated renderer."""

import argparse
import json
from pathlib import Path
from examples.render_run_explorer import render


CHECKS = r"""
<script>
(()=>{try{
 const byId=id=>document.getElementById(id), api=window.explorer;
 const original=JSON.parse(byId('run-data').textContent),check=(ok,label)=>{if(!ok)throw Error(label)};
 const visit=i=>{byId('scrubber').value=i;byId('scrubber').dispatchEvent(new Event('input'))};
 check(api.state.episode==='base-astra_direct','direct editor selected by manifest');
 check(document.querySelectorAll('#episodes button').length===5,'five experimental conditions');
 for(const key of [...new Set(original.events.map(e=>e.episode))]){api.select(key);for(let i=0;i<api.state.eventCount;i++)visit(i)}
 api.select('base-astra_direct');
 check(byId('coding-value').textContent==='2 / 4'&&byId('retention-value').textContent==='4 / 4','measured unchanged result');
 check(byId('steps-value').textContent==='0 / 8','zero optimizer steps');
 check(byId('tensor-code').textContent.includes('outer(d,v)'),'actual authored matrix program');
 check(byId('mechanism').textContent.includes('falsifiable steering hypothesis'),'editor uncertainty preserved');
 check(byId('access-cost').textContent.includes('4 probe requests (3 successful)'),'invalid probe not presented as a forward pass');
 check(byId('access-cost').textContent.includes('2782244'),'actual changed scalar entries');
 check(byId('predicted').textContent.startsWith('+12.5 pp'),'prediction preserved');
 check(byId('observed').textContent==='0.0 pp / 0.0 pp','prediction failed to materialize');
 check(byId('case-prediction').textContent.includes('+10.0 pp'),'case prediction preserved');
 const events=original.events.filter(e=>e.episode==='base-astra_direct');
 const failed=events.findIndex(e=>e.kind==='diagnostic'&&e.diagnostic.result?.error);
 visit(failed);check(byId('event-description').textContent.includes('failed'),'failed diagnostic visible');
 check(byId('tensors').textContent.includes('No numerical edit'),'no future edit in diagnostic frame');
 const probe=events.findIndex(e=>e.kind==='diagnostic'&&e.diagnostic.kind==='probe'&&!e.diagnostic.result?.error);
 visit(probe);check(byId('diagnostic-output').textContent.includes('activations'),'actual activation values inspectable');
 const inspection=events.findIndex(e=>e.kind==='diagnostic'&&e.diagnostic.kind==='inspect');
 visit(inspection);check(byId('tensor-code').textContent.includes('weight('),'inspection source visible');
 api.select('base-ordinary_lora');
 check(byId('coding-value').textContent==='2 / 4','accepted baseline restored');
 check(byId('program-note').textContent.includes('158 / 256'),'accepted programs inspected by default');
 byId('candidate-code').checked=true;byId('candidate-code').dispatchEvent(new Event('change'));
 check(byId('program-note').textContent.includes('135 / 256')&&byId('program-note').textContent.startsWith('Raw candidate'),'raw regression remains inspectable');
 check(byId('tensors').textContent.includes('rejected'),'discarded weight update labeled');
 check(document.querySelectorAll('#comparison .comparison-row').length===5,'every control shown');
 check(byId('comparison').textContent.includes('135/256'),'raw comparison exposes regression');
 byId('comparison-stage').value='final';byId('comparison-stage').dispatchEvent(new Event('change'));
 check(!byId('comparison').textContent.includes('135/256'),'accepted comparison differs from raw');
 byId('comparison-stage').value='candidate_final';byId('comparison-stage').dispatchEvent(new Event('change'));
 api.select('base-astra_direct');byId('candidate-code').checked=false;byId('candidate-code').dispatchEvent(new Event('change'));
 byId('direct-evidence').open=true;document.querySelector('.program').open=true;
 check(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow with code and evidence open');
 byId('direct-evidence').querySelector('summary').focus();check(document.activeElement.tagName==='SUMMARY','keyboard evidence access');
 const hostile=structuredClone(original);hostile.events.find(e=>e.kind==='proposed').decision.action.code='<img src=x onerror="window.injected=true">';
 api.load(hostile);check(!byId('tensor-code').querySelector('img')&&!window.injected,'authored code cannot execute as HTML');
 api.load(original);api.select('base-astra_direct');
 console.log('DIRECT_EXPLORER_QA_PASS '+JSON.stringify({events:original.events.length,width:innerWidth,checks:['all events','inspection','probes','failed requests','authored program','predictions','raw vs accepted','controls','case regressions','no overflow','keyboard access','injection text']}));
}catch(error){console.error('DIRECT_EXPLORER_QA_FAIL '+error.stack)}})();
</script>
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.write_text(render(json.loads(args.snapshot.read_text())).replace('</body>',CHECKS+'</body>'))
    print(args.out)


if __name__=='__main__': main()
