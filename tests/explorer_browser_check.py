"""Prepare browser interaction checks for T3's isolated renderer; no browser launch."""

import argparse
import json
from pathlib import Path

from examples.render_run_explorer import render


CHECKS = r"""
<script>
(async()=>{
try {
 const check=(ok,label)=>{if(!ok)throw Error(label)}, original=JSON.parse(document.getElementById('run-data').textContent);
 const byId=id=>document.getElementById(id), api=window.explorer;
 check(api.state.status==='completed','actual completed run loaded');
 check(document.querySelectorAll('#episodes button').length===4,'all four episodes');
 const visit=index=>{byId('scrubber').value=index;byId('scrubber').dispatchEvent(new Event('input'))};
 for(const key of [...new Set(original.events.map(e=>e.episode))]){
  api.select(key);for(let i=0;i<api.state.eventCount;i++)visit(i);
 }
 api.select('inherited-astra');visit(1);
 check(byId('coding-value').textContent==='2 / 4','starting score');
 check(byId('tensors').textContent.includes('No numerical edit'),'baseline has no future weights');
 check(document.querySelectorAll('#trajectory .point').length===1,'trajectory has no future measurements');
 visit(5);check(byId('coding-value').textContent==='4 / 4','intermediate coding gain');
 check(byId('retention-value').textContent==='3 / 4','intermediate retained-skill loss visible');
 check(document.querySelectorAll('.tensor').length===4,'four measured tensors');
 visit(7);check(byId('coding-value').textContent==='3 / 4'&&byId('retention-value').textContent==='4 / 4','recovery tradeoff');
 check(byId('observed').textContent==='-25.0 pp / +25.0 pp','prediction compared with immediate outcome');
 document.querySelectorAll('#trajectory .point')[2].dispatchEvent(new MouseEvent('click'));
 check(api.state.cursor===5&&byId('coding-value').textContent==='4 / 4','trajectory point jumps to measured edit');
 visit(11);byId('previous').click();check(byId('coding-note').textContent.startsWith('Development'),'previous event hides final metric');
 byId('next').click();check(byId('coding-note').textContent.startsWith('Final'),'next restores terminal metric');
 byId('program-kind').value='retention';byId('program-kind').dispatchEvent(new Event('change'));
 check(document.querySelectorAll('.program').length===4,'retention program inspection');
 byId('program-kind').value='coding';byId('program-kind').dispatchEvent(new Event('change'));
 document.querySelector('.program').open=true;document.querySelectorAll('.tensor').forEach(n=>n.open=true);
 byId('scrubber').focus();check(document.activeElement===byId('scrubber'),'keyboard focus');
 check(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow with details open');
 check(document.querySelectorAll('#comparison .comparison-row').length===3,'all final comparators');
 check([...document.querySelectorAll('#comparison .comparison-row')].at(-1).textContent.includes('5/8'),'real Astra total');
 check(byId('comparison').textContent.includes('387/512 coding cases'),'partial-case regression stays visible');
 api.select('base-astra');visit(7);check(byId('tensors').textContent.includes('Rollback restored'),'rollback does not show discarded weights as active');
 const hostile=structuredClone(original);
 hostile.events.filter(e=>e.episode==='inherited-astra'&&e.kind==='proposed').at(-1).decision.policy.rationale='<img src=x onerror="window.injected=true">';
 api.load(hostile);api.select('inherited-astra');
 check(!byId('rationale').querySelector('img')&&!window.injected,'editor text is not executable HTML');
 const before=api.state.eventCount;try{api.load({schema:'wrong'})}catch{}check(api.state.eventCount===before,'invalid import preserves current run');
 const broken=structuredClone(original);broken.events.at(-1).record.final.coding.rows[0].grade=null;
 api.select('inherited-ordinary_lora');const saved=api.state;try{api.load(broken)}catch{}
 check(api.state.episode===saved.episode&&byId('coding-value').textContent==='2 / 4','incomplete event import restores prior display');
 const file=new File([JSON.stringify(original)],'live.json',{type:'application/json'}), transfer=new DataTransfer();transfer.items.add(file);
 byId('import').files=transfer.files;byId('import').dispatchEvent(new Event('change'));
 await new Promise(resolve=>setTimeout(resolve,50));check(!byId('rationale').textContent.includes('<img'),'file import loaded valid data');
 const active=structuredClone(original);active.status='running';active.events=active.events.slice(0,4);
 let contents=active,tick=null;const realInterval=window.setInterval;
 window.setInterval=(fn,ms)=>{if(ms===2000){tick=fn;return 987654}return realInterval(fn,ms)};
 await api.followHandle({getFile:async()=>new File([JSON.stringify(contents)],'live.json',{type:'application/json'})});
 check(api.state.following&&byId('connection').textContent.includes('Following'),'following activates only after successful file read');
 contents=original;await tick();check(api.state.status==='completed'&&!api.state.following,'new snapshot arrives and completed run stops polling');
 window.setInterval=realInterval;
 api.select('inherited-astra');byId('play').click();check(byId('play').textContent==='Pause replay','play starts');byId('play').click();check(byId('play').textContent==='Play replay','pause stops');
 api.select('inherited-astra');
 console.log('EXPLORER_QA_PASS '+JSON.stringify({events:original.events.length,width:innerWidth,checks:['replay','trajectory','recovery','rollback','programs','numeric weights','predictions','focus','overflow','XSS','file import','invalid imports','follow updates','completion','play/pause']}));
 document.documentElement.dataset.qa='passed';
}catch(error){console.error('EXPLORER_QA_FAIL '+error.stack);document.documentElement.dataset.qa='failed'}
})();
</script>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    page = render(json.loads(args.snapshot.read_text())).replace("</body>", CHECKS+"</body>")
    args.out.write_text(page)
    print(f"Prepared isolated-browser checks: {args.out}")


if __name__ == "__main__":
    main()
