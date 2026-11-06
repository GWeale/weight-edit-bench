"""Exercise the offline replay in an isolated headless browser, without a server."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from weight_edit_lab import Config
from weight_edit_lab.agents import Baseline
from weight_edit_lab.replay import render_episode
from weight_edit_lab.runner import run_episode

SCRIPT = r"""
<script>
try {
 const check=(ok,label)=>{if(!ok)throw new Error(label)};
 check(document.querySelectorAll('#weights tr').length===25,'all weights displayed');
 check(document.getElementById('next').disabled,'end state disables next');
 const last=document.getElementById('weights').textContent;
 document.getElementById('previous').click();
 check(document.getElementById('step').value==='2','previous steps backward');
 check(document.getElementById('weights').textContent!==last,'weights change with frame');
 document.getElementById('next').click();
 check(document.getElementById('weights').textContent===last,'next restores exact frame');
 const slider=document.getElementById('step');slider.value='0';slider.dispatchEvent(new Event('input'));
 check(document.getElementById('previous').disabled,'initial state disables previous');
 slider.focus();check(document.activeElement===slider,'slider accepts keyboard focus');
 document.querySelector('details').open=true;
 check(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow with weight table open');
 check(!window.injected,'untrusted agent name stayed text');
 check(document.querySelectorAll('#function path').length===2,'both function curves render');
 check(document.querySelectorAll('#loss path').length===2,'both loss curves render');
 document.documentElement.dataset.qa='passed';
 document.documentElement.dataset.viewport=innerWidth;
}catch(e){document.documentElement.dataset.qa='failed';document.documentElement.dataset.reason=e.message;}
</script>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome")
    parser.add_argument("--prepare-only", action="store_true",
                        help="write the test fixture without launching a browser")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.prepare_only:
        if sys.platform == "darwin":
            parser.error("Direct Chrome launches are disabled on macOS: they caused a startup "
                         "crash dialog in the agent runtime. Use --prepare-only and T3's "
                         "isolated HTML renderer, or run this check on Linux.")
        if not args.chrome:
            parser.error("--chrome is required unless --prepare-only is used")
    args.out.mkdir(parents=True, exist_ok=True)
    record = run_episode(Baseline("adam", 0.03), 0, "adapt", "test", Config(horizon=3))
    record["agent"] = "<script>window.injected=true</script>"
    page = args.out / "fixture.html"
    page.write_text(render_episode(record).replace("</body>", SCRIPT + "</body>"))
    if args.prepare_only:
        print(f"Prepared browser fixture (not browser-verified): {page.resolve()}")
        return
    for width in (1100, 500):
        with tempfile.TemporaryDirectory(prefix="weight-edit-browser-") as profile:
            result = subprocess.run([args.chrome, "--headless=new", "--no-first-run", "--no-default-browser-check",
                                     "--disable-background-networking", "--disable-component-update",
                                     f"--user-data-dir={profile}", f"--window-size={width},1100",
                                     "--virtual-time-budget=1500", "--dump-dom",
                                     f"--screenshot={args.out.resolve() / f'{width}.png'}", page.resolve().as_uri()],
                                    capture_output=True, text=True, timeout=30)
            (args.out / f"{width}.dom.html").write_text(result.stdout)
            (args.out / f"{width}.stderr.txt").write_text(result.stderr)
            if (result.returncode or 'data-qa="passed"' not in result.stdout
                    or f'data-viewport="{width}"' not in result.stdout):
                raise RuntimeError(f"browser verification failed at {width}px; see saved DOM (exit {result.returncode})")
            print(f"Replay interactions, text isolation, focus, and layout passed at {width}px")


if __name__ == "__main__":
    main()
