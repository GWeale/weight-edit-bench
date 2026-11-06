# Contributing

Start with the [benchmark contract](docs/public/benchmark.md) and
[agent API](docs/public/agents.md). Small fixes, stronger task generators,
independent replications and new editor integrations are welcome.

```sh
python -m pip install -e '.[coding,gym,dev]'
python -m unittest discover -s tests -v
python -m build
```

The core package has no runtime dependencies. Tests that require optional
libraries skip when those libraries are absent. The full test suite uses tiny
local fixtures and does not download model weights or contact a model API.
Browser check scripts are optional manual checks; they are not run by unittest.

For Harbor changes, use Python 3.12+, install `.[harbor]`, and run
`bash examples/harbor-smoke.sh runs/harbor-smoke` with Docker available. This runs
real CPU oracle/no-edit trials and checks their verifier artifacts. It downloads
the base container image but no model weights. Unit tests alone do not establish
that Harbor can build and run a task. Keep coding-image qualification separate.

For a new agent, use the public observations and record its actions and budget.
Do not inspect scoring inputs, terminal labels or another condition's results
while choosing an edit. Report failures and rejected candidates. A failed run is
not a zero-loss success. Keep development and final evaluation separate.

For a task or scoring change, add a reference solution, independent oracle checks,
edge cases and a protocol version change when existing scores would change.
Do not compare scores from different protocols as if they were paired trials.

Pull requests should explain the behavior changed and the checks run. Include
regression tests for substantive behavior. Keep checkpoints, credentials, raw
agent conversations, personal paths and paid-service configuration out of Git.
Contributions are accepted under the repository's MIT license.

This is an early research benchmark. New results should include the source
commit, model revision, dtype, device, seeds, budgets, decoding settings, raw and
accepted outcomes, and relevant training costs. Claims of RL benefit require
frozen-editor and tuned baseline controls across independent training seeds.
