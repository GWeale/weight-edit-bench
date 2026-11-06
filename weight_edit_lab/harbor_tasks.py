"""Export self-contained Harbor tasks; no running service or provider required."""

import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from .environment import Config
from .harbor_runtime import PROTOCOL
from .runner import atomic_json, source_hash
from .tasks import SPLITS

HARBOR_VERSION = "0.25.0"
BASE_IMAGE = "python:3.13.9-slim-bookworm"


def export_task(destination, *, track="cpu", seed=0, mode="retain", split="test", model=None,
                max_new_tokens=512, family_partition="train"):
    if type(seed) is not int or seed < 0 or mode not in ("adapt", "retain") or split not in SPLITS:
        raise ValueError("invalid seed, mode or split")
    if track not in ("cpu", "coding") or family_partition not in ("train", "evaluation"):
        raise ValueError("invalid track or family partition")
    if track == "coding" and (split != "test" or mode != "retain"):
        raise ValueError("coding tasks use retention and --family-partition; --split applies only to CPU tasks")
    if type(max_new_tokens) is not int or not 16 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be in 16..512")
    spec = {"protocol": PROTOCOL, "track": track, "seed": seed, "split": split,
            "source_hash": source_hash(), "harbor_version": HARBOR_VERSION}
    if track == "cpu":
        spec.update(mode=mode, config=asdict(Config()), retention_limit=.05)
    else:
        from .coding.environment import CodingConfig
        from .coding.direct_env import DirectConfig
        if (not isinstance(model, dict) or not isinstance(model.get("model_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["model_id"])
                or not isinstance(model.get("revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", model["revision"])):
            raise ValueError("coding tasks require a model ID and immutable 40-hex revision")
        spec.update(model={"model_id": model["model_id"], "revision": model["revision"], "path": "/opt/student"},
            max_new_tokens=max_new_tokens, access=asdict(DirectConfig()),
            config=asdict(CodingConfig(horizon=3, task_suite="algorithmic", case_weight=.25,
                family_partition=family_partition, max_optimizer_steps=0, max_edit_evaluations=3)))
    package = Path(__file__).parent
    license_path = package.parent/"LICENSE"
    if not license_path.is_file():
        from importlib.metadata import distribution
        dist = distribution("weight-edit-bench")
        license_path = next((Path(dist.locate_file(p)) for p in dist.files or []
                             if p.name == "LICENSE" and ".dist-info" in str(p)), None)
        if license_path is None:
            raise RuntimeError("the installed distribution's MIT license was not found")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    environment, tests, solution = [destination/name for name in ("environment", "tests", "solution")]
    for path in (environment, tests, solution): path.mkdir()
    shutil.copytree(package, environment/"weight_edit_lab", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(license_path, environment/"LICENSE")
    atomic_json(environment/"spec.json", spec)
    atomic_json(tests/"spec.json", spec)
    atomic_json(environment/"submission.json", {"protocol": PROTOCOL, "actions": []})
    docker = f"FROM {BASE_IMAGE}\nWORKDIR /app\nENV PYTHONPATH=/opt/bench PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n"
    if track == "coding":
        docker += ("RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu\n"
                   "RUN pip install --no-cache-dir transformers==4.57.6 safetensors==0.7.0\n")
        downloader = ("from huggingface_hub import snapshot_download\n"
            f"snapshot_download({model['model_id']!r}, revision={model['revision']!r}, local_dir='/opt/student', token=False, "
            "allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE'])\n")
        (environment/"download_model.py").write_text(downloader)
        docker += "COPY download_model.py /opt/download_model.py\nRUN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python /opt/download_model.py\nENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
    docker += "COPY weight_edit_lab /opt/bench/weight_edit_lab\nCOPY LICENSE /opt/bench/LICENSE\nCOPY spec.json submission.json /app/\n"
    (environment/"Dockerfile").write_text(docker)
    memory = 1024 if track == "cpu" else 4096
    timeout = 120 if track == "cpu" else 1800
    (destination/"task.toml").write_text(f'''schema_version = "1.4"
artifacts = ["/app/submission.json"]

[task]
name = "weight-edit-bench/{track}-{mode}-{seed}"
version = "0.2.0"
description = "Improve a neural network through a replayable numerical edit transcript"
authors = [{{name = "George Weale"}}]

[metadata]
category = "machine-learning"
tags = ["weight-editing", "reinforcement-learning", "{track}"]
benchmark_protocol = "{PROTOCOL}"
reference_role = "{'optimizer reference' if track == 'cpu' else 'edit smoke baseline; positive reward not guaranteed'}"

[agent]
timeout_sec = {max(timeout, 300)}

[verifier]
environment_mode = "separate"
timeout_sec = {timeout}
network_mode = "no-network"

[environment]
build_timeout_sec = {600 if track == 'cpu' else 1800}
cpus = 2
memory_mb = {memory}
storage_mb = {2048 if track == 'cpu' else 8192}
gpus = 0
''')
    (tests/"test.sh").write_text("#!/bin/bash\nset -euo pipefail\npython -I /tests/verify.py\n")
    (tests/"verify.py").write_text("import sys\nsys.path.insert(0, '/opt/bench')\nfrom weight_edit_lab.harbor_runtime import verify\nverify('/tests/spec.json', '/app/submission.json', '/logs/verifier')\n")
    (solution/"solve.sh").write_text("#!/bin/bash\nset -euo pipefail\npython -m weight_edit_lab.harbor_runtime reference\n")
    action = '{"delta": [one finite number per weight]}' if track == "cpu" else '{"kind": "edit", "code": "bounded tensor program"}'
    objective = (f"Reduce final test MSE{' plus retention MSE' if mode == 'retain' else ''}. "
        + ("Retained-behavior MSE must stay at or below 0.05. " if mode == "retain" else "")
        + "Reward is clipped relative improvement minus 0.001 per edit. "
        if track == "cpu" else "Improve coding accuracy while preserving every previously correct retention function and partial-case accuracy. ")
    (destination/"instruction.md").write_text(f'''# Edit the student's numerical weights

{objective}The final verifier replays your submitted edits on a fresh student;
modifying the local library, model specification or printed scores does not affect grading.

Your deliverable is `/app/submission.json`, a UTF-8 JSON object with exactly
`"protocol": "{PROTOCOL}"` and `"actions": [...]`. It starts as an empty no-edit
submission. Each action uses {action}.

Inspect the current public observation and replay feedback with:

```sh
python -m weight_edit_lab.harbor_runtime observe
```

Edit the actions array and call observe again to test your candidate. For CPU
tasks, `{{"stop": true}}` ends adaptation. Coding tasks support `inspect`, `probe`,
`edit`, `rollback` and `stop`; the observation supplies the full tensor language
and remaining budgets. CPU tasks allow 20 edits. Coding tasks allow three editing
actions and 16 total tool turns. Do not append actions after termination.

The public observation includes support examples and permitted feedback. Final
test scores are computed separately after your attempt. The submitted transcript
must fit within 256 KiB and the declared episode budget. The separate verifier
grades only this transcript. A missing, malformed or invalid submission
earns zero. Infrastructure failures are recorded as errors.

Exploration inside the agent sandbox is allowed within Harbor's time/resource
limits. The transcript limits do not measure discarded attempts or all agent
compute. Public generators and task seeds are not secret; this task is not a
contamination-proof evaluation. Write the final file before your attempt ends.
''')
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--track", choices=("cpu", "coding"), default="cpu")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--split", choices=SPLITS, default="test", help="CPU task namespace; coding uses --family-partition")
    parser.add_argument("--model", type=Path, help="Pinned model manifest for coding tasks")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--family-partition", choices=("train", "evaluation"), default="train")
    args = parser.parse_args(argv)
    seeds = [int(x) for x in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or len(seeds) > 100:
        raise ValueError("choose 1..100 distinct seeds")
    if args.out.exists():
        raise ValueError("choose a fresh task export directory")
    model = json.loads(args.model.read_text()) if args.model else None
    for seed in seeds:
        for mode in (("adapt", "retain") if args.track == "cpu" else ("retain",)):
            path = args.out/f"{args.track}-{mode}-{seed}"
            export_task(path, track=args.track, seed=seed, mode=mode, split=args.split,
                model=model, max_new_tokens=args.max_new_tokens, family_partition=args.family_partition)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
