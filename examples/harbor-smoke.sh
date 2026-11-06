#!/usr/bin/env bash
set -euo pipefail

output=${1:-runs/harbor-smoke}
if [[ -e "$output" ]]; then
  echo "Choose a fresh output directory: $output" >&2
  exit 1
fi
docker info >/dev/null
docker compose version >/dev/null
docker buildx version >/dev/null
python -m weight_edit_lab.harbor_tasks --out "$output/tasks" --seeds 0
for agent in oracle nop; do
  harbor run --path "$output/tasks" --agent "$agent" --env docker \
    --n-concurrent 1 --n-attempts 1 --max-retries 0 \
    --override-cpus 2 --override-memory-mb 1024 --override-gpus 0 \
    --jobs-dir "$output/$agent"
done
python - "$output" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for agent in ('oracle', 'nop'):
    for path in (root/agent).rglob('result.json'):
        result = json.loads(path.read_text())
        if result.get('exception_info'):
            raise SystemExit(f'{agent}: infrastructure exception at {path}')
    rewards = sorted((root/agent).rglob('reward.json'))
    if len(rewards) != 2:
        raise SystemExit(f'{agent}: expected two completed verifier results, found {len(rewards)}')
    for path in rewards:
        reward = json.loads(path.read_text())['reward']
        details = json.loads(path.with_name('details.json').read_text())
        passed = 0 < reward <= 1 if agent == 'oracle' else reward == 0
        if not passed or details['status'] != 'graded':
            raise SystemExit(f'{agent}: unexpected result at {path}: {reward}, {details["status"]}')
        print(f'{agent}: {path.parent.parent.name}: reward={reward:.6f}, graded')
print('Harbor CPU controls passed: two positive references and two zero-reward no-edit trials.')
PY
