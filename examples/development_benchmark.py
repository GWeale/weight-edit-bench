"""Run the frozen, bounded harder-task development benchmark on local weights."""

import argparse
import json
from pathlib import Path

from weight_edit_lab.coding.development import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--editor-model", type=Path, required=True)
    parser.add_argument("--parent-patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    summary = run_benchmark(json.loads(args.model.read_text()), json.loads(args.editor_model.read_text()),
        args.parent_patch, args.out, args.device)
    print(json.dumps({k: summary[k] for k in ("training_headroom", "rl_executed", "editor_training", "terminal_results", "cost")}, indent=2))


if __name__ == "__main__":
    main()
