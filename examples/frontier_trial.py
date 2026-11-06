"""A bounded frozen-Astra pilot; a separate controller answers public requests."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from weight_edit_lab.coding.actions import starter_catalog
from weight_edit_lab.coding.backend import Student
from weight_edit_lab.coding.development import recipe_policy
from weight_edit_lab.coding.environment import CodingConfig
from weight_edit_lab.coding.external import FileEditor, RunJournal
from weight_edit_lab.coding.policy_runner import run_policy_episode
from weight_edit_lab.runner import atomic_json, source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--starting-patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a fresh trial directory; no implicit resumption")
    args.out.mkdir(parents=True)
    model = json.loads(args.model.read_text())
    config = CodingConfig(horizon=3, task_suite="algorithmic", curriculum="bridge",
        case_weight=0., max_optimizer_steps=24, max_edit_evaluations=3)
    manifest = {"protocol": "frozen-frontier-development-v1", "source_hash": source_hash(),
        "student": {k: model[k] for k in ("model_id", "revision")},
        "editor": "gpt-6-astra", "editor_weights_trained": False,
        "editor_interface": "fresh native Astra agent per student state, file exchange",
        "config": asdict(config), "seed": 2, "states": ["base", "inherited"],
        "conditions": ["astra", "ordinary_lora"], "max_new_tokens": 512,
        "parent_patch_sha256": hashlib.sha256(args.starting_patch.read_bytes()).hexdigest(),
        "reserved_families_opened": False,
        "scope": "Development demonstration on previously examined task parameters; not an untouched test, scaling study, or mechanistic interpretability result."}
    atomic_json(args.out/"manifest.json", manifest)
    journal = RunJournal(args.out/"live.json", manifest)
    began = time.monotonic()
    bridge = None
    records = []
    try:
        student = Student(model, device=args.device, max_new_tokens=512)
        for state in manifest["states"]:
            patch = args.starting_patch if state == "inherited" else None
            # Finish every policy decision for a context before opening its
            # terminal results. A fresh Astra agent is used for the next state.
            for condition in manifest["conditions"]:
                key = f"{state}-{condition}"
                if condition == "astra":
                    bridge = FileEditor(args.out/"exchange"/state, student.num_layers)
                    policy = bridge
                else:
                    policy = recipe_policy(starter_catalog(student.num_layers)[1])
                print(f"Starting {key}", flush=True)
                record = run_policy_episode(student, policy, 2, args.out/f"{key}.json",
                    config=config, starting_patch=patch, policy_id=condition, final=False,
                    on_event=lambda event, key=key: journal.emit(key, event))
                records.append((key, record))
                if bridge is not None:
                    bridge.close("completed")
                    bridge = None
        # Final tests are opened only after ALL controller decisions are done.
        from weight_edit_lab.coding.environment import episode_problems
        from weight_edit_lab.coding.headroom import paired_changes
        for key, record in records:
            journal.emit(key, {"kind": "terminal_started"})
            student.restore()
            if key.startswith("inherited"):
                student.load_patch(args.starting_patch)
            if student.checkpoint_id() != record["lineage"]["episode_start"]:
                raise RuntimeError("terminal baseline identity differs")
            baseline = {kind: student.evaluate(episode_problems(config, split, 2))
                for kind, split in (("coding", "test"), ("retention", "retention_test"))}
            student.restore()
            hashes = student.load_patch(args.out/record["patch"]["file"])
            if hashes != record["patch"]["tensor_hashes"] or student.checkpoint_id() != record["lineage"]["accepted"]:
                raise RuntimeError("accepted patch identity differs")
            record["baseline_final"] = baseline
            record["final"] = {kind: student.evaluate(episode_problems(config, split, 2))
                for kind, split in (("coding", "test"), ("retention", "retention_test"))}
            record["final_changes"] = paired_changes(baseline, record["final"])
            atomic_json(args.out/f"{key}-terminal.json", record)
            journal.emit(key, {"kind": "terminal_completed", "record": record})
        journal.finish("completed", wall_seconds=time.monotonic()-began)
        print("Completed frozen frontier trial", flush=True)
    except BaseException as exc:
        if bridge is not None:
            bridge.close("failed")
        journal.finish("failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        atomic_json(args.out/"failure.json", {"error": type(exc).__name__, "detail": str(exc)[:300]})
        raise


if __name__ == "__main__":
    main()
