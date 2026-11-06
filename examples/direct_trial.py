"""One authored direct patch, two matched controls, no edit and a LoRA reference."""

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

from weight_edit_lab.coding.actions import starter_catalog
from weight_edit_lab.coding.backend import Student
from weight_edit_lab.coding.direct_controls import matched_controls
from weight_edit_lab.coding.direct_editor import DirectFileEditor
from weight_edit_lab.coding.environment import CodingConfig, episode_problems
from weight_edit_lab.coding.external import RunJournal
from weight_edit_lab.coding.headroom import paired_changes
from weight_edit_lab.coding.policy_runner import run_policy_episode
from weight_edit_lab.runner import atomic_json, source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a fresh direct trial directory")
    args.out.mkdir(parents=True)
    model = json.loads(args.model.read_text())
    config = CodingConfig(horizon=1, task_suite="algorithmic", curriculum="bridge",
        case_weight=.25, max_optimizer_steps=8, max_edit_evaluations=1)
    seed = 37
    labels = {"base-astra_direct": "Astra · direct patch", "base-no_edit": "No edit",
              "base-ordinary_lora": "Fixed LoRA", "base-random_sign": "Random-sign control",
              "base-wrong_layer": "Other-layer control"}
    manifest = {"protocol": "direct-weight-programming-v2", "source_hash": source_hash(),
        "student": {k: model[k] for k in ("model_id", "revision")}, "editor": "gpt-6-astra",
        "editor_weights_trained": False, "edit_track": "direct_program", "config": asdict(config),
        "seed": seed, "states": ["base"], "conditions": list(labels), "episode_labels": labels,
        "default_episode": "base-astra_direct", "max_new_tokens": 512,
        "inspection_limit": 8, "probe_request_limit": 4, "probe_modules_per_request": 2, "authored_patch_limit": 1,
        "protect_case_accuracy": True, "control_seed": 3701,
        "reserved_families_opened": False,
        "scope": "One frozen Astra editor, one pretrained student, seed 37 on four known coding families. Direct patch only; no editor RL or claim of causal understanding. Controls match requested delta norms before dtype rounding. LoRA uses gradients and is a separate reference, not compute-matched."}
    atomic_json(args.out/"manifest.json", manifest)
    journal = RunJournal(args.out/"live.json", manifest)
    started = time.monotonic()
    bridge, records, candidates = None, {}, {}
    try:
        student = Student(model, device=args.device, max_new_tokens=512)
        def run(key, policy, step_fn=None):
            candidate_info = {}
            def observe(event):
                if event["kind"] == "evaluated":
                    candidates[key] = student.snapshot()
                    path = args.out/f"{key}-candidate.safetensors"
                    candidate_info.update(candidate_development=copy.deepcopy(event["record"]["development"]),
                        candidate_patch={"file": path.name, "tensor_hashes": student.save_patch(path)})
                journal.emit(key, event)
            record = run_policy_episode(student, policy, seed, args.out/f"{key}.json",
                config=config, policy_id=key, protect_case_accuracy=True, on_event=observe, step_fn=step_fn)
            record.update(candidate_info)
            records[key] = record
            atomic_json(args.out/f"{key}.json", record)
            print(f"Adaptation closed: {key}", flush=True)
            return record

        key = "base-astra_direct"
        bridge = DirectFileEditor(args.out/"exchange", student, on_event=lambda e: journal.emit(key, e))
        direct = run(key, bridge)
        direct["access_cost"] = bridge.costs()
        atomic_json(args.out/"editor-access.json", {"cost": bridge.costs(), "trace": bridge.trace})
        bridge.close("completed")
        bridge = None
        # The author is now closed; controls and terminal tests cannot influence it.
        run("base-no_edit", lambda _: {"action": {"kind": "stop"}})
        action = starter_catalog(student.num_layers)[1]["action"]
        run("base-ordinary_lora", lambda _: {"action": action, "policy": {"rationale": "Fixed eight-step LoRA reference"}})
        student.restore()
        values, control_info = matched_controls(student, candidates.get(key, {}), manifest["control_seed"])
        atomic_json(args.out/"control-manifest.json", control_info)
        for label, tensors in values.items():
            if not tensors:
                journal.value.setdefault("unavailable_controls", {})[label] = "Author made no nonzero numerical patch"
                journal.write()
                continue
            try:
                student.apply_tensor_values(tensors, validate_only=True)
            except ValueError as exc:
                journal.value.setdefault("unavailable_controls", {})[label] = str(exc)
                journal.write()
                continue
            record = run(f"base-{label}", lambda _, label=label: {"action": {"kind": "matched_control", "label": label}},
                         step_fn=lambda env, _, label=label, tensors=tensors: env.step_control(tensors, label))
            record["control"] = control_info

        def score():
            return {kind: student.evaluate(episode_problems(config, split, seed))
                    for kind, split in (("coding", "test"), ("retention", "retention_test"))}
        student.restore()
        baseline = score()
        for key, record in records.items():
            journal.emit(key, {"kind": "terminal_started"})
            student.restore()
            student.load_patch(args.out/record["candidate_patch"]["file"])
            if student.checkpoint_id() != record["lineage"]["candidate"]:
                raise RuntimeError("raw candidate identity differs")
            record["candidate_final"] = score()
            student.restore()
            hashes = student.load_patch(args.out/record["patch"]["file"])
            if hashes != record["patch"]["tensor_hashes"] or student.checkpoint_id() != record["lineage"]["accepted"]:
                raise RuntimeError("accepted patch identity differs")
            record["baseline_final"] = copy.deepcopy(baseline)
            record["final"] = score()
            record["final_changes"] = paired_changes(baseline, record["final"])
            record["candidate_final_changes"] = paired_changes(baseline, record["candidate_final"])
            atomic_json(args.out/f"{key}-terminal.json", record)
            journal.emit(key, {"kind": "terminal_completed", "record": record})
        journal.finish("completed", wall_seconds=time.monotonic()-started)
        print("Direct-program trial complete", flush=True)
    except BaseException as exc:
        if bridge:
            bridge.close("failed")
        journal.finish("failed", error=f"{type(exc).__name__}: {str(exc)[:400]}")
        raise


if __name__ == "__main__":
    main()
