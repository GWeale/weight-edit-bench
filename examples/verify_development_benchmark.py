"""Fresh-process audit of matrix identities and saved editor decisions."""

import argparse
import gc
import json
from pathlib import Path

from weight_edit_lab.coding.development import matrix_summary
from weight_edit_lab.coding.editor import EditorPolicy
from weight_edit_lab.coding.hard_tasks import EVAL_FAMILIES
from weight_edit_lab.coding.headroom import paired_changes
from weight_edit_lab.runner import atomic_json, source_hash


def verify_evaluation(evaluation, count):
    rows = evaluation["rows"]
    if len(rows) != count or any(r["grade"]["total"] != 64 for r in rows):
        raise RuntimeError("unexpected program or test-case count")
    if any(r["grade"]["correct"] != (r["grade"]["passed"] == 64) for r in rows):
        raise RuntimeError("full correctness differs from passed test cases")
    accuracy = sum(r["grade"]["correct"] for r in rows)/count
    case_accuracy = sum(r["grade"]["passed"] for r in rows)/(64*count)
    if evaluation["accuracy"] != accuracy or evaluation["case_accuracy"] != case_accuracy:
        raise RuntimeError("aggregate accuracy differs from individual programs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--editor-model", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    summary = json.loads((root/"summary.json").read_text())
    manifest = json.loads((root/"manifest.json").read_text())
    editor_manifest = json.loads(args.editor_model.read_text())
    if manifest != summary["manifest"] or source_hash() != manifest["source_hash"]:
        raise RuntimeError("source or manifest identity differs")
    if {k: editor_manifest[k] for k in ("model_id", "revision")} != manifest["editor"]:
        raise RuntimeError("editor identity differs")
    if (root/"failure.json").exists():
        raise RuntimeError("run has a recorded failure")
    groups = {}
    records_checked = 0
    reserved_rows = 0
    for partition, seeds in (("training", manifest["train_seeds"]), ("validation", manifest["validation_seeds"])):
        groups[partition] = {}
        for state, identity in manifest["starting_checkpoints"].items():
            for seed in seeds:
                key = f"{state}-seed-{seed}"
                records = {}
                for candidate in manifest["catalog"]:
                    label = candidate["label"]
                    r = json.loads((root/partition/f"{key}-{label}.json").read_text())
                    if (r["source_hash"] != manifest["source_hash"] or r["config"] != manifest["config"]
                            or r["lineage"]["episode_start"] != identity or "final" in r
                            or r["seed"] != seed or r["policy_id"] != f"recipe-{label}"
                            or not r["done"] or r["self_editing_verified"]):
                        raise RuntimeError("matrix episode identity or feedback boundary differs")
                    reserved_rows += sum(any(row["name"].endswith("-"+f) for f in EVAL_FAMILIES)
                        for kind in ("coding", "retention") for row in r["baseline_development"][kind]["rows"])
                    for stage in ("baseline_development", "development"):
                        for evaluation in r[stage].values():
                            verify_evaluation(evaluation, manifest["config"]["problems_per_split"])
                    records[label] = r
                    records_checked += 1
                groups[partition][key] = records
        calculated = matrix_summary(groups[partition], manifest["catalog"])
        recorded = summary["training_headroom" if partition == "training" else "validation_headroom"]
        if calculated["summary"] != recorded:
            raise RuntimeError("headroom summary does not match its complete matrix")
    if reserved_rows or manifest["reserved_families_opened"]:
        raise RuntimeError("reserved task families were opened")
    frozen = json.loads((root/"frozen-decisions.json").read_text())
    if frozen != summary["frozen_decisions"]:
        raise RuntimeError("reported decisions differ from the frozen decisions")
    terminal_checked = set()
    for row in summary["terminal_results"]:
        record = json.loads((root/row["record"]).read_text())
        baseline = json.loads((root/"validation"/f"{row['context']}-A-terminal.json").read_text())["final"]
        chosen = groups["validation"][row["context"]][row["recipe"]]
        if (row["recipe"] != frozen[row["context"]][row["condition"]]
                or record["lineage"] != chosen["lineage"] or record["source_hash"] != manifest["source_hash"]
                or record["baseline_final"] != baseline
                or paired_changes(baseline, record["final"]) != row["final_changes"]):
            raise RuntimeError("reported terminal result differs from its frozen choice or baseline")
        for kind in ("coding", "retention"):
            verify_evaluation(record["final"][kind], manifest["config"]["problems_per_split"])
            if (row[f"{kind}_before"] != baseline[kind]["accuracy"]
                    or row[f"{kind}_after"] != record["final"][kind]["accuracy"]):
                raise RuntimeError("reported terminal accuracy differs from its record")
        terminal_checked.add(row["record"])
    verified = {}
    for seed in manifest["editor_seeds"] if summary["rl_executed"] else (0,):
        policy = EditorPolicy(editor_manifest, manifest["catalog"], device=args.device, seed=seed)
        conditions = [("frozen_editor", "initial.safetensors")] if seed == 0 else []
        if summary["rl_executed"]: conditions.append((f"trained_editor_{seed}", "trained.safetensors"))
        for condition, filename in conditions:
            policy.load(root/f"editor-seed-{seed}"/filename)
            if summary["rl_executed"]:
                training = json.loads((root/f"editor-seed-{seed}"/"training.json").read_text())
                expected = training["weights_after" if filename == "trained.safetensors" else "weights_before"]
                if policy.fingerprints() != expected:
                    raise RuntimeError("saved editor tensor hashes differ")
                changed = sum(training["weights_before"][k] != v for k, v in training["weights_after"].items())
                reported = next(r for r in summary["editor_training"] if r["seed"] == seed)
                if changed != reported["changed_adapter_tensors"] or len(training["history"]) != manifest["rl"]["updates"]:
                    raise RuntimeError("editor training record differs from the reported work")
            for key, rs in groups["validation"].items():
                action, _ = policy.choose(rs["A"]["initial_observation"])
                if action["label"] != summary["frozen_decisions"][key][condition]:
                    raise RuntimeError("freshly reloaded editor made a different decision")
            verified[condition] = len(groups["validation"])
        device = str(policy.backend.device)
        del policy
        gc.collect()
        if device == "mps":
            import torch
            torch.mps.empty_cache()
    result = {"source_and_manifest_match": True, "matrix_records_verified": records_checked,
              "headroom_recomputed": True, "reloaded_editor_decisions_match": verified,
              "terminal_records_verified": len(terminal_checked), "reserved_development_rows": reserved_rows,
              "scope": "Saved editor output verification and matrix audit; optimizer training was not rerun."}
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
