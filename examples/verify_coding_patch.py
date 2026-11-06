"""Verify a saved student's tensor bits and generated programs in a fresh process."""

import argparse
import json
from pathlib import Path

from weight_edit_lab.coding.backend import Student, tensor_digest
from weight_edit_lab.coding.environment import CodingConfig, episode_problems
from weight_edit_lab.runner import atomic_json, source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--starting-patch", type=Path, help="Also verify an interactive episode's inherited rollback target")
    args = parser.parse_args()
    record = json.loads(args.episode.read_text())
    manifest = json.loads(args.model.read_text())
    if {k: manifest[k] for k in ("model_id", "revision")} != record["model"]:
        raise ValueError("episode/model identity mismatch")
    decoding = record["decoding"]
    student = Student(manifest, device=args.device, max_new_tokens=decoding["max_new_tokens"],
                      batch_size=decoding["batch_size"])
    if str(student.dtype) != decoding["dtype"]:
        raise ValueError("verify on a device with the recorded dtype")
    hashes = student.load_patch(args.episode.parent / record["patch"]["file"])
    if hashes != record["patch"]["tensor_hashes"]:
        raise RuntimeError("patch tensor hashes differ")
    lineage_matches = None
    if "lineage" in record:
        lineage_matches = student.checkpoint_id() == record["lineage"]["accepted"]
        if not lineage_matches:
            raise RuntimeError("accepted checkpoint lineage differs after reload")
    baseline_hashes = {name: tensor_digest(t) for name,t in student.originals.items()}
    changed = sum(hashes[name] != baseline_hashes[name] for name in hashes)
    verified = {}
    for key, split in (("coding", "test"), ("retention", "retention_test")):
        evaluation = student.evaluate(episode_problems(CodingConfig(**record["config"]),split, record["seed"]))
        expected = record["final"][key]
        fields = ("name", "code", "grade", "tokens", "truncated")
        actual_rows = [{k: r[k] for k in fields} for r in evaluation["rows"]]
        expected_rows = [{k: r[k] for k in fields} for r in expected["rows"]]
        # Diagnostic text was added later; compare stable grading semantics.
        for row in actual_rows + expected_rows:
            row["grade"] = {k:row["grade"][k] for k in ("passed","total","correct","error")}
        if actual_rows != expected_rows:
            raise RuntimeError(f"reloaded {key} outputs differ")
        verified[key] = len(actual_rows)
    student.restore()
    for name, expected in baseline_hashes.items():
        if tensor_digest(student.model.get_submodule(name).weight) != expected:
            raise RuntimeError("rollback failed to restore original tensor bits")
    inherited_rollback = None
    if args.starting_patch:
        student.load_patch(args.starting_patch)
        parent = student.snapshot()
        parent_id = student.checkpoint_id()
        if parent_id != record["lineage"]["episode_start"]:
            raise RuntimeError("inherited checkpoint differs from the recorded start")
        student.load_patch(args.episode.parent / record["patch"]["file"])
        student.restore_snapshot(parent)
        inherited_rollback = student.checkpoint_id() == parent_id
        if not inherited_rollback:
            raise RuntimeError("rollback failed to preserve the inherited checkpoint")
        student.restore()
    result = {"tensor_hashes_match": True, "changed_student_tensors": changed,
        "identical_generated_programs": verified, "rollback_bit_exact": True,
        "accepted_lineage_matches": lineage_matches, "inherited_rollback_bit_exact": inherited_rollback,
        "source_matches_episode": record.get("source_hash") == source_hash(),
        "scope": "checkpoint output verification, not bit-exact replay of optimizer training"}
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
