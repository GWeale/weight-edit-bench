"""Re-execute public probes and the authored patch, then verify saved outputs."""

import argparse
import json
from pathlib import Path

import torch

from weight_edit_lab.coding.backend import Student, tensor_digest
from weight_edit_lab.coding.environment import CodingConfig, episode_problems
from weight_edit_lab.runner import atomic_json, source_hash
from weight_edit_lab.tasks import digest


def stable(evaluation):
    return [{k: row[k] for k in ("name", "code", "grade", "tokens", "truncated")}
            for row in evaluation["rows"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preserve the prior verification; choose a new output")
    record = json.loads((args.run/"base-astra_direct-terminal.json").read_text())
    access = json.loads((args.run/"editor-access.json").read_text())
    student = Student(json.loads(args.model.read_text()), device=args.device,
        max_new_tokens=record["decoding"]["max_new_tokens"], batch_size=record["decoding"]["batch_size"])
    if str(student.dtype) != record["decoding"]["dtype"]:
        raise ValueError("verification requires the recorded dtype")
    initial_id = student.checkpoint_id()
    probe_count, inspection_count, decision = 0, 0, None
    for event in access["trace"]:
        request = json.loads((args.run/"exchange"/f"request-{event['index']}.json").read_text())
        response = json.loads((args.run/"exchange"/f"response-{event['index']}.json").read_text())
        if digest(request) != event["request_sha256"] or response != event["response"]:
            raise RuntimeError("exchange provenance mismatch")
        if event.get("result", {}).get("error"):
            continue
        if event["kind"] == "probe":
            result = student.probe_activations(event["result"]["probe_id"], response["prompt"], response["modules"])
            if result != event["result"]:
                raise RuntimeError("probe outputs differ on replay")
            probe_count += 1
        elif event["kind"] == "inspect":
            if student.tensor_program(response["code"]).output != event["result"]["output"]:
                raise RuntimeError("weight-inspection output differs on replay")
            inspection_count += 1
        elif event["kind"] == "patch" and event.get("validated"):
            decision = response
    if decision is None:
        raise ValueError("this run contains no authored numerical patch to verify")
    if student.checkpoint_id() != initial_id:
        raise RuntimeError("diagnostics changed the initial checkpoint")
    program = student.tensor_program(decision["code"], writable=True)
    before = {name: student.model.get_submodule(name).weight.detach().cpu().clone() for name in program.writes}
    requested = {name: value-before[name].float() for name, value in program.writes.items()}
    edit = student.edit({"kind": "tensor_program", "code": decision["code"]}, [], [])
    actual_hashes = {name: tensor_digest(t) for name,t in student.snapshot().items()}
    if actual_hashes != record["candidate_patch"]["tensor_hashes"] or student.checkpoint_id() != record["lineage"]["candidate"]:
        raise RuntimeError("authored program did not reproduce candidate tensor bits")
    rounding, probe_shifts = {}, {}
    for name, original in before.items():
        actual = student.model.get_submodule(name).weight.detach().cpu().float()-original.float()
        rounding[name] = {"requested_delta_l2": requested[name].norm().item(), "realized_delta_l2": actual.norm().item(),
            "relative_delta_rounding_error": ((actual-requested[name]).norm()/requested[name].norm().clamp_min(1e-12)).item()}
        for probe_id, entry in student.probe_cache.items():
            if name in entry["values"]:
                probe_shifts.setdefault(probe_id,{})[name] = (actual@entry["values"][name]["input"]).norm().item()
    verified = {}
    config = CodingConfig(**record["config"])
    for stage, patch_key in (("candidate_final", "candidate_patch"), ("final", "patch")):
        student.restore()
        hashes = student.load_patch(args.run/record[patch_key]["file"])
        if hashes != record[patch_key]["tensor_hashes"]:
            raise RuntimeError("saved checkpoint hashes differ")
        verified[stage] = {}
        for kind, split in (("coding","test"),("retention","retention_test")):
            result = student.evaluate(episode_problems(config, split, record["seed"]))
            if stable(result) != stable(record[stage][kind]):
                raise RuntimeError(f"saved {stage} {kind} programs differ")
            verified[stage][kind] = len(result["rows"])
    student.restore()
    if student.checkpoint_id() != initial_id or any(not torch.equal(student.model.get_submodule(name).weight.detach().cpu(), value) for name,value in before.items()):
        raise RuntimeError("rollback did not restore initial tensor bits")
    result = {"replayed_successful_probes": probe_count, "replayed_inspections": inspection_count,
        "program_reproduces_candidate_tensor_bits": True, "verified_programs": verified,
        "rollback_bit_exact": True, "source_matches_episode": record["source_hash"] == source_hash(),
        "optimizer_steps_in_authored_patch": edit["steps"], "rounding": rounding,
        "probe_output_shift_l2_after_dtype_rounding": probe_shifts,
        "scope": "Same-runtime numerical replay and checkpoint output verification. Probe shifts are local linear-layer effects on frozen pre-edit vectors, not evidence of a behavioral mechanism."}
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
