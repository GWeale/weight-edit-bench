"""A bounded, development-only trial of the board's editing benchmark.

This trains the existing finite-action editor only when a preregistered training
matrix shows strict-correctness selection headroom. Reserved families stay closed.
"""

import copy
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from ..runner import atomic_json, source_hash
from ..tasks import digest
from .actions import recipe_actions, starter_catalog
from .environment import CodingConfig, CodingEditEnv, episode_problems
from .headroom import paired_changes, summarize_matrix
from .policy_runner import run_policy_episode


TRAIN_SEEDS = (0, 1)
VALIDATION_SEEDS = (2,)
EDITOR_SEEDS = (0, 1, 2)


def recipe_policy(candidate):
    actions = recipe_actions(candidate)
    index = 0

    def decide(observation):
        nonlocal index
        action = actions[index] if index < len(actions) else {"kind": "stop"}
        index += 1
        return {"action": copy.deepcopy(action)}
    return decide


def matrix_summary(groups, catalog):
    contexts = []
    for key, records in groups.items():
        first = records[catalog[0]["label"]]
        instructions = {kind: sorted(r["instruction"] for r in first["baseline_development"][kind]["rows"])
                        for kind in ("coding", "retention")}
        instructions["support"] = sorted(r["instruction"] for r in first["initial_observation"]["support"])
        signature = digest({"start": first["lineage"]["episode_start"], "instructions": instructions})
        cells = {}
        for candidate in catalog:
            label = candidate["label"]
            r = records[label]
            if r["lineage"]["episode_start"] != first["lineage"]["episode_start"]:
                raise ValueError("candidate starting checkpoints differ")
            # Paired original generations must agree, independent of measured runtime.
            for kind in ("coding", "retention"):
                if r["baseline_development"][kind]["rows"] != first["baseline_development"][kind]["rows"]:
                    raise ValueError("candidate baseline generations differ")
            if r["initial_observation"] != first["initial_observation"]:
                raise ValueError("candidate initial observations differ")
            changes = paired_changes(r["baseline_development"], r["development"])
            cells[label] = {"accepted_gain": changes["coding"]["gain"],
                "accepted": r["finalization"]["accepted"], "candidate_changes": r["finalization"]["candidate_changes"],
                "reward": r["total_reward"], "budget_used": r["budget_used"],
                "accepted_checkpoint": r["lineage"]["accepted"]}
        contexts.append({"context": key, "semantic_context": signature, "cells": cells})
    summary = summarize_matrix(contexts, [c["label"] for c in catalog])
    # This threshold is an execution gate for a tiny pilot, not statistical significance.
    summary["rl_gate_passed"] = summary["selection_headroom"] >= .05 and summary["editable_contexts"] >= 2
    return {"summary": summary, "contexts": contexts}


def behavioral_features(observation):
    return [observation["development"][kind][metric]
            for kind in ("coding", "retention") for metric in ("accuracy", "case_accuracy")]


def behavioral_choice(observation, contexts, rewards):
    query = behavioral_features(observation)
    nearest = min(range(len(contexts)), key=lambda i: sum((a-b)**2 for a, b in
                  zip(query, behavioral_features(contexts[i]))))
    return max(range(len(rewards[nearest])), key=rewards[nearest].__getitem__)


def run_benchmark(student_manifest, editor_manifest, parent_patch, out, device="auto"):
    from .backend import Student
    from .editor import EditorPolicy, train_editor
    from .experiment import versions
    import torch

    out, parent_patch = Path(out), Path(parent_patch)
    started = time.monotonic()
    cfg = CodingConfig(horizon=3, task_suite="algorithmic", family_partition="train", curriculum="bridge",
                       case_weight=0., max_optimizer_steps=24, max_edit_evaluations=3)
    student = Student(student_manifest, device=device, max_new_tokens=512)
    catalog = starter_catalog(student.num_layers, with_recovery=True)
    catalog[0] = {"label": "A", "name": "stop_without_editing", "action": {"kind": "stop"}}
    parent_hash = hashlib.sha256(parent_patch.read_bytes()).hexdigest()
    states = {"base": None, "inherited": parent_patch}
    checkpoint_ids = {}
    for state, path in states.items():
        student.restore()
        if path is not None:
            student.load_patch(path)
        checkpoint_ids[state] = student.checkpoint_id()
    if len(set(checkpoint_ids.values())) != 2:
        raise ValueError("this protocol requires two different student states")
    manifest = {"protocol": "gated-development-benchmark-v1", "source_hash": source_hash(), "versions": versions(),
        "student": {k: student_manifest[k] for k in ("model_id", "revision")},
        "editor": {k: editor_manifest[k] for k in ("model_id", "revision")},
        "parent_patch_sha256": parent_hash, "starting_checkpoints": checkpoint_ids, "config": asdict(cfg),
        "train_seeds": list(TRAIN_SEEDS), "validation_seeds": list(VALIDATION_SEEDS),
        "editor_seeds": list(EDITOR_SEEDS), "catalog": catalog,
        "decoding": {"max_new_tokens": 512, "device": str(student.device), "dtype": str(student.dtype)},
        "rl": {"updates": 24, "group_size": 8, "epochs": 2, "lr": .0003, "clip": .2, "beta": .02},
        "gate": {"minimum_selection_headroom": .05, "minimum_editable_contexts": 2},
        "retention_tolerance": 0., "reserved_families_opened": False,
        "scope": "Two related student states; four shared bridge families. Validation changes parameters, not task families or pretrained models."}
    manifest_path = out/"manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("run identity changed; choose a fresh output directory")
    else:
        atomic_json(manifest_path, manifest)
    if (out/"failure.json").exists():
        raise RuntimeError("this run records a failed attempt; inspect it before starting a new run")

    def collect(partition, seeds):
        groups = {}
        for state, patch in states.items():
            for seed in seeds:
                key = f"{state}-seed-{seed}"
                records = {}
                for candidate in catalog:
                    label = candidate["label"]
                    path = out/partition/f"{key}-{label}.json"
                    if path.exists():
                        record = json.loads(path.read_text())
                        expected_actions = recipe_actions(candidate)
                        if expected_actions[-1]["kind"] != "stop":
                            expected_actions += [{"kind": "stop"}]
                        if (record["source_hash"] != manifest["source_hash"] or record["config"] != asdict(cfg)
                                or record["seed"] != seed or record["policy_id"] != f"recipe-{label}"
                                or record["lineage"]["episode_start"] != checkpoint_ids[state]
                                or [h["action"] for h in record["history"]] != expected_actions
                                or "final" in record or not record["done"]):
                            raise ValueError("cached episode differs from the frozen protocol")
                    else:
                        print(f"{partition}: {key} recipe {label}", flush=True)
                        record = run_policy_episode(student, recipe_policy(candidate), seed, path,
                            config=cfg, starting_patch=patch, policy_id=f"recipe-{label}", final=False)
                    records[label] = record
                    candidate_gain = record["finalization"]["candidate_changes"]["coding"]["gain"]
                    print(f"  raw gain={candidate_gain:+.2f}; accepted={record['finalization']['accepted']}; "
                          f"reward={record['total_reward']:+.4f}", flush=True)
                groups[key] = records
        return groups

    try:
        train_groups = collect("training", TRAIN_SEEDS)
        audit = matrix_summary(train_groups, catalog)
        atomic_json(out/"training-headroom.json", audit)
        print(f"Training headroom: {audit['summary']}", flush=True)
        contexts = [records["A"]["initial_observation"] for records in train_groups.values()]
        rewards = [[records[c["label"]]["total_reward"] for c in catalog] for records in train_groups.values()]
        means = [sum(row[j] for row in rewards)/len(rewards) for j in range(len(catalog))]
        fixed = max(range(len(catalog)), key=means.__getitem__)
        atomic_json(out/"training-inputs.json", {"contexts": contexts, "rewards": rewards, "best_fixed": fixed})

        # Validation observations can be used at adaptation time, but no candidate
        # outcomes or terminal labels are available when decisions are frozen.
        validation_observations = {}
        for state, patch in states.items():
            for seed in VALIDATION_SEEDS:
                obs, _ = CodingEditEnv(student, cfg).reset(seed=seed, starting_patch=patch)
                validation_observations[f"{state}-seed-{seed}"] = obs
        decisions = {key: {"no_edit": "A", "ordinary_lora": "B", "tuned_fixed": catalog[fixed]["label"],
                          "behavioral_selector": catalog[behavioral_choice(obs, contexts, rewards)]["label"]}
                     for key, obs in validation_observations.items()}
        training_results = []
        editor_runs = EDITOR_SEEDS if audit["summary"]["rl_gate_passed"] else (0,)
        for editor_seed in editor_runs:
            policy = EditorPolicy(editor_manifest, catalog, device=device, seed=editor_seed)
            folder = out/f"editor-seed-{editor_seed}"
            policy.save(folder/"initial.safetensors")
            if editor_seed == 0:
                for key, obs in validation_observations.items():
                    decisions[key]["frozen_editor"] = policy.choose(obs)[0]["label"]
            if audit["summary"]["rl_gate_passed"]:
                result_path = folder/"training.json"
                if result_path.exists():
                    training = json.loads(result_path.read_text())
                    policy.load(folder/"trained.safetensors")
                    if policy.fingerprints() != training["weights_after"]:
                        raise ValueError("saved editor training weights differ")
                else:
                    def progress(row):
                        atomic_json(folder/"progress.json", row)
                        print(f"Editor seed {editor_seed}: update {row['update']+1}/24; reward={row['mean_reward']:+.4f}", flush=True)
                    training_start = time.monotonic()
                    training = train_editor(policy, contexts, lambda i, j: rewards[i][j], seed=editor_seed,
                        progress=progress, **manifest["rl"])
                    training["wall_seconds"] = time.monotonic()-training_start
                    policy.save(folder/"trained.safetensors")
                    atomic_json(result_path, training)
                for key, obs in validation_observations.items():
                    decisions[key][f"trained_editor_{editor_seed}"] = policy.choose(obs)[0]["label"]
                training_results.append({"seed": editor_seed, "changed_adapter_tensors": training["changed_adapter_tensors"],
                    "initial_expected_reward": sum(sum(p*r for p, r in zip(ps, rs))
                        for ps, rs in zip(training["reference_probabilities"], rewards))/len(rewards),
                    "final_expected_reward": sum(sum(p*r for p, r in zip(ps, rs))
                        for ps, rs in zip(training["final_probabilities"], rewards))/len(rewards),
                    "wall_seconds": training["wall_seconds"]})
            del policy
            gc.collect()
            if str(student.device) == "mps": torch.mps.empty_cache()
        decisions_path = out/"frozen-decisions.json"
        if decisions_path.exists() and json.loads(decisions_path.read_text()) != decisions:
            raise ValueError("resumed policy decisions changed")
        atomic_json(decisions_path, decisions)
        atomic_json(out/"editor-results.json", training_results)

        validation_groups = collect("validation", VALIDATION_SEEDS)
        validation_audit = matrix_summary(validation_groups, catalog)
        atomic_json(out/"validation-headroom.json", validation_audit)
        terminal_rows = []
        for key, choices in decisions.items():
            baseline_final = None
            evaluated = {}
            for label in ["A"] + sorted(set(choices.values())-{"A"}):
                record = copy.deepcopy(validation_groups[key][label])
                path = out/"validation"/f"{key}-{label}-terminal.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["lineage"] != validation_groups[key][label]["lineage"]:
                        raise ValueError("terminal checkpoint identity differs")
                else:
                    student.restore()
                    hashes = student.load_patch(out/"validation"/record["patch"]["file"])
                    if hashes != record["patch"]["tensor_hashes"] or student.checkpoint_id() != record["lineage"]["accepted"]:
                        raise RuntimeError("accepted student patch differs on reload")
                    record["final"] = {kind: student.evaluate(episode_problems(cfg, split, record["seed"]))
                        for kind, split in (("coding", "test"), ("retention", "retention_test"))}
                    record["baseline_final"] = baseline_final or copy.deepcopy(record["final"])
                    record["final_changes"] = paired_changes(record["baseline_final"], record["final"])
                    atomic_json(path, record)
                if label == "A": baseline_final = record["final"]
                evaluated[label] = record
            for condition, label in choices.items():
                r = evaluated[label]
                terminal_rows.append({"context": key, "condition": condition, "recipe": label,
                    "candidate_accepted": r["finalization"]["accepted"], "development_reward": r["total_reward"],
                    "coding_before": baseline_final["coding"]["accuracy"], "coding_after": r["final"]["coding"]["accuracy"],
                    "retention_before": baseline_final["retention"]["accuracy"], "retention_after": r["final"]["retention"]["accuracy"],
                    "final_changes": paired_changes(baseline_final, r["final"]),
                    "record": str(Path("validation")/f"{key}-{label}-terminal.json")})
        all_records = [r for groups in (train_groups, validation_groups) for rs in groups.values() for r in rs.values()]
        summary = {"manifest": manifest, "training_headroom": audit["summary"],
            "rl_executed": bool(training_results), "editor_training": training_results,
            "frozen_decisions": decisions, "validation_headroom": validation_audit["summary"], "terminal_results": terminal_rows,
            "cost": {"matrix_episodes": len(all_records), "optimizer_steps": sum(r["budget_used"]["optimizer_steps"] for r in all_records),
                "post_edit_scoring_rounds": sum(r["budget_used"]["edit_evaluations"] for r in all_records),
                "initial_scoring_rounds": len(all_records), "invocation_wall_seconds": time.monotonic()-started,
                "paid_api_calls": 0},
            "limitations": [manifest["scope"], "Two related student states are not two independent pretrained models.",
                "Finite-action contextual-bandit RL; no trained sequential policy or descendant acting as editor.",
                "Development selection and final parameters still share four synthetic task families.",
                "Cost counters do not establish matched total FLOPs or amortized benefit.",
                "Interpreter restrictions remain; no causal interpretability or AGI claim."]}
        atomic_json(out/"summary.json", summary)
        return summary
    except Exception as exc:
        atomic_json(out/"failure.json", {"error": type(exc).__name__, "detail": str(exc)[:500],
            "source_hash": source_hash(), "wall_seconds": time.monotonic()-started})
        raise
