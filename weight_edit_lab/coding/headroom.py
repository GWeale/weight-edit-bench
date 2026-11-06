"""Development-only diagnostics for whether choosing an edit can matter.

The hindsight ceiling uses observed outcomes. It is not an executable editor,
an unbiased estimate of transfer, or a substitute for a held-out comparison.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from ..tasks import digest
from ..runner import source_hash
from .actions import recipe_actions


def paired_changes(before, after):
    """Count lost previously correct programs, even if other programs improve."""
    result = {}
    for skill in ("coding", "retention"):
        old, new = before[skill], after[skill]
        a = {r["name"]: r for r in old["rows"]}
        b = {r["name"]: r for r in new["rows"]}
        if not a or a.keys() != b.keys() or len(a) != len(old["rows"]) or len(b) != len(new["rows"]):
            raise ValueError("paired evaluations need identical unique problem names")
        for rows, evaluation in ((a, old), (b, new)):
            if any(type(r["grade"]["correct"]) is not bool for r in rows.values()):
                raise ValueError("correctness must be boolean")
            accuracy = sum(r["grade"]["correct"] for r in rows.values()) / len(rows)
            if not math.isclose(accuracy, evaluation["accuracy"], abs_tol=1e-9):
                raise ValueError("aggregate accuracy disagrees with program grades")
        lost = [name for name in a if a[name]["grade"]["correct"] and not b[name]["grade"]["correct"]]
        gained = [name for name in a if not a[name]["grade"]["correct"] and b[name]["grade"]["correct"]]
        result[skill] = {"before": old["accuracy"], "after": new["accuracy"],
            "gain": new["accuracy"] - old["accuracy"], "lost_programs": sorted(lost),
            "gained_programs": sorted(gained), "regression_fraction": len(lost) / len(a)}
    return result


def acceptance(changes, max_retention_regression=0.):
    if (type(max_retention_regression) not in (int, float)
            or not math.isfinite(max_retention_regression) or not 0 <= max_retention_regression <= 1):
        raise ValueError("retention regression tolerance must be in 0..1")
    return (changes["coding"]["gain"] >= 0
            and changes["retention"]["regression_fraction"] <= max_retention_regression)


def summarize_matrix(contexts, labels):
    """Give each semantic context equal weight, averaging nested seed repeats."""
    if not contexts or not labels or len(set(labels)) != len(labels):
        raise ValueError("a nonempty rectangular action matrix is required")
    grouped = defaultdict(list)
    for row in contexts:
        if set(row["cells"]) != set(labels):
            raise ValueError("incomplete candidate matrix")
        grouped[row["semantic_context"]].append(row)
    means = {label: sum(sum(r["cells"][label]["accepted_gain"] for r in repeats)/len(repeats)
                        for repeats in grouped.values())/len(grouped) for label in labels}
    fixed = max(labels, key=means.__getitem__)
    oracle = sum(sum(max(r["cells"][label]["accepted_gain"] for label in labels)
                     for r in repeats)/len(repeats) for repeats in grouped.values())/len(grouped)
    return {"episode_contexts": len(contexts), "semantic_contexts": len(grouped),
        "nested_repetitions": len(contexts)-len(grouped), "fixed_recipe": fixed,
        "fixed_accepted_gain": means[fixed], "hindsight_accepted_gain": oracle,
        "selection_headroom": oracle-means[fixed], "fixed_recipe_gains": means,
        "editable_contexts": sum(any(c["accepted_gain"] > 0 for c in r["cells"].values()) for r in contexts)}


def audit_pilot(directory, max_retention_regression=0., max_optimizer_steps=24, max_edit_evaluations=3):
    """Read a complete training grid; never read evaluation files or reward tables."""
    acceptance({"coding": {"gain": 0}, "retention": {"regression_fraction": 0}}, max_retention_regression)
    for limit in (max_optimizer_steps, max_edit_evaluations):
        if type(limit) is not int or limit < 0:
            raise ValueError("budgets must be nonnegative integers")
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["protocol"] != "categorical-coding-pilot-v1":
        raise ValueError("expected a categorical pilot manifest")
    if manifest["config"].get("family_partition", "train") != "train":
        raise ValueError("headroom analysis refuses final-family data")
    seeds, catalog = manifest["train_seeds"], manifest["catalog"]
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) & set(manifest["eval_seeds"]):
        raise ValueError("training and final seeds must be nonempty, unique and disjoint")
    labels = [c["label"] for c in catalog]
    if any(not isinstance(label, str) or not label.isalnum() for label in labels):
        raise ValueError("invalid catalog label")
    contexts = []
    for seed in seeds:
        row = {"seed": seed, "cells": {}}
        reference = None
        for candidate in catalog:
            label = candidate["label"]
            record = json.loads((directory/"training"/f"seed-{seed}-action-{label}.json").read_text())
            if ("final" in record or record["config"].get("family_partition", "train") != "train"
                    or record["seed"] != seed or not record["done"]):
                raise ValueError("only closed training records without terminal results are allowed")
            if (record["model"] != manifest["student"] or record["source_hash"] != manifest["source_hash"]
                    or [s["action"] for s in record["history"]] != recipe_actions(candidate)):
                raise ValueError("record model, source or action differs from the declared comparison")
            # Horizon varies with recipe length. All other data/reward settings must match.
            cfg = {k: v for k, v in record["config"].items() if k != "horizon"}
            expected_cfg = {k: v for k, v in manifest["config"].items() if k != "horizon"}
            if cfg != expected_cfg:
                raise ValueError("candidate task configuration differs")
            baseline = record["baseline_development"]
            signature = {"baseline": baseline, "support": record["initial_observation"]["support"],
                         "decoding": record["decoding"]}
            # Runtime is not part of the paired behavioral state.
            for value in signature["baseline"].values():
                value.pop("seconds", None)
            if reference is not None and signature != reference:
                raise ValueError("candidate baselines, support or decoding differ")
            reference = signature
            changes = paired_changes(baseline, record["development"])
            steps = sum(s["edit"]["steps"] for s in record["history"])
            queries = sum(s["action"]["kind"] not in ("noop", "stop") for s in record["history"])
            within_budget = steps <= max_optimizer_steps and queries <= max_edit_evaluations
            accepted = acceptance(changes, max_retention_regression) and within_budget
            row["cells"][label] = {"changes": changes, "accepted": accepted,
                "accepted_gain": changes["coding"]["gain"] if accepted else 0.,
                "optimizer_steps": steps, "edit_evaluations": queries, "within_budget": within_budget,
                "shaped_reward": record["total_reward"],
                "edit_seconds": sum(s["edit"].get("seconds", 0) for s in record["history"])}
        instructions = {skill: sorted(r["instruction"] for r in reference["baseline"][skill]["rows"])
                        for skill in ("coding", "retention")}
        instructions["support"] = sorted(r["instruction"] for r in reference["support"])
        row["semantic_context"] = digest({"model": manifest["student"], "instructions": instructions})
        row["instruction_hashes"] = {k: [digest(s) for s in values] for k, values in instructions.items()}
        contexts.append(row)
    # A prospective stop/no-edit option is always available at zero cost.
    labels = ["STOP"] + labels
    for row in contexts:
        row["cells"]["STOP"] = {"accepted_gain": 0., "accepted": True,
            "optimizer_steps": 0, "edit_evaluations": 0, "within_budget": True}
    return {"protocol": "development-edit-headroom-v1", "input_source_hash": manifest["source_hash"],
        "analysis_source_hash": source_hash(),
        "student": manifest["student"], "task_config": manifest["config"],
        "limits": {"max_retention_regression": max_retention_regression,
            "max_optimizer_steps": max_optimizer_steps, "max_edit_evaluations": max_edit_evaluations},
        "summary": summarize_matrix(contexts, labels), "contexts": contexts,
        "interpretation": "Development hindsight diagnostic. Both fixed and contextual recipes use the same post-edit acceptance rule, reverting rejected candidates to the start. Not a held-out editor result.",
        "limitations": ["Semantic hashes detect identical instruction sets, not independent task families.",
            "Legacy pilot records declare one original student; they do not prove full checkpoint identity.",
            "Optimizer-step and scoring-round caps are not equal FLOPs or equal total training cost.",
            "No confidence intervals: a tiny, reused development matrix is not confirmatory evidence."]}
