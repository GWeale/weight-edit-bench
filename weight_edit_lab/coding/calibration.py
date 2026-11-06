"""Measure task difficulty on development families before spending on editor RL."""

from collections import Counter
from pathlib import Path
import time
import json

from ..runner import atomic_json, source_hash
from ..tasks import digest
from .hard_tasks import ANCHORS, EVAL_FAMILIES, PROTOCOL, TRAIN_FAMILIES, BRIDGE_FAMILIES, STRESS_FAMILIES, anchor_problem, make_problem
from .interpreter import grade, extract_prefilled_function
from .tasks import problems


def failure_category(row):
    result = row["grade"]
    if result["correct"]:
        return "correct"
    if result["error"] == "RejectedProgram":
        detail = result.get("error_detail", "")
        if "invalid syntax" in detail:
            return "truncated" if row["truncated"] else "syntax"
        if "solve function" in detail or "function signature" in detail:
            return "interface_error"
        if any(x in detail for x in ("budget", "limit", "too large", "too complex")):
            return "resource_limit"
        return "unsupported_subset"
    return "runtime_error" if result["error"] else "wrong_answer"


def calibration_groups(seeds):
    # Calibration NEVER calls make_problem for the four reserved families.
    return {
        "micro": [(seed, p) for seed in seeds for p in problems("score",seed)],
        "bridge": [(seed,make_problem(family,"score",seed)) for seed in seeds for family in BRIDGE_FAMILIES],
        "stress": [(seed,make_problem(family,"score",seed)) for seed in seeds for family in STRESS_FAMILIES],
        "retention": [(seed,anchor_problem(family,"retention_score",seed)) for seed in seeds for family in ANCHORS],
    }


def calibrate(manifest, out, seeds=(0,1,2,3), device="auto", max_new_tokens=512, reuse_generations=None):
    from .experiment import versions
    if (not seeds or len(seeds)>16 or len(set(seeds))!=len(seeds)
            or any(type(s) is not int or s<0 for s in seeds)):
        raise ValueError("use one to 16 unique nonnegative calibration seeds")
    out = Path(out)
    if (out / "manifest.json").exists():
        raise ValueError("choose a fresh calibration directory")
    groups = calibration_groups(seeds)
    for group in groups.values():
        for _,p in group:
            if not grade(p.solution,p.cases,budget=p.instruction_budget)["correct"]:
                raise RuntimeError(f"reference solution disagrees with oracle: {p.name}")
    definition = {"protocol":PROTOCOL,"source_hash":source_hash(),"versions":versions(),
        "model":{k:manifest[k] for k in ("model_id","revision")}, "seeds":list(seeds),
        "partition":"development calibration, not final evaluation", "max_new_tokens":max_new_tokens,
        "reserved_families_not_evaluated":list(EVAL_FAMILIES),
        "task_hash":digest({k:[dict(name=p.name,instruction=p.instruction,solution=p.solution,cases=p.cases,completion_prefix=p.completion_prefix)
                              for _,p in group] for k,group in groups.items()})}
    previous = None
    if reuse_generations:
        previous = json.loads((Path(reuse_generations)/"manifest.json").read_text())
        for key in ("protocol","model","seeds","max_new_tokens","task_hash"):
            if previous[key] != definition[key]:
                raise ValueError(f"cached generation protocol mismatch: {key}")
        definition["generation_source_hash"] = previous.get("generation_source_hash",previous["source_hash"])
        definition["regraded_from"] = str(Path(reuse_generations).name)
    atomic_json(out / "manifest.json",definition)
    started = time.monotonic()
    student = None
    if previous is None:
        from .backend import Student
        student = Student(manifest,device=device,max_new_tokens=max_new_tokens)
    all_rows, cohorts = [], {}
    for cohort, group in groups.items():
        print(f"Calibrating {cohort}: {len(group)} programs",flush=True)
        rows = []
        cached = {r["name"]:r for r in json.loads((Path(reuse_generations)/f"{cohort}.json").read_text())} if previous else None
        for start in range(0,len(group),4):
            batch = group[start:start+4]
            if cached is not None:
                generated = []
                for _,p in batch:
                    row = dict(cached[p.name])
                    if row["instruction"] != p.instruction:
                        raise ValueError("cached instruction mismatch")
                    evaluated = extract_prefilled_function(row["code"]) if p.completion_prefix else row["code"]
                    row.update(evaluated_code=evaluated,grade=grade(evaluated,p.cases,p.instruction_budget))
                    generated.append(row)
                result = {"rows":generated}
            else:
                result = student.evaluate([p for _,p in batch])
            for (seed,p),row in zip(batch,result["rows"]):
                row.update(cohort=cohort,seed=seed,family=p.family,case_count=len(p.cases),
                           instruction_budget=p.instruction_budget)
                row["failure_category"] = failure_category(row)
                rows.append(row)
            atomic_json(out / f"{cohort}.json",rows)
            print(f"  {min(start+4,len(group))}/{len(group)} generated",flush=True)
        cohorts[cohort] = summarize(rows)
        all_rows.extend(rows)
    family_rows = {family:summarize([r for r in all_rows if r["family"]==family])
                   for family in TRAIN_FAMILIES}
    previous_summary = json.loads((Path(reuse_generations)/"summary.json").read_text()) if previous else None
    summary = {"manifest":definition,"cohorts":cohorts,"families":family_rows,
        "seconds":time.monotonic()-started,"device":str(student.device) if student else previous_summary["device"],
        "dtype":str(student.dtype) if student else previous_summary["dtype"], "reused_generations":previous is not None,
        "references_verified":sum(map(len,groups.values())),
        "interpretation":"All development families retained, regardless of score. No final-family model outputs inspected.",
        "limitations":["Synthetic array algorithms, not repository-level coding.",
                       "Shared templates make prompt instances dependent.",
                       "Unsupported Python features and token/resource limits are reported separately.",
                       "This calibrates difficulty; it does not measure improvement from weight editing."]}
    atomic_json(out / "summary.json",summary)
    return summary


def summarize(rows):
    return {"programs":len(rows),"correct":sum(r["grade"]["correct"] for r in rows),
        "accuracy":sum(r["grade"]["correct"] for r in rows)/len(rows),
        "cases":sum(r["grade"]["total"] for r in rows),
        "case_accuracy":sum(r["grade"]["passed"] for r in rows)/sum(r["grade"]["total"] for r in rows),
        "failure_categories":dict(Counter(r["failure_category"] for r in rows)),
        "raw_token_capped":sum(r["truncated"] for r in rows),
        "function_syntax_or_token_failures":sum(r["failure_category"] in ("syntax","truncated") for r in rows),
        "max_generated_tokens":max(r["tokens"] for r in rows)}
