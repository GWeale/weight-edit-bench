"""Bounded local experiment. No paid endpoints, hidden workers, or automatic resume."""

import importlib.metadata
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..runner import atomic_json, source_hash
from .actions import recipe_actions, starter_catalog
from .backend import Student
from .editor import EditorPolicy, train_editor
from .environment import CodingConfig, CodingEditEnv, episode_problems, metric


def load_manifest(path):
    return json.loads(Path(path).read_text())


def versions():
    return {p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")}


def run_episode(student, actions, seed, out, starting_patch=None, final=True, config=None):
    config = replace(config or CodingConfig(), horizon=max(1,len(actions)))
    env = CodingEditEnv(student, config)
    observation, _ = env.reset(seed=seed, starting_patch=starting_patch)
    initial = observation
    for action in actions:
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close_episode()
    record = env.record(include_final=final)
    record.update(source_hash=source_hash(), versions=versions(), initial_observation=initial)
    out = Path(out)
    patch = out.with_suffix(".safetensors")
    record["patch"] = {"file": patch.name, "tensor_hashes": student.save_patch(patch)}
    atomic_json(out, record)
    return record


def run_pilot(student_manifest, editor_manifest, out, train_seeds=(0, 1), eval_seeds=(20, 21),
              updates=8, group_size=8, device="auto", max_new_tokens=192, with_recovery=False, task_suite="micro", curriculum="bridge", editor_seed=0):
    out = Path(out)
    if (out / "manifest.json").exists():
        raise ValueError("pilot directory already contains a run; choose a fresh output path")
    if (not train_seeds or not eval_seeds or len(set(train_seeds)) != len(train_seeds)
            or len(set(eval_seeds)) != len(eval_seeds) or set(train_seeds) & set(eval_seeds)
            or any(type(s) is not int or s < 0 for s in (*train_seeds, *eval_seeds))):
        raise ValueError("training and evaluation seeds must be unique, nonempty, and disjoint")
    if len(train_seeds) > 16 or len(eval_seeds) > 16 or not 1 <= updates <= 1000 or not 2 <= group_size <= 32:
        raise ValueError("pilot budget exceeded")
    if type(editor_seed) is not int or editor_seed < 0:
        raise ValueError("editor seed must be nonnegative")
    started = time.monotonic()
    student = Student(student_manifest, device=device, max_new_tokens=max_new_tokens)
    catalog = starter_catalog(student.num_layers, with_recovery=with_recovery)
    cfg = CodingConfig(horizon=2 if with_recovery else 1,task_suite=task_suite,curriculum=curriculum)
    eval_cfg = replace(cfg,family_partition="evaluation")
    manifest = {"protocol": "categorical-coding-pilot-v1", "source_hash": source_hash(), "versions": versions(),
        "student": {k: student_manifest[k] for k in ("model_id", "revision")},
        "editor": {k: editor_manifest[k] for k in ("model_id", "revision")},
        "train_seeds": list(train_seeds), "eval_seeds": list(eval_seeds), "config": asdict(cfg),
        "evaluation_config":asdict(eval_cfg),
        "updates": updates, "group_size": group_size, "catalog": catalog, "with_recovery": with_recovery,
        "editor_seed": editor_seed,
        "max_new_tokens": max_new_tokens, "device": str(student.device), "dtype": str(student.dtype),
        "teacher": "Astra-authored starter action catalog; this program makes no Astra API calls",
        "scope": ("twelve editor-training families (four bridge/eight stress), four reserved editor-evaluation families; within-episode parameter splits"
                  if task_suite=="algorithmic" else "parameter-disjoint instances of four shared templates; not unseen-family generalization")}
    atomic_json(out / "manifest.json", manifest)
    policy = EditorPolicy(editor_manifest, catalog, device=device, seed=editor_seed)
    policy.save(out / "editor-initial.safetensors")
    contexts = []
    for seed in train_seeds:
        print(f"Preparing training context {seed}", flush=True)
        env = CodingEditEnv(student, cfg)
        observation, _ = env.reset(seed=seed)
        contexts.append(observation)
    atomic_json(out / "training-contexts.json", contexts)
    cache, counters = {}, {"oracle_episodes": 0, "reward_cache_hits": 0}

    def reward_fn(index, action_index):
        key = (index, action_index)
        if key in cache:
            counters["reward_cache_hits"] += 1
            return cache[key]
        seed, candidate = train_seeds[index], catalog[action_index]
        print(f"Evaluating train seed={seed} action={candidate['label']} ({candidate['name']})", flush=True)
        record = run_episode(student, recipe_actions(candidate), seed,
            out / "training" / f"seed-{seed}-action-{candidate['label']}.json", final=False, config=cfg)
        reward = record["total_reward"]
        cache[key] = reward
        counters["oracle_episodes"] += 1
        print(f"Student reward {reward:+.4f}; coding={record['development']['coding']['accuracy']:.2f}; "
              f"retention={record['development']['retention']['accuracy']:.2f}", flush=True)
        return reward

    def progress(row):
        atomic_json(out / "latest-update.json", row)
        print(f"Editor update {row['update']+1}/{updates}: reward={row['mean_reward']:+.4f}, "
              f"KL={row['kl_to_reference']:.5f}", flush=True)

    training = train_editor(policy, contexts, reward_fn, updates=updates, group_size=group_size, progress=progress, seed=editor_seed)
    policy.save(out / "editor-trained.safetensors")
    atomic_json(out / "editor-training.json", training)
    # Complete the same finite grid for an honestly tuned fixed-action baseline.
    # These extra rewards do not update the trained editor.
    oracle_before_baseline = counters["oracle_episodes"]
    table = [[reward_fn(i, j) for j in range(len(catalog))] for i in range(len(contexts))]
    means = [sum(row[j] for row in table)/len(table) for j in range(len(catalog))]
    best_fixed = max(range(len(means)), key=means.__getitem__)
    atomic_json(out / "training-reward-table.json", {"rewards": table, "mean": means, "best_fixed": best_fixed})
    eval_rows, verified_reload = [], False
    for seed in eval_seeds:
        env = CodingEditEnv(student, eval_cfg)
        observation, _ = env.reset(seed=seed)
        policy.load(out / "editor-initial.safetensors")
        frozen, frozen_p = policy.choose(observation)
        policy.load(out / "editor-trained.safetensors")
        learned, learned_p = policy.choose(observation)
        choices = {"no_edit": catalog[0], "frozen_editor": frozen, "trained_editor": learned,
                   "tuned_fixed": catalog[best_fixed]}
        for condition, candidate in choices.items():
            print(f"Held-out seed={seed} condition={condition} action={candidate['label']}", flush=True)
            path = out / "evaluation" / f"seed-{seed}-{condition}.json"
            record = run_episode(student, recipe_actions(candidate), seed, path,config=eval_cfg)
            eval_rows.append({"seed": seed, "condition": condition, "action": candidate["label"],
                "development_reward": record["total_reward"], "final": {k: metric(v) for k,v in record["final"].items()},
                "probabilities": learned_p if condition == "trained_editor" else frozen_p if condition == "frozen_editor" else None})
            if condition == "trained_editor" and not verified_reload:
                student.restore()
                hashes = student.load_patch(path.with_suffix(".safetensors"))
                if hashes != record["patch"]["tensor_hashes"]:
                    raise RuntimeError("saved student patch did not reload exactly")
                repeated = student.evaluate(episode_problems(eval_cfg,"test", seed))
                if [r["code"] for r in repeated["rows"]] != [r["code"] for r in record["final"]["coding"]["rows"]]:
                    raise RuntimeError("reloaded patch changed greedy evaluation outputs")
                verified_reload = True
    # A second application uses the already edited student and new feedback.
    # Both editing rounds are frozen before inspecting terminal test results.
    cycle_seed = max((*train_seeds, *eval_seeds)) + 1
    print(f"Repeated-edit and fixed-recovery smoke, seed={cycle_seed}", flush=True)
    env = CodingEditEnv(student, replace(eval_cfg,horizon=4 if with_recovery else 2))
    observation, _ = env.reset(seed=cycle_seed)
    cycle_choices = []
    for _ in range(2):
        candidate, probabilities = policy.choose(observation)
        cycle_choices.append({"label": candidate["label"], "probabilities": probabilities})
        for action in recipe_actions(candidate):
            observation, _, _, _, _ = env.step(action)
    env.close_episode()
    cycle = env.record(include_final=True)
    cycle["choices"] = cycle_choices
    cycle["patch_hashes"] = student.save_patch(out / "cycle-edited.safetensors")
    atomic_json(out / "cycle-edited.json", cycle)
    recovery_action = {"kind": "lora", "layers": list(range(student.num_layers-2, student.num_layers)),
        "module": "mlp.down_proj", "rank": 4, "steps": 4, "lr": .001, "data": "rehearsal"}
    recovery = run_episode(student, [recovery_action], cycle_seed, out / "cycle-recovered.json",
                           starting_patch=out / "cycle-edited.safetensors", config=eval_cfg)
    summary = {"manifest": manifest, "training": {"changed_adapter_tensors": training["changed_adapter_tensors"],
        "initial_expected_reward": [sum(p*r for p,r in zip(probs, row)) for probs,row in zip(training["reference_probabilities"],table)],
        "final_expected_reward": [sum(p*r for p,r in zip(probs, row)) for probs,row in zip(training["final_probabilities"],table)],
        "best_fixed_action": catalog[best_fixed]["label"]}, "evaluation": eval_rows,
        "repeated_edit": {"seed": cycle_seed, "choices": cycle_choices,
            "after_two_edits": {k: metric(v) for k,v in cycle["final"].items()},
            "after_fixed_sft_recovery": {k: metric(v) for k,v in recovery["final"].items()}},
        "verification": {"patch_reload_exact_and_same_outputs": verified_reload,
            "editor_weights_changed": training["changed_adapter_tensors"] > 0},
        "cost": {**counters, "oracle_episodes_before_fixed_baseline": oracle_before_baseline,
            "wall_seconds": time.monotonic()-started, "additional_paid_api_calls": 0,
            "interactive_astra_tokens": "not available to the local runner"},
        "limitations": ["One editor training seed and one pretrained student checkpoint.",
            manifest["scope"] + ". Synthetic templates are not evidence of real-project coding competence.",
            "Finite recipes and categorical RL; no unconstrained weight-program generation.",
            "Interpreter accepts a restricted Python subset, not arbitrary Python programs.",
            "Weight statistics and location controls do not establish interpretability.",
            "Recovery uses supervised rehearsal, not on-policy distillation.",
            "Repeated edits update the student, not the hosted Astra model."]}
    atomic_json(out / "summary.json", summary)
    return summary
