from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .agents import Baseline, RatePolicy
from .environment import Config, WeightEditEnv
from .tasks import MODES, PROTOCOL, SPLITS, digest, rng_for


def atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_hash() -> str:
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*"))
                   if p.suffix in (".py", ".html")})


def run_episode(agent, seed: int, mode: str, split: str, config: Config) -> dict:
    env = WeightEditEnv(config, mode)
    observation, _ = env.reset(seed=seed, split=split)
    started = time.perf_counter()
    agent_seconds = 0.0
    calls = 0
    while not observation["done"]:
        decision_started = time.perf_counter()
        action = agent.act(observation)
        agent_seconds += time.perf_counter() - decision_started
        calls += 1
        observation, _, _, _, _ = env.step(action)
    record = env.record()
    record["runtime"] = {"episode_seconds": time.perf_counter() - started,
                         "agent_seconds": agent_seconds, "agent_calls": calls}
    record["checkpoint"] = env.checkpoint()
    return record


def interval(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    ci = None
    if len(values) >= 2:
        rng = rng_for("bootstrap", len(values))
        samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(1000))
        ci = [samples[24], samples[974]]
    return {"mean": statistics.mean(values), "ci95": ci, "n": len(values)}


def summarize(records: list[dict], manifest: dict) -> dict:
    rows = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        groups.setdefault((r["mode"], r["agent"]), []).append(r)
    for (mode, agent), group in sorted(groups.items()):
        valid = [r for r in group if r["status"] == "complete"]
        rows.append({"mode": mode, "agent": agent, "planned": len(group), "completed": len(valid),
                     "errors": len(group) - len(valid),
                     "metrics": {metric: interval([r["result"][metric] for r in valid]) for metric in
                                 ("final_test_mse", "test_improvement", "retention_test_mse",
                                  "final_test_objective", "return", "weight_change_norm")},
                     "agent_calls": sum(r["runtime"]["agent_calls"] for r in valid)})
    pairs = []
    complete = {(r["seed"], r["mode"], r["agent"]): r for r in records if r["status"] == "complete"}
    for mode in MODES:
        for agent in manifest["agents"]:
            if agent == "adam":
                continue
            differences = []
            for seed in manifest["seeds"]:
                r, reference = complete.get((seed, mode, agent)), complete.get((seed, mode, "adam"))
                if r and reference:
                    if r["task_hash"] != reference["task_hash"]:
                        raise ValueError("paired agents did not receive the same task")
                    differences.append(r["result"]["final_test_objective"] - reference["result"]["final_test_objective"])
            pairs.append({"mode": mode, "agent": agent, "reference": "adam",
                          "metric": "final_test_objective_difference", "negative_favors_agent": True,
                          "missing_pairs": len(manifest["seeds"]) - len(differences), **interval(differences)})
    mode_pairs = []
    for agent in manifest["agents"]:
        samples = {"final_test_mse": [], "retention_test_mse": []}
        for seed in manifest["seeds"]:
            adapt, retain = complete.get((seed, "adapt", agent)), complete.get((seed, "retain", agent))
            if adapt and retain:
                if adapt["task_hash"] != retain["task_hash"]:
                    raise ValueError("paired modes did not receive the same task")
                for metric in samples:
                    samples[metric].append(retain["result"][metric] - adapt["result"][metric])
        mode_pairs.append({"agent": agent, "difference": "retain_minus_adapt",
                           "metrics": {metric: interval(values) for metric, values in samples.items()}})
    return {"manifest": manifest, "rows": rows, "paired_vs_adam": pairs, "paired_modes": mode_pairs,
            "uncertainty": "95% percentile bootstrap, 1000 resamples of task seeds within each mode. "
                           "No interval with fewer than two seeds. Errors excluded from means and reported. "
                           "One policy training seed does not establish robustness across training runs. "
                           "Reward scoring examples are adaptive feedback; final test examples are terminal-only."}


def evaluate(out: Path, seeds: list[int], split: str = "test", config: Config | None = None,
             policy_document: dict | None = None, tuning: dict | None = None) -> dict:
    config = config or Config()
    if (not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds)
            or split not in SPLITS):
        raise ValueError("evaluation needs unique nonnegative seeds and a known split")
    rates = {mode: {"sgd": 0.08, "adam": 0.03} for mode in MODES}
    if tuning is not None:
        if (tuning.get("format") != "weight-edit-tuning-v1" or tuning.get("split") != "validation"
                or tuning.get("config") != asdict(config)):
            raise ValueError("incompatible tuning document")
        rates = tuning["rates"]
        for mode in MODES:
            for name in ("sgd", "adam"):
                Baseline(name, rates[mode][name])
    agents = ["noop", "random", "sgd", "adam"]
    if policy_document is not None:
        RatePolicy.load(policy_document)
        training_config = policy_document.get("training", {}).get("config")
        if training_config is not None and training_config != asdict(config):
            raise ValueError("policy was trained with a different environment configuration")
        agents.append("learned_rate")
    manifest = {"protocol": PROTOCOL, "source_hash": source_hash(), "config": asdict(config),
                "seeds": seeds, "split": split, "agents": agents, "rates": rates,
                "baseline_tuning": tuning, "policy": policy_document}
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("resume refused: code, policy, seeds, configuration, or tuning changed; use a new output directory")
    atomic_json(manifest_path, manifest)
    records = []
    for seed in seeds:
        for mode in MODES:
            for name in agents:
                path = out / "episodes" / f"{mode}-{name}-{seed}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if (record.get("manifest_hash") != digest(manifest)
                            or (record.get("seed"), record.get("mode"), record.get("agent")) != (seed, mode, name)):
                        raise ValueError("episode does not match its manifest")
                else:
                    agent = (RatePolicy.load(policy_document) if name == "learned_rate"
                             else Baseline(name, rates[mode].get(name, 0.08), seed))
                    try:
                        record = run_episode(agent, seed, mode, split, config)
                        record["status"] = "complete"
                    except (ValueError, RuntimeError, ArithmeticError) as exc:
                        record = {"seed": seed, "mode": mode, "split": split, "status": "error",
                                  "error": f"{type(exc).__name__}: {exc}"}
                    record.update(agent=name, manifest_hash=digest(manifest))
                    atomic_json(path, record)
                records.append(record)
    summary = summarize(records, manifest)
    atomic_json(out / "summary.json", summary)
    return summary


def tune(seeds: int = 12, config: Config | None = None) -> dict:
    if type(seeds) is not int or seeds < 1:
        raise ValueError("tuning seeds must be positive")
    config = config or Config()
    rates, rows = {}, []
    for mode in MODES:
        rates[mode] = {}
        for name, candidates in (("sgd", (0.005, 0.02, 0.08, 0.2, 0.5)),
                                  ("adam", (0.005, 0.01, 0.03, 0.08, 0.2))):
            scores = []
            for rate in candidates:
                values = [run_episode(Baseline(name, rate, s), s, mode, "validation", config)
                          ["result"]["final_test_objective"] for s in range(seeds)]
                score = statistics.mean(values)
                rows.append({"mode": mode, "agent": name, "rate": rate, "validation_objective": score})
                scores.append((score, rate))
            rates[mode][name] = min(scores)[1]
    return {"format": "weight-edit-tuning-v1", "split": "validation", "seeds": seeds,
            "config": asdict(config), "source_hash": source_hash(), "rates": rates, "trials": rows}
