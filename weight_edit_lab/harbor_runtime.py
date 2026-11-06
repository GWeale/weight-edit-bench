"""Agent-side feedback and trusted replay grading for Harbor submissions.

The verifier consumes data, never imports agent-authored Python, and starts from
the task's own specification. Agent-side results are advisory, not authority.
"""

import argparse
import json
import math
from pathlib import Path

from .environment import Config, WeightEditEnv
from .runner import atomic_json

PROTOCOL = "weight-edit-harbor-v1"
MAX_BYTES = 262144


class InvalidSubmission(ValueError):
    pass


def read_submission(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise InvalidSubmission("submission must be a regular JSON file of at most 256 KiB")
    try:
        def nonfinite(value):
            raise ValueError("nonfinite JSON number")
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=nonfinite)
    except (ValueError, UnicodeError, OSError, RecursionError) as exc:
        raise InvalidSubmission("submission is not finite UTF-8 JSON") from exc


def validate_submission(document, limit):
    if (not isinstance(document, dict) or set(document) != {"protocol", "actions"}
            or document["protocol"] != PROTOCOL or not isinstance(document["actions"], list)
            or len(document["actions"]) > limit):
        raise InvalidSubmission("provide protocol and an actions array within the task budget")
    return document["actions"]


def make_env(spec):
    if spec["protocol"] != PROTOCOL:
        raise ValueError("unknown task protocol")
    if spec["track"] == "cpu":
        return WeightEditEnv(Config(**spec["config"]), spec["mode"])
    if spec["track"] == "coding":
        from .coding.backend import Student
        from .coding.direct_env import DirectConfig, DirectEditEnv
        from .coding.environment import CodingConfig
        student = Student(spec["model"], device="cpu", max_new_tokens=spec["max_new_tokens"], batch_size=1)
        return DirectEditEnv(student, CodingConfig(**spec["config"]), DirectConfig(**spec["access"]))
    raise ValueError("unknown track")


def replay(spec, submission, *, close=False):
    limit = spec["config"]["horizon"]+1 if spec["track"] == "cpu" else spec["access"]["max_turns"]
    actions = validate_submission(submission, limit)
    env = make_env(spec)
    observation, _ = (env.reset(seed=spec["seed"], split=spec["split"])
                      if spec["track"] == "cpu" else env.reset(seed=spec["seed"]))
    done = False
    for action in actions:
        if done:
            raise InvalidSubmission("actions after episode termination are not permitted")
        try:
            observation, _, terminated, truncated, info = env.step(action)
        except (ValueError, TypeError, KeyError, IndexError, SyntaxError, ZeroDivisionError) as exc:
            # DirectEditEnv invalidates itself after a scorer/runtime failure.
            # Such failures are infrastructure errors, not zero-reward answers.
            if spec["track"] == "coding" and not env.ready:
                raise
            raise InvalidSubmission("an edit action failed validation") from exc
        if info.get("invalid_action"):
            raise InvalidSubmission("an edit action failed validation: " + str(observation["feedback"]))
        done = terminated or truncated
    if close and not done:
        observation, _, _, _, _ = env.step({"stop": True} if spec["track"] == "cpu" else {"kind": "stop"})
    return env, observation


def grade(spec, submission):
    env, _ = replay(spec, submission, close=True)
    if spec["track"] == "cpu":
        metrics = env.result()
        improvement = (metrics["initial_test_mse"]-metrics["final_test_objective"])/max(metrics["initial_test_mse"], .05)
        retention_ok = spec["mode"] == "adapt" or metrics["retention_test_mse"] <= spec["retention_limit"]
        reward = max(0., min(1., improvement-spec["config"]["step_cost"]*metrics["edits"])) if retention_ok else 0.
        return {"reward": reward, "status": "graded", "retention_ok": retention_ok, "metrics": metrics}
    from .coding.headroom import paired_changes
    record = env.record(include_final=True)
    results = record["evaluation"]
    changes = paired_changes(results["baseline"], results["accepted"])
    gains = {k: results["accepted"][k]["case_accuracy"]-results["baseline"][k]["case_accuracy"] for k in changes}
    retention_ok = not changes["retention"]["lost_programs"] and gains["retention"] >= -1e-12
    coding_ok = changes["coding"]["gain"] >= 0 and gains["coding"] >= -1e-12
    gain = changes["coding"]["gain"] + spec["config"]["case_weight"]*gains["coding"]
    reward = max(0., min(1., gain)) if retention_ok and coding_ok else 0.
    return {"reward": reward, "status": "graded", "retention_ok": retention_ok,
            "final_changes": changes, "case_gains": gains, "episode": record}


def verify(spec_path, submission_path, output):
    spec = json.loads(Path(spec_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Remove stale reward channels before starting. Infrastructure failures leave
    # no reward; Harbor records an error instead of a fabricated agent failure.
    for name in ("reward.json", "reward.txt"):
        (output/name).unlink(missing_ok=True)
    try:
        result = grade(spec, read_submission(submission_path))
    except InvalidSubmission as exc:
        result = {"reward": 0., "status": "invalid_submission", "error": str(exc)[:2000]}
    if not math.isfinite(result["reward"]):
        raise RuntimeError("nonfinite verifier reward")
    atomic_json(output/"details.json", result)
    atomic_json(output/"reward.json", {"reward": result["reward"]})
    return result


def reference(spec, output):
    """CPU reference optimizer; coding reference is only an edit smoke baseline."""
    document = {"protocol": PROTOCOL, "actions": []}
    if spec["track"] == "coding":
        document["actions"] = [{"kind": "edit", "code":
            "W=weight('model.layers.0.mlp.down_proj'); write('model.layers.0.mlp.down_proj',W*0.99)"}, {"kind": "stop"}]
    else:
        from .agents import Baseline
        env = make_env(spec)
        observation, _ = env.reset(seed=spec["seed"], split=spec["split"])
        policy = Baseline("adam", rate=.03)
        while not observation["done"]:
            action = policy.act(observation)
            document["actions"].append(action)
            observation, _, _, _, _ = env.step(action)
    atomic_json(Path(output), document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("observe", "reference", "verify"))
    parser.add_argument("--spec", type=Path, default=Path("/app/spec.json"))
    parser.add_argument("--submission", type=Path, default=Path("/app/submission.json"))
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify(args.spec, args.submission, args.output)
        print(json.dumps({k: result[k] for k in ("status", "reward")}))
    elif args.command == "reference":
        reference(json.loads(args.spec.read_text()), args.submission)
        print("Reference transcript saved. This does not establish learned editing ability.")
    else:
        _, observation = replay(json.loads(args.spec.read_text()), read_submission(args.submission))
        print(json.dumps(observation, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
