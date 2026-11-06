"""A transparent adaptive baseline for the sequential runner, not an RL policy."""

import argparse
import json
from pathlib import Path

from weight_edit_lab.coding.backend import Student
from weight_edit_lab.coding.environment import CodingConfig
from weight_edit_lab.coding.policy_runner import run_policy_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--starting-patch", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--final", action="store_true", help="Open terminal parameters of training families after all decisions")
    args = parser.parse_args()
    student = Student(json.loads(args.model.read_text()), device=args.device, max_new_tokens=512)
    action = {"kind": "lora", "layers": [student.num_layers-2, student.num_layers-1],
              "module": "mlp.down_proj", "rank": 4, "steps": 4, "lr": .001, "data": "support"}

    def decide(observation):
        if not observation["action_history"]:
            return {"action": action}
        old = observation["baseline_development"]
        now = observation["development"]
        regressed = any(now[k]["accuracy"] < old[k]["accuracy"] for k in ("coding", "retention"))
        return {"action": {"kind": "rollback" if regressed and len(observation["action_history"]) == 1 else "stop"}}

    result = run_policy_episode(student, decide, args.seed, args.out,
        config=CodingConfig(horizon=3, task_suite="algorithmic", curriculum="bridge",
                            max_optimizer_steps=8, max_edit_evaluations=2),
        starting_patch=args.starting_patch, policy_id="scripted_edit_then_check", final=args.final)
    print(json.dumps({"decisions": len(result["decisions"]), "lineage": result["lineage"],
        "finalization": result["finalization"], "budget_used": result["budget_used"]}, indent=2))


if __name__ == "__main__":
    main()
