"""Stateful editing episodes; final coding labels never enter editor observations."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from .actions import validate_action
from .tasks import problems


class BudgetExceeded(ValueError):
    """A valid action would exceed the remaining episode budget."""


@dataclass(frozen=True)
class CodingConfig:
    horizon: int = 2
    problems_per_split: int = 4
    case_weight: float = .1
    retention_weight: float = 1.
    step_cost: float = .0005
    action_cost: float = .001
    task_suite: str = "micro"
    family_partition: str = "train"
    curriculum: str = "bridge"
    max_optimizer_steps: int = 192
    max_edit_evaluations: int = 8

    def __post_init__(self):
        import math
        if type(self.horizon) is not int or not 1 <= self.horizon <= 8:
            raise ValueError("horizon must be in 1..8")
        if self.problems_per_split != 4:
            raise ValueError("each editing episode uses four problems per split")
        if self.task_suite not in ("micro", "algorithmic") or self.family_partition not in ("train", "evaluation"):
            raise ValueError("invalid task suite or family partition")
        if self.curriculum not in ("bridge", "mixed", "stress"):
            raise ValueError("invalid curriculum")
        for value, limit in ((self.max_optimizer_steps, 192), (self.max_edit_evaluations, 8)):
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError("invalid cumulative episode budget")
        for x in (self.case_weight, self.retention_weight, self.step_cost, self.action_cost):
            if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
                raise ValueError("reward weights must be finite and nonnegative")


def metric(evaluation):
    return {key: evaluation[key] for key in ("accuracy", "case_accuracy")}


def episode_problems(config, split, seed):
    if config.task_suite == "algorithmic":
        from .hard_tasks import hard_problems
        return hard_problems(split,seed,config.problems_per_split,config.family_partition,config.curriculum)
    return problems(split,seed,config.problems_per_split)


class CodingEditEnv:
    def __init__(self, student, config=None):
        self.student, self.config = student, config or CodingConfig()
        self.ready = False

    def reset(self, *, seed=0, starting_patch=None):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be nonnegative")
        self.student.restore()
        if starting_patch is not None:
            self.student.load_patch(starting_patch)
        self.starting_state = self.student.snapshot()
        self.seed = seed
        self.support = episode_problems(self.config,"support", seed)
        self.rehearsal = episode_problems(self.config,"rehearsal", seed)
        self.score = episode_problems(self.config,"score", seed)
        self.retention = episode_problems(self.config,"retention_score", seed)
        self.support_result = self.student.evaluate(self.support)
        self.baseline = self._scores()
        self.current = copy.deepcopy(self.baseline)
        self.inspection = self.student.inspect()
        self.initial_support = copy.deepcopy(self.support_result)
        self.initial_inspection = copy.deepcopy(self.inspection)
        self.history, self.total_reward = [], 0.
        self.optimizer_steps, self.edit_evaluations = 0, 0
        self.finalization = None
        self.done, self.ready, self._final = False, True, None
        return self.observation(), {"config": asdict(self.config)}

    def _scores(self):
        return {"coding": self.student.evaluate(self.score), "retention": self.student.evaluate(self.retention)}

    def _objective(self, scores):
        cfg = self.config
        def value(x):
            return x["accuracy"] + cfg.case_weight*x["case_accuracy"]
        return (value(scores["coding"]) - value(self.baseline["coding"])
                - cfg.retention_weight*max(0., value(self.baseline["retention"])-value(scores["retention"])))

    def observation(self):
        if not self.ready:
            raise RuntimeError("reset the environment first")
        return copy.deepcopy({"protocol": "coding-edit-v1", "seed": self.seed, "done": self.done,
            "remaining_actions": self.config.horizon-len(self.history),
            "budget": {"optimizer_steps_used": self.optimizer_steps,
                "optimizer_steps_remaining": self.config.max_optimizer_steps-self.optimizer_steps,
                "edit_evaluations_used": self.edit_evaluations,
                "edit_evaluations_remaining": self.config.max_edit_evaluations-self.edit_evaluations},
            "baseline_development": {k: metric(v) for k,v in self.baseline.items()},
            "action_history": [{"action": h["action"], "reward": h["reward"],
                "development": h["development"]} for h in self.history],
            "support": [{"instruction": p.instruction, "solution": p.solution,
                "student_code": r.get("evaluated_code",r["code"]), "grade": r["grade"]} for p, r in zip(self.support, self.support_result["rows"])],
            "development": {k: metric(v) for k,v in self.current.items()},
            "weight_statistics": self.inspection,
            "last_edit": self.history[-1]["edit"] if self.history else None})

    def step(self, action):
        if not self.ready or self.done:
            raise RuntimeError("step requires an active episode")
        action = validate_action(action, self.student.num_layers)
        steps = action["steps"] if action["kind"] == "lora" else 0
        evaluations = int(action["kind"] not in ("noop", "stop"))
        if (self.optimizer_steps+steps > self.config.max_optimizer_steps
                or self.edit_evaluations+evaluations > self.config.max_edit_evaluations):
            raise BudgetExceeded("action exceeds cumulative optimizer or development-evaluation budget")
        previous = self._objective(self.current)
        if action["kind"] == "rollback":
            self.student.restore_snapshot(self.starting_state)
            edit = {"kind": "rollback", "steps": 0, "delta_norm": 0., "tensors": [],
                    "target": "episode_start"}
        else:
            edit = self.student.edit(action, self.support, self.rehearsal, seed=self.seed)
        return self._finish_step(action, edit, previous, evaluations)

    def step_control(self, values, label):
        """Trusted benchmark-only intervention; unavailable in the editor action API."""
        if not self.ready or self.done:
            raise RuntimeError("step requires an active episode")
        if label not in ("random_sign", "wrong_layer"):
            raise ValueError("unknown matched control")
        if self.edit_evaluations >= self.config.max_edit_evaluations:
            raise BudgetExceeded("control exceeds development-evaluation budget")
        previous = self._objective(self.current)
        edit = self.student.apply_tensor_values(values)
        return self._finish_step({"kind": "matched_control", "label": label}, edit, previous, 1)

    def _finish_step(self, action, edit, previous, evaluations):
        self.optimizer_steps += edit["steps"]
        self.edit_evaluations += evaluations
        if action["kind"] not in ("noop", "stop"):
            self.current = self._scores()
            self.support_result = self.student.evaluate(self.support)
            self.inspection = self.student.inspect()
        cost = self.config.step_cost*edit["steps"] + self.config.action_cost*(action["kind"] != "stop")
        reward = self._objective(self.current) - previous - cost
        self.total_reward += reward
        self.history.append({"action": action, "edit": edit, "reward": reward,
            "development": {k: metric(v) for k,v in self.current.items()}})
        terminated = action["kind"] == "stop"
        truncated = not terminated and len(self.history) >= self.config.horizon
        self.done = terminated or truncated
        return self.observation(), reward, terminated, truncated, {"cost": cost}

    def finalize(self, max_retention_regression=0., protect_case_accuracy=False):
        """Apply a development-only acceptance gate before any terminal tests."""
        from .headroom import paired_changes, acceptance
        if not self.ready or self._final is not None or self.finalization is not None:
            raise RuntimeError("finalize exactly once, before terminal evaluation")
        changes = paired_changes(self.baseline, self.current)
        accepted = acceptance(changes, max_retention_regression)
        case_gains = {k: self.current[k]["case_accuracy"]-self.baseline[k]["case_accuracy"] for k in self.current}
        if protect_case_accuracy and any(x < -1e-12 for x in case_gains.values()):
            accepted = False
        adjustment = 0.
        if not accepted:
            adjustment = -self._objective(self.current)
            self.student.restore_snapshot(self.starting_state)
            self.current = copy.deepcopy(self.baseline)
            self.support_result = copy.deepcopy(self.initial_support)
            self.inspection = copy.deepcopy(self.initial_inspection)
            self.total_reward += adjustment
        self.finalization = {"accepted": accepted, "candidate_changes": changes,
            "max_retention_regression": max_retention_regression, "reward_adjustment": adjustment,
            "rollback_target": "episode_start" if not accepted else None}
        if protect_case_accuracy:
            self.finalization.update(protect_case_accuracy=True, candidate_case_gains=case_gains)
        self.close_episode()
        return copy.deepcopy(self.finalization)

    def close_episode(self):
        if not self.ready:
            raise RuntimeError("reset first")
        self.done = True

    def final_evaluation(self):
        if not self.ready or not self.done:
            raise RuntimeError("final evaluation is available only after closing the episode")
        if self._final is None:
            self._final = {"coding": self.student.evaluate(episode_problems(self.config,"test", self.seed)),
                           "retention": self.student.evaluate(episode_problems(self.config,"retention_test", self.seed))}
        return copy.deepcopy(self._final)

    def record(self, include_final=False):
        if not self.ready:
            raise RuntimeError("reset first")
        result = {"protocol": "coding-edit-v1", "seed": self.seed, "config": asdict(self.config),
            "model": {k: self.student.manifest[k] for k in ("model_id", "revision")},
            "decoding": {"device": str(self.student.device), "dtype": str(self.student.dtype),
                "batch_size": self.student.batch_size, "max_new_tokens": self.student.max_new_tokens, "sampling": False},
            "baseline_development": self.baseline, "history": self.history,
            "development": self.current, "total_reward": self.total_reward, "done": self.done}
        result["budget_used"] = {"optimizer_steps": self.optimizer_steps,
            "edit_evaluations": self.edit_evaluations, "initial_scoring_rounds": 1}
        result["finalization"] = self.finalization
        if include_final:
            result["final"] = self.final_evaluation()
        return copy.deepcopy(result)
