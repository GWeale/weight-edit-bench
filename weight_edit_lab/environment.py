from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .model import Network
from .tasks import MODES, PROTOCOL, Task


@dataclass(frozen=True)
class Config:
    hidden: int = 8
    horizon: int = 20
    max_delta_norm: float = 0.35
    weight_bound: float = 8.0
    retention_weight: float = 1.0
    step_cost: float = 0.001

    def __post_init__(self) -> None:
        for name, maximum in (("hidden", 64), ("horizon", 1000)):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        for name in ("max_delta_norm", "weight_bound", "retention_weight", "step_cost"):
            v = getattr(self, name)
            if type(v) not in (float, int) or not 0 <= v <= 100 or not math.isfinite(v):
                raise ValueError(f"{name} must be finite and in 0..100")
        if self.max_delta_norm == 0 or self.weight_bound < 1.5:
            raise ValueError("delta norm must be positive and weight bound must be at least 1.5")


class WeightEditEnv:
    """Gym-style reset/step API without third-party dependencies. Observations are JSON-safe."""

    def __init__(self, config: Config | None = None, mode: str = "adapt"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.config = config or Config()
        self.mode = mode
        self.network = Network(self.config.hidden)
        self._task: Task | None = None
        self._weights: list[float] = []
        self._done = False

    @property
    def retention(self) -> float:
        return self.config.retention_weight if self.mode == "retain" else 0.0

    def reset(self, *, seed: int = 0, split: str = "train") -> tuple[dict, dict]:
        self._task = Task.generate(seed, split, self.network)
        self._weights = list(self._task.initial_weights)
        self._step = 0
        self._done = False
        self._return = 0.0
        self._last_delta = [0.0] * self.network.size
        self._clipped = 0
        self._gradient_calls = 0
        self._score_calls = 0
        self._last_score = self._objective()
        self._initial_score = self._last_score
        self._scale = max(self._initial_score, 0.05)
        self._frames = [self._frame()]
        return self._observation(), {"protocol": PROTOCOL, "task_hash": self._task.fingerprint}

    def _require_reset(self) -> Task:
        if self._task is None:
            raise RuntimeError("call reset before using the environment")
        return self._task

    def _objective(self) -> float:
        task = self._require_reset()
        self._score_calls += 1
        return (self.network.loss(self._weights, task.score)
                + self.retention * self.network.loss(self._weights, task.anchor_score))

    def _observation(self) -> dict:
        task = self._require_reset()
        self._gradient_calls += 1
        loss, gradient = self.network.loss_gradient(self._weights, task.support)
        anchor_loss, anchor_gradient = self.network.loss_gradient(self._weights, task.anchor_support)
        return {"protocol": PROTOCOL, "mode": self.mode, "step": self._step,
                "steps_remaining": self.config.horizon - self._step, "config": asdict(self.config),
                "weights": list(self._weights), "parameter_names": self.network.names,
                "support": [list(p) for p in task.support],
                "anchor_support": [list(p) for p in task.anchor_support],
                "support_loss": loss, "anchor_loss": anchor_loss,
                "gradient": [g + self.retention * a for g, a in zip(gradient, anchor_gradient)],
                "last_delta": list(self._last_delta), "done": self._done}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        self._require_reset()
        if self._done:
            raise RuntimeError("episode ended; reset before another step")
        if not isinstance(action, dict):
            raise ValueError("action must be an object containing delta or stop")
        if action == {"stop": True} and type(action["stop"]) is bool:
            self._done = True
            self._last_delta = [0.0] * self.network.size
            self._frames.append(self._frame(action=action, reward=0.0))
            return self._observation(), 0.0, True, False, self.result()
        if set(action) != {"delta"} or not isinstance(action["delta"], list):
            raise ValueError("action must contain exactly delta: [numbers], or stop: true")
        delta = action["delta"]
        if len(delta) != self.network.size or any(type(x) not in (float, int) or abs(x) > 1000
                                                or not math.isfinite(x) for x in delta):
            raise ValueError(f"delta must have {self.network.size} finite numbers, each in -1000..1000")
        norm = math.hypot(*delta)
        scale = min(1.0, self.config.max_delta_norm / norm) if norm else 1.0
        bound = self.config.weight_bound
        proposed = [max(-bound, min(bound, w + d * scale)) for w, d in zip(self._weights, delta)]
        applied = [n - w for n, w in zip(proposed, self._weights)]
        clipped = scale < 1 or any(abs(w + d * scale) > bound for w, d in zip(self._weights, delta))
        self._weights = proposed
        self._last_delta = applied
        self._step += 1
        self._clipped += int(clipped)
        score = self._objective()
        reward = (self._last_score - score) / self._scale - self.config.step_cost
        self._last_score = score
        self._return += reward
        self._done = self._step == self.config.horizon
        self._frames.append(self._frame(action={"delta": list(delta)}, reward=reward, clipped=clipped))
        info = self.result() if self._done else {"clipped": clipped, "applied_delta_norm": math.hypot(*applied)}
        return self._observation(), reward, False, self._done, info

    def _frame(self, *, action: dict | None = None, reward: float = 0.0, clipped: bool = False) -> dict:
        task = self._require_reset()
        return {"step": self._step, "weights": list(self._weights), "action": action,
                "applied_delta": list(self._last_delta), "reward": reward, "clipped": clipped,
                "support_loss": self.network.loss(self._weights, task.support),
                "score_loss": self.network.loss(self._weights, task.score),
                "anchor_score_loss": self.network.loss(self._weights, task.anchor_score)}

    def result(self) -> dict:
        task = self._require_reset()
        if not self._done:
            raise RuntimeError("final test metrics are available only after the episode ends")
        initial = self.network.loss(task.initial_weights, task.test)
        final = self.network.loss(self._weights, task.test)
        anchor = self.network.loss(self._weights, task.anchor_test)
        return {"initial_test_mse": initial, "final_test_mse": final,
                "test_improvement": initial - final, "retention_test_mse": anchor,
                "final_test_objective": final + self.retention * anchor,
                "return": self._return, "edits": self._step, "clipped_edits": self._clipped,
                "weight_change_norm": math.dist(task.initial_weights, self._weights),
                "support_gradient_calls": self._gradient_calls,
                "anchor_gradient_calls": self._gradient_calls, "reward_score_queries": self._score_calls}

    def record(self) -> dict:
        task = self._require_reset()
        result = self.result()
        # Terminal-only audit data must never be sent back to an editing agent.
        return {"protocol": PROTOCOL, "config": asdict(self.config), "mode": self.mode,
                "seed": task.seed, "split": task.split, "task_hash": task.fingerprint,
                "task": asdict(task), "frames": self._frames, "result": result}

    def checkpoint(self) -> dict:
        self._require_reset()
        return {"format": "weight-edit-network-v1", "hidden": self.network.hidden,
                "weights": list(self._weights)}
