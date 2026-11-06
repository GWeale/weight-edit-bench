from __future__ import annotations

import math
from dataclasses import dataclass, field

from .tasks import PROTOCOL, rng_for

RATES = (0.0, 0.005, 0.02, 0.08, 0.2, 0.5)
FEATURES = ("bias", "progress", "log_support_loss", "log_gradient_norm", "retention_mode")


@dataclass
class Baseline:
    name: str
    rate: float = 0.08
    seed: int = 0
    _m: list[float] = field(default_factory=list)
    _v: list[float] = field(default_factory=list)
    _t: int = 0

    def __post_init__(self) -> None:
        if self.name not in ("noop", "random", "sgd", "adam"):
            raise ValueError(f"unknown baseline: {self.name}")
        if type(self.rate) not in (float, int) or not 0 < self.rate <= 10 or not math.isfinite(self.rate):
            raise ValueError("baseline rate must be finite and in (0, 10]")
        self._rng = rng_for("baseline", self.seed)

    def act(self, observation: dict) -> dict:
        gradient = observation["gradient"]
        n = len(gradient)
        if self.name == "noop":
            return {"delta": [0.0] * n}
        if self.name == "random":
            direction = [self._rng.gauss(0, 1) for _ in gradient]
            norm = math.hypot(*direction)
            return {"delta": [self.rate * d / norm for d in direction]}
        if self.name == "sgd":
            return {"delta": [-self.rate * g for g in gradient]}
        if not self._m:
            self._m, self._v = [0.0] * n, [0.0] * n
        self._t += 1
        self._m = [0.9 * m + 0.1 * g for m, g in zip(self._m, gradient)]
        self._v = [0.999 * v + 0.001 * g * g for v, g in zip(self._v, gradient)]
        return {"delta": [-self.rate * (m / (1 - 0.9 ** self._t))
                           / (math.sqrt(v / (1 - 0.999 ** self._t)) + 1e-8)
                           for m, v in zip(self._m, self._v)]}


def features(observation: dict) -> list[float]:
    def bounded_log(v: float) -> float:
        return max(-2.0, min(2.0, math.log(max(v, 1e-8)))) / 2
    return [1.0, observation["step"] / observation["config"]["horizon"],
            bounded_log(observation["support_loss"]),
            bounded_log(math.hypot(*observation["gradient"])), float(observation["mode"] == "retain")]


@dataclass
class RatePolicy:
    """REINFORCE policy over gradient step sizes; the action sent to the env is a full delta vector."""

    weights: list[list[float]]

    @classmethod
    def initial(cls) -> RatePolicy:
        return cls([[0.0] * len(FEATURES) for _ in RATES])

    def probabilities(self, x: list[float]) -> list[float]:
        logits = [sum(w * f for w, f in zip(row, x)) for row in self.weights]
        exps = [math.exp(v - max(logits)) for v in logits]
        total = sum(exps)
        return [v / total for v in exps]

    def choose(self, observation: dict, rng=None) -> tuple[dict, tuple]:
        x = features(observation)
        probabilities = self.probabilities(x)
        index = (rng.choices(range(len(RATES)), weights=probabilities)[0] if rng is not None
                 else max(range(len(RATES)), key=lambda i: probabilities[i]))
        return {"delta": [-RATES[index] * g for g in observation["gradient"]]}, (x, probabilities, index)

    def act(self, observation: dict) -> dict:
        return self.choose(observation)[0]

    def export(self) -> dict:
        return {"format": "weight-edit-rate-policy-v1", "protocol": PROTOCOL,
                "rates": list(RATES), "features": list(FEATURES), "weights": self.weights}

    @classmethod
    def load(cls, document: dict) -> RatePolicy:
        if (document.get("format") != "weight-edit-rate-policy-v1" or document.get("protocol") != PROTOCOL
                or document.get("rates") != list(RATES) or document.get("features") != list(FEATURES)):
            raise ValueError("incompatible policy checkpoint")
        matrix = document.get("weights")
        if (not isinstance(matrix, list) or len(matrix) != len(RATES)
                or any(not isinstance(row, list) or len(row) != len(FEATURES)
                       or any(type(v) not in (float, int) or abs(v) > 1000 or not math.isfinite(v) for v in row)
                       for row in matrix)):
            raise ValueError("invalid policy weights")
        return cls([list(row) for row in matrix])
