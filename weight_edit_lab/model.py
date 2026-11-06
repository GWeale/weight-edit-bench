from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Points = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Network:
    """Scalar-input tanh MLP. Layout: input weights, hidden biases, output weights, bias."""

    hidden: int = 8

    def __post_init__(self) -> None:
        if type(self.hidden) is not int or not 1 <= self.hidden <= 64:
            raise ValueError("hidden must be an integer in 1..64")

    @classmethod
    def from_checkpoint(cls, document: dict) -> tuple[Network, list[float]]:
        if document.get("format") != "weight-edit-network-v1":
            raise ValueError("unknown network checkpoint format")
        network = cls(document.get("hidden"))
        weights = document.get("weights")
        if (not isinstance(weights, list) or len(weights) != network.size
                or any(type(w) not in (int, float) or abs(w) > 100 or not math.isfinite(w) for w in weights)):
            raise ValueError("invalid network checkpoint weights")
        return network, list(weights)

    @property
    def size(self) -> int:
        return 3 * self.hidden + 1

    @property
    def names(self) -> list[str]:
        return ([f"input.{i}" for i in range(self.hidden)]
                + [f"hidden_bias.{i}" for i in range(self.hidden)]
                + [f"output.{i}" for i in range(self.hidden)] + ["output_bias"])

    def predict(self, weights: Sequence[float], x: float) -> float:
        h = self.hidden
        return weights[-1] + sum(weights[2 * h + i] * math.tanh(weights[i] * x + weights[h + i])
                                 for i in range(h))

    def loss(self, weights: Sequence[float], points: Points) -> float:
        return sum((self.predict(weights, x) - y) ** 2 for x, y in points) / len(points)

    def loss_gradient(self, weights: Sequence[float], points: Points) -> tuple[float, list[float]]:
        h = self.hidden
        gradient = [0.0] * self.size
        loss = 0.0
        for x, y in points:
            activations = [math.tanh(weights[i] * x + weights[h + i]) for i in range(h)]
            error = weights[-1] + sum(weights[2 * h + i] * activations[i] for i in range(h)) - y
            loss += error * error
            scale = 2 * error / len(points)
            gradient[-1] += scale
            for i, activation in enumerate(activations):
                upstream = scale * weights[2 * h + i] * (1 - activation * activation)
                gradient[i] += upstream * x
                gradient[h + i] += upstream
                gradient[2 * h + i] += scale * activation
        return loss / len(points), gradient
