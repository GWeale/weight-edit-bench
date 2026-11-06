from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass

from .model import Network, Points

PROTOCOL = "weight-edit-v1"
SPLITS = ("train", "validation", "test", "ood")
MODES = ("adapt", "retain")


def rng_for(*keys: object) -> random.Random:
    encoded = json.dumps([PROTOCOL, *keys], separators=(",", ":"), allow_nan=False)
    return random.Random(int.from_bytes(hashlib.sha256(encoded.encode()).digest(), "big"))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    seed: int
    split: str
    initial_weights: tuple[float, ...]
    support: Points
    score: Points
    test: Points
    anchor_support: Points
    anchor_score: Points
    anchor_test: Points

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    @classmethod
    def generate(cls, seed: int, split: str, network: Network) -> Task:
        if type(seed) is not int or seed < 0 or split not in SPLITS:
            raise ValueError("seed must be a nonnegative integer and split must be known")
        init = rng_for(split, seed, "weights")
        h = network.hidden
        weights = tuple([init.uniform(-1.5, 1.5) for _ in range(h)]
                        + [init.uniform(-1, 1) for _ in range(h)]
                        + [init.uniform(-0.25, 0.25) for _ in range(h)] + [0.0])
        task_rng = rng_for(split, seed, "function")
        amplitude = task_rng.uniform(0.3, 1.2)
        frequency = task_rng.uniform(2.5, 4) if split == "ood" else task_rng.uniform(0.5, 1.5)
        phase = task_rng.uniform(-math.pi, math.pi)
        slope = task_rng.uniform(-0.3, 0.3)

        def sample(kind: str, n: int, anchor: bool = False) -> Points:
            rng = rng_for(split, seed, kind)
            if anchor:
                xs = [rng.choice((-1, 1)) * rng.uniform(2.5, 4) for _ in range(n)]
                return tuple((x, network.predict(weights, x)) for x in xs)
            xs = [rng.uniform(-2, 2) for _ in range(n)]
            return tuple((x, amplitude * math.sin(frequency * x + phase) + slope * x) for x in xs)

        return cls(seed, split, weights, sample("support", 24), sample("score", 48),
                   sample("test", 128), sample("anchor-support", 16, True),
                   sample("anchor-score", 32, True), sample("anchor-test", 64, True))
