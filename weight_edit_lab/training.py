from __future__ import annotations

import math
import statistics
from dataclasses import asdict

from .agents import FEATURES, RATES, RatePolicy
from .environment import Config, WeightEditEnv
from .tasks import MODES, rng_for


def train(episodes: int = 300, seed: int = 0, config: Config | None = None,
          learning_rate: float = 0.03) -> tuple[dict, list[dict]]:
    if type(episodes) is not int or not 1 <= episodes <= 100000:
        raise ValueError("episodes must be an integer in 1..100000")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning rate must be in (0, 1]")
    config = config or Config()
    policy = RatePolicy.initial()
    rng = rng_for("trainer", seed)
    baseline = [0.0] * config.horizon
    history = []
    for episode in range(episodes):
        mode = MODES[episode % len(MODES)]
        env = WeightEditEnv(config, mode)
        observation, _ = env.reset(seed=seed * 100000 + episode, split="train")
        transitions = []
        while not observation["done"]:
            action, policy_data = policy.choose(observation, rng)
            observation, reward, _, _, _ = env.step(action)
            transitions.append((*policy_data, reward))
        returns = [0.0] * len(transitions)
        future = 0.0
        for t in reversed(range(len(transitions))):
            future += transitions[t][-1]
            returns[t] = future
        gradient = [[0.0] * len(FEATURES) for _ in RATES]
        for t, (x, probabilities, chosen, _) in enumerate(transitions):
            advantage = max(-5.0, min(5.0, returns[t] - baseline[t]))
            for a, row in enumerate(gradient):
                for f in range(len(FEATURES)):
                    row[f] += advantage * (float(a == chosen) - probabilities[a]) * x[f]
            baseline[t] = 0.95 * baseline[t] + 0.05 * returns[t]
        for a, row in enumerate(policy.weights):
            for f in range(len(FEATURES)):
                row[f] += learning_rate * gradient[a][f] / len(transitions)
        history.append({"episode": episode, "mode": mode, "return": returns[0]})
    document = policy.export()
    document["training"] = {"algorithm": "REINFORCE with reward-to-go and a lagged per-step baseline",
                            "split": "train", "episodes": episodes, "seed": seed,
                            "config": asdict(config), "learning_rate": learning_rate,
                            "initial_policy": "uniform over six gradient step sizes",
                            "last_50_mean_return": statistics.mean(r["return"] for r in history[-50:])}
    return document, history
