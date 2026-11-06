"""Run from the repository root: python3 -m examples.custom_editor."""

from pathlib import Path

from weight_edit_lab import Config, Network, WeightEditEnv
from weight_edit_lab.runner import atomic_json

env = WeightEditEnv(Config(), mode="retain")
observation, _ = env.reset(seed=42, split="test")
while not observation["done"]:
    action = {"delta": [-0.08 * g for g in observation["gradient"]]}
    observation, reward, terminated, truncated, info = env.step(action)

checkpoint = env.checkpoint()
atomic_json(Path("runs/custom.network.json"), checkpoint)
network, weights = Network.from_checkpoint(checkpoint)
print("Saved edited network prediction at x=0.5:", network.predict(weights, 0.5))
print("Final metrics:", env.result())
