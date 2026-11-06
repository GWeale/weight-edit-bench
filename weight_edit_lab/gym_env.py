"""Optional Gymnasium adapter for the dependency-free numerical environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import Config, WeightEditEnv
from .tasks import SPLITS


class WeightEditGymEnv(gym.Env):
    """Continuous full-weight deltas. Final tests are evaluator-only, outside info."""

    metadata = {"render_modes": []}

    def __init__(self, *, mode="adapt", split="train", config=None):
        if split not in SPLITS:
            raise ValueError("unknown split")
        self.core = WeightEditEnv(Config(**config) if isinstance(config, dict) else config, mode)
        self.split = split
        size = self.core.network.size
        self.action_space = spaces.Box(-1000., 1000., (size,), dtype=np.float64)
        unbounded = lambda shape: spaces.Box(-np.inf, np.inf, shape, dtype=np.float64)
        self.observation_space = spaces.Dict({
            "weights": unbounded((size,)), "gradient": unbounded((size,)),
            "last_delta": unbounded((size,)), "support": unbounded((24, 2)),
            "anchor_support": unbounded((16, 2)), "losses": unbounded((2,)),
            "progress": spaces.Box(0., 1., (1,), dtype=np.float64),
            "retention_weight": spaces.Box(0., 100., (1,), dtype=np.float64)})

    def _observation(self, observation):
        result = {k: np.asarray(observation[k], dtype=np.float64)
                  for k in ("weights", "gradient", "last_delta", "support", "anchor_support")}
        result.update(losses=np.array([observation["support_loss"], observation["anchor_loss"]]),
            progress=np.array([observation["step"]/self.core.config.horizon]),
            retention_weight=np.array([self.core.retention]))
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("set the split and config when constructing the environment")
        task_seed = int(self.np_random.integers(0, 2**63-1))
        observation, _ = self.core.reset(seed=task_seed, split=self.split)
        return self._observation(observation), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action):
            raise ValueError("delta is outside the declared action space")
        observation, reward, terminated, truncated, _ = self.core.step({"delta": action.tolist()})
        # Do not leak terminal test metrics through a trainer's info channel.
        return self._observation(observation), reward, terminated, truncated, {}

    def final_metrics(self):
        return self.core.result()


gym.register(id="WeightEditBench-v0", entry_point="weight_edit_lab.gym_env:WeightEditGymEnv")
