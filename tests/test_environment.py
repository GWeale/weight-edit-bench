import copy
import math
import unittest
from dataclasses import replace

from weight_edit_lab import Config, Network, WeightEditEnv
from weight_edit_lab.tasks import Task


class EnvironmentTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences_for_every_weight(self):
        network = Network(3)
        task = Task.generate(4, "train", network)
        weights = list(task.initial_weights)
        _, gradient = network.loss_gradient(weights, task.support)
        for i in range(network.size):
            plus, minus = list(weights), list(weights)
            plus[i] += 1e-6
            minus[i] -= 1e-6
            finite_difference = (network.loss(plus, task.support) - network.loss(minus, task.support)) / 2e-6
            self.assertAlmostEqual(gradient[i], finite_difference, places=7)

    def test_actions_change_real_parameters_and_predictions(self):
        env = WeightEditEnv(Config(horizon=2))
        before, _ = env.reset(seed=5)
        delta = [0.0] * len(before["weights"])
        delta[-1] = 0.1
        after, _, _, _, _ = env.step({"delta": delta})
        self.assertAlmostEqual(after["weights"][-1] - before["weights"][-1], 0.1)
        net = env.network
        self.assertAlmostEqual(net.predict(after["weights"], 0.7) - net.predict(before["weights"], 0.7), 0.1)

    def test_gradient_action_improves_support_loss(self):
        env = WeightEditEnv()
        obs, _ = env.reset(seed=4)
        after, *_ = env.step({"delta": [-0.01 * g for g in obs["gradient"]]})
        self.assertLess(after["support_loss"], obs["support_loss"])

    def test_noop_preserves_model_and_pays_cost(self):
        env = WeightEditEnv(Config(horizon=1))
        before, _ = env.reset()
        after, reward, terminated, truncated, info = env.step({"delta": [0.0] * len(before["weights"])})
        self.assertEqual(after["weights"], before["weights"])
        self.assertEqual(reward, -env.config.step_cost)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["test_improvement"], 0)

    def test_invalid_actions_do_not_mutate_episode(self):
        env = WeightEditEnv()
        obs, _ = env.reset()
        invalid = [{"delta": [0]}, {"delta": [float("nan")] * 25}, {"delta": [True] * 25},
                   {"delta": [float("inf")] * 25}, {"delta": [1001.0] * 25}, {"delta": [10**1000] * 25}, {"stop": 1},
                   {"delta": [0.0] * 25, "stop": True}, [0.0] * 25]
        for action in invalid:
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.checkpoint()["weights"], obs["weights"])
        next_obs, *_ = env.step({"delta": [0.0] * 25})
        self.assertEqual(next_obs["step"], 1)

    def test_global_norm_clip_is_observable(self):
        env = WeightEditEnv()
        obs, _ = env.reset()
        after, _, _, _, info = env.step({"delta": [100.0] * 25})
        self.assertTrue(info["clipped"])
        self.assertAlmostEqual(math.dist(obs["weights"], after["weights"]), env.config.max_delta_norm)

    def test_coordinate_bounds_hold(self):
        env = WeightEditEnv(Config(horizon=5, weight_bound=1.5, max_delta_norm=100))
        obs, _ = env.reset()
        for _ in range(5):
            obs, *_ = env.step({"delta": [100.0] * 25})
        self.assertTrue(all(abs(w) <= 1.5 for w in obs["weights"]))

    def test_final_tests_are_terminal_only(self):
        env = WeightEditEnv()
        obs, _ = env.reset()
        self.assertFalse({"test", "score", "anchor_test", "anchor_score", "seed", "split"} & set(obs))
        for method in (env.result, env.record):
            with self.assertRaises(RuntimeError):
                method()
        _, reward, terminated, truncated, info = env.step({"stop": True})
        self.assertEqual(reward, 0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["edits"], 0)
        with self.assertRaises(RuntimeError):
            env.step({"stop": True})

    def test_final_labels_cannot_influence_observations_or_rewards(self):
        a, b = WeightEditEnv(Config(horizon=2)), WeightEditEnv(Config(horizon=2))
        obs_a, _ = a.reset(seed=6)
        obs_b, _ = b.reset(seed=6)
        b._task = replace(b._task, test=tuple((x, y + 10) for x, y in b._task.test))
        for _ in range(2):
            self.assertEqual(obs_a, obs_b)
            action = {"delta": [-0.02 * g for g in obs_a["gradient"]]}
            obs_a, reward_a, *_ = a.step(action)
            obs_b, reward_b, *_ = b.step(action)
            self.assertEqual(reward_a, reward_b)
        self.assertNotEqual(a.result()["final_test_mse"], b.result()["final_test_mse"])

    def test_reward_telescopes_and_cost_is_separate(self):
        env = WeightEditEnv(Config(horizon=4), "retain")
        obs, _ = env.reset(seed=2)
        rewards = []
        for _ in range(4):
            obs, reward, *_ = env.step({"delta": [-0.02 * g for g in obs["gradient"]]})
            rewards.append(reward)
        frames = env.record()["frames"]
        initial = frames[0]["score_loss"] + frames[0]["anchor_score_loss"]
        final = frames[-1]["score_loss"] + frames[-1]["anchor_score_loss"]
        self.assertAlmostEqual(sum(rewards), (initial-final)/max(initial, 0.05)-4*env.config.step_cost)

    def test_pairing_and_split_isolation(self):
        adapt, retain = WeightEditEnv(), WeightEditEnv(mode="retain")
        a, ia = adapt.reset(seed=3, split="test")
        b, ib = retain.reset(seed=3, split="test")
        self.assertEqual(ia["task_hash"], ib["task_hash"])
        self.assertEqual(a["weights"], b["weights"])
        self.assertEqual(a["support"], b["support"])
        _, train_info = adapt.reset(seed=3, split="train")
        self.assertNotEqual(ia["task_hash"], train_info["task_hash"])

    def test_observations_are_detached(self):
        env = WeightEditEnv()
        obs, _ = env.reset()
        original = copy.deepcopy(obs)
        obs["weights"][0] = 100
        obs["support"][0][1] = 100
        obs["config"]["horizon"] = 1
        self.assertEqual(env.checkpoint()["weights"], original["weights"])
        unchanged, *_ = env.step({"delta": [0.0] * 25})
        self.assertEqual(unchanged["support"], original["support"])
        self.assertEqual(unchanged["support_loss"], original["support_loss"])
        self.assertEqual(unchanged["config"], original["config"])
        again, _ = env.reset()
        self.assertEqual(again, original)

    def test_saved_checkpoint_reproduces_predictions(self):
        env = WeightEditEnv(Config(horizon=1))
        obs, _ = env.reset()
        obs, *_ = env.step({"delta": [-0.03 * g for g in obs["gradient"]]})
        network, weights = Network.from_checkpoint(env.checkpoint())
        for x in (-3, -0.1, 1, 3):
            self.assertEqual(network.predict(weights, x), env.network.predict(obs["weights"], x))

    def test_lifecycle_and_config_validation(self):
        with self.assertRaises(RuntimeError):
            WeightEditEnv().step({"stop": True})
        for settings in ({"horizon": 0}, {"hidden": True}, {"max_delta_norm": 0}, {"step_cost": float("nan")}):
            with self.assertRaises(ValueError):
                Config(**settings)


if __name__ == "__main__":
    unittest.main()
