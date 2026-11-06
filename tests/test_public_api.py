import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_editing_benchmark import MemoryStudent
from weight_edit_lab.coding.direct_env import DirectConfig, DirectEditEnv
from weight_edit_lab.coding.environment import CodingConfig


class DirectStudent(MemoryStudent):
    def tensor_catalog(self):
        return [{"name": "model.layers.0.mlp.down_proj", "shape": [4, 4]}]

    def tensor_program(self, code, writable=False):
        if code == "invalid":
            raise ValueError("invalid program")
        return SimpleNamespace(output={"state": self.state}, writes={"target": code})

    def apply_tensor_values(self, values, validate_only=False):
        if values["target"] == "oversize":
            raise ValueError("excessive norm")

    def probe_activations(self, probe_id, prompt, modules):
        if len(modules) > 2:
            raise ValueError("at most two modules")
        self.diagnostic_forward_passes = getattr(self, "diagnostic_forward_passes", 0)+1
        self.probe_cache[probe_id] = {"state": self.state}
        return {"probe_id": probe_id}

    def edit(self, action, support, rehearsal, seed):
        if action["kind"] == "tensor_program":
            self.state += 2 if action["code"] == "regress" else 1
        return {"kind": action["kind"], "steps": 0}


class DirectEnvTests(unittest.TestCase):
    def make(self, horizon=3, **access):
        return DirectEditEnv(DirectStudent(), CodingConfig(horizon=horizon,
            max_optimizer_steps=0, max_edit_evaluations=horizon), DirectConfig(**access))

    def test_multiturn_inspect_edit_stop_and_final_separation(self):
        env = self.make()
        with self.assertRaises(RuntimeError): env.step({"kind": "stop"})
        obs, _ = env.reset(seed=9)
        self.assertNotIn("seed", obs["student"])
        obs["student"]["support"].clear()
        self.assertEqual(len(env.observation()["student"]["support"]), 4)
        with self.assertRaises(RuntimeError): env.evaluate()
        obs, reward, ended, truncated, _ = env.step({"kind": "inspect", "code": "inspect"})
        self.assertEqual(obs["feedback"]["output"], {"state": 0})
        self.assertLess(reward, 0)
        self.assertFalse(ended or truncated)
        obs, reward, ended, truncated, info = env.step({"kind": "edit", "code": "improve"})
        self.assertGreater(reward, 0)
        self.assertFalse(ended or truncated)
        self.assertNotIn("evaluation", info)
        obs, _, ended, truncated, info = env.step({"kind": "stop"})
        self.assertTrue(ended and obs["done"])
        self.assertFalse(truncated)
        self.assertTrue(info["finalization"]["accepted"])
        self.assertFalse(any(n.startswith(("test-", "retention_test-")) for n in env.env.student.seen))
        report = env.record(include_final=True)
        self.assertGreater(report["evaluation"]["accepted"]["coding"]["accuracy"],
                           report["evaluation"]["baseline"]["coding"]["accuracy"])
        self.assertEqual(report["costs"]["optimizer_steps"], 0)
        self.assertAlmostEqual(sum(t["reward"] for t in report["transitions"]), report["total_reward"])
        json.dumps(report, allow_nan=False)
        with self.assertRaises(RuntimeError): env.step({"kind": "stop"})

    def test_rejected_candidate_is_saved_and_reward_is_adjusted(self):
        env = self.make(horizon=1)
        env.reset()
        _, reward, ended, truncated, info = env.step({"kind": "edit", "code": "regress"})
        self.assertFalse(ended)
        self.assertTrue(truncated)
        self.assertFalse(info["finalization"]["accepted"])
        self.assertEqual(env.env.student.state, 0)
        self.assertLess(reward, 0)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"run"
            record = env.save(target, include_final=True)
            self.assertEqual(json.loads((target/"candidate.safetensors").read_text()), 2)
            self.assertEqual(json.loads((target/"accepted.safetensors").read_text()), 0)
            self.assertNotEqual(record["evaluation"]["candidate"], record["evaluation"]["accepted"])
            with self.assertRaises(FileExistsError): env.save(target)
        self.assertEqual(env.env.student.state, 0)

    def test_bad_requests_consume_turns_without_mutation_or_forward(self):
        env = self.make(max_turns=3, probes=1)
        env.reset()
        _, _, _, _, info = env.step({"kind": "probe", "prompt": "test", "modules": ["a", "b", "c"]})
        self.assertEqual(info["costs"]["probe_requests"], 1)
        self.assertEqual(info["costs"]["probe_forward_passes"], 0)
        env.step({"kind": "edit", "code": "oversize"})
        obs, reward, ended, truncated, info = env.step({"kind": "lora"})
        self.assertTrue(truncated and obs["done"] and info["invalid_action"])
        self.assertFalse(ended)
        self.assertLess(reward, 0)
        self.assertEqual(env.env.student.state, 0)
        self.assertEqual(info["costs"]["edit_evaluations"], 0)
        self.assertEqual(env.record()["transitions"][1]["action"]["code"], "oversize")

    def test_probe_reset_and_rollback_restore_the_start(self):
        env = self.make()
        env.reset()
        env.step({"kind": "probe", "prompt": "test", "modules": ["a"]})
        self.assertEqual(env.costs()["probe_forward_passes"], 1)
        env.step({"kind": "edit", "code": "improve"})
        env.step({"kind": "rollback"})
        self.assertEqual(env.env.student.state, 0)
        env.reset(seed=2)
        self.assertEqual(env.costs()["probe_forward_passes"], 0)
        self.assertEqual(env.env.student.probe_cache, {})

    def test_config_and_non_json_actions(self):
        for args in ({"probes": 9}, {"max_turns": 0}, {"diagnostic_cost": float("nan")}):
            with self.assertRaises(ValueError): DirectConfig(**args)
        with self.assertRaises(ValueError): DirectEditEnv(DirectStudent(), CodingConfig())
        env = self.make(max_turns=1)
        env.reset()
        obs, _, _, truncated, info = env.step({"kind": object()})
        self.assertTrue(truncated and info["invalid_action"])
        json.dumps(env.record(), allow_nan=False)

    def test_scoring_failure_is_fatal_not_an_invalid_action_reward(self):
        env = self.make()
        env.reset()
        with patch.object(env.env.student, "evaluate", side_effect=ValueError("scorer failure")):
            with self.assertRaisesRegex(ValueError, "scorer failure"):
                env.step({"kind": "edit", "code": "improve"})
        with self.assertRaises(RuntimeError): env.step({"kind": "stop"})
        with self.assertRaises(RuntimeError): env.record()
        env.reset()
        self.assertEqual(env.env.student.state, 0)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional coding runtime")
class RealDirectEnvTests(unittest.TestCase):
    def test_real_tensor_edit_through_public_api_without_optimizer(self):
        import torch
        from tests.test_direct_programming import TensorProgramTests
        from tests.test_editing_benchmark import evaluation
        fixture = TensorProgramTests()
        fixture.setUp()
        student = fixture.student
        student.max_new_tokens, student.batch_size = 64, 1
        def evaluate(problems):
            result = evaluation([True]*len(problems), "tiny")
            for row, problem in zip(result["rows"], problems):
                row["name"] = problem.name
            return result
        student.evaluate = evaluate
        env = DirectEditEnv(student, CodingConfig(horizon=1, max_optimizer_steps=0))
        original = student.model.get_submodule(fixture.name).weight.clone()
        env.reset()
        with patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")), \
             patch.object(torch.Tensor, "backward", side_effect=AssertionError("gradients forbidden")):
            obs, _, _, truncated, info = env.step({"kind": "edit", "code":
                f"W=weight('{fixture.name}'); write('{fixture.name}',W*0.99)"})
        self.assertTrue(truncated and info["finalization"]["accepted"])
        self.assertEqual(obs["feedback"]["edit"]["steps"], 0)
        self.assertTrue(torch.equal(student.model.get_submodule(fixture.name).weight, original*.99))
        env.reset()
        self.assertTrue(torch.equal(student.model.get_submodule(fixture.name).weight, original))


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "optional gym extra")
class GymTests(unittest.TestCase):
    def test_standard_checker_and_registration(self):
        import gymnasium as gym
        import weight_edit_lab.gym_env
        from gymnasium.utils.env_checker import check_env
        env = gym.make("WeightEditBench-v0", mode="retain", config={"horizon": 2})
        check_env(env.unwrapped, skip_render_check=True)
        obs, _ = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))
        with self.assertRaises(RuntimeError): env.unwrapped.final_metrics()
        for _ in range(2):
            obs, _, terminated, truncated, info = env.step(-.05*obs["gradient"])
            self.assertEqual(info, {})
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertIn("final_test_mse", env.unwrapped.final_metrics())
        env.close()

    def test_seeded_episodes_and_invalid_actions(self):
        import numpy as np
        from weight_edit_lab.gym_env import WeightEditGymEnv
        a, b = WeightEditGymEnv(), WeightEditGymEnv()
        first, _ = a.reset(seed=17)
        second, _ = b.reset(seed=17)
        for key in first: np.testing.assert_array_equal(first[key], second[key])
        before = a.core.checkpoint()
        with self.assertRaises(ValueError): a.step(np.full(25, np.nan))
        self.assertEqual(a.core.checkpoint(), before)
        with self.assertRaises(ValueError): a.reset(options={"split": "test"})


if __name__ == "__main__":
    unittest.main()
