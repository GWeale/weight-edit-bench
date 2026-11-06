import copy
import importlib.util
import unittest

from weight_edit_lab.coding.actions import starter_catalog, validate_action
from weight_edit_lab.coding.environment import CodingConfig, CodingEditEnv
from weight_edit_lab.coding.interpreter import grade
from weight_edit_lab.coding.tasks import problems


class CodingTaskTests(unittest.TestCase):
    def test_reference_programs_and_parameter_separation(self):
        instructions = {s: set() for s in ("support", "score", "test")}
        for split in (*instructions, "retention_score", "retention_test", "rehearsal"):
            for seed in range(10):
                for p in problems(split, seed):
                    self.assertTrue(grade(p.solution, p.cases)["correct"], p.name)
                    if split in instructions:
                        instructions[split].add(p.instruction)
        self.assertFalse(instructions["support"] & instructions["score"])
        self.assertFalse(instructions["support"] & instructions["test"])
        self.assertFalse(instructions["score"] & instructions["test"])
        self.assertNotEqual(problems("retention_score", 0)[0].cases, problems("retention_test", 0)[0].cases)

    def test_rejects_host_access_and_resource_abuse(self):
        programs = [
            "import os\ndef solve(xs):\n    return 0",
            "def solve(xs):\n    return open('/etc/passwd').read()",
            "def solve(xs):\n    return xs.__class__",
            "def solve(xs):\n    while True:\n        pass",
            "def solve(xs):\n    return [1] * 1000000000000",
            "def solve(xs):\n    return '%1000000000000s' % 1",
            "def solve(xs):\n    a=[0]\n    a[0]=a\n    return a",
            "def solve(xs):\n    a=[]\n    b=[a]\n    a.append(b)\n    return a",
            "def solve(xs):\n    return [i for i in range(1000000000000)]",
            "def solve(xs):\n    return min(xs)",
        ]
        for source in programs:
            result = grade(source, (([], 0),))
            self.assertFalse(result["correct"], source)
            self.assertIsNotNone(result["error"], source)

    def test_loops_and_short_circuit_have_python_semantics(self):
        source = """def solve(xs):
    total = 0
    for x in xs:
        if x < 0:
            continue
        if x > 10:
            break
        total += x
    return total if xs and xs[0] == 1 else 0
"""
        self.assertTrue(grade(source, (([], 0), ([1,-3,4,20,2],5), ([2],0)))["correct"])

    def test_action_bounds_and_copy(self):
        catalog = starter_catalog(24)
        for entry in catalog:
            self.assertEqual(validate_action(entry["action"], 24), entry["action"])
        action = copy.deepcopy(catalog[1]["action"])
        copied = validate_action(action, 24)
        copied["layers"].append(20)
        self.assertEqual(len(action["layers"]), 2)
        for key, value in (("layers", [-1]), ("steps", 25), ("rank", 0), ("lr", float("nan")),
                           ("data", "test"), ("module", "lm_head")):
            invalid = dict(action, **{key: value})
            with self.assertRaises(ValueError):
                validate_action(invalid, 24)


class FakeStudent:
    num_layers = 24
    def __init__(self):
        self.seen, self.edited = [], False
    def restore(self):
        self.edited = False
    def snapshot(self):
        return self.edited
    def restore_snapshot(self, state):
        self.edited = state
    def inspect(self):
        return [{"shape": [4,4], "rms": .1}]
    def edit(self, action, support, rehearsal, seed):
        self.edited = action["kind"] == "lora"
        return {"kind": action["kind"], "steps": 8 if self.edited else 0}
    def evaluate(self, ps):
        self.seen.extend(p.name for p in ps)
        accuracy = 1. if self.edited or ps[0].name.startswith("retention") else .5
        return {"accuracy": accuracy, "case_accuracy": accuracy,
            "rows": [{"code": "def solve(xs): return 0", "grade": {"correct": False}} for p in ps]}


class CodingContractTests(unittest.TestCase):
    def test_final_is_unavailable_to_active_editor(self):
        student = FakeStudent()
        env = CodingEditEnv(student, CodingConfig(horizon=1))
        observation, _ = env.reset(seed=0)
        observation["support"][0]["solution"] = "tampered"
        self.assertNotEqual(env.observation()["support"][0]["solution"], "tampered")
        with self.assertRaises(RuntimeError):
            env.final_evaluation()
        next_obs, reward, terminal, truncated, info = env.step(starter_catalog(24)[1]["action"])
        self.assertAlmostEqual(reward, .55-.005)
        self.assertFalse(terminal)
        self.assertTrue(truncated)
        self.assertNotIn("final", next_obs)
        self.assertNotIn("final", info)
        self.assertFalse(any(n.startswith(("test-", "retention_test-")) for n in student.seen))
        env.final_evaluation()
        self.assertTrue(any(n.startswith("test-") for n in student.seen))
        with self.assertRaises(RuntimeError):
            env.step({"kind": "noop"})

    def test_reset_restores_model_and_noop_does_not_gain_reward(self):
        env = CodingEditEnv(FakeStudent(), CodingConfig(horizon=1))
        env.reset(seed=1)
        env.step(starter_catalog(24)[1]["action"])
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs["development"]["coding"]["accuracy"], .5)
        self.assertLessEqual(env.step({"kind": "noop"})[1], 0)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional coding runtime")
class PolicyLearningTests(unittest.TestCase):
    def test_group_relative_update_favors_rewarded_action(self):
        import torch
        from weight_edit_lab.coding.editor import train_editor
        from weight_edit_lab.coding.backend import tensor_digest
        class Policy:
            def __init__(self):
                self.logits = torch.nn.Parameter(torch.zeros(2))
                self.parameters = [self.logits]
            def probabilities(self, observation):
                return torch.softmax(self.logits, dim=0)
            def fingerprints(self):
                return {"logits": tensor_digest(self.logits)}
        policy = Policy()
        result = train_editor(policy, [{}], lambda ctx, action: action, updates=6, group_size=16, lr=.005)
        self.assertGreater(policy.probabilities({})[1].item(), .5)
        self.assertEqual(result["changed_adapter_tensors"], 1)
        constant = Policy()
        result = train_editor(constant, [{}], lambda ctx, action: 1., updates=3, group_size=8)
        self.assertEqual(result["changed_adapter_tensors"], 0)

    def test_low_rank_update_preserves_frozen_base_until_merge(self):
        import torch
        from weight_edit_lab.coding.backend import LowRankLinear
        base = torch.nn.Linear(3, 2, bias=False)
        before = base.weight.detach().clone()
        wrapper = LowRankLinear(base, 2, 0)
        x = torch.tensor([[1., 2., -1.]])
        self.assertTrue(torch.equal(wrapper(x), base(x)))
        optimizer = torch.optim.SGD((wrapper.a, wrapper.b), lr=.1)
        loss = (wrapper(x)-torch.tensor([[2.,-2.]])).square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(before, base.weight))
        self.assertFalse(torch.equal(wrapper(x), base(x)))
        expected = wrapper(x).detach()
        with torch.no_grad():
            base.weight.add_(wrapper.delta())
        self.assertTrue(torch.allclose(base(x), expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
