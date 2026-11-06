import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_edit_lab.coding.actions import starter_catalog
from weight_edit_lab.coding.environment import BudgetExceeded, CodingConfig, CodingEditEnv
from weight_edit_lab.coding.headroom import acceptance, audit_pilot, paired_changes, summarize_matrix
from weight_edit_lab.tasks import digest


def evaluation(flags, prefix):
    return {"accuracy": sum(flags)/len(flags), "case_accuracy": sum(flags)/len(flags),
        "rows": [{"name": f"{prefix}-{i}", "instruction": f"{prefix} problem {i}",
                  "code": "def solve(xs): return 0", "grade": {"correct": bool(flag)}}
                 for i, flag in enumerate(flags)]}


class MemoryStudent:
    num_layers, batch_size, max_new_tokens = 24, 4, 512
    manifest = {"model_id": "test/model", "revision": "pinned"}
    device, dtype = "cpu", "float32"

    def __init__(self):
        self.state, self.edit_calls, self.seen = 0, 0, []

    def restore(self): self.state = 0
    def snapshot(self): return self.state
    def restore_snapshot(self, state): self.state = state
    def checkpoint_id(self): return digest({"state": self.state})
    def inspect(self): return [{"rms": self.state}]
    def load_patch(self, path): self.state = json.loads(Path(path).read_text())
    def save_patch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state))
        return {"fake": self.checkpoint_id()}

    def edit(self, action, support, rehearsal, seed):
        self.edit_calls += 1
        if action["kind"] == "lora": self.state += 1
        return {"kind": action["kind"], "steps": action.get("steps", 0)}

    def evaluate(self, ps):
        self.seen.extend(p.name for p in ps)
        retention = "retention" in ps[0].name or "rehearsal" in ps[0].name
        flags = ([1, 0, 1, 0] if self.state >= 2 else [1, 1, 0, 0]) if retention else [i < 2+self.state for i in range(4)]
        result = evaluation(flags, "retention" if retention else "coding")
        for row, p in zip(result["rows"], ps):
            row.update(name=p.name, instruction=p.instruction)
        return result


class HeadroomTests(unittest.TestCase):
    def test_net_retention_can_hide_a_lost_skill(self):
        before = {"coding": evaluation([1, 0], "c"), "retention": evaluation([1, 0], "r")}
        after = {"coding": evaluation([1, 1], "c"), "retention": evaluation([0, 1], "r")}
        changes = paired_changes(before, after)
        self.assertEqual(changes["retention"]["gain"], 0)
        self.assertEqual(changes["retention"]["lost_programs"], ["r-0"])
        self.assertFalse(acceptance(changes))
        self.assertTrue(acceptance(changes, .5))

    def test_duplicate_prompts_do_not_add_weight_to_a_context(self):
        a = {"semantic_context": "a", "cells": {"A": {"accepted_gain": .5}, "B": {"accepted_gain": 0.}}}
        b = {"semantic_context": "b", "cells": {"A": {"accepted_gain": 0.}, "B": {"accepted_gain": .5}}}
        report = summarize_matrix([a, a, a, b], ["A", "B"])
        self.assertEqual(report["semantic_contexts"], 2)
        self.assertEqual(report["nested_repetitions"], 2)
        self.assertEqual(report["fixed_accepted_gain"], .25)
        self.assertEqual(report["hindsight_accepted_gain"], .5)
        self.assertEqual(report["selection_headroom"], .25)
        with self.assertRaises(ValueError): summarize_matrix([a], ["A", "B", "C"])

    def test_audit_uses_training_records_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"training").mkdir()
            catalog = starter_catalog(24)[:2]
            manifest = {"protocol": "categorical-coding-pilot-v1", "source_hash": "frozen",
                "config": {"horizon": 1, "family_partition": "train"}, "student": MemoryStudent.manifest,
                "train_seeds": [0], "eval_seeds": [99], "catalog": catalog}
            (root/"manifest.json").write_text(json.dumps(manifest))
            baseline = {"coding": evaluation([1, 0], "c"), "retention": evaluation([1, 1], "r")}
            for candidate in catalog:
                after = copy.deepcopy(baseline)
                if candidate["label"] == "B": after["coding"] = evaluation([1, 1], "c")
                r = {"config": manifest["config"], "seed": 0, "done": True,
                    "model": MemoryStudent.manifest, "source_hash": "frozen", "decoding": {"sampling": False},
                    "initial_observation": {"support": [{"instruction": "training only"}]},
                    "baseline_development": baseline, "development": after, "total_reward": .1,
                    "history": [{"action": candidate["action"], "edit": {"steps": 8 if candidate["label"] == "B" else 0}}]}
                (root/"training"/f"seed-0-action-{candidate['label']}.json").write_text(json.dumps(r))
            # No evaluation directory exists; the audit must not need it.
            self.assertEqual(audit_pilot(root)["summary"]["fixed_accepted_gain"], .5)
            self.assertEqual(audit_pilot(root, max_optimizer_steps=4)["summary"]["fixed_accepted_gain"], 0)
            path = root/"training/seed-0-action-B.json"
            r = json.loads(path.read_text())
            r["final"] = {}
            path.write_text(json.dumps(r))
            with self.assertRaises(ValueError): audit_pilot(root)
            del r["final"]
            r["config"]["family_partition"] = "evaluation"
            path.write_text(json.dumps(r))
            with self.assertRaises(ValueError): audit_pilot(root)


class SequentialContractTests(unittest.TestCase):
    def test_cumulative_budget_rejects_before_mutation(self):
        student = MemoryStudent()
        env = CodingEditEnv(student, CodingConfig(horizon=3, max_optimizer_steps=4))
        env.reset()
        action = dict(starter_catalog(24)[1]["action"], steps=3)
        env.step(action)
        with self.assertRaises(BudgetExceeded): env.step(action)
        self.assertEqual(student.state, 1)
        self.assertEqual(student.edit_calls, 1)
        self.assertEqual(env.observation()["budget"]["optimizer_steps_remaining"], 1)
        env = CodingEditEnv(student, CodingConfig(horizon=3, max_edit_evaluations=1))
        env.reset()
        env.step(action)
        with self.assertRaises(BudgetExceeded): env.step({"kind": "rollback"})

    def test_rollback_preserves_an_inherited_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"parent.json"
            path.write_text("1")
            student = MemoryStudent()
            env = CodingEditEnv(student, CodingConfig(horizon=3))
            env.reset(starting_patch=path)
            env.step(starter_catalog(24)[1]["action"])
            self.assertEqual(student.state, 2)
            env.step({"kind": "rollback"})
            self.assertEqual(student.state, 1)
            self.assertLess(env.total_reward, 0)

    def test_acceptance_gate_rejects_hidden_retention_loss_without_free_reward(self):
        student = MemoryStudent()
        env = CodingEditEnv(student, CodingConfig(horizon=2))
        env.reset()
        for _ in range(2): env.step(starter_catalog(24)[1]["action"])
        result = env.finalize()
        self.assertFalse(result["accepted"])
        self.assertEqual(student.state, 0)
        self.assertEqual(env.optimizer_steps, 16)
        self.assertAlmostEqual(env.total_reward, -.01)
        self.assertFalse(any("test-" in name for name in student.seen))
        with self.assertRaises(RuntimeError): env.finalize()


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional coding runtime")
class PolicyRunnerTests(unittest.TestCase):
    def test_decisions_receive_fresh_feedback_and_stop_ends_calls(self):
        from weight_edit_lab.coding.policy_runner import run_policy_episode
        student = MemoryStudent()
        seen = []
        def decide(obs):
            seen.append(copy.deepcopy(obs))
            self.assertNotIn("final", obs)
            self.assertFalse(any("test-" in name for name in student.seen))
            obs["support"][0]["instruction"] = "policy attempted mutation"
            return {"action": starter_catalog(24)[1]["action"] if len(seen) == 1 else {"kind": "stop"}}
        with tempfile.TemporaryDirectory() as directory:
            r = run_policy_episode(student, decide, 0, Path(directory)/"episode.json", final=True)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["development"]["coding"]["accuracy"], .5)
        self.assertEqual(seen[1]["development"]["coding"]["accuracy"], .75)
        self.assertNotEqual(r["decisions"][0]["observation"]["support"][0]["instruction"], "policy attempted mutation")
        self.assertEqual(r["stopped_by"], "stop")
        self.assertEqual(r["final_changes"]["coding"]["gain"], .25)
        self.assertNotEqual(r["lineage"]["episode_start"], r["lineage"]["accepted"])
        self.assertFalse(r["self_editing_verified"])

    def test_budget_refusal_is_logged_and_does_not_apply_an_edit(self):
        from weight_edit_lab.coding.policy_runner import run_policy_episode
        student = MemoryStudent()
        with tempfile.TemporaryDirectory() as directory:
            r = run_policy_episode(student, lambda obs: {"action": starter_catalog(24)[1]["action"]},
                0, Path(directory)/"episode.json", config=CodingConfig(horizon=3, max_optimizer_steps=8))
        self.assertEqual(student.edit_calls, 1)
        self.assertEqual(r["stopped_by"], "budget_rejected")
        self.assertEqual(r["budget_used"]["optimizer_steps"], 8)
        self.assertNotIn("final", r)

    def test_tensor_snapshot_reload_and_inherited_rollback_are_exact(self):
        import torch
        from weight_edit_lab.coding.backend import Student
        student = Student.__new__(Student)
        student.device, student.dtype = torch.device("cpu"), torch.float32
        student.manifest, student.num_layers, student.originals = MemoryStudent.manifest, 1, {}
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(3, 2, bias=False)
        model.model.layers = torch.nn.ModuleList([layer])
        student.model = model
        original = student.checkpoint_id()
        action = {"kind": "scale_columns", "layer": 0, "module": "mlp.down_proj", "columns": [0], "factor": 1.1}
        student.edit(action, [], [])
        parent, parent_id = student.snapshot(), student.checkpoint_id()
        self.assertNotEqual(original, parent_id)
        student.edit(action, [], [])
        student.restore_snapshot(parent)
        self.assertEqual(student.checkpoint_id(), parent_id)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"parent.safetensors"
            student.save_patch(path)
            student.restore()
            self.assertEqual(student.checkpoint_id(), original)
            student.load_patch(path)
            self.assertEqual(student.checkpoint_id(), parent_id)
        bad = {k: v.clone().fill_(float("nan")) for k, v in parent.items()}
        with self.assertRaises(ValueError): student.restore_snapshot(bad)
        self.assertEqual(student.checkpoint_id(), parent_id)
