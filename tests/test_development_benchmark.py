import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_editing_benchmark import MemoryStudent, evaluation
from weight_edit_lab.coding.development import behavioral_choice, run_benchmark


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional coding runtime")
class DevelopmentBenchmarkTests(unittest.TestCase):
    def exercise(self, improving):
        instances, training_seeds = [], []

        class Student(MemoryStudent):
            def __init__(self, *args, **kwargs):
                super().__init__()
                instances.append(self)

            def load_patch(self, path):
                super().load_patch(path)
                return {"fake": self.checkpoint_id()}

            def edit(self, action, support, rehearsal, seed):
                self.edit_calls += 1
                base = 10*(self.state//10)
                if action["kind"] == "lora":
                    if action["data"] == "rehearsal":
                        self.state = base+(2 if self.state%10 == 2 else 3)
                    elif action["lr"] == .003:
                        self.state = base+2
                    elif action["module"] == "mlp.down_proj" and action["layers"] == [22, 23]:
                        self.state = base+1
                    else:
                        self.state = base+3
                elif action["kind"] == "scale_columns":
                    self.state = base+3
                return {"kind": action["kind"], "steps": action.get("steps", 0)}

            def evaluate(self, ps):
                self.seen.extend(p.name for p in ps)
                seed = int(ps[0].name.split("-")[2])
                retention = "retention" in ps[0].name or "rehearsal" in ps[0].name
                win = improving and ((seed == 0 and self.state%10 == 1) or (seed == 1 and self.state%10 == 2))
                flags = [1]*4 if retention else [i < 1+2*win for i in range(4)]
                result = evaluation(flags, "retention" if retention else "coding")
                for row, p in zip(result["rows"], ps): row.update(name=p.name, instruction=p.instruction)
                return result

        class Policy:
            def __init__(self, manifest, catalog, device, seed):
                self.catalog, self.seed, self.version = catalog, seed, 0

            def choose(self, obs):
                self.assert_no_final(obs)
                return self.catalog[1+self.version], [1/len(self.catalog)]*len(self.catalog)

            def assert_no_final(self, obs):
                if "final" in obs: raise AssertionError("terminal labels reached the editor")

            def fingerprints(self): return {"adapter": f"{self.seed}-{self.version}"}
            def save(self, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self.version))
            def load(self, path): self.version = json.loads(path.read_text())

        def train(policy, contexts, reward_fn, seed, **kwargs):
            self.assertTrue(improving, "RL ran without selection headroom")
            self.assertEqual({c["seed"] for c in contexts}, {0, 1})
            self.assertEqual(len(contexts), 4)
            training_seeds.append(seed)
            observed = [[reward_fn(i, j) for j in range(8)] for i in range(4)]
            self.assertTrue(any(max(row) > 0 for row in observed))
            before = policy.fingerprints()
            policy.version = 1
            return {"weights_before": before, "weights_after": policy.fingerprints(), "changed_adapter_tensors": 1,
                "reference_probabilities": [[1/8]*8 for _ in contexts],
                "final_probabilities": [[0, 0, 1, 0, 0, 0, 0, 0] for _ in contexts]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root/"parent.json"
            parent.write_text("10")
            with patch("weight_edit_lab.coding.backend.Student", Student), patch("weight_edit_lab.coding.editor.EditorPolicy", Policy), \
                    patch("weight_edit_lab.coding.editor.train_editor", train), patch("builtins.print"):
                result = run_benchmark(MemoryStudent.manifest, MemoryStudent.manifest, parent, root/"run", "cpu")
                replay = run_benchmark(MemoryStudent.manifest, MemoryStudent.manifest, parent, root/"run", "cpu")
                self.assertEqual(result["terminal_results"], replay["terminal_results"])
                self.assertEqual(instances[-1].edit_calls, 0, "resume repeated cached optimizer work")
                with patch("weight_edit_lab.coding.development.source_hash", return_value="changed"):
                    with self.assertRaises(ValueError):
                        run_benchmark(MemoryStudent.manifest, MemoryStudent.manifest, parent, root/"run", "cpu")
            self.assertEqual(result["cost"]["matrix_episodes"], 48)
            self.assertEqual(result["rl_executed"], improving)
            self.assertEqual(training_seeds, [0, 1, 2] if improving else [])
            self.assertFalse(result["manifest"]["reserved_families_opened"])
            self.assertFalse(any(any(f in name for f in ("coin_change", "lis_gap", "balanced_signs", "clipped_water"))
                                 for student in instances for name in student.seen))
            decisions = json.loads((root/"run/frozen-decisions.json").read_text())
            self.assertEqual(set(decisions), {"base-seed-2", "inherited-seed-2"})
            self.assertEqual(set(decisions["base-seed-2"]), {"no_edit", "ordinary_lora", "tuned_fixed", "behavioral_selector", "frozen_editor"}
                             | ({"trained_editor_0", "trained_editor_1", "trained_editor_2"} if improving else set()))
            self.assertEqual(result["training_headroom"]["semantic_contexts"], 4)
            return result

    def test_train_gate_holdout_and_resume(self):
        result = self.exercise(True)
        self.assertEqual(result["training_headroom"]["selection_headroom"], .25)

    def test_no_headroom_skips_rl_but_finishes_frozen_comparison(self):
        result = self.exercise(False)
        self.assertEqual(result["training_headroom"]["selection_headroom"], 0)

    def test_behavioral_selector_uses_training_contexts_and_rewards(self):
        a = {"development": {k: {"accuracy": .25, "case_accuracy": .5} for k in ("coding", "retention")}}
        b = copy.deepcopy(a)
        b["development"]["coding"]["accuracy"] = .75
        self.assertEqual(behavioral_choice(b, [a, b], [[1, 0], [0, 1]]), 1)
