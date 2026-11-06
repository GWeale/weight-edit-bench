import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_edit_lab.harbor_runtime import PROTOCOL, InvalidSubmission, grade, read_submission, reference, replay, verify
from weight_edit_lab.harbor_tasks import export_task


class HarborTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = self.root/"task"
        self.spec = export_task(self.task, seed=0)

    def test_export_has_real_harbor_structure_and_separate_verifier(self):
        import tomllib
        config = tomllib.loads((self.task/"task.toml").read_text())
        self.assertEqual(config["schema_version"], "1.4")
        self.assertEqual(config["verifier"]["environment_mode"], "separate")
        self.assertEqual(config["verifier"]["network_mode"], "no-network")
        self.assertEqual(config["artifacts"], ["/app/submission.json"])
        for name in ("instruction.md", "environment/Dockerfile", "solution/solve.sh", "tests/test.sh", "tests/verify.py"):
            self.assertTrue((self.task/name).is_file())
        self.assertEqual((self.task/"tests/spec.json").read_bytes(), (self.task/"environment/spec.json").read_bytes())
        self.assertFalse((self.task/"environment/tests").exists())
        self.assertIn("MIT License", (self.task/"environment/LICENSE").read_text())
        with self.assertRaises(FileExistsError): export_task(self.task)

    def test_reference_improves_and_nop_gets_zero(self):
        document = reference(self.spec, self.root/"reference.json")
        self.assertGreater(grade(self.spec, document)["reward"], .1)
        self.assertEqual(grade(self.spec, {"protocol": PROTOCOL, "actions": []})["reward"], 0.)
        env, observation = replay(self.spec, document)
        self.assertTrue(observation["done"])
        self.assertNotIn("final_test_mse", observation)
        self.assertGreater(env.result()["weight_change_norm"], 0.)

    def test_malformed_oversized_symlink_and_forged_scores(self):
        target = self.root/"submission.json"
        for text in ("not-json", "{\"x\": NaN}", " "*262145):
            target.write_text(text)
            with self.assertRaises(InvalidSubmission): read_submission(target)
        target.write_text("["*2000+"0"+"]"*2000)
        nested = verify(self.task/"tests/spec.json", target, self.root/"nested-logs")
        self.assertEqual(nested["status"], "invalid_submission")
        self.assertEqual(nested["reward"], 0.)
        target.unlink()
        target.symlink_to(self.task/"environment/submission.json")
        with self.assertRaises(InvalidSubmission): read_submission(target)
        target.unlink()
        document = {"protocol": PROTOCOL, "actions": [], "reward": 1.}
        target.write_text(json.dumps(document))
        result = verify(self.task/"tests/spec.json", target, self.root/"logs")
        self.assertEqual(result["status"], "invalid_submission")
        self.assertEqual(json.loads((self.root/"logs/reward.json").read_text()), {"reward": 0.})

    def test_action_budgets_validation_and_terminal_state(self):
        for actions in ([{"delta": [1.]*24}], [{"delta": [float("inf")]*25}],
                        [{"stop": True}, {"stop": True}], [{"delta": [0.]*25}]*22):
            with self.assertRaises(InvalidSubmission): grade(self.spec, {"protocol": PROTOCOL, "actions": actions})

    def test_infrastructure_failures_leave_no_reward(self):
        output=self.root/"logs";output.mkdir()
        (output/"reward.json").write_text('{"reward": 1}')
        with patch("weight_edit_lab.harbor_runtime.make_env", side_effect=RuntimeError("device failed")):
            with self.assertRaises(RuntimeError):
                verify(self.task/"tests/spec.json", self.task/"environment/submission.json", output)
        self.assertFalse((output/"reward.json").exists())

    def test_grader_ignores_modified_agent_spec(self):
        document = reference(self.spec, self.root/"submission.json")
        (self.task/"environment/spec.json").write_text('{"reward": 1}')
        result = verify(self.task/"tests/spec.json", self.root/"submission.json", self.root/"logs")
        self.assertGreater(result["reward"], .1)

    def test_coding_task_pins_revision_and_excludes_local_paths(self):
        model={"model_id":"Qwen/Qwen2.5-Coder-0.5B-Instruct", "revision":"a"*40, "path":"/private/model"}
        target=self.root/"coding"
        spec=export_task(target,track="coding",model=model,max_new_tokens=64)
        self.assertEqual(spec["model"]["path"], "/opt/student")
        self.assertEqual(spec["config"]["max_optimizer_steps"], 0)
        self.assertIn("'a"+'a'*39+"'", (target/"environment/download_model.py").read_text())
        self.assertNotIn("/private/model", (target/"environment/spec.json").read_text())
        with self.assertRaises(ValueError): export_task(self.root/"bad",track="coding",model=dict(model,revision="main"))
        with self.assertRaises(ValueError): export_task(self.root/"bad-split",track="coding",model=model,split="train")
        with self.assertRaises(ValueError): export_task(self.root/"bad-mode",track="coding",model=model,mode="adapt")

    def test_coding_replay_rewards_improvement_and_preserves_rejected_candidate(self):
        from tests.test_public_api import DirectEnvTests
        spec = dict(self.spec, track="coding", access={"max_turns": 16}, config={"case_weight": .25})
        for code, expected in (("improve", .3125), ("regress", 0.)):
            env = DirectEnvTests().make()
            with patch("weight_edit_lab.harbor_runtime.make_env", return_value=env):
                result = grade(spec, {"protocol": PROTOCOL, "actions": [
                    {"kind": "edit", "code": code}, {"kind": "stop"}]})
            self.assertEqual(result["reward"], expected)
            self.assertEqual(result["episode"]["costs"]["optimizer_steps"], 0)
            if code == "regress":
                outcomes = result["episode"]["evaluation"]
                self.assertNotEqual(outcomes["candidate"], outcomes["accepted"])
                self.assertEqual(outcomes["baseline"], outcomes["accepted"])

    def test_final_retention_loss_zeroes_reward_even_if_development_accepts(self):
        from tests.test_public_api import DirectEnvTests
        from tests.test_editing_benchmark import evaluation
        env = DirectEnvTests().make()
        student = env.env.student
        original = student.evaluate

        def evaluate(problems):
            if student.state and problems[0].name.startswith("retention_test-"):
                result = evaluation([0, 1, 1, 0], "retention")
                for row, problem in zip(result["rows"], problems):
                    row.update(name=problem.name, instruction=problem.instruction)
                return result
            return original(problems)

        student.evaluate = evaluate
        with patch("weight_edit_lab.harbor_runtime.make_env", return_value=env):
            result = grade(dict(self.spec, track="coding", access={"max_turns": 16}, config={"case_weight": .25}),
                {"protocol": PROTOCOL, "actions": [{"kind": "edit", "code": "improve"}]})
        self.assertTrue(env.env.finalization["accepted"])
        self.assertFalse(result["retention_ok"])
        self.assertEqual(result["reward"], 0.)
        self.assertTrue(result["final_changes"]["retention"]["lost_programs"])

    @unittest.skipUnless(importlib.util.find_spec("harbor"), "optional Harbor runtime")
    def test_installed_harbor_accepts_the_task(self):
        from harbor.models.task.task import Task
        task = Task(self.task)
        self.assertEqual(task.config.verifier.environment_mode, "separate")
        self.assertEqual(task.config.environment.gpus, 0)


if __name__ == "__main__": unittest.main()
