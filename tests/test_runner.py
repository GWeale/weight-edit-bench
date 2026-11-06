import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_edit_lab import Config
from weight_edit_lab.agents import Baseline, RatePolicy
from weight_edit_lab.cli import main
from weight_edit_lab.replay import render_episode, verify_episode
from weight_edit_lab.report import render_report
from weight_edit_lab.runner import evaluate, run_episode, tune
from weight_edit_lab.training import train


class RunnerTests(unittest.TestCase):
    def test_numerical_replay_detects_tampering(self):
        record = run_episode(Baseline("adam", 0.03), 1, "retain", "test", Config(horizon=3))
        verify_episode(json.loads(json.dumps(record)))
        corrupt = copy.deepcopy(record)
        corrupt["frames"][1]["weights"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "frames"):
            verify_episode(corrupt)

    def test_replay_is_offline_and_escapes_embedded_data(self):
        record = run_episode(Baseline("noop"), 1, "adapt", "test", Config(horizon=1))
        record["agent"] = "</script><script>alert(1)</script>"
        html = render_episode(record)
        self.assertNotIn(record["agent"], html)
        self.assertNotIn("__EPISODE_JSON__", html)
        self.assertNotIn('src="http', html)
        self.assertIn("\\u003c/script", html)

    def test_batch_is_paired_and_resumes_without_rerunning(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            config = Config(horizon=2)
            result = evaluate(out, [0, 1], config=config)
            self.assertEqual(sum(r["completed"] for r in result["rows"]), 16)
            with patch("weight_edit_lab.runner.run_episode", side_effect=AssertionError("reran")):
                again = evaluate(out, [0, 1], config=config)
            self.assertEqual(result, again)
            hashes = {json.loads(p.read_text())["task_hash"] for p in (out / "episodes").glob("*-0.json")}
            self.assertEqual(len(hashes), 1)
            with self.assertRaisesRegex(ValueError, "resume refused"):
                evaluate(out, [0, 1], config=Config(horizon=3))

    def test_failures_are_counted_and_not_scores(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("weight_edit_lab.runner.run_episode", side_effect=RuntimeError("deliberate failure")):
                result = evaluate(Path(d), [1], config=Config(horizon=1))
            self.assertTrue(all(row["errors"] == 1 for row in result["rows"]))
            self.assertTrue(all(row["metrics"]["final_test_mse"]["mean"] is None for row in result["rows"]))
            self.assertTrue(all(row["n"] == 0 for row in result["paired_vs_adam"]))

    def test_training_updates_policy_reproducibly_and_roundtrips(self):
        config = Config(horizon=3)
        document, history = train(8, 2, config)
        again, history_again = train(8, 2, config)
        self.assertEqual(document, again)
        self.assertEqual(history, history_again)
        self.assertNotEqual(document["weights"], RatePolicy.initial().weights)
        self.assertEqual(RatePolicy.load(document).weights, document["weights"])
        self.assertEqual(document["training"]["split"], "train")

    def test_checkpoint_rejects_wrong_shapes_and_nonfinite_weights(self):
        for value in ([[]], [[float("nan")] * 5 for _ in range(6)]):
            doc = RatePolicy.initial().export()
            doc["weights"] = value
            with self.assertRaises(ValueError):
                RatePolicy.load(doc)

    def test_tuning_uses_validation_and_frozen_rates(self):
        document = tune(1, Config(horizon=1))
        self.assertEqual(document["split"], "validation")
        self.assertEqual(len(document["trials"]), 20)
        with tempfile.TemporaryDirectory() as d:
            summary = evaluate(Path(d), [10], config=Config(horizon=1), tuning=document)
            self.assertEqual(summary["manifest"]["rates"], document["rates"])
        document["split"] = "test"
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            evaluate(Path(d), [1], config=Config(horizon=1), tuning=document)

    def test_batch_rejects_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            evaluate(Path(d), [1, 1])

    def test_report_contains_observed_data_and_rejects_duplicate_splits(self):
        with tempfile.TemporaryDirectory() as d:
            summary = evaluate(Path(d), [1], config=Config(horizon=1))
            html = render_report([summary])
            self.assertNotIn("__SUMMARIES_JSON__", html)
            self.assertIn(str(summary["rows"][0]["metrics"]["final_test_mse"]["mean"]), html)
            with self.assertRaises(ValueError):
                render_report([summary, summary])

    def test_stop_is_replayable_and_keeps_the_last_checkpoint(self):
        class Stopper:
            def act(self, obs):
                return {"stop": True} if obs["step"] else {"delta": [-0.02*g for g in obs["gradient"]]}
        record = run_episode(Stopper(), 2, "retain", "test", Config(horizon=4))
        verify_episode(record)
        self.assertEqual(record["result"]["edits"], 1)
        self.assertEqual(record["frames"][-1]["applied_delta"], [0.0]*25)
        self.assertEqual(record["runtime"]["agent_calls"], 2)

    def test_cli_episode_checkpoint_and_replay(self):
        with tempfile.TemporaryDirectory() as d:
            episode = Path(d) / "episode.json"
            html = Path(d) / "replay.html"
            self.assertEqual(main(["episode", "--horizon", "2", "--out", str(episode)]), 0)
            self.assertTrue(episode.with_suffix(".network.json").exists())
            self.assertEqual(main(["verify", str(episode)]), 0)
            self.assertEqual(main(["replay", str(episode), "--out", str(html)]), 0)
            self.assertIn("Weight Edit Lab", html.read_text())


if __name__ == "__main__":
    unittest.main()
