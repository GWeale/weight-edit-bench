import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.test_editing_benchmark import MemoryStudent
from weight_edit_lab.coding.environment import CodingConfig
from weight_edit_lab.coding.external import FileEditor, RunJournal, validate_response
from weight_edit_lab.coding.policy_runner import run_policy_episode
from weight_edit_lab.runner import atomic_json


class ExternalEditorTests(unittest.TestCase):
    def response(self, request_id):
        return {"request_id": request_id, "action": {"kind": "stop"},
                "rationale": "Preserve the starting model.",
                "prediction": {"coding_gain": 0., "retention_gain": 0.}}

    def test_public_exchange_binds_response_and_excludes_hidden_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = FileEditor(Path(folder)/"exchange", 24, timeout=2, poll_seconds=.01)
            def answer():
                path = bridge.directory/"request-0.json"
                for _ in range(200):
                    if path.exists(): break
                    time.sleep(.01)
                request = json.loads(path.read_text())
                atomic_json(bridge.directory/"response-0.json", self.response(request["request_id"]))
            thread = threading.Thread(target=answer)
            thread.start()
            result = bridge({"development": {"coding": {"accuracy": .5}}, "final": {"secret": 42}, "token": "hidden"})
            thread.join()
            request = json.loads((bridge.directory/"request-0.json").read_text())
            self.assertEqual(set(request["observation"]), {"development"})
            self.assertEqual(result["action"], {"kind": "stop"})
            self.assertFalse(result["policy"]["producer_weights_verified"])
            bridge.close("completed")
            self.assertEqual(json.loads((bridge.directory/"closed.json").read_text())["decisions_requested"], 1)

    def test_stale_invalid_and_nonfinite_responses_fail_closed(self):
        with self.assertRaises(ValueError): validate_response(self.response("old"), "new", 24)
        r = self.response("id")
        r["prediction"]["coding_gain"] = float("nan")
        with self.assertRaises(ValueError): validate_response(r, "id", 24)
        r = self.response("id")
        r["action"] = {"kind": "execute_shell"}
        with self.assertRaises(ValueError): validate_response(r, "id", 24)

    def test_timeout_preserves_unanswered_request(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = FileEditor(Path(folder)/"exchange", 24, timeout=.02, poll_seconds=.01)
            with self.assertRaises(TimeoutError): bridge({"development": {}})
            self.assertTrue((bridge.directory/"request-0.json").exists())

    def test_observer_is_detached_and_final_scores_arrive_after_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            events = []
            student = MemoryStudent()
            def observe(event):
                events.append(event["kind"])
                if "record" in event:
                    event["record"]["development"]["coding"]["accuracy"] = -99
            def decide(observation):
                self.assertNotIn("terminal_started", events)
                self.assertNotIn("final", observation)
                return {"action": {"kind": "stop"}}
            result = run_policy_episode(student, decide, 0, Path(folder)/"episode.json",
                config=CodingConfig(task_suite="algorithmic"), final=True, on_event=observe)
            self.assertEqual(result["development"]["coding"]["accuracy"], .5)
            self.assertEqual(events, ["starting", "baseline", "proposed", "evaluated", "finalized", "terminal_started", "completed"])
            self.assertIn("final", result)

    def test_journal_preserves_events_and_failed_status(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"live.json"
            journal = RunJournal(path, {"editor_weights_trained": False})
            journal.emit("base-astra", {"kind": "starting"})
            journal.finish("failed", error="timeout")
            value = json.loads(path.read_text())
            self.assertEqual(value["status"], "failed")
            self.assertEqual(len(value["events"]), 1)
            with self.assertRaises(ValueError): RunJournal(path, {})


if __name__ == "__main__":
    unittest.main()
