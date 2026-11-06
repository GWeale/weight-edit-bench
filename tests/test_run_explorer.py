import copy
import json
import unittest

from examples.render_run_explorer import render


class ExplorerExportTests(unittest.TestCase):
    def test_export_preserves_evidence_without_embedding_executable_text(self):
        malicious = '</script><img src=x onerror="alert(1)"> &'
        snapshot = {"schema": "weight-edit-explorer-v1", "events": [{
            "kind": "evaluated", "episode": "inherited-astra",
            "decision": {"observation": {"private": "omit from export"},
                         "policy": {"rationale": malicious}},
            "record": {"patch": "private/checkpoint", "versions": {"unused": True},
                       "development": {"coding": {"accuracy": .75, "rows": [
                           {"code": "raw duplicate", "evaluated_code": malicious,
                            "grade": {"passed": 64, "total": 64}}]}},
                       "history": [{"edit": "earlier"}, {"edit": "latest"}]}}]}
        original = copy.deepcopy(snapshot)
        page = render(snapshot)
        payload = page.split('<script id="run-data" type="application/json">')[1].split('</script>')[0]
        self.assertNotIn('<', payload)
        self.assertNotIn('&', payload)
        self.assertEqual(snapshot, original)
        view = json.loads(payload)
        event = view["events"][0]
        self.assertNotIn("observation", event["decision"])
        self.assertNotIn("patch", event["record"])
        self.assertEqual(event["decision"]["policy"]["rationale"], malicious)
        self.assertEqual(event["record"]["development"]["coding"]["accuracy"], .75)
        self.assertEqual(event["record"]["development"]["coding"]["rows"][0]["evaluated_code"], malicious)
        self.assertEqual(event["record"]["history"], [{"edit": "latest"}])

    def test_refuse_wrong_schema_and_nonfinite_data(self):
        with self.assertRaises(ValueError):
            render({"schema": "other", "events": []})
        with self.assertRaises(ValueError):
            render({"schema": "weight-edit-explorer-v1", "events": [], "value": float("nan")})


if __name__ == "__main__":
    unittest.main()
