"""File exchange for a frozen external editor and an observable run journal.

The editor receives only public observations. This is a workflow boundary, not
OS isolation against an agent with arbitrary filesystem access.
"""

import copy
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..runner import atomic_json
from ..tasks import digest
from .actions import MODULES, validate_action


PUBLIC_FIELDS = ("support", "development", "baseline_development", "weight_statistics",
                 "last_edit", "remaining_actions", "budget", "action_history")


def validate_response(response, request_id, num_layers):
    if not isinstance(response, dict) or set(response) != {"request_id", "action", "rationale", "prediction"}:
        raise ValueError("response requires request_id, action, rationale and prediction")
    if response["request_id"] != request_id:
        raise ValueError("response belongs to a different request")
    rationale = response["rationale"]
    if not isinstance(rationale, str) or not 1 <= len(rationale) <= 2000:
        raise ValueError("provide a brief decision rationale, at most 2000 characters")
    prediction = response["prediction"]
    if not isinstance(prediction, dict) or set(prediction) != {"coding_gain", "retention_gain"}:
        raise ValueError("predict changes in coding and retention accuracy")
    for value in prediction.values():
        if type(value) not in (float, int) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("predicted accuracy changes must be finite values in [-1,1]")
    return {"action": validate_action(response["action"], num_layers),
            "policy": {"rationale": rationale, "prediction": copy.deepcopy(prediction)}}


class FileEditor:
    """A controller writes one atomic response file for each fresh request."""

    def __init__(self, directory, num_layers, model="gpt-6-astra", timeout=600, poll_seconds=.5):
        self.directory, self.num_layers = Path(directory), num_layers
        self.model, self.timeout, self.poll_seconds = model, timeout, poll_seconds
        self.index = 0
        if timeout <= 0 or poll_seconds <= 0:
            raise ValueError("positive exchange timeout and polling interval required")
        self.directory.mkdir(parents=True, exist_ok=False)

    def __call__(self, observation):
        index = self.index
        self.index += 1
        request = {"protocol": "external-weight-editor-v1", "request_id": uuid.uuid4().hex,
            "decision": index, "model": self.model,
            "observation": {k: copy.deepcopy(observation[k]) for k in PUBLIC_FIELDS if k in observation},
            "action_space": {"num_layers": self.num_layers, "modules": list(MODULES),
                "kinds": ["lora", "scale_columns", "rollback", "stop"],
                "lora": {"layers": "1 to 4 distinct layer indices", "rank": [1, 8], "steps": [1, 24],
                         "lr": [1e-5, .005], "data": ["support", "rehearsal"]},
                "scale_columns": {"columns": "1 to 32 valid column indices", "factor": [.8, 1.2]}},
            "objective": "Improve fully correct coding programs, preserving all previously correct retention programs. Costs remain charged after rollback. Final tests are inaccessible.",
            "prediction_semantics": "Predict the change from CURRENT development accuracy after this action, in units of 0 to 1. Predictions are recorded, not used as reward.",
            "response_example": {"request_id": "COPY_REQUEST_ID", "action": {"kind": "stop"},
                "rationale": "Brief justification, not a private reasoning transcript.",
                "prediction": {"coding_gain": 0., "retention_gain": 0.}}}
        atomic_json(self.directory/f"request-{index}.json", request)
        response_path = self.directory/f"response-{index}.json"
        started = time.monotonic()
        while not response_path.exists():
            if time.monotonic()-started >= self.timeout:
                raise TimeoutError(f"external editor did not answer decision {index}")
            time.sleep(self.poll_seconds)
        if response_path.stat().st_size > 65536:
            raise ValueError("external response exceeds 64 KiB")
        result = validate_response(json.loads(response_path.read_text()), request["request_id"], self.num_layers)
        result["policy"].update(model=self.model, interface="file-exchange",
            request_sha256=digest(request), producer_weights_verified=False)
        return result

    def close(self, status):
        atomic_json(self.directory/"closed.json", {"status": status, "decisions_requested": self.index})


class RunJournal:
    def __init__(self, path, manifest):
        self.path = Path(path)
        if self.path.exists():
            raise ValueError("journal already exists; preserve the previous run")
        self.value = {"schema": "weight-edit-explorer-v1", "run_id": uuid.uuid4().hex,
            "title": "Astra edits a small coding model", "mode": "frozen_editor",
            "status": "running", "manifest": copy.deepcopy(manifest), "events": []}
        self.write()

    def write(self):
        self.value["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, self.value)

    def emit(self, episode, event):
        self.value["events"].append({"id": len(self.value["events"]), "episode": episode,
            "at": datetime.now(timezone.utc).isoformat(), **copy.deepcopy(event)})
        self.write()

    def finish(self, status, **details):
        self.value.update(status=status, **copy.deepcopy(details))
        self.write()
