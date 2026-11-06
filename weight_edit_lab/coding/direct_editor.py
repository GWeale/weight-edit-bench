"""Public file exchange for a frozen editor with bounded tensor/probe access."""

import copy
import json
import math
import time
import uuid
from pathlib import Path

from ..runner import atomic_json
from ..tasks import digest
from .external import PUBLIC_FIELDS


LANGUAGE = """A restricted Python-expression language, interpreted without exec/eval.
Statements: variable assignments and function calls only. No imports, attributes,
loops, functions, file/network access, optimizer, gradients, fitting or generation.
Tensor arithmetic: + - * / @, integer/list/tensor indices and slices (e.g. W[:, ids]).
Functions (positional arguments only):
weight(full_module_name): current CPU float32 tensor; at most four per program.
activation(probe_id, full_module_name, 'input' or 'output'): recorded last-token vector.
array(list), shape(tensor), transpose(tensor), outer(vector,vector), unit(tensor),
abs(tensor), sqrt(tensor), sum/mean/std/norm(tensor[, integer_axis]),
topk(vector, count): indices of largest values, at most 128 (use abs explicitly).
columns(W, integer_indices, replacement_matrix): COPY of W with those columns replaced.
emit(label, value): show a scalar or <=1024 tensor elements; <=12 labels, 40KB total.
write(full_module_name, replacement_tensor): stage a complete replacement; patch only.
Inspection and patch variables are local to each program. Re-read weights as needed.
Programs: <=12000 characters, 64 statements, tensors <=8 million entries, bounded work.
Each probe request accepts one or two distinct modules and one prompt of at most
4000 characters / 512 tokens after chat formatting. Invalid requests consume the
probe-request budget but do not count as executed forward passes.
All replacements are checked together, cast to the student's dtype and then applied.
The cumulative delta norm of each tensor must be <=10% of its pretrained norm;
oversized/nonfinite edits are rejected, never optimized or silently clipped.
All four edit modules are exposed in tensor_catalog. No other tensor is writable.
For example, an inspection can use W=weight('model.layers.0.mlp.down_proj');
ids=topk(norm(W,0),8); emit('columns',ids); emit('norms',norm(W,0)[ids]).
This example is a syntax illustration, NOT a recommended edit or mechanistic hypothesis.
"""


class DirectFileEditor:
    def __init__(self, directory, student, on_event=None, timeout=600):
        self.directory, self.student = Path(directory), student
        self.directory.mkdir(parents=True, exist_ok=False)
        self.on_event, self.timeout = on_event, timeout
        self.calls, self.inspections, self.probes, self.patch_attempts = 0, 0, 0, 0
        self.trace = []
        self.forward_start = getattr(student, "diagnostic_forward_passes", 0)

    def close(self, status):
        atomic_json(self.directory/"closed.json", {"status": status, "calls": self.calls,
            "inspections": self.inspections, "probe_requests": self.probes,
            "probe_forward_passes": self.costs()["probe_forward_passes"],
            "patch_attempts": self.patch_attempts})

    def __call__(self, observation):
        feedback = None
        while self.calls < 16:
            index = self.calls
            request = {"protocol": "direct-weight-editor-v2", "request_id": uuid.uuid4().hex,
                "index": index, "model": "gpt-6-astra", "language": LANGUAGE,
                "observation": {k: copy.deepcopy(observation[k]) for k in PUBLIC_FIELDS if k in observation},
                "tensor_catalog": self.student.tensor_catalog(), "previous_result": feedback,
                "remaining": {"inspections": 8-self.inspections, "probe_requests": 4-self.probes,
                              "patch_validation_attempts": 2-self.patch_attempts, "accepted_patches": 1},
                "objective": "Improve coding without losing any fully correct retention function or reducing coding/retention case accuracy. One numerical patch only in this pilot. Final tests and control results stay closed until you finish.",
                "responses": {
                    "inspect": {"request_id": "COPY", "kind": "inspect", "code": "emit('shape',shape(weight('model.layers.0.mlp.down_proj')))"},
                    "probe": {"request_id": "COPY", "kind": "probe", "prompt": "Your own practice text", "modules": ["model.layers.0.mlp.down_proj"]},
                    "patch": {"request_id": "COPY", "kind": "patch", "code": "AUTHOR_A_TENSOR_PROGRAM", "rationale": "brief justification",
                        "hypothesis": "testable mechanism and what evidence supports it",
                        "prediction": {"coding_gain": 0., "retention_gain": 0., "coding_case_gain": 0., "retention_case_gain": 0.},
                        "control_prediction": {"random_sign_coding_gain": 0., "wrong_layer_coding_gain": 0.}},
                    "stop": {"request_id": "COPY", "kind": "stop", "rationale": "Why no edit is justified"}},
                "prediction_units": "Changes in accuracy relative to the current student, in [-1,1], recorded before evaluation. Controls use the same patch delta on random signs at the same locations, or deterministic other layers. They are not shown during editing."}
            atomic_json(self.directory/f"request-{index}.json", request)
            response_path = self.directory/f"response-{index}.json"
            started = time.monotonic()
            while not response_path.exists():
                if time.monotonic()-started >= self.timeout:
                    raise TimeoutError(f"direct editor did not answer request {index}")
                time.sleep(.5)
            if response_path.stat().st_size > 65536:
                raise ValueError("response exceeds 64 KiB")
            response = json.loads(response_path.read_text())
            if not isinstance(response, dict) or response.get("request_id") != request["request_id"]:
                raise ValueError("response does not belong to this request")
            self.calls += 1
            kind = response.get("kind")
            expected = set(request["responses"].get(kind, {}))
            if not expected or set(response) != expected:
                raise ValueError("unexpected response kind or fields")
            event = {"request_sha256": digest(request), "kind": kind, "index": index,
                     "response": response, "checkpoint_before": self.student.checkpoint_id()}
            try:
                if kind == "inspect":
                    if self.inspections >= 8:
                        raise ValueError("inspection budget exhausted")
                    self.inspections += 1
                    program = self.student.tensor_program(response["code"])
                    feedback = {"kind": "inspection", "output": program.output, "work": program.work}
                elif kind == "probe":
                    if self.probes >= 4:
                        raise ValueError("probe budget exhausted")
                    self.probes += 1
                    feedback = self.student.probe_activations(f"probe-{self.probes}", response["prompt"], response["modules"])
                elif kind == "patch":
                    if self.patch_attempts >= 2:
                        raise ValueError("patch validation budget exhausted")
                    self.patch_attempts += 1
                    for field in ("rationale", "hypothesis"):
                        if not isinstance(response[field], str) or not 1 <= len(response[field]) <= 2000:
                            raise ValueError("brief rationale and testable hypothesis required")
                    for field in ("prediction", "control_prediction"):
                        if not isinstance(response[field], dict) or set(response[field]) != set(request["responses"]["patch"][field]):
                            raise ValueError("prediction fields differ from contract")
                        if any(type(x) not in (int,float) or not math.isfinite(x) or not -1 <= x <= 1 for x in response[field].values()):
                            raise ValueError("predictions must be finite accuracy changes in [-1,1]")
                    program = self.student.tensor_program(response["code"], writable=True)
                    self.student.apply_tensor_values(program.writes, validate_only=True)
                    event["validated"] = True
                    self.trace.append(event)
                    return {"action": {"kind": "tensor_program", "code": response["code"]},
                        "policy": {"model": "gpt-6-astra", "interface": "direct-file-exchange",
                            "request_sha256": digest(request), "producer_weights_verified": False,
                            **{k: response[k] for k in ("rationale", "hypothesis", "prediction", "control_prediction")},
                            "access_cost": self.costs()}}
                else:
                    if not isinstance(response["rationale"], str) or not 1 <= len(response["rationale"]) <= 2000:
                        raise ValueError("brief stop rationale required")
                    return {"action": {"kind": "stop"}, "policy": {"model": "gpt-6-astra",
                        "rationale": response["rationale"], "access_cost": self.costs()}}
            except (ValueError, TypeError, IndexError, RuntimeError, SyntaxError, ZeroDivisionError) as exc:
                feedback = {"error": type(exc).__name__, "detail": str(exc)[:500], "budget_charged": True}
            if self.student.checkpoint_id() != event["checkpoint_before"]:
                raise RuntimeError("diagnostic request unexpectedly changed model weights")
            event["result"] = feedback
            self.trace.append(event)
            atomic_json(self.directory/f"result-{index}.json", event)
            if self.on_event:
                self.on_event({"kind": "diagnostic", "diagnostic": event})
        raise RuntimeError("direct editor exhausted its request budget without finishing")

    def costs(self):
        return {"requests": self.calls, "inspection_programs": self.inspections,
                "probe_requests": self.probes,
                "probe_forward_passes": getattr(self.student, "diagnostic_forward_passes", 0)-self.forward_start,
                "patch_validation_attempts": self.patch_attempts}
