"""Provider-independent, bounded tool actions for direct weight-program policies."""

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .direct_editor import LANGUAGE
from .environment import CodingConfig, CodingEditEnv, episode_problems
from .external import PUBLIC_FIELDS
from ..runner import atomic_json, source_hash


@dataclass(frozen=True)
class DirectConfig:
    max_turns: int = 16
    inspections: int = 8
    probes: int = 4
    diagnostic_cost: float = .0001
    invalid_action_cost: float = .001

    def __post_init__(self):
        for name, low, high in (("max_turns", 1, 128), ("inspections", 0, 64), ("probes", 0, 8)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        for name in ("diagnostic_cost", "invalid_action_cost"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in 0..1")


class DirectEditEnv:
    """JSON reset/step interface. No provider SDK, file exchange, or optimizer.

    This object owns one Student for the duration of an episode. The API separates
    adaptation from final tests; in-process Python is not a security boundary.
    """

    protocol = "direct-edit-bench-v1"

    def __init__(self, student, config=None, access=None):
        config = config or CodingConfig(horizon=3, task_suite="algorithmic",
            case_weight=.25, max_optimizer_steps=0, max_edit_evaluations=3)
        if config.max_optimizer_steps != 0:
            raise ValueError("the direct track requires max_optimizer_steps=0")
        self.env = CodingEditEnv(student, config)
        self.access = access or DirectConfig()
        self.ready = False

    def reset(self, *, seed=0, starting_patch=None):
        self.ready = False
        self.env.student.probe_cache = {}
        self.env.reset(seed=seed, starting_patch=starting_patch)
        self.start_id = self.env.student.checkpoint_id()
        self.forward_start = getattr(self.env.student, "diagnostic_forward_passes", 0)
        self.turns = self.inspections = self.probes = 0
        self.total_reward = 0.
        self.done, self.feedback, self.transitions, self._evaluation = False, None, [], None
        self.ready = True
        return self.observation(), {"protocol": self.protocol, "access": asdict(self.access),
                                    "config": asdict(self.env.config)}

    def observation(self):
        if not self.ready:
            raise RuntimeError("reset first")
        public = self.env.observation()
        return copy.deepcopy({"protocol": self.protocol, "done": self.done,
            "student": {k: public[k] for k in PUBLIC_FIELDS if k in public},
            "language": LANGUAGE, "tensor_catalog": self.env.student.tensor_catalog(),
            "remaining": {"turns": self.access.max_turns-self.turns,
                "inspections": self.access.inspections-self.inspections,
                "probes": self.access.probes-self.probes}, "feedback": self.feedback})

    def _prepare(self, action):
        if not isinstance(action, dict) or len(json.dumps(action, allow_nan=False)) > 20000:
            raise ValueError("action must be a JSON object of at most 20000 characters")
        fields = {"inspect": {"kind", "code"}, "probe": {"kind", "prompt", "modules"},
                  "edit": {"kind", "code"}, "rollback": {"kind"}, "stop": {"kind"}}
        kind = action.get("kind")
        if not isinstance(kind, str) or kind not in fields or set(action) != fields[kind]:
            raise ValueError("use inspect(code), probe(prompt, modules), edit(code), rollback, or stop")
        student = self.env.student
        if kind == "inspect":
            if self.inspections >= self.access.inspections:
                raise ValueError("inspection budget exhausted")
            self.inspections += 1
            self.feedback = {"output": student.tensor_program(action["code"]).output}
        elif kind == "probe":
            if self.probes >= self.access.probes:
                raise ValueError("probe budget exhausted")
            self.probes += 1
            self.feedback = student.probe_activations(f"probe-{self.probes}", action["prompt"], action["modules"])
        else:
            if kind != "stop" and self.env.edit_evaluations >= self.env.config.max_edit_evaluations:
                raise ValueError("edit evaluation budget exhausted")
            if kind == "edit":
                # Validate before entering the scoring transition. Runtime/scoring
                # failures must propagate, not masquerade as invalid agent actions.
                program = student.tensor_program(action["code"], writable=True)
                student.apply_tensor_values(program.writes, validate_only=True)
                return {"kind": "tensor_program", "code": action["code"]}
            return {"kind": kind}
        return None

    def step(self, action):
        if not self.ready or self.done:
            raise RuntimeError("step requires an active episode")
        try:
            return self._step(action)
        except Exception:
            # A scorer/device failure can occur after weights change. Require a
            # full reset rather than letting a caller continue from partial state.
            self.ready = False
            raise

    def _step(self, action):
        self.turns += 1
        terminated, truncated = False, False
        self.feedback = None
        invalid = False
        try:
            encoded = json.dumps(action, allow_nan=False)
            recorded_action = json.loads(encoded) if len(encoded) <= 20000 else None
        except (ValueError, TypeError):
            recorded_action = None
        try:
            prepared = self._prepare(action)
        except (ValueError, TypeError, KeyError, SyntaxError, IndexError, ZeroDivisionError) as exc:
            self.feedback = {"error": type(exc).__name__, "message": str(exc)[:1000]}
            reward, invalid = -self.access.invalid_action_cost, True
        else:
            if prepared is None:
                reward = -self.access.diagnostic_cost
            else:
                _, reward, terminated, truncated, _ = self.env.step(prepared)
                self.feedback = {"edit": self.env.history[-1]["edit"]}
        truncated = truncated or (not terminated and self.turns >= self.access.max_turns)
        if terminated or truncated:
            self._candidate = self.env.student.snapshot()
            self.candidate_id = self.env.student.checkpoint_id()
            self.candidate_development = copy.deepcopy(self.env.current)
            gate = self.env.finalize(protect_case_accuracy=True)
            reward += gate["reward_adjustment"]
            self._accepted = self.env.student.snapshot()
            self.accepted_id = self.env.student.checkpoint_id()
            self.done = True
        self.total_reward += reward
        # Preserve invalid JSON-safe proposals too, but never repr() arbitrary
        # objects or retain oversized payloads in the audit record.
        self.transitions.append({"turn": self.turns, "action": recorded_action,
            "feedback": copy.deepcopy(self.feedback), "reward": reward,
            "invalid": invalid, "terminated": terminated, "truncated": truncated})
        info = {"invalid_action": invalid, "costs": self.costs()}
        if self.done:
            info["finalization"] = copy.deepcopy(self.env.finalization)
        return self.observation(), reward, terminated, truncated, info

    def costs(self):
        return {"turns": self.turns, "inspection_requests": self.inspections,
            "probe_requests": self.probes,
            "probe_forward_passes": getattr(self.env.student, "diagnostic_forward_passes", 0)-self.forward_start,
            "edit_evaluations": self.env.edit_evaluations, "optimizer_steps": self.env.optimizer_steps}

    def evaluate(self):
        """Evaluator-only terminal tests. Never call from a training policy."""
        if not self.ready or not self.done:
            raise RuntimeError("final tests require a closed episode")
        if self._evaluation is None:
            results = {}
            try:
                for label, snapshot in (("baseline", self.env.starting_state),
                                        ("candidate", self._candidate), ("accepted", self._accepted)):
                    self.env.student.restore_snapshot(snapshot)
                    results[label] = {key: self.env.student.evaluate(episode_problems(self.env.config, split, self.env.seed))
                        for key, split in (("coding", "test"), ("retention", "retention_test"))}
            finally:
                self.env.student.restore_snapshot(self._accepted)
            self._evaluation = results
        return copy.deepcopy(self._evaluation)

    def record(self, *, include_final=False):
        if not self.ready or not self.done:
            raise RuntimeError("records require a closed episode")
        record = self.env.record()
        record.update(protocol=self.protocol, source_hash=source_hash(), access=asdict(self.access),
            transitions=copy.deepcopy(self.transitions), total_reward=self.total_reward,
            candidate_development=copy.deepcopy(self.candidate_development), costs=self.costs(),
            lineage={"start": self.start_id, "candidate": self.candidate_id, "accepted": self.accepted_id})
        if include_final:
            record["evaluation"] = self.evaluate()
        return record

    def save(self, directory, *, include_final=False):
        """Save both candidates and accepted weights; refuse to overwrite a run."""
        record = self.record(include_final=include_final)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            for label, snapshot in (("candidate", self._candidate), ("accepted", self._accepted)):
                self.env.student.restore_snapshot(snapshot)
                record[label+"_patch"] = {"file": label+".safetensors",
                    "tensor_hashes": self.env.student.save_patch(directory/(label+".safetensors"))}
        finally:
            self.env.student.restore_snapshot(self._accepted)
        atomic_json(directory/"episode.json", record)
        return record
