"""Interactive editing trajectories; this does not train a sequential RL policy."""

import copy
import json
import time
from pathlib import Path

from ..runner import atomic_json, source_hash
from .environment import BudgetExceeded, CodingConfig, CodingEditEnv, episode_problems
from .headroom import paired_changes, acceptance


def run_policy_episode(student, decide, seed, out, *, config=None, starting_patch=None,
                       policy_id="external_callback", max_retention_regression=0., final=False,
                       on_event=None, protect_case_accuracy=False, step_fn=None):
    """Call decide again after each edit, and save the accepted weight checkpoint.

    decide receives a copied observation and returns {action, policy?}. The
    optional policy object is JSON diagnostics, such as sampled probabilities.
    It is reported metadata, never evidence of the directive producer's weights.
    """
    from .experiment import versions
    def emit(kind, **payload):
        if on_event is not None:
            on_event(copy.deepcopy({"kind": kind, **payload}))
    out = Path(out)
    if out.exists() or out.with_suffix(".safetensors").exists():
        raise ValueError("choose a fresh policy episode path")
    # Reject invalid acceptance thresholds before any model work.
    acceptance({"coding": {"gain": 0}, "retention": {"regression_fraction": 0}}, max_retention_regression)
    config = config or CodingConfig(horizon=3, task_suite="algorithmic",
        max_optimizer_steps=24, max_edit_evaluations=3)
    env = CodingEditEnv(student, config)
    emit("starting", policy_id=policy_id, seed=seed)
    observation, _ = env.reset(seed=seed, starting_patch=starting_patch)
    initial = copy.deepcopy(observation)
    starting_id = student.checkpoint_id()
    emit("baseline", record=env.record(), checkpoint=starting_id)
    decisions = []
    stopped_by = "horizon"
    for _ in range(config.horizon):
        before = student.checkpoint_id()
        started = time.monotonic()
        decision = decide(copy.deepcopy(observation))
        seconds = time.monotonic()-started
        if (not isinstance(decision, dict) or "action" not in decision
                or set(decision)-{"action", "policy"}):
            raise ValueError("policy must return action and optional JSON policy diagnostics")
        decision = json.loads(json.dumps(decision, allow_nan=False))
        emit("proposed", decision=decision, target_before=before)
        row = {"observation": observation, "decision": decision, "policy_seconds": seconds,
               "target_before": before}
        try:
            next_observation, reward, terminated, truncated, info = (
                env.step(decision["action"]) if step_fn is None else step_fn(env, decision["action"]))
        except BudgetExceeded as exc:
            row.update(status="budget_rejected", error=str(exc), target_after=student.checkpoint_id())
            decisions.append(row)
            emit("budget_rejected", decision=row)
            stopped_by = "budget_rejected"
            break
        row.update(status="applied", reward=reward, cost=info["cost"],
                   target_after=student.checkpoint_id())
        decisions.append(row)
        emit("evaluated", decision=row, record=env.record())
        observation = next_observation
        if terminated or truncated:
            stopped_by = "stop" if terminated else "horizon"
            break
    candidate_id = student.checkpoint_id()
    env.finalize(max_retention_regression, protect_case_accuracy=protect_case_accuracy)
    accepted_id = student.checkpoint_id()
    emit("finalized", record=env.record(), candidate=candidate_id, accepted=accepted_id)
    record = env.record()
    record.update(protocol="interactive-coding-edit-v1", source_hash=source_hash(), versions=versions(),
        initial_observation=initial, decisions=decisions, stopped_by=stopped_by,
        policy_id=policy_id, editor_track="external_or_scripted", self_editing_verified=False,
        lineage={"episode_start": starting_id, "candidate": candidate_id, "accepted": accepted_id})
    if final:
        emit("terminal_started")
        # The controller has ended. Evaluate both checkpoints on the same terminal
        # problems, and restore the accepted weights even if evaluation fails.
        accepted_state = student.snapshot()
        try:
            student.restore_snapshot(env.starting_state)
            record["baseline_final"] = {key: student.evaluate(episode_problems(config, split, seed))
                for key, split in (("coding", "test"), ("retention", "retention_test"))}
        finally:
            student.restore_snapshot(accepted_state)
        record["final"] = env.final_evaluation()
        record["final_changes"] = paired_changes(record["baseline_final"], record["final"])
    patch = out.with_suffix(".safetensors")
    record["patch"] = {"file": patch.name, "tensor_hashes": student.save_patch(patch)}
    atomic_json(out, record)
    emit("completed", record=record)
    return record
