from __future__ import annotations

import json
from pathlib import Path

from .environment import Config, WeightEditEnv
from .tasks import PROTOCOL


def verify_episode(record: dict) -> None:
    if record.get("protocol") != PROTOCOL or record.get("status", "complete") != "complete":
        raise ValueError("only completed episodes from this protocol can be replayed")
    env = WeightEditEnv(Config(**record["config"]), record["mode"])
    _, info = env.reset(seed=record["seed"], split=record["split"])
    if info["task_hash"] != record["task_hash"]:
        raise ValueError("task fingerprint differs from this runtime")
    for frame in record["frames"][1:]:
        env.step(frame["action"])
    expected = env.record()
    # JSON canonicalization makes tuples in generated tasks comparable to loaded lists.
    for key in ("frames", "task", "result"):
        if json.dumps(expected[key], sort_keys=True) != json.dumps(record[key], sort_keys=True):
            raise ValueError(f"replay mismatch in {key}")
    if "checkpoint" in record and env.checkpoint() != record["checkpoint"]:
        raise ValueError("replay mismatch in checkpoint")


def render_episode(record: dict) -> str:
    verify_episode(record)
    data = json.dumps(record, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return Path(__file__).with_name("replay.html").read_text().replace("__EPISODE_JSON__", data)
