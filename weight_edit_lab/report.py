import json
from pathlib import Path

from .tasks import PROTOCOL


def render_report(summaries: list[dict]) -> str:
    if not summaries or any(s.get("manifest", {}).get("protocol") != PROTOCOL for s in summaries):
        raise ValueError("report requires summaries from this protocol")
    splits = [s["manifest"]["split"] for s in summaries]
    if len(set(splits)) != len(splits):
        raise ValueError("report accepts one summary per split")
    encoded = json.dumps(summaries, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return Path(__file__).with_name("report.html").read_text().replace("__SUMMARIES_JSON__", encoded)
