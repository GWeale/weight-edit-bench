"""Build the standalone interactive website from a recorded or active run."""

import argparse
import copy
import json
from pathlib import Path


def render(snapshot):
    if snapshot.get("schema") != "weight-edit-explorer-v1":
        raise ValueError("expected an explorer snapshot")
    template = Path(__file__).resolve().parents[1]/"web"/"run-explorer"/"index.html"
    # Exclude duplicated controller observations, runtime metadata and patch
    # paths. Keep measured data needed for every replay event, not raw logs.
    view = copy.deepcopy(snapshot)
    record_fields = {"baseline_development", "development", "history", "total_reward", "budget_used",
                     "finalization", "final", "baseline_final", "lineage", "candidate_development",
                     "candidate_final", "access_cost", "control"}
    for event in view["events"]:
        if "record" in event:
            event["record"] = {k: v for k, v in event["record"].items() if k in record_fields}
            record = event["record"]
            record["history"] = record.get("history", [])[-1:]
            for stage in ("baseline_development", "development", "final", "baseline_final", "candidate_development", "candidate_final"):
                for evaluation in record.get(stage, {}).values():
                    for row in evaluation.get("rows", []):
                        if "evaluated_code" in row:
                            row.pop("code", None)
        if isinstance(event.get("decision"), dict):
            event["decision"].pop("observation", None)
    payload = json.dumps(view, separators=(",", ":"), ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return template.read_text().replace("__RUN_DATA__", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(json.loads(args.snapshot.read_text())))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
