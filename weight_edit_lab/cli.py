from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import Baseline
from .environment import Config
from .replay import render_episode, verify_episode
from .report import render_report
from .runner import atomic_json, evaluate, run_episode, source_hash, tune
from .training import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edit actual neural-network weights and measure adaptation.")
    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train", help="Train the starter step-size policy with REINFORCE on train tasks")
    train_p.add_argument("--episodes", type=int, default=300)
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--out", type=Path, required=True)
    tune_p = sub.add_parser("tune", help="Select SGD/Adam rates on validation tasks only")
    tune_p.add_argument("--seeds", type=int, default=12)
    tune_p.add_argument("--out", type=Path, required=True)
    eval_p = sub.add_parser("evaluate", help="Run matched adaptation/retention comparisons, with safe resume")
    eval_p.add_argument("--seeds", type=int, default=20)
    eval_p.add_argument("--seed-start", type=int, default=0)
    eval_p.add_argument("--split", choices=("train", "validation", "test", "ood"), default="test")
    eval_p.add_argument("--policy", type=Path)
    eval_p.add_argument("--tuning", type=Path)
    eval_p.add_argument("--out", type=Path, required=True)
    episode_p = sub.add_parser("episode", help="Run one baseline episode and save its edited-network checkpoint")
    episode_p.add_argument("--agent", choices=("noop", "random", "sgd", "adam"), default="adam")
    episode_p.add_argument("--rate", type=float, default=0.03)
    episode_p.add_argument("--seed", type=int, default=0)
    episode_p.add_argument("--mode", choices=("adapt", "retain"), default="adapt")
    episode_p.add_argument("--split", choices=("train", "validation", "test", "ood"), default="test")
    episode_p.add_argument("--out", type=Path, required=True)
    replay_p = sub.add_parser("replay", help="Verify every numerical edit and export an offline interactive replay")
    replay_p.add_argument("episode", type=Path)
    replay_p.add_argument("--out", type=Path, required=True)
    verify_p = sub.add_parser("verify", help="Re-execute logged actions and compare every frame and final result")
    verify_p.add_argument("episode", type=Path)
    report_p = sub.add_parser("report", help="Render matched summary results as an offline comparison chart")
    report_p.add_argument("summaries", nargs="+", type=Path)
    report_p.add_argument("--out", type=Path, required=True)
    for p in (train_p, tune_p, eval_p, episode_p):
        p.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        config = Config(horizon=args.horizon) if hasattr(args, "horizon") else None
        if args.command == "train":
            document, history = train(args.episodes, args.seed, config)
            document["training"]["source_hash"] = source_hash()
            atomic_json(args.out, document)
            atomic_json(args.out.with_suffix(".history.json"), history)
            print(f"Trained on {args.episodes} train episodes. Policy: {args.out}")
        elif args.command == "tune":
            document = tune(args.seeds, config)
            atomic_json(args.out, document)
            print(json.dumps(document["rates"], indent=2))
        elif args.command == "evaluate":
            if args.seeds < 1 or args.seed_start < 0:
                raise ValueError("seeds must be positive and seed-start nonnegative")
            policy = json.loads(args.policy.read_text()) if args.policy else None
            tuning = json.loads(args.tuning.read_text()) if args.tuning else None
            summary = evaluate(args.out, list(range(args.seed_start, args.seed_start + args.seeds)),
                               args.split, config, policy, tuning)
            for row in summary["rows"]:
                mse = row["metrics"]["final_test_mse"]["mean"]
                metric = f"{mse:.6f}" if mse is not None else "unavailable"
                print(f"{row['mode']:6} {row['agent']:12} test MSE {metric}  "
                      f"complete {row['completed']}/{row['planned']} errors {row['errors']}")
            print(f"Summary: {args.out / 'summary.json'}")
            if any(r["errors"] for r in summary["rows"]):
                return 1
        elif args.command == "episode":
            record = run_episode(Baseline(args.agent, args.rate, args.seed), args.seed, args.mode, args.split, config)
            record.update(agent=args.agent, status="complete", source_hash=source_hash())
            atomic_json(args.out, record)
            atomic_json(args.out.with_suffix(".network.json"), record["checkpoint"])
            print(json.dumps(record["result"], indent=2))
        elif args.command == "verify":
            verify_episode(json.loads(args.episode.read_text()))
            print("Replay verified: every edit, frame, checkpoint, and final metric matches.")
        elif args.command == "replay":
            html = render_episode(json.loads(args.episode.read_text()))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(html)
            print(f"Verified offline replay: {args.out}")
        elif args.command == "report":
            html = render_report([json.loads(p.read_text()) for p in args.summaries])
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(html)
            print(f"Offline comparison chart: {args.out}")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
