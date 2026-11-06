import argparse
import json
import os
import sys
from pathlib import Path

from ..runner import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded coding-model weight editing and editor RL")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download a public model and pin its exact revision")
    download.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    download.add_argument("--revision")
    download.add_argument("--cache", type=Path, default=Path("runs/hf-cache"))
    download.add_argument("--out", type=Path, required=True)
    episode = sub.add_parser("episode", help="Execute JSON actions against actual student weights")
    episode.add_argument("--actions", type=Path, required=True, help="JSON array of validated actions")
    episode.add_argument("--seed", type=int, default=0)
    episode.add_argument("--starting-patch", type=Path)
    episode.add_argument("--out", type=Path, required=True)
    direct = sub.add_parser("direct", help="Run provider-independent direct tensor actions; no optimizer")
    direct.add_argument("--actions", type=Path, required=True, help="JSON array of inspect/probe/edit/rollback/stop actions")
    direct.add_argument("--seed", type=int, default=0)
    direct.add_argument("--out", type=Path, required=True, help="Fresh episode directory")
    direct.add_argument("--horizon", type=int, default=3)
    direct.add_argument("--max-turns", type=int, default=16)
    direct.add_argument("--final", action="store_true", help="Open final tests after decisions end; do not use for training")
    direct.add_argument("--family-partition", choices=("train", "evaluation"), default="train")
    direct.set_defaults(task_suite="algorithmic")
    direct.add_argument("--curriculum", choices=("bridge", "mixed", "stress"), default="bridge")
    pilot = sub.add_parser("pilot", help="Train an editor adapter, compare controls, and repeat an edit")
    pilot.add_argument("--editor-model", type=Path, help="Pinned editor model manifest; defaults to student")
    pilot.add_argument("--train-seeds", default="0,1")
    pilot.add_argument("--eval-seeds", default="20,21")
    pilot.add_argument("--updates", type=int, default=8)
    pilot.add_argument("--group-size", type=int, default=8)
    pilot.add_argument("--editor-seed", type=int, default=0)
    pilot.add_argument("--with-recovery", action="store_true", help="Also reward edit-plus-recovery and recovery-only recipes")
    pilot.add_argument("--out", type=Path, required=True)
    calibration = sub.add_parser("calibrate",help="Measure harder development tasks without editing weights or opening final families")
    calibration.add_argument("--seeds",default="0,1,2,3")
    calibration.add_argument("--out",type=Path,required=True)
    calibration.add_argument("--reuse-generations",type=Path,help="Regrade recorded code only when model, prompts, seeds and task hash match")
    headroom = sub.add_parser("headroom", help="Audit editability and selection headroom using training records only")
    headroom.add_argument("directory", type=Path)
    headroom.add_argument("--out", type=Path, required=True)
    headroom.add_argument("--max-retention-regression", type=float, default=0.)
    headroom.add_argument("--max-optimizer-steps", type=int, default=24)
    headroom.add_argument("--max-edit-evaluations", type=int, default=3)
    for p in (episode, pilot, calibration, direct):
        p.add_argument("--model", type=Path, required=True, help="Downloaded student model manifest")
        p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        p.add_argument("--max-new-tokens", type=int)
    for p in (episode,pilot):
        p.add_argument("--task-suite",choices=("micro","algorithmic"),default="micro")
        p.add_argument("--curriculum",choices=("bridge","mixed","stress"),default="bridge")
    episode.add_argument("--family-partition",choices=("train","evaluation"),default="train")
    args = parser.parse_args(argv)
    if hasattr(args,"max_new_tokens") and args.max_new_tokens is None:
        args.max_new_tokens = 512 if args.command=="calibrate" or args.task_suite=="algorithmic" else 192
    try:
        if args.command == "download":
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            from huggingface_hub import HfApi, snapshot_download
            revision = HfApi().model_info(args.model_id, revision=args.revision, token=False).sha
            path = snapshot_download(args.model_id, revision=revision, token=False,
                cache_dir=str(args.cache.resolve()), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE", "README.md"])
            atomic_json(args.out, {"model_id": args.model_id, "revision": revision, "path": path})
            print(f"Pinned model manifest: {args.out}")
        elif args.command == "headroom":
            from .headroom import audit_pilot
            if args.out.exists():
                raise ValueError("choose a fresh headroom report path")
            report = audit_pilot(args.directory, args.max_retention_regression,
                                args.max_optimizer_steps, args.max_edit_evaluations)
            atomic_json(args.out, report)
            print(json.dumps(report["summary"], indent=2))
        elif args.command == "calibrate":
            from .calibration import calibrate
            from .experiment import load_manifest
            summary = calibrate(load_manifest(args.model),args.out,
                seeds=tuple(int(x) for x in args.seeds.split(",")),device=args.device,max_new_tokens=args.max_new_tokens,
                reuse_generations=args.reuse_generations)
            print(json.dumps(summary["cohorts"],indent=2))
        elif args.command == "direct":
            from .backend import Student
            from .direct_env import DirectConfig, DirectEditEnv
            from .environment import CodingConfig
            from .experiment import load_manifest
            if args.out.exists():
                raise ValueError("choose a fresh direct episode directory")
            actions = json.loads(args.actions.read_text())
            access = DirectConfig(max_turns=args.max_turns)
            if not isinstance(actions, list) or not 1 <= len(actions) <= access.max_turns:
                raise ValueError("actions must be a nonempty array within the turn budget")
            config = CodingConfig(horizon=args.horizon, task_suite="algorithmic",
                curriculum=args.curriculum, family_partition=args.family_partition,
                case_weight=.25, max_optimizer_steps=0, max_edit_evaluations=args.horizon)
            student = Student(load_manifest(args.model), device=args.device, max_new_tokens=args.max_new_tokens)
            env = DirectEditEnv(student, config, access)
            env.reset(seed=args.seed)
            used = 0
            for action in actions:
                _, _, terminated, truncated, _ = env.step(action)
                used += 1
                if terminated or truncated:
                    break
            if not env.done:
                env.step({"kind": "stop"})
            record = env.save(args.out, include_final=args.final)
            print(json.dumps({"total_reward": record["total_reward"], "costs": record["costs"],
                "accepted": record["finalization"]["accepted"], "unused_actions": len(actions)-used,
                "final_tests_opened": args.final}, indent=2))
        elif args.command == "episode":
            from .backend import Student
            from .experiment import load_manifest, run_episode
            from .environment import CodingConfig
            actions = json.loads(args.actions.read_text())
            if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
                raise ValueError("actions must be an array of one to eight edits")
            student = Student(load_manifest(args.model), device=args.device, max_new_tokens=args.max_new_tokens)
            record = run_episode(student, actions, args.seed, args.out, args.starting_patch,
                                 config=CodingConfig(task_suite=args.task_suite,family_partition=args.family_partition,curriculum=args.curriculum))
            print(json.dumps({"reward": record["total_reward"], "final": {
                k: {m: v[m] for m in ("accuracy", "case_accuracy")} for k,v in record["final"].items()}}, indent=2))
        elif args.command == "pilot":
            from .experiment import load_manifest, run_pilot
            summary = run_pilot(load_manifest(args.model), load_manifest(args.editor_model or args.model), args.out,
                train_seeds=tuple(int(x) for x in args.train_seeds.split(",")),
                eval_seeds=tuple(int(x) for x in args.eval_seeds.split(",")), updates=args.updates,
                group_size=args.group_size, device=args.device, max_new_tokens=args.max_new_tokens,
                with_recovery=args.with_recovery,task_suite=args.task_suite,curriculum=args.curriculum,
                editor_seed=args.editor_seed)
            print(json.dumps(summary["verification"], indent=2))
            print(f"Pilot result: {args.out / 'summary.json'}")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
