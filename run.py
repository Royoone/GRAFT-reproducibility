"""Small command-line entry point for the fixed GRAFT manuscript experiments."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIVE = ("A375", "A549", "HT29", "MCF7", "PC3")
EXTERNAL = ("HepG2", "MCF10A", "Jurkat", "K562_L1000", "HAP1")
SEEDS = (52, 229, 564)


def execute(command):
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / "results/cache/matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / "results/cache"))
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


def module(name, *args):
    execute([sys.executable, "-m", name, *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("verify", help="Check packaged file hashes and fixed paper contracts")
    fig = subs.add_parser("figures", help="Redraw paper panels from their frozen source data")
    fig.add_argument("--only", nargs="+", help="Task IDs listed in docs/figures.md")
    bench = subs.add_parser("benchmark", help="Recompute GRAFT predictions using packaged checkpoints")
    bench.add_argument("--target", choices=(*FIVE, *EXTERNAL), nargs="+", default=[*FIVE, *EXTERNAL])
    bench.add_argument("--seed", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--output", type=Path, default=Path("results/benchmark_predictions"))
    train = subs.add_parser("train", help="Train a fixed paper experiment from source responses")
    train.add_argument("--experiment", choices=["internal", "external", "adaptation", "ipsc"], required=True)
    train.add_argument("--target", choices=FIVE, help="Required for internal/adaptation source folds")
    train.add_argument("--seed", type=int, choices=SEEDS, default=52)
    train.add_argument("--device", default="cpu")
    train.add_argument("--output", type=Path, default=Path("results/backbone_training"))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "verify":
        module("tools.verify")
    elif args.command == "figures":
        tasks = json.loads((ROOT / "figures/figure_tasks.json").read_text())["tasks"]
        known = {t["id"] for t in tasks}
        if args.only and set(args.only) - known:
            parser.error("Unknown figure tasks: " + ", ".join(sorted(set(args.only) - known)))
        for task in tasks:
            if task["id"] not in (args.only or known):
                continue
            execute([x.replace("{python}", sys.executable) for x in task["command"]])
            absent = [p for p in task["outputs"] if not (ROOT / p).is_file()]
            if absent:
                raise RuntimeError("Missing expected figure outputs: " + ", ".join(absent))
        print("Requested figure tasks completed. Outputs: results/figures/", flush=True)
    elif args.command == "benchmark":
        for target in args.target:
            for seed in args.seed:
                module("experiments.backbone.predict", "--target", target, "--seed", seed,
                       "--device", args.device, "--output", args.output)
    elif args.command == "train":
        if args.experiment in ("internal", "adaptation") and not args.target:
            parser.error("--target is required for this experiment")
        if args.experiment in ("external", "ipsc") and args.target:
            parser.error("This experiment uses all five source contexts; omit --target")
        target = args.target or "external"
        mapping, initialization = {
            "internal": ("cross_moment_residual", "direct"),
            "external": ("saturating", "direct"),
            "adaptation": ("saturating", "pruned"),
            "ipsc": ("cross_moment_residual", "direct"),
        }[args.experiment]
        command = ["--data", "data/processed/sources", "--output", args.output / args.experiment,
                   "--target", target, "--seed", args.seed, "--mapping", mapping,
                   "--initialization", initialization, "--steps", 200, "--device", args.device]
        if args.experiment == "internal":
            command.append("--evaluate")
        module("experiments.backbone.train", *command)


if __name__ == "__main__":
    main()
