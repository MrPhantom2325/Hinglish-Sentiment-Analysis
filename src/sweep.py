"""Run the transformer experiments in phases. Resumable: finished runs are skipped.

Phase A  (LR search):  every model x lr in config `training.lr_grid`, seed 42.
Phase B  (final runs): best LR per model by dev macro-F1 (from the Phase A run JSONs),
                       seeds in `training.final_seeds`. Seed 42 is reused from Phase A.
Learning curve (opt.): train fractions x seeds at the best LR. Fraction 1.0 is reused from Phase B.

Each run is a separate `python -m src.train` process, so GPU memory is freed between runs and
one failing run does not stop the rest.

Usage:
    python -m src.sweep --phase A
    python -m src.sweep --phase B
    python -m src.sweep --phase none --learning_curve
    python -m src.sweep --models muril hingbert --dry_run      # print the plan only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import format_lr, load_config, pred_path, resolve_path, run_json_path, run_name, setup_logging

log = logging.getLogger(__name__)

ALL_MODELS = ["mbert", "muril", "hingbert"]


@dataclass(frozen=True)
class Run:
    """One planned fine-tuning run."""

    model: str
    lr: float
    seed: int
    fraction: float = 1.0
    phase: str = "manual"
    max_steps: int = -1

    @property
    def name(self) -> str:
        return run_name(self.model, format_lr(self.lr), self.seed, self.fraction, self.max_steps)

    def done(self, results_dir: Path) -> bool:
        """A run is complete once its prediction file exists."""
        return pred_path(results_dir, self.name).exists()

    def cli_args(self) -> list[str]:
        args = ["--model", self.model, "--lr", str(self.lr), "--seed", str(self.seed),
                "--train_fraction", str(self.fraction), "--phase", self.phase]
        if self.max_steps > 0:
            args += ["--max_steps", str(self.max_steps)]
        return args


# --------------------------------------------------------------------------- planning


def plan_phase_a(models: list[str], cfg: dict[str, Any], max_steps: int = -1) -> list[Run]:
    """Every model x LR at the LR-search seed."""
    seed = int(cfg["training"]["lr_search_seed"])
    return [Run(m, float(lr), seed, 1.0, "lr_search", max_steps)
            for m in models for lr in cfg["training"]["lr_grid"]]


def lr_search_results(results_dir: Path, model: str, cfg: dict[str, Any], max_steps: int = -1) -> dict[float, float]:
    """{lr: dev macro-F1} for the Phase A runs of `model` that have finished."""
    out = {}
    for run in plan_phase_a([model], cfg, max_steps):
        path = run_json_path(results_dir, run.name)
        if run.done(results_dir) and path.exists():
            out[run.lr] = float(json.loads(path.read_text())["dev_macro_f1"])
    return out


def best_lr(results_dir: Path, model: str, cfg: dict[str, Any], max_steps: int = -1) -> float | None:
    """Best LR by dev macro-F1 (ties -> smaller LR); None until every Phase A run has finished."""
    res = lr_search_results(results_dir, model, cfg, max_steps)
    if len(res) < len(cfg["training"]["lr_grid"]):
        return None
    return max(res, key=lambda lr: (res[lr], -lr))


def plan_phase_b(models: list[str], cfg: dict[str, Any], results_dir: Path, max_steps: int = -1) -> list[Run]:
    """Final seeds at each model's best LR (models without a complete Phase A are left out)."""
    runs = []
    for m in models:
        lr = best_lr(results_dir, m, cfg, max_steps)
        if lr is None:
            log.warning("%s: Phase A incomplete, cannot choose an LR yet; skipping Phase B", m)
            continue
        log.info("%s: best LR %s (dev macro-F1 %.4f)", m, format_lr(lr),
                 lr_search_results(results_dir, m, cfg, max_steps)[lr])
        runs += [Run(m, lr, int(s), 1.0, "final", max_steps) for s in cfg["training"]["final_seeds"]]
    return runs


def plan_learning_curve(models: list[str], cfg: dict[str, Any], results_dir: Path,
                        max_steps: int = -1) -> list[Run]:
    """Train fractions x seeds at each model's best LR."""
    lc = cfg["training"]["learning_curve"]
    runs = []
    for m in models:
        lr = best_lr(results_dir, m, cfg, max_steps)
        if lr is None:
            log.warning("%s: Phase A incomplete; skipping learning curve", m)
            continue
        phase = "learning_curve"
        runs += [Run(m, lr, int(s), float(f), phase if f < 1 else "final", max_steps)
                 for f in lc["fractions"] for s in lc["seeds"]]
    return runs


def print_plan(title: str, runs: list[Run], results_dir: Path) -> None:
    """Log which runs will execute and which are skipped."""
    todo = [r for r in runs if not r.done(results_dir)]
    log.info("=== %s: %d runs, %d to run, %d already done (skip)", title, len(runs), len(todo), len(runs) - len(todo))
    for r in runs:
        log.info("   %-5s %s", "skip" if r.done(results_dir) else "RUN", r.name)


# --------------------------------------------------------------------------- execution


def execute(runs: list[Run], results_dir: Path, passthrough: list[str]) -> list[str]:
    """Run each unfinished run as a subprocess; return the names of failed runs."""
    failed = []
    todo = [r for r in runs if not r.done(results_dir)]
    for i, r in enumerate(todo, 1):
        cmd = [sys.executable, "-m", "src.train", *r.cli_args(), *passthrough]
        log.info("[%d/%d] %s", i, len(todo), " ".join(cmd[1:]))
        code = subprocess.run(cmd, check=False).returncode
        if code != 0 or not r.done(results_dir):
            log.error("run failed (exit %d): %s", code, r.name)
            failed.append(r.name)
    return failed


def main() -> None:
    p = argparse.ArgumentParser(description="Resumable LR search + multi-seed final runs.")
    p.add_argument("--phase", choices=["A", "B", "both", "none"], default="both")
    p.add_argument("--learning_curve", action="store_true")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    p.add_argument("--dry_run", action="store_true", help="print the plan, run nothing")
    p.add_argument("--max_steps", type=int, default=-1, help="testing only: cut every run short")
    p.add_argument("--config", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    args = p.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    results_dir = resolve_path(args.out_dir or cfg["paths"]["results_dir"])
    passthrough: list[str] = []
    for flag in ("config", "data_dir", "out_dir", "checkpoint_dir", "epochs", "batch_size"):
        value = getattr(args, flag)
        if value is not None:
            passthrough += [f"--{flag}", str(value)]

    phases = []
    if args.phase in ("A", "both"):
        phases.append(("Phase A (LR search)", lambda: plan_phase_a(args.models, cfg, args.max_steps)))
    if args.phase in ("B", "both"):
        phases.append(("Phase B (final seeds)",
                       lambda: plan_phase_b(args.models, cfg, results_dir, args.max_steps)))
    if args.learning_curve:
        phases.append(("Learning curve",
                       lambda: plan_learning_curve(args.models, cfg, results_dir, args.max_steps)))

    failed: list[str] = []
    for title, plan in phases:
        runs = plan()  # planned lazily: Phase B needs Phase A's results
        print_plan(title, runs, results_dir)
        if not args.dry_run:
            failed += execute(runs, results_dir, passthrough)
    if failed:
        log.error("%d run(s) failed: %s", len(failed), ", ".join(failed))
        sys.exit(1)
    log.info("sweep finished" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
