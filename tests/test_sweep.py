"""Tests for the sweep planner (no training)."""

from __future__ import annotations

import json

from src.sweep import Run, best_lr, plan_learning_curve, plan_phase_a, plan_phase_b
from src.utils import load_config, pred_path, run_json_path


def _fake_run(results_dir, run: Run, dev_f1: float) -> None:
    """Create the two files a finished run leaves behind."""
    p = run_json_path(results_dir, run.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"dev_macro_f1": dev_f1}))
    q = pred_path(results_dir, run.name)
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text("id,split,gold,pred,score_neg,score_neu,score_pos\n")


def test_phase_a_plan():
    cfg = load_config()
    runs = plan_phase_a(["mbert", "muril"], cfg)
    assert len(runs) == 2 * len(cfg["training"]["lr_grid"])
    assert {r.seed for r in runs} == {42}
    assert runs[0].name == "mbert_2e-05_42"


def test_best_lr_waits_for_complete_phase_a_and_breaks_ties_low(tmp_path):
    cfg = load_config()
    a = plan_phase_a(["muril"], cfg)
    _fake_run(tmp_path, a[0], 0.70)
    assert best_lr(tmp_path, "muril", cfg) is None  # 1 of 3 done
    _fake_run(tmp_path, a[1], 0.72)
    _fake_run(tmp_path, a[2], 0.72)
    assert best_lr(tmp_path, "muril", cfg) == a[1].lr  # tie 3e-5 vs 5e-5 -> smaller


def test_phase_b_reuses_seed_42_and_skips_incomplete_models(tmp_path):
    cfg = load_config()
    for i, run in enumerate(plan_phase_a(["hingbert"], cfg)):
        _fake_run(tmp_path, run, [0.6, 0.65, 0.62][i])
    b = plan_phase_b(["hingbert", "mbert"], cfg, tmp_path)  # mbert has no Phase A runs
    assert {r.model for r in b} == {"hingbert"}
    assert [r.seed for r in b] == cfg["training"]["final_seeds"]
    assert b[0].name == "hingbert_3e-05_42" and b[0].done(tmp_path)  # same file as Phase A
    assert not b[1].done(tmp_path)


def test_learning_curve_and_steps_suffix(tmp_path):
    cfg = load_config()
    for run in plan_phase_a(["mbert"], cfg, max_steps=5):
        _fake_run(tmp_path, run, 0.5)
    lc = plan_learning_curve(["mbert"], cfg, tmp_path, max_steps=5)
    lcfg = cfg["training"]["learning_curve"]
    assert len(lc) == len(lcfg["fractions"]) * len(lcfg["seeds"])
    assert all(r.name.endswith("_steps5") for r in lc)
    assert "mbert_2e-05_42_frac0.1_steps5" in {r.name for r in lc}
    # a short run never shares a file name with a full run
    assert Run("mbert", 2e-5, 42, max_steps=30).name != Run("mbert", 2e-5, 42).name
