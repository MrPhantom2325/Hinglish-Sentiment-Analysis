"""Fine-tune one (model, lr, seed[, train fraction]) run and write the standard outputs.

Writes results/metrics/run_{name}.json and then results/preds/preds_{name}.csv (dev + test).
If the prediction file already exists, the run is skipped, so re-running is always safe.

Usage:
    python -m src.train --model mbert --lr 2e-5 --seed 42
    python -m src.train --model mbert --train_fraction 0.02 --max_steps 20 --batch_size 8   # Mac smoke test
"""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.preprocess import clean_for_transformer
from src.utils import (
    compute_metrics,
    format_lr,
    get_device,
    label_map,
    load_config,
    now_iso,
    pred_path,
    predictions_frame,
    resolve_path,
    run_json_path,
    run_name,
    set_seed,
    setup_logging,
    write_json,
    write_predictions,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data


class TweetDataset(torch.utils.data.Dataset):
    """Tokenized tweets (no padding; DataCollatorWithPadding pads each batch)."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


def stratified_subsample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Stratified (by label) subsample of `df`; fraction 1.0 returns `df` unchanged."""
    if fraction >= 1.0:
        return df
    sub, _ = train_test_split(df, train_size=fraction, stratify=df["label"], random_state=seed)
    return sub.sort_index()


def load_data(data_dir: Path, labels: dict[str, int]) -> dict[str, pd.DataFrame]:
    """Processed splits with cleaned text and integer labels."""
    dfs = {}
    for split in ("train", "dev", "test"):
        df = pd.read_csv(data_dir / f"{split}.csv", dtype={"id": str})
        df["clean"] = df["text"].map(clean_for_transformer)
        df["y"] = df["label"].str.lower().map(labels).astype(int)
        dfs[split] = df
    return dfs


# --------------------------------------------------------------------------- metrics


def metrics_from_logits(eval_pred: Any) -> dict[str, float]:
    """Trainer `compute_metrics`: macro-F1 and accuracy from logits."""
    logits, labels = eval_pred
    return compute_metrics(labels, np.asarray(logits).argmax(-1))


# --------------------------------------------------------------------------- best-epoch selection


class KeepBestInMemory(TrainerCallback):
    """Keep a CPU copy of the weights from the evaluation with the highest dev macro-F1.

    Used instead of Trainer's `load_best_model_at_end`: in transformers 5.x the Trainer saves
    mBERT's LayerNorm weights under their legacy names (gamma/beta) and then does not load them
    back, so the "best" model would silently mix weights from two epochs. Restoring an
    in-memory copy with `strict=True` avoids that for every model.
    """

    def __init__(self, metric: str = "eval_macro_f1") -> None:
        self.metric = metric
        self.best_score = -float("inf")
        self.best_epoch: float | None = None
        self.best_step: int | None = None
        self.best_state: dict[str, torch.Tensor] | None = None

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict | None = None,
                    model: Any = None, **kwargs: Any) -> None:
        score = (metrics or {}).get(self.metric)
        if score is None or model is None or score <= self.best_score:
            return
        self.best_score, self.best_epoch, self.best_step = score, state.epoch, state.global_step
        self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    def restore(self, model: torch.nn.Module) -> None:
        """Load the best weights back into `model` (all keys must match)."""
        assert self.best_state is not None, "no evaluation ran, nothing to restore"
        model.load_state_dict(self.best_state, strict=True)


# --------------------------------------------------------------------------- run


def build_parser() -> argparse.ArgumentParser:
    """CLI for one fine-tuning run."""
    p = argparse.ArgumentParser(description="Fine-tune one transformer run.")
    p.add_argument("--model", required=True, choices=["mbert", "muril", "hingbert"])
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=None, help="default: config training.epochs")
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--data_dir", default=None, help="processed data dir (default: config)")
    p.add_argument("--out_dir", default=None, help="results dir (default: config)")
    p.add_argument("--checkpoint_dir", default=None, help="temporary checkpoints (default: config)")
    p.add_argument("--max_steps", type=int, default=-1, help=">0: stop early (smoke test); file gets _steps{n}")
    p.add_argument("--keep_checkpoint", action="store_true", help="keep the best model instead of deleting it")
    p.add_argument("--phase", default="manual", help="label stored in the run JSON (set by sweep.py)")
    p.add_argument("--config", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    max_length = args.max_length or tcfg["max_length"]
    batch_size = args.batch_size or tcfg["batch_size"]
    grad_accum = args.grad_accum or tcfg["grad_accum"]
    data_dir = resolve_path(args.data_dir or cfg["paths"]["processed_dir"])
    results_dir = resolve_path(args.out_dir or cfg["paths"]["results_dir"])

    name = run_name(args.model, format_lr(args.lr), args.seed, args.train_fraction, args.max_steps)
    out_pred = pred_path(results_dir, name)
    if out_pred.exists():
        log.info("skip: %s already exists", out_pred)
        return

    device = get_device()
    on_cuda = device == "cuda"
    model_id = cfg["models"][args.model]
    labels = label_map(cfg)
    id2label = {v: k for k, v in labels.items()}
    log.info("run %s | model=%s | device=%s", name, model_id, device)

    set_seed(args.seed)
    transformers.set_seed(args.seed)

    dfs = load_data(data_dir, labels)
    train_df = stratified_subsample(dfs["train"], args.train_fraction, tcfg["subsample_seed"])
    log.info("train=%d dev=%d test=%d", len(train_df), len(dfs["dev"]), len(dfs["test"]))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ds = {
        "train": TweetDataset(train_df["clean"].tolist(), train_df["y"].tolist(), tokenizer, max_length),
        "dev": TweetDataset(dfs["dev"]["clean"].tolist(), dfs["dev"]["y"].tolist(), tokenizer, max_length),
        "test": TweetDataset(dfs["test"]["clean"].tolist(), dfs["test"]["y"].tolist(), tokenizer, max_length),
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, id2label=id2label, label2id=labels
    )

    ckpt_dir = resolve_path(args.checkpoint_dir or tcfg["checkpoint_dir"]) / name
    # Evaluate on dev every epoch; KeepBestInMemory keeps the best epoch's weights.
    # Epoch-based evaluation never fires when --max_steps stops training inside the first
    # epoch, so short runs evaluate once at the last step instead.
    strategy = "steps" if args.max_steps > 0 else "epoch"
    keep_best = KeepBestInMemory()
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=args.lr,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=tcfg["warmup_ratio"],  # a float < 1 is a ratio of total steps (transformers 5.x)
        weight_decay=tcfg["weight_decay"],
        eval_strategy=strategy,
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_strategy="no",  # best weights are kept in memory (see KeepBestInMemory)
        logging_strategy="steps",
        logging_steps=50,
        fp16=on_cuda,
        dataloader_num_workers=2 if on_cuda else 0,
        dataloader_pin_memory=on_cuda,
        use_cpu=device == "cpu",
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics_from_logits,
        callbacks=[keep_best],
    )

    t0 = time.perf_counter()
    train_out = trainer.train()
    train_time = time.perf_counter() - t0
    log.info("trained in %.1fs on %s (trainer device: %s)", train_time, device, trainer.args.device)
    keep_best.restore(trainer.model)
    log.info("restored best weights: epoch %.2f, step %d, dev macro-F1 %.4f",
             keep_best.best_epoch, keep_best.best_step, keep_best.best_score)

    frames, metrics = [], {}
    for split in ("dev", "test"):
        out = trainer.predict(ds[split])
        logits = np.asarray(out.predictions)
        frames.append(predictions_frame(dfs[split]["id"], split, dfs[split]["y"], logits))
        for k, v in compute_metrics(dfs[split]["y"], logits.argmax(1)).items():
            metrics[f"{split}_{k}"] = v
    log.info("dev macro-F1=%.4f acc=%.4f | test macro-F1=%.4f acc=%.4f",
             metrics["dev_macro_f1"], metrics["dev_accuracy"], metrics["test_macro_f1"], metrics["test_accuracy"])

    eval_history = [
        {k: h[k] for k in ("epoch", "step", "eval_macro_f1", "eval_accuracy", "eval_loss") if k in h}
        for h in trainer.state.log_history
        if "eval_macro_f1" in h
    ]
    run = {
        "model": args.model,
        "model_id": model_id,
        "lr": format_lr(args.lr),
        "seed": args.seed,
        "fraction": args.train_fraction,
        "max_steps": args.max_steps,
        "phase": args.phase,
        "config": {
            "learning_rate": args.lr,
            "epochs": epochs,
            "max_length": max_length,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "warmup_ratio": tcfg["warmup_ratio"],
            "weight_decay": tcfg["weight_decay"],
            "fp16": on_cuda,
            "n_train": len(train_df),
            "subsample_seed": tcfg["subsample_seed"] if args.train_fraction < 1 else None,
        },
        **metrics,
        "best_epoch": keep_best.best_epoch,
        "best_step": keep_best.best_step,
        "best_dev_macro_f1_during_training": keep_best.best_score,
        "eval_history": eval_history,
        "global_steps": train_out.global_step,
        "train_time_sec": round(train_time, 1),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if on_cuda else None,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "timestamp": now_iso(),
    }
    # JSON first, predictions last: the prediction file marks a run as complete.
    write_json(run, run_json_path(results_dir, name))
    write_predictions(out_pred, frames)
    log.info("wrote %s", out_pred)

    if args.keep_checkpoint:
        best_dir = ckpt_dir / "best"
        trainer.save_model(str(best_dir))
        log.info("kept best model at %s", best_dir)
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        log.info("deleted checkpoint dir %s", ckpt_dir)


if __name__ == "__main__":
    main()
