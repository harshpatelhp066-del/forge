"""Train a Forge GPT on a tokenized corpus.

    python scripts/prepare_data.py --vocab-size 1024
    python scripts/train.py --steps 2500

Everything, the autodiff, the layers, the optimizer, the tokenizer, is Forge.
NumPy is the only numerical dependency; matplotlib is used once at the end to
draw the loss curve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An untrained model emits random bytes, which decode to U+FFFD replacement
# characters, and the default Windows console codepage (cp1252) cannot encode
# those, so printing a sample would crash the run. Force UTF-8 on the streams.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from forge import nn  # noqa: E402
from forge.data import DataLoader  # noqa: E402
from forge.model import GPT, GPTConfig  # noqa: E402
from forge.optim import Adam, clip_grad_norm  # noqa: E402
from forge.tokenizer import load_tokenizer  # noqa: E402
from forge.train import (CosineWarmupSchedule, LossLogger, estimate_loss,  # noqa: E402
                         format_duration, load_checkpoint, plot_loss_curve,
                         save_checkpoint)

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--tokens", type=Path, default=ROOT / "data/tokens_bpe_1024.npy")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer_bpe_1024.json")
    p.add_argument("--val-fraction", type=float, default=0.1)
    # model
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-tie", action="store_true", help="untie input/output embeddings")
    # optimisation
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--beta2", type=float, default=0.99)
    # bookkeeping
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--sample-every", type=int, default=0,
                   help="generate a sample every N steps (0 = only at checkpoints)")
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", type=Path, default=None,
                   help="resume from a checkpoint; --steps stays the TOTAL for the "
                        "schedule, so the LR curve continues rather than restarting")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    nn.set_seed(args.seed)

    # ---- data -------------------------------------------------------------- #
    if not args.tokens.exists():
        print(f"error: {args.tokens} not found. Run scripts/prepare_data.py first.",
              file=sys.stderr)
        return 1
    tokens = np.load(args.tokens)
    tok = load_tokenizer(args.tokenizer)

    start_step = 0
    if args.resume is not None:
        with np.load(args.resume, allow_pickle=False) as z:
            start_step = int(json.loads(str(z["meta"]))["step"])
        print(f"resuming from {args.resume.name} at step {start_step}")

    # Offset the loader's seed when resuming so the second leg does not replay the
    # exact batch sequence the first leg already trained on.
    loader = DataLoader(tokens, block_size=args.block_size, batch_size=args.batch_size,
                        val_fraction=args.val_fraction, seed=args.seed + start_step)
    print(loader.summary())

    # ---- model ------------------------------------------------------------- #
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, d_model=args.d_model, dropout=args.dropout,
        tie_embeddings=not args.no_tie,
    )
    model = GPT(cfg)
    print(model.parameter_summary())

    opt = Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2),
               weight_decay=args.weight_decay)
    schedule = CosineWarmupSchedule(args.lr, args.warmup, args.steps, args.min_lr_ratio)

    print(f"\ntokens/step {args.batch_size * args.block_size:,}   "
          f"total {args.steps * args.batch_size * args.block_size / 1e6:.1f}M   "
          f"({args.steps * args.batch_size * args.block_size / len(loader.train_tokens):.1f} "
          f"epochs over the training split)")
    print(f"baseline loss for a uniform model: ln({tok.vocab_size}) = "
          f"{np.log(tok.vocab_size):.4f}\n")

    csv_path = args.out / "loss_curve.csv"
    prompt = "\n"
    best_val = float("inf")
    elapsed_offset = 0.0
    sample_log: list[dict] = []

    if args.resume is not None:
        load_checkpoint(args.resume, model, opt)
        # Recover the best validation loss and elapsed time already banked, so a
        # resumed run does not overwrite a better checkpoint or report a wall
        # clock covering only its own leg.
        if csv_path.exists():
            import csv as _csv
            with csv_path.open(encoding="utf-8") as fh:
                for row in _csv.DictReader(fh):
                    if int(row["step"]) > start_step:
                        continue
                    if row.get("val_loss"):
                        best_val = min(best_val, float(row["val_loss"]))
                    if row.get("elapsed_s"):
                        elapsed_offset = max(elapsed_offset, float(row["elapsed_s"]))
            print(f"  best val so far {best_val:.4f}, "
                  f"{format_duration(elapsed_offset)} already spent")

    start = time.time() - elapsed_offset

    with LossLogger(csv_path, resume=args.resume is not None) as logger:
        model.train()
        for step in range(start_step, args.steps):
            lr = schedule(step)
            opt.lr = lr

            x, y = loader.random_batch("train")
            opt.zero_grad()
            _, loss = model(x, targets=y)
            loss.backward()

            grad_norm = clip_grad_norm(model.parameters(), args.grad_clip)
            if not np.isfinite(grad_norm):
                # Do not step on a poisoned gradient: one inf would turn every
                # parameter it touches into NaN and the run would never recover.
                print(f"step {step}: non-finite gradient norm, skipping update")
                continue
            opt.step()

            train_loss = loss.item()
            val_loss = ""
            if step % args.eval_every == 0 or step == args.steps - 1:
                val_loss = estimate_loss(model, loader, "val", args.eval_batches, seed=99)
                elapsed = time.time() - start
                done = step + 1 - start_step
                eta = (time.time() - start - elapsed_offset) / done * (args.steps - step - 1)
                print(f"step {step:5d}/{args.steps}  train {train_loss:.4f}  "
                      f"val {val_loss:.4f}  lr {lr:.2e}  |g| {grad_norm:6.3f}  "
                      f"{format_duration(elapsed)} elapsed, ~{format_duration(eta)} left")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(args.out / "best.npz", model, opt, step,
                                    {"val_loss": val_loss, "train_loss": train_loss})

            logger.log(step=step, train_loss=f"{train_loss:.6f}",
                       val_loss=f"{val_loss:.6f}" if val_loss != "" else "",
                       lr=f"{lr:.8f}", grad_norm=f"{grad_norm:.6f}",
                       elapsed_s=f"{time.time() - start:.1f}",
                       tokens=(step + 1) * args.batch_size * args.block_size)

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(args.out / f"step_{step:06d}.npz", model, opt, step,
                                {"train_loss": train_loss})
                text = tok.decode(model.generate(
                    tok.encode(prompt) or [0], max_new_tokens=200, temperature=0.8,
                    top_k=40, rng=np.random.default_rng(args.seed))[0])
                sample_log.append({"step": step, "train_loss": train_loss,
                                   "val_loss": val_loss if val_loss != "" else None,
                                   "text": text})
                print(f"  --- sample @ step {step} " + "-" * 40)
                print("  " + text.replace("\n", "\n  ")[:600])
                print("  " + "-" * 58)

    elapsed = time.time() - start
    final_train = estimate_loss(model, loader, "train", args.eval_batches, seed=7)
    final_val = estimate_loss(model, loader, "val", args.eval_batches, seed=99)
    save_checkpoint(args.out / "final.npz", model, opt, args.steps,
                    {"train_loss": final_train, "val_loss": final_val})

    print(f"\ntrained {args.steps} steps in {format_duration(elapsed)} "
          f"({elapsed / args.steps:.2f}s/step)")
    print(f"final  train {final_train:.4f}   val {final_val:.4f}   best val {best_val:.4f}")

    plot_loss_curve(csv_path, args.out / "loss_curve.png",
                    title=f"Forge GPT — {model.num_parameters()/1e6:.2f}M params, "
                          f"{cfg.n_layer}L/{cfg.n_head}H/{cfg.d_model}D, "
                          f"tiny-shakespeare")
    print(f"wrote {args.out / 'loss_curve.png'}")

    (args.out / "samples_during_training.json").write_text(
        json.dumps(sample_log, indent=2), encoding="utf-8")
    (args.out / "run_summary.json").write_text(json.dumps({
        "config": cfg.to_dict(),
        "parameters": model.num_parameters(),
        "parameters_non_embedding": model.num_parameters(non_embedding=True),
        "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "warmup": args.warmup, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "dropout": args.dropout,
        "final_train_loss": final_train, "final_val_loss": final_val,
        "best_val_loss": best_val, "uniform_baseline": float(np.log(tok.vocab_size)),
        "elapsed_seconds": elapsed, "seconds_per_step": elapsed / args.steps,
        "train_tokens": int(len(loader.train_tokens)),
        "val_tokens": int(len(loader.val_tokens)),
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
