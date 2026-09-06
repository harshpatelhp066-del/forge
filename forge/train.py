"""Training utilities: learning-rate schedule, checkpointing, evaluation, logging.

The training loop itself lives in ``scripts/train.py``; this module holds the
pieces that are worth testing independently of a long run.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .tensor import no_grad

__all__ = [
    "CosineWarmupSchedule",
    "save_checkpoint",
    "load_checkpoint",
    "estimate_loss",
    "LossLogger",
    "plot_loss_curve",
    "format_duration",
]


class CosineWarmupSchedule:
    """Linear warmup, then cosine decay to ``min_lr``.

    **Why warmup.** Adam's second-moment estimate ``v`` is meaningless for the
    first few dozen steps -- it is an average over a handful of gradients from a
    randomly initialised model, so ``1/√v̂`` is a badly scaled step size in a
    direction that is mostly noise. Warming the learning rate up from ~0 keeps
    those steps small enough not to matter. Skipping warmup on a transformer
    typically shows up as a loss spike in the first ~100 steps that the run never
    fully recovers from.

    **Why cosine.** The decay is smooth and ends at a genuinely small value, so
    late training takes small steps and settles into a minimum rather than
    bouncing around it. A step decay does the same thing but introduces a
    discontinuity that shows up as a visible kink in the loss curve; linear decay
    spends too long at large learning rates.

    ``min_lr`` is a *floor*, not zero: a zero learning rate at the end wastes the
    last steps entirely.
    """

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1):
        if warmup_steps < 0 or total_steps <= 0:
            raise ValueError("warmup_steps must be >= 0 and total_steps > 0")
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})"
            )
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio

    def __call__(self, step: int) -> float:
        """Learning rate at ``step`` (0-based)."""
        if step < self.warmup_steps:
            # +1 so step 0 gets a non-zero rate rather than a wasted step.
            return self.base_lr * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.base_lr - self.min_lr)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(path, model, optimizer=None, step: int = 0,
                    meta: dict | None = None) -> None:
    """Write model (and optionally optimizer) state to a single ``.npz``.

    Parameter names contain dots, which ``np.savez`` accepts as archive member
    names, so the state dict maps over directly under a ``param/`` prefix.
    Config and metadata ride along as a JSON string in a 0-d array -- keeping
    everything in one file means a checkpoint can be loaded without needing the
    original config to be reconstructed by hand.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {f"param/{k}": v for k, v in model.state_dict().items()}
    if optimizer is not None:
        state = optimizer.state_dict()
        for i, a in enumerate(state["m"]):
            arrays[f"opt_m/{i}"] = a
        for i, a in enumerate(state["v"]):
            arrays[f"opt_v/{i}"] = a
        arrays["opt_t"] = np.array(state["t"])

    payload = {"step": step, "config": model.cfg.to_dict() if hasattr(model, "cfg") else {}}
    payload.update(meta or {})
    arrays["meta"] = np.array(json.dumps(payload))

    # Write to a temporary file and rename, so an interrupted save cannot leave a
    # truncated checkpoint where a valid one used to be.  The handle is opened
    # here rather than passing a path: np.savez silently appends ".npz" to a
    # filename that lacks it, which would write "best.npz.tmp.npz" and leave the
    # rename below pointing at a file that does not exist.
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)


def load_checkpoint(path, model=None, optimizer=None) -> dict:
    """Load a checkpoint. Returns the metadata dict.

    ``model`` and ``optimizer`` are populated in place when given.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if model is not None:
            state = {k[len("param/"):]: z[k] for k in z.files if k.startswith("param/")}
            model.load_state_dict(state)
        if optimizer is not None and "opt_t" in z.files:
            n = sum(1 for k in z.files if k.startswith("opt_m/"))
            optimizer.load_state_dict({
                "t": int(z["opt_t"]),
                "m": [z[f"opt_m/{i}"] for i in range(n)],
                "v": [z[f"opt_v/{i}"] for i in range(n)],
            })
    return meta


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def estimate_loss(model, loader, split: str = "val", n_batches: int = 20,
                  seed: int | None = None) -> float:
    """Mean loss over ``n_batches`` batches, in eval mode and without a tape.

    Runs under :class:`~forge.tensor.no_grad` and in ``eval()`` so dropout is off
    -- otherwise the reported validation loss would be measured on a randomly
    thinned network and would read worse than the model actually is.
    """
    was_training = model.training
    model.eval()
    rng_state = loader.rng.bit_generator.state if seed is not None else None
    if seed is not None:
        loader.rng = np.random.default_rng(seed)
    try:
        total = 0.0
        with no_grad():
            for _ in range(n_batches):
                x, y = loader.random_batch(split)
                _, loss = model(x, targets=y)
                total += loss.item()
        return total / n_batches
    finally:
        model.train(was_training)
        if rng_state is not None:
            loader.rng.bit_generator.state = rng_state


# --------------------------------------------------------------------------- #
# Logging and plotting
# --------------------------------------------------------------------------- #

class LossLogger:
    """Appends training metrics to a CSV, flushing every row.

    Flushing on every row is deliberate: a run that dies at step 1800 of 2500
    should still leave a usable loss curve behind.
    """

    FIELDS = ["step", "train_loss", "val_loss", "lr", "grad_norm", "elapsed_s", "tokens"]

    def __init__(self, path, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and resume
        self._fh = self.path.open("a" if exists else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if not exists:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, **row) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.FIELDS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def plot_loss_curve(csv_path, out_path, title: str = "Forge training") -> None:
    """Render the CSV as a two-panel loss/learning-rate figure."""
    import matplotlib
    matplotlib.use("Agg")           # no display on a headless box
    import matplotlib.pyplot as plt

    steps, train, val_steps, val, lrs = [], [], [], [], []
    with Path(csv_path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            s = int(row["step"])
            steps.append(s)
            train.append(float(row["train_loss"]))
            lrs.append(float(row["lr"]))
            if row.get("val_loss"):
                val_steps.append(s)
                val.append(float(row["val_loss"]))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax.plot(steps, train, lw=1.0, color="#4C7BD9", alpha=0.75, label="train")
    if val:
        ax.plot(val_steps, val, lw=1.8, color="#D95F4C", marker="o", ms=3.5,
                label="validation")
        best = int(np.argmin(val))
        ax.scatter([val_steps[best]], [val[best]], s=90, facecolors="none",
                   edgecolors="#B03A28", lw=1.6, zorder=5)
        # When the best value is the last one -- which is what "validation never
        # diverged" looks like -- a right-anchored label would run off the axes.
        near_right = val_steps[best] > 0.75 * max(steps)
        ax.annotate(f"best val {val[best]:.4f} @ step {val_steps[best]}",
                    (val_steps[best], val[best]), textcoords="offset points",
                    xytext=(-12, 20) if near_right else (10, 14),
                    ha="right" if near_right else "left",
                    fontsize=9, color="#B03A28")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False)

    ax2.plot(steps, lrs, lw=1.2, color="#5B9E6B")
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("step")
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.savefig(Path(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
