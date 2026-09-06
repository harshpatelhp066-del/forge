"""Batching a token stream into (input, target) pairs for next-token prediction.

The task is: given tokens ``x[0..T-1]``, predict ``x[1..T]``. So a training
example is a window of the corpus and its own one-position shift, every
position in the window contributes a prediction, which is why a decoder-only
transformer gets `T` training signals per sequence rather than one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["DataLoader", "load_corpus"]


def load_corpus(path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"corpus at {path} is empty")
    return text


class DataLoader:
    """Splits a token array into train/val and serves ``(x, y)`` batches.

    Parameters
    ----------
    tokens:
        1-D integer array of the whole corpus.
    block_size:
        Context length ``T``. Each example is ``T`` inputs and ``T`` targets.
    batch_size:
        Number of sequences per batch.
    val_fraction:
        Fraction of the corpus held out for validation.
    seed:
        Seeds the shuffling RNG, so a run is reproducible.
    """

    def __init__(self, tokens, block_size: int, batch_size: int,
                 val_fraction: float = 0.1, seed: int = 1337):
        tokens = np.asarray(tokens)
        if tokens.ndim != 1:
            raise ValueError(f"expected a 1-D token array, got shape {tokens.shape}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # A *contiguous* split, not a random one. Sampling held-out windows from
        # throughout the corpus would let a validation window overlap a training
        # window by up to block_size-1 tokens, so validation loss would partly
        # measure memorisation of text the model had already been trained on.
        # Splitting at a single cut point makes the two sets genuinely disjoint.
        n_val = int(len(tokens) * val_fraction)
        self.train_tokens = tokens[: len(tokens) - n_val]
        self.val_tokens = tokens[len(tokens) - n_val:]

        for name, arr in (("train", self.train_tokens), ("val", self.val_tokens)):
            if len(arr) < block_size + 1:
                raise ValueError(
                    f"{name} split has {len(arr)} tokens, which is fewer than "
                    f"block_size + 1 = {block_size + 1}; use a shorter context, "
                    f"a smaller val_fraction, or more data"
                )

    # ------------------------------------------------------------------ #

    def _split(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train_tokens
        if split in ("val", "valid", "validation"):
            return self.val_tokens
        raise ValueError(f"unknown split {split!r}; expected 'train' or 'val'")

    def n_windows(self, split: str = "train") -> int:
        """Number of distinct starting offsets available in a split."""
        return len(self._split(split)) - self.block_size

    def _gather(self, data: np.ndarray, offsets: np.ndarray):
        """Vectorised window gather: offsets -> (B, T) inputs and (B, T) targets."""
        idx = offsets[:, None] + np.arange(self.block_size + 1)[None, :]
        windows = data[idx]                       # (B, T+1)
        return windows[:, :-1], windows[:, 1:]    # x, y, y is x shifted by one

    def random_batch(self, split: str = "train", batch_size: int | None = None):
        """Sample a batch of windows uniformly at random, with replacement.

        This is the standard sampler for language-model training: with far more
        distinct windows than training steps, sampling with replacement is
        indistinguishable from an epoch schedule and needs no bookkeeping. The
        epoch iterator below is used where exact coverage matters (validation).
        """
        data = self._split(split)
        B = batch_size or self.batch_size
        offsets = self.rng.integers(0, len(data) - self.block_size, size=B)
        return self._gather(data, offsets)

    def epoch(self, split: str = "train", batch_size: int | None = None,
              stride: int | None = None, shuffle: bool = True, drop_last: bool = True):
        """Iterate over the split once, in shuffled order.

        ``stride`` defaults to ``block_size``, giving non-overlapping windows so
        that one pass sees each token exactly once. The window *order* is
        shuffled, which is what stops consecutive batches from being consecutive
        text, correlated batches make the gradient estimate correlated too, and
        Adam's second-moment estimate then tracks a moving target.
        """
        data = self._split(split)
        B = batch_size or self.batch_size
        stride = stride or self.block_size

        starts = np.arange(0, len(data) - self.block_size, stride)
        if shuffle:
            self.rng.shuffle(starts)

        for i in range(0, len(starts), B):
            chunk = starts[i:i + B]
            if drop_last and len(chunk) < B:
                return
            yield self._gather(data, chunk)

    def n_batches(self, split: str = "train", batch_size: int | None = None,
                  stride: int | None = None) -> int:
        B = batch_size or self.batch_size
        stride = stride or self.block_size
        return len(np.arange(0, len(self._split(split)) - self.block_size, stride)) // B

    def summary(self) -> str:
        return (f"train tokens {len(self.train_tokens):,}  "
                f"val tokens {len(self.val_tokens):,}  "
                f"block_size {self.block_size}  batch_size {self.batch_size}  "
                f"train windows {self.n_windows('train'):,}")
