"""A GPT-style decoder-only Transformer, assembled from Phase 2 primitives.

Architecture (pre-norm, as in GPT-2 and every transformer since):

    tokens ──► token embedding ──┐
                                 ├──► + ──► dropout ──► [ Block ] x N ──► LayerNorm ──► logits
    positions ──► pos embedding ─┘                                                        ▲
                                                                     tied to token embedding

    Block(x):  x = x + Attention(LayerNorm(x))
               x = x + MLP(LayerNorm(x))

Pre-norm rather than post-norm is the one structural choice here that decides
whether the model trains at all at this depth; the reasoning is in
PHASE_3_NOTES.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from . import nn
from .tensor import Tensor, no_grad

__all__ = ["GPTConfig", "MLP", "Block", "GPT"]


@dataclass
class GPTConfig:
    """Model geometry. Depth, width, heads and context are all free parameters."""

    vocab_size: int = 65
    block_size: int = 128          # context length
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int | None = None        # defaults to 4 * d_model, the standard ratio
    dropout: float = 0.1
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_head={self.n_head}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


class MLP(nn.Module):
    """Position-wise feedforward: expand by 4x, GELU, project back.

    The expansion carries most of the model's parameters (8·d² per block against
    attention's 4·d²) and is where the bulk of the per-token computation happens.
    """

    def __init__(self, cfg: GPTConfig, proj_init_std: float):
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, init_std=cfg.init_std)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, init_std=proj_init_std)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(nn.gelu(self.fc(x))))


class Block(nn.Module):
    """One pre-norm transformer block.

    Both sub-layers are wrapped as ``x + f(LayerNorm(x))``. The residual branch
    is an identity path from input to output, so gradient reaches the earliest
    block undiminished regardless of depth, the LayerNorm sits *inside* the
    branch, not astride the shortcut.
    """

    def __init__(self, cfg: GPTConfig, proj_init_std: float):
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiHeadSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_head,
            dropout=cfg.dropout,
            max_seq_len=cfg.block_size,
            init_std=cfg.init_std,
            proj_init_std=proj_init_std,
        )
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg, proj_init_std)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, cfg: GPTConfig):
        self.cfg = cfg

        # Residual-branch projections are initialised at std/sqrt(2*n_layer).
        # Each block adds two branches into the residual stream, so without this
        # the stream's variance grows like 2*n_layer and the activations at the
        # final layer are already large before a single step is taken.
        proj_init_std = cfg.init_std / math.sqrt(2 * cfg.n_layer)

        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model, init_std=cfg.init_std)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model, init_std=cfg.init_std)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [Block(cfg, proj_init_std) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(cfg.d_model)

        if cfg.tie_embeddings:
            # Weight tying: the output projection *is* the token embedding,
            # transposed. Not a separate parameter, gradient from both uses
            # accumulates into the one tensor, and named_parameters()
            # de-duplicates so the optimizer steps it once.
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size,
                                     bias=False, init_std=cfg.init_std)

    # ------------------------------------------------------------------ #

    def forward(self, idx, targets=None):
        """``idx`` is ``(B, T)`` integer token ids.

        Returns ``logits`` of shape ``(B, T, vocab)``; if ``targets`` is given,
        returns ``(logits, loss)`` with the mean next-token cross-entropy.
        """
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx).astype(np.int64)
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}"
            )

        tok = self.wte(idx)                              # (B, T, C)
        pos = self.wpe(np.arange(T, dtype=np.int64))     # (T, C), broadcast over batch
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if self.lm_head is None:
            logits = x @ self.wte.weight.transpose()     # (B, T, vocab)
        else:
            logits = self.lm_head(x)

        if targets is None:
            return logits
        return logits, nn.cross_entropy(logits, targets)

    # ------------------------------------------------------------------ #

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = super().num_parameters()
        if non_embedding:
            n -= self.wpe.weight.size
            if self.lm_head is None:
                n -= self.wte.weight.size
        return n

    def parameter_summary(self) -> str:
        cfg = self.cfg
        lines = [
            f"layers          {cfg.n_layer}",
            f"heads           {cfg.n_head}  (head_dim {cfg.d_model // cfg.n_head})",
            f"d_model         {cfg.d_model}",
            f"d_ff            {cfg.d_ff}",
            f"context         {cfg.block_size}",
            f"vocab           {cfg.vocab_size}",
            f"tied embeddings {cfg.tie_embeddings}",
            f"parameters      {self.num_parameters():,} "
            f"({self.num_parameters(non_embedding=True):,} non-embedding)",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #

    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, rng: np.random.Generator | None = None):
        """Autoregressively sample ``max_new_tokens`` continuations of ``idx``.

        ``idx`` is ``(B, T)``; returns ``(B, T + max_new_tokens)``.

        ``temperature`` divides the logits before softmax: below 1 sharpens the
        distribution towards the argmax, above 1 flattens it. ``top_k`` restricts
        sampling to the k most likely tokens, which is what stops the long tail of
        thousands of near-zero-probability tokens from collectively owning a
        meaningful share of the probability mass.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0 (use top_k=1 for greedy decoding)")
        rng = rng if rng is not None else np.random.default_rng()
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]

        was_training = self.training
        self.eval()
        try:
            with no_grad():
                for _ in range(max_new_tokens):
                    # Crop to the context window: the model has no positional
                    # embedding beyond block_size, so the prefix must slide.
                    window = idx[:, -self.cfg.block_size:]
                    logits = self.forward(window).data[:, -1, :]   # (B, vocab)
                    logits = logits.astype(np.float64) / temperature

                    if top_k is not None:
                        k = min(top_k, logits.shape[-1])
                        kth = np.partition(logits, -k, axis=-1)[:, -k][:, None]
                        logits = np.where(logits < kth, -np.inf, logits)

                    logits -= logits.max(axis=-1, keepdims=True)
                    probs = np.exp(logits)
                    probs /= probs.sum(axis=-1, keepdims=True)

                    nxt = np.array([rng.choice(len(p), p=p) for p in probs], dtype=np.int64)
                    idx = np.concatenate([idx, nxt[:, None]], axis=1)
        finally:
            self.train(was_training)
        return idx
