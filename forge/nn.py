"""Neural network primitives, built entirely on the Phase 1 Tensor.

Nothing in this file reaches past :mod:`forge.tensor` for a derivative.  Every
layer is composed from ops that the autodiff engine already knows how to
differentiate, which means the gradient of a LayerNorm or a multi-head attention
block is assembled by the engine rather than hand-derived here -- the certified
op gradients from Phase 1 are what make that safe.
"""

from __future__ import annotations

import math

import numpy as np

from .tensor import Tensor, default_dtype

__all__ = [
    "Parameter",
    "Module",
    "Linear",
    "Embedding",
    "LayerNorm",
    "Dropout",
    "Sequential",
    "MultiHeadSelfAttention",
    "softmax",
    "log_softmax",
    "gelu",
    "relu",
    "cross_entropy",
    "set_seed",
    "get_rng",
]


# --------------------------------------------------------------------------- #
# Randomness
# --------------------------------------------------------------------------- #

_RNG = np.random.default_rng(1337)


def set_seed(seed: int) -> None:
    """Seed the RNG used for initialisation and dropout."""
    global _RNG
    _RNG = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _RNG


# --------------------------------------------------------------------------- #
# Parameter / Module
# --------------------------------------------------------------------------- #

class Parameter(Tensor):
    """A Tensor that an optimizer is allowed to update.

    The only difference from a plain Tensor is intent: ``requires_grad`` is
    always on, and :meth:`Module.parameters` collects these and nothing else.
    """

    def __init__(self, data):
        super().__init__(data, requires_grad=True)

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape})"


class Module:
    """Base class holding parameters and sub-modules.

    Parameters and children are discovered by walking ``__dict__`` rather than
    through an explicit registration call, so a subclass just assigns attributes
    in ``__init__`` and everything downstream (optimizer, checkpointing,
    train/eval switching) picks them up.
    """

    training: bool = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # -- traversal ---------------------------------------------------------- #

    def _children(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield f"{name}.{i}", item
            elif isinstance(value, dict):
                for k, item in value.items():
                    if isinstance(item, Module):
                        yield f"{name}.{k}", item

    def named_parameters(self, prefix: str = ""):
        """Yield ``(qualified_name, Parameter)`` for this module and its children.

        De-duplicated by identity, so a weight shared between two modules (the
        tied embedding in Phase 3) is yielded once and therefore stepped once by
        the optimizer.
        """
        seen: set[int] = set()
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter) and id(value) not in seen:
                seen.add(id(value))
                yield (prefix + name, value)
        for name, child in self._children():
            for qual, p in child.named_parameters(prefix + name + "."):
                if id(p) not in seen:
                    seen.add(id(p))
                    yield (qual, p)

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def modules(self):
        yield self
        for _, child in self._children():
            yield from child.modules()

    # -- state ------------------------------------------------------------- #

    def train(self, mode: bool = True) -> "Module":
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def state_dict(self) -> dict:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict) -> None:
        own = dict(self.named_parameters())
        missing = set(own) - set(state)
        unexpected = set(state) - set(own)
        if missing or unexpected:
            raise KeyError(f"state_dict mismatch; missing={sorted(missing)}, "
                           f"unexpected={sorted(unexpected)}")
        for name, p in own.items():
            incoming = np.asarray(state[name])
            if incoming.shape != p.data.shape:
                raise ValueError(
                    f"shape mismatch for {name}: checkpoint {incoming.shape} "
                    f"vs model {p.data.shape}"
                )
            p.data[...] = incoming.astype(p.data.dtype)


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #

class Linear(Module):
    """Dense layer: ``y = x @ W + b``.

    The weight is stored ``(in_features, out_features)`` so the forward pass is a
    plain right-multiply with no transpose -- one fewer op on the tape than the
    PyTorch convention, and the batched-matmul gradient from Phase 1 covers it
    for any number of leading batch dimensions.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init_std: float = 0.02):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(_RNG.normal(0.0, init_std, (in_features, out_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """Token lookup table: a gather from a ``(num_embeddings, dim)`` matrix.

    Backward is the scatter-add certified in Phase 1, which is what makes a
    repeated token accumulate gradient instead of overwriting it.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float = 0.02):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(_RNG.normal(0.0, init_std, (num_embeddings, embedding_dim)))

    def forward(self, idx) -> Tensor:
        idx = idx.data if isinstance(idx, Tensor) else np.asarray(idx)
        return self.weight[idx.astype(np.int64)]


class LayerNorm(Module):
    """Normalise the last dimension to zero mean and unit variance, then affine.

    Built from ``mean``, ``sub``, ``mul``, ``sqrt`` and ``div`` -- the backward
    pass (which is genuinely fiddly to derive by hand, because mean and variance
    both depend on every element) falls out of the engine.

    ``eps`` sits *inside* the square root, matching the standard formulation.
    Outside, it would stop dividing by zero but would no longer be a variance
    floor, and the gradient near zero variance would still blow up.
    """

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Parameter(np.ones(dim))
            self.bias = Parameter(np.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        centred = x - mu
        var = (centred * centred).mean(axis=-1, keepdims=True)
        xhat = centred / (var + self.eps).sqrt()
        if self.affine:
            xhat = xhat * self.weight + self.bias
        return xhat


class Dropout(Module):
    """Inverted dropout.

    Scaling by ``1/(1-p)`` at *training* time means evaluation is an exact
    identity -- no rescaling to remember, and no train/eval mismatch if someone
    forgets to call ``.eval()``... except that they would then still be dropping
    units.  Hence the explicit train/eval tests in Phase 2.
    """

    def __init__(self, p: float = 0.1):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (_RNG.random(x.shape) < keep).astype(default_dtype()) / keep
        return x * Tensor(mask)


# --------------------------------------------------------------------------- #
# Activations and losses (functions, not layers)
# --------------------------------------------------------------------------- #

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax.

    ``exp`` overflows in float32 above ~88, and attention logits routinely exceed
    that, so the row max is subtracted first.  Softmax is invariant to that shift,
    ``softmax(x) == softmax(x - c)``, so the result is unchanged.

    The max is taken on the raw NumPy buffer, i.e. treated as a constant rather
    than differentiated through.  That is not an approximation: the shift cancels
    exactly in the forward value, so its derivative contribution is exactly zero.
    Detaching just avoids putting a ``max`` node and its scatter on the tape.
    """
    m = Tensor(x.data.max(axis=axis, keepdims=True))
    e = (x - m).exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Stable ``log(softmax(x))``.

    Computed as ``z - log(sum(exp(z)))`` with ``z = x - max(x)`` rather than by
    taking the log of a softmax.  Softmax underflows to exactly 0.0 for
    sufficiently negative logits, and ``log(0)`` is ``-inf``; this form never
    forms the small probability in the first place, so the loss stays finite even
    when the model is confidently wrong.
    """
    m = Tensor(x.data.max(axis=axis, keepdims=True))
    z = x - m
    return z - z.exp().sum(axis=axis, keepdims=True).log()


def gelu(x: Tensor) -> Tensor:
    """GELU, tanh approximation (the GPT-2 formulation).

    The exact form uses ``erf``, which NumPy does not provide as a ufunc; a
    ``math.erf`` loop or a SciPy dependency would both be worse than the tanh
    approximation, which agrees with the exact form to within ~1e-3 absolute
    across the whole range and is what GPT-2 itself shipped.
    """
    c = math.sqrt(2.0 / math.pi)
    inner = (x + (x * x * x) * 0.044715) * c
    return x * 0.5 * (inner.tanh() + 1.0)


def relu(x: Tensor) -> Tensor:
    return x.relu()


def cross_entropy(logits: Tensor, targets, ignore_index: int | None = None) -> Tensor:
    """Mean negative log-likelihood over a batch of next-token predictions.

    ``logits`` is ``(..., vocab)``; ``targets`` holds integer class ids with the
    matching leading shape.  Uses :func:`log_softmax`, so no probability is ever
    materialised and the loss cannot become ``inf`` from an underflowed softmax.
    """
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets)
    targets = targets.astype(np.int64)
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)

    logp = log_softmax(flat_logits, axis=-1)
    rows = np.arange(flat_targets.shape[0])
    picked = logp[(rows, flat_targets)]

    if ignore_index is not None:
        keep = (flat_targets != ignore_index).astype(default_dtype())
        n = float(keep.sum())
        if n == 0.0:
            raise ValueError("every target was ignore_index; loss is undefined")
        return -(picked * Tensor(keep)).sum() / n
    return -picked.mean()


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #

# Fill value for masked attention logits.  -1e9 rather than -inf: a row that is
# entirely masked would make softmax produce 0/0 = NaN with -inf, whereas -1e9
# degrades to a uniform distribution.  Causal masking never produces such a row
# (position i can always see itself), but the finite constant means a future
# padding mask cannot silently poison the whole batch with NaN.  After the max
# subtraction in softmax, exp(-1e9) underflows to exactly 0.0, so the masked
# weight is zero, not merely small.
NEG_INF = -1e9


def causal_mask(seq_len: int) -> np.ndarray:
    """``(1, 1, T, T)`` boolean mask, True where attention must be blocked.

    Entry ``[i, j]`` is True when ``j > i`` -- strictly-upper-triangular -- so a
    query at position ``i`` may attend to keys at ``0..i`` inclusive and nothing
    later.  The diagonal is False: a token always sees itself.
    """
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)[None, None, :, :]


class MultiHeadSelfAttention(Module):
    """Multi-head scaled dot-product self-attention with causal masking.

    Shapes through the block, for batch ``B``, sequence ``T``, model width ``C``
    and ``H`` heads of size ``d = C/H``:

        x        (B, T, C)
        qkv      (B, T, 3C)      one fused projection, then sliced
        q,k,v    (B, H, T, d)    reshape to (B,T,H,d) then swap T and H
        scores   (B, H, T, T)    q @ kᵀ * d^-0.5, then causal mask, then softmax
        out      (B, H, T, d)    scores @ v
        y        (B, T, C)       heads concatenated, then output projection
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 512, init_std: float = 0.02,
                 proj_init_std: float | None = None):
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # 1/sqrt(d) keeps the logits' variance at ~1 regardless of head width.
        # Without it, dot products grow like d and softmax saturates, which
        # drives the gradient through it to ~0 and the block never learns.
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = Linear(d_model, 3 * d_model, init_std=init_std)
        self.proj = Linear(d_model, d_model,
                           init_std=proj_init_std if proj_init_std is not None else init_std)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)
        self.max_seq_len = max_seq_len
        self._mask = causal_mask(max_seq_len)

    def forward(self, x: Tensor, return_attention: bool = False):
        B, T, C = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        qkv = self.qkv(x)
        q = qkv[:, :, 0 * C:1 * C]
        k = qkv[:, :, 1 * C:2 * C]
        v = qkv[:, :, 2 * C:3 * C]

        def split_heads(t: Tensor) -> Tensor:
            return t.reshape(B, T, self.n_heads, self.head_dim).transpose((0, 2, 1, 3))

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scores = (q @ k.transpose((0, 1, 3, 2))) * self.scale
        scores = scores.masked_fill(self._mask[:, :, :T, :T], NEG_INF)
        weights = softmax(scores, axis=-1)
        attn = self.attn_dropout(weights)

        out = attn @ v                                       # (B, H, T, d)
        out = out.transpose((0, 2, 1, 3)).reshape(B, T, C)   # concat heads
        y = self.resid_dropout(self.proj(out))
        if return_attention:
            return y, weights
        return y
