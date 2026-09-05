"""Forge autodiff engine.

A minimal reverse-mode automatic differentiation library built on NumPy.

The design is the classic one: every :class:`Tensor` produced by an operation
keeps a reference to its parents plus a closure that knows how to push gradient
from the output back to those parents.  Calling :meth:`Tensor.backward` walks the
graph in reverse topological order and invokes each closure exactly once, so
every node's ``.grad`` is complete before it is used.

Everything else in Forge -- layers, attention, the optimizer, the GPT model --
is written on top of this file.  Nothing here imports a framework.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Tensor", "set_default_dtype", "default_dtype", "no_grad", "is_grad_enabled"]


# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #

_DEFAULT_DTYPE = np.dtype(np.float32)
_GRAD_ENABLED = True


def set_default_dtype(dtype) -> None:
    """Set the dtype every new Tensor is cast to.

    Training runs in float32 (half the memory, and meaningfully faster BLAS than
    float64).  The gradient checker flips this to float64 because central finite
    differences burn roughly half the available significant digits, and float32
    does not have enough of them left to certify an analytical gradient.
    """
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype)


def default_dtype() -> np.dtype:
    return _DEFAULT_DTYPE


class no_grad:
    """Context manager that disables graph construction.

    Used for evaluation and generation, where building a tape would waste both
    memory and time.  Implemented as a global flag consulted by :func:`Tensor._make`;
    ops still run, they just do not record parents.
    """

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


# --------------------------------------------------------------------------- #
# Broadcasting helpers
# --------------------------------------------------------------------------- #

def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` after NumPy broadcasting.

    Broadcasting copies in the forward direction, so its adjoint is a sum.  Two
    things can happen when an array of shape ``shape`` is broadcast: leading axes
    are prepended (sum them away entirely) and size-1 axes are stretched (sum
    them but keep the axis).  This handles both.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


def _expand_dims_for_reduction(grad: np.ndarray, shape: tuple, axis, keepdims: bool) -> np.ndarray:
    """Undo a reduction: put the reduced axes back so ``grad`` can broadcast."""
    if axis is None:
        return np.broadcast_to(grad.reshape((1,) * len(shape)), shape)
    if not keepdims:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(a % len(shape) for a in axes)
        for a in sorted(axes):
            grad = np.expand_dims(grad, a)
    return np.broadcast_to(grad, shape)


def _noop() -> None:
    """Backward function for leaves and detached nodes."""
    return None


# --------------------------------------------------------------------------- #
# Tensor
# --------------------------------------------------------------------------- #

class Tensor:
    """An n-dimensional array that remembers how it was computed.

    Parameters
    ----------
    data:
        Anything ``np.asarray`` accepts, or another Tensor (its buffer is reused).
    requires_grad:
        Whether gradient should be accumulated into ``.grad`` for this node.
        Operation results set this automatically to ``any(parent.requires_grad)``.
    """

    # Make NumPy defer to our __radd__/__rmul__/... instead of trying to
    # broadcast a Tensor into an object array element by element.
    __array_priority__ = 100.0

    def __init__(self, data, requires_grad: bool = False, _children=(), _op: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        if arr.dtype != _DEFAULT_DTYPE:
            arr = arr.astype(_DEFAULT_DTYPE)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = bool(requires_grad)
        self._backward = _noop
        self._prev: tuple = tuple(_children)
        self._op: str = _op

    # -- basic properties --------------------------------------------------- #

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, op={self._op!r})"

    # ``__eq__`` is overloaded below to build masks, so keep identity hashing.
    __hash__ = object.__hash__

    def item(self) -> float:
        return self.data.item()

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> "Tensor":
        """Return a Tensor sharing this buffer but cut out of the graph."""
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate(self, g: np.ndarray) -> None:
        """Add ``g`` into ``.grad``, allocating on the first contribution."""
        if self.grad is None:
            self.grad = np.array(g, dtype=self.data.dtype, copy=True)
        else:
            self.grad += g

    # -- graph construction ------------------------------------------------- #

    @staticmethod
    def _make(data, parents, op: str, backward_fn) -> "Tensor":
        """Build an output Tensor, wiring it into the graph only if needed."""
        needs = _GRAD_ENABLED and any(p.requires_grad for p in parents)
        if not needs:
            return Tensor(data, requires_grad=False, _op=op)
        out = Tensor(data, requires_grad=True, _children=parents, _op=op)
        out._backward = backward_fn(out)
        return out

    def backward(self, gradient=None) -> None:
        """Reverse-mode sweep, populating ``.grad`` on every node that needs it.

        ``gradient`` seeds the output adjoint; it defaults to ones, which is only
        meaningful for a scalar output, so a non-scalar output without an explicit
        seed raises rather than silently implying a sum.
        """
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() on a non-scalar Tensor requires an explicit `gradient` "
                    f"argument (this tensor has shape {self.shape})"
                )
            gradient = np.ones_like(self.data)
        gradient = np.asarray(gradient, dtype=self.data.dtype)
        if gradient.shape != self.data.shape:
            raise ValueError(
                f"gradient shape {gradient.shape} does not match tensor shape {self.shape}"
            )

        # Iterative post-order DFS.  Recursion would blow the stack: a 6-layer
        # transformer forward pass builds a graph thousands of nodes deep, well
        # past CPython's default 1000-frame limit.
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))

        self.grad = np.array(gradient, copy=True)
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------ #
    # Arithmetic
    # ------------------------------------------------------------------ #

    def _coerce(self, other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other) -> "Tensor":
        other = self._coerce(other)
        a, b = self, other

        def make_bw(out):
            def bw():
                g = out.grad
                if a.requires_grad:
                    a._accumulate(_unbroadcast(g, a.data.shape))
                if b.requires_grad:
                    b._accumulate(_unbroadcast(g, b.data.shape))
            return bw

        return Tensor._make(a.data + b.data, (a, b), "add", make_bw)

    def __mul__(self, other) -> "Tensor":
        other = self._coerce(other)
        a, b = self, other

        def make_bw(out):
            def bw():
                g = out.grad
                if a.requires_grad:
                    a._accumulate(_unbroadcast(g * b.data, a.data.shape))
                if b.requires_grad:
                    b._accumulate(_unbroadcast(g * a.data, b.data.shape))
            return bw

        return Tensor._make(a.data * b.data, (a, b), "mul", make_bw)

    def __neg__(self) -> "Tensor":
        a = self

        def make_bw(out):
            def bw():
                a._accumulate(-out.grad)
            return bw

        return Tensor._make(-a.data, (a,), "neg", make_bw)

    def __sub__(self, other) -> "Tensor":
        other = self._coerce(other)
        a, b = self, other

        def make_bw(out):
            def bw():
                g = out.grad
                if a.requires_grad:
                    a._accumulate(_unbroadcast(g, a.data.shape))
                if b.requires_grad:
                    b._accumulate(_unbroadcast(-g, b.data.shape))
            return bw

        return Tensor._make(a.data - b.data, (a, b), "sub", make_bw)

    def __truediv__(self, other) -> "Tensor":
        other = self._coerce(other)
        a, b = self, other

        def make_bw(out):
            def bw():
                g = out.grad
                if a.requires_grad:
                    a._accumulate(_unbroadcast(g / b.data, a.data.shape))
                if b.requires_grad:
                    b._accumulate(_unbroadcast(-g * a.data / (b.data * b.data), b.data.shape))
            return bw

        return Tensor._make(a.data / b.data, (a, b), "div", make_bw)

    def __pow__(self, other) -> "Tensor":
        other = self._coerce(other)
        a, b = self, other
        val = a.data ** b.data

        def make_bw(out):
            def bw():
                g = out.grad
                if a.requires_grad:
                    a._accumulate(_unbroadcast(g * b.data * (a.data ** (b.data - 1)), a.data.shape))
                if b.requires_grad:
                    # d/db a**b = a**b * ln(a); only defined for a > 0.
                    safe_log = np.log(np.where(a.data > 0, a.data, 1.0))
                    b._accumulate(_unbroadcast(g * val * safe_log, b.data.shape))
            return bw

        return Tensor._make(val, (a, b), "pow", make_bw)

    def __matmul__(self, other) -> "Tensor":
        other = self._coerce(other)
        a, b = self, other
        a_nd, b_nd = a.data.ndim, b.data.ndim
        if a_nd == 0 or b_nd == 0:
            raise ValueError("matmul does not accept 0-d operands")

        def make_bw(out):
            def bw():
                # Promote 1-D operands to 2-D so one code path covers every case,
                # then fold the promoted axis back out of the resulting gradient.
                A = a.data if a_nd > 1 else a.data[None, :]
                B = b.data if b_nd > 1 else b.data[:, None]
                G = out.grad
                if a_nd == 1:
                    G = np.expand_dims(G, -2)
                if b_nd == 1:
                    G = np.expand_dims(G, -1)
                if a.requires_grad:
                    gA = G @ np.swapaxes(B, -1, -2)
                    a._accumulate(_unbroadcast(gA, A.shape).reshape(a.data.shape))
                if b.requires_grad:
                    gB = np.swapaxes(A, -1, -2) @ G
                    b._accumulate(_unbroadcast(gB, B.shape).reshape(b.data.shape))
            return bw

        return Tensor._make(a.data @ b.data, (a, b), "matmul", make_bw)

    # Reflected variants -- scalars and NumPy arrays on the left-hand side.
    def __radd__(self, other):
        return self._coerce(other) + self

    def __rmul__(self, other):
        return self._coerce(other) * self

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __rpow__(self, other):
        return self._coerce(other) ** self

    def __rmatmul__(self, other):
        return self._coerce(other) @ self

    # ------------------------------------------------------------------ #
    # Elementwise unary maths
    # ------------------------------------------------------------------ #

    def exp(self) -> "Tensor":
        a = self
        val = np.exp(a.data)

        def make_bw(out):
            def bw():
                a._accumulate(out.grad * val)
            return bw

        return Tensor._make(val, (a,), "exp", make_bw)

    def log(self) -> "Tensor":
        a = self

        def make_bw(out):
            def bw():
                a._accumulate(out.grad / a.data)
            return bw

        return Tensor._make(np.log(a.data), (a,), "log", make_bw)

    def sqrt(self) -> "Tensor":
        a = self
        val = np.sqrt(a.data)

        def make_bw(out):
            def bw():
                a._accumulate(out.grad * 0.5 / val)
            return bw

        return Tensor._make(val, (a,), "sqrt", make_bw)

    def tanh(self) -> "Tensor":
        a = self
        val = np.tanh(a.data)

        def make_bw(out):
            def bw():
                a._accumulate(out.grad * (1.0 - val * val))
            return bw

        return Tensor._make(val, (a,), "tanh", make_bw)

    def abs(self) -> "Tensor":
        a = self

        def make_bw(out):
            def bw():
                a._accumulate(out.grad * np.sign(a.data))
            return bw

        return Tensor._make(np.abs(a.data), (a,), "abs", make_bw)

    def relu(self) -> "Tensor":
        a = self
        mask = (a.data > 0).astype(a.data.dtype)

        def make_bw(out):
            def bw():
                a._accumulate(out.grad * mask)
            return bw

        return Tensor._make(a.data * mask, (a,), "relu", make_bw)

    def sigmoid(self) -> "Tensor":
        a = self
        # Branch-free stable logistic: exp() of a large positive argument overflows.
        pos = a.data >= 0
        z = np.empty_like(a.data)
        z[pos] = 1.0 / (1.0 + np.exp(-a.data[pos]))
        e = np.exp(a.data[~pos])
        z[~pos] = e / (1.0 + e)

        def make_bw(out):
            def bw():
                a._accumulate(out.grad * z * (1.0 - z))
            return bw

        return Tensor._make(z, (a,), "sigmoid", make_bw)

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        a = self
        shape = a.data.shape

        def make_bw(out):
            def bw():
                a._accumulate(_expand_dims_for_reduction(out.grad, shape, axis, keepdims))
            return bw

        return Tensor._make(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum", make_bw)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        a = self
        shape = a.data.shape
        if axis is None:
            n = a.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = int(np.prod([shape[ax] for ax in axes]))

        def make_bw(out):
            def bw():
                a._accumulate(_expand_dims_for_reduction(out.grad, shape, axis, keepdims) / n)
            return bw

        return Tensor._make(a.data.mean(axis=axis, keepdims=keepdims), (a,), "mean", make_bw)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        a = self
        shape = a.data.shape
        val = a.data.max(axis=axis, keepdims=keepdims)

        def make_bw(out):
            def bw():
                expanded_val = _expand_dims_for_reduction(np.asarray(val), shape, axis, keepdims)
                # Ties split the gradient evenly.  That keeps the adjoint a valid
                # subgradient and matches what a symmetric finite difference
                # measures at a tie, so the gradient checker stays meaningful.
                hits = (a.data == expanded_val).astype(a.data.dtype)
                counts = _expand_dims_for_reduction(
                    hits.sum(axis=axis, keepdims=keepdims), shape, axis, keepdims
                )
                g = _expand_dims_for_reduction(out.grad, shape, axis, keepdims)
                a._accumulate(g * hits / counts)
            return bw

        return Tensor._make(val, (a,), "max", make_bw)

    def min(self, axis=None, keepdims: bool = False) -> "Tensor":
        return -((-self).max(axis=axis, keepdims=keepdims))

    def var(self, axis=None, keepdims: bool = False) -> "Tensor":
        """Population (biased, ddof=0) variance, built from primitives."""
        mu = self.mean(axis=axis, keepdims=True)
        centred = self - mu
        return (centred * centred).mean(axis=axis, keepdims=keepdims)

    # ------------------------------------------------------------------ #
    # Shape manipulation
    # ------------------------------------------------------------------ #

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self
        orig = a.data.shape

        def make_bw(out):
            def bw():
                a._accumulate(out.grad.reshape(orig))
            return bw

        return Tensor._make(a.data.reshape(shape), (a,), "reshape", make_bw)

    def view(self, *shape) -> "Tensor":
        return self.reshape(*shape)

    def flatten(self, start_dim: int = 0) -> "Tensor":
        shape = self.data.shape
        new = shape[:start_dim] + (int(np.prod(shape[start_dim:])),)
        return self.reshape(new)

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        a = self
        inverse = tuple(int(i) for i in np.argsort(axes))

        def make_bw(out):
            def bw():
                a._accumulate(out.grad.transpose(inverse))
            return bw

        return Tensor._make(a.data.transpose(axes), (a,), "transpose", make_bw)

    def permute(self, *axes) -> "Tensor":
        return self.transpose(*axes)

    def swapaxes(self, ax1: int, ax2: int) -> "Tensor":
        axes = list(range(self.data.ndim))
        axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
        return self.transpose(tuple(axes))

    def broadcast_to(self, shape) -> "Tensor":
        a = self
        orig = a.data.shape

        def make_bw(out):
            def bw():
                a._accumulate(_unbroadcast(out.grad, orig))
            return bw

        return Tensor._make(np.broadcast_to(a.data, shape).copy(), (a,), "broadcast_to", make_bw)

    def __getitem__(self, idx) -> "Tensor":
        """Indexing / gather.

        The adjoint of a gather is a scatter-add.  ``np.add.at`` is the buffered
        (duplicate-safe) form -- plain fancy-index assignment silently keeps only
        the last write when an index repeats, which is exactly what happens when
        a token appears twice in a batch.
        """
        a = self
        key = idx.data.astype(np.int64) if isinstance(idx, Tensor) else idx

        def make_bw(out):
            def bw():
                buf = np.zeros_like(a.data)
                np.add.at(buf, key, out.grad)
                a._accumulate(buf)
            return bw

        return Tensor._make(a.data[key], (a,), "getitem", make_bw)

    # ------------------------------------------------------------------ #
    # Comparisons and masking
    # ------------------------------------------------------------------ #

    def _compare(self, other, fn, name) -> "Tensor":
        other_data = other.data if isinstance(other, Tensor) else np.asarray(other)
        # Masks are constants: a comparison has zero gradient almost everywhere.
        return Tensor(
            fn(self.data, other_data).astype(_DEFAULT_DTYPE), requires_grad=False, _op=name
        )

    def __gt__(self, other):
        return self._compare(other, np.greater, "gt")

    def __ge__(self, other):
        return self._compare(other, np.greater_equal, "ge")

    def __lt__(self, other):
        return self._compare(other, np.less, "lt")

    def __le__(self, other):
        return self._compare(other, np.less_equal, "le")

    def __eq__(self, other):
        return self._compare(other, np.equal, "eq")

    def __ne__(self, other):
        return self._compare(other, np.not_equal, "ne")

    @staticmethod
    def where(condition, x, y) -> "Tensor":
        """Elementwise select. ``condition`` is treated as a constant mask."""
        cond = condition.data if isinstance(condition, Tensor) else np.asarray(condition)
        cond = cond.astype(bool)
        x = x if isinstance(x, Tensor) else Tensor(x)
        y = y if isinstance(y, Tensor) else Tensor(y)

        def make_bw(out):
            def bw():
                g = out.grad
                if x.requires_grad:
                    x._accumulate(_unbroadcast(np.where(cond, g, 0.0), x.data.shape))
                if y.requires_grad:
                    y._accumulate(_unbroadcast(np.where(cond, 0.0, g), y.data.shape))
            return bw

        return Tensor._make(np.where(cond, x.data, y.data), (x, y), "where", make_bw)

    def masked_fill(self, mask, value: float) -> "Tensor":
        """Replace entries where ``mask`` is truthy with ``value``.

        Gradient flows only through the *unmasked* entries -- the filled ones are
        constants.  This is what makes causal masking gradient-tight: a future
        position cannot leak signal backwards through the mask.
        """
        a = self
        m = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
        m = m.astype(bool)
        keep = (~m).astype(a.data.dtype)

        def make_bw(out):
            def bw():
                a._accumulate(_unbroadcast(out.grad * keep, a.data.shape))
            return bw

        return Tensor._make(np.where(m, value, a.data), (a,), "masked_fill", make_bw)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @staticmethod
    def _norm_shape(shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return tuple(shape[0])
        return shape

    @staticmethod
    def zeros(*shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros(Tensor._norm_shape(shape)), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.ones(Tensor._norm_shape(shape)), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad: bool = False, rng=None) -> "Tensor":
        shape = Tensor._norm_shape(shape)
        rng = rng if rng is not None else np.random.default_rng()
        return Tensor(rng.standard_normal(shape), requires_grad=requires_grad)

    @staticmethod
    def zeros_like(t: "Tensor", requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros_like(t.data), requires_grad=requires_grad)

    @staticmethod
    def ones_like(t: "Tensor", requires_grad: bool = False) -> "Tensor":
        return Tensor(np.ones_like(t.data), requires_grad=requires_grad)

    @staticmethod
    def concat(tensors, axis: int = 0) -> "Tensor":
        tensors = tuple(tensors)
        sizes = [t.data.shape[axis] for t in tensors]
        bounds = np.cumsum([0] + sizes)

        def make_bw(out):
            def bw():
                for i, t in enumerate(tensors):
                    if not t.requires_grad:
                        continue
                    sl = [slice(None)] * out.grad.ndim
                    sl[axis] = slice(bounds[i], bounds[i + 1])
                    t._accumulate(out.grad[tuple(sl)])
            return bw

        return Tensor._make(
            np.concatenate([t.data for t in tensors], axis=axis), tensors, "concat", make_bw
        )
