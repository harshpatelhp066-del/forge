"""Numerical gradient checking for the Forge autodiff engine.

The whole project rests on Phase 1 being right, and "the loss went down" is not
evidence that a gradient is correct, a systematically wrong gradient can still
descend, just to the wrong place. So every operation is certified against a
central finite difference of the same function.

Two details matter for this to be a real test rather than a rubber stamp:

1. **Central differences, in float64.**  The forward difference
   ``(f(x+h) - f(x)) / h`` has O(h) truncation error; the central difference
   ``(f(x+h) - f(x-h)) / 2h`` has O(h^2), which buys about four extra digits for
   free. Catastrophic cancellation in the numerator then costs roughly half the
   mantissa, so float32 (7 digits) leaves ~3 digits of signal, not enough to
   distinguish a correct gradient from a subtly wrong one.  float64 leaves ~8.

2. **A random scalar projection, not a plain sum.**  Reducing the output with
   ``out.sum()`` only ever tests ``J^T @ 1``. A transposed or mis-permuted
   Jacobian can pass that by accident. Reducing with ``(out * w).sum()`` for a
   fixed random ``w`` tests ``J^T @ w``; over several random draws that pins down
   the full Jacobian.

3. **A combined relative/absolute criterion.**  A finite difference has an
   irreducible absolute noise floor of roughly ``eps * |f| / h``, about 2e-11
   for an O(1) loss in float64, because the numerator subtracts two nearly
   equal numbers. Judging a gradient entry of size 1e-8 by relative error alone
   therefore reports a "failure" of 1e-3 that is entirely an artefact of the
   measurement. The criterion here is the usual ``allclose`` form,
   ``|a - n| <= atol + rtol * max(|a|, |n|)``: ``rtol`` does the real work on
   entries above the floor, and ``atol`` stops the floor from generating noise.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, default_dtype, set_default_dtype


def relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Symmetric relative error, the standard gradient-check metric.

    Using ``|a| + |b|`` in the denominator keeps the measure well behaved when
    both quantities are near zero, where a plain ratio would explode.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.abs(a - b)
    den = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.max(num / den))


def numerical_gradient(scalar_fn, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of ``scalar_fn`` w.r.t. every entry of ``x``.

    ``x`` is perturbed in place and restored, so ``scalar_fn`` must read the same
    buffer it is handed (which is how Forge Tensors behave).
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=[["readonly"]])
    while not it.finished:
        idx = it.multi_index
        original = x[idx].copy()
        x[idx] = original + h
        f_plus = scalar_fn()
        x[idx] = original - h
        f_minus = scalar_fn()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2.0 * h)
        it.iternext()
    return grad


def check_gradient(
    fn,
    inputs,
    h: float = 1e-5,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    n_projections: int = 2,
    seed: int = 0,
    raise_on_fail: bool = True,
):
    """Compare analytical against numerical gradients for ``fn``.

    Parameters
    ----------
    fn:
        Callable taking the Tensors in ``inputs`` and returning a single Tensor
        of any shape.
    inputs:
        Sequence of Tensors. Those with ``requires_grad=True`` are checked.
    h:
        Finite-difference step.  1e-5 sits near the float64 optimum: truncation
        error falls as h^2 while round-off grows as 1/h, and the two cross around
        1e-5 to 1e-6 for well-scaled inputs.
    rtol, atol:
        Pass criterion ``|a - n| <= atol + rtol * max(|a|, |n|)``, applied
        elementwise. ``atol`` is the finite-difference noise floor (see the
        module docstring), not slack for a wrong gradient.
    n_projections:
        Number of random output projections to test (see module docstring).

    Returns
    -------
    dict mapping input index -> worst relative error observed.
    """
    prev_dtype = default_dtype()
    set_default_dtype(np.float64)
    try:
        inputs = [
            t if isinstance(t, Tensor) else Tensor(t)
            for t in inputs
        ]
        # Re-wrap so buffers are guaranteed float64 and owned by this call.
        inputs = [
            Tensor(np.array(t.data, dtype=np.float64), requires_grad=t.requires_grad)
            for t in inputs
        ]
        rng = np.random.default_rng(seed)

        probe = fn(*inputs)
        if not isinstance(probe, Tensor):
            raise TypeError("fn must return a Tensor")
        out_shape = probe.shape

        worst: dict[int, float] = {}
        for p in range(n_projections):
            # p == 0 uses an all-ones projection (the plain sum) so a failure is
            # easy to reason about; later projections are random.
            w = np.ones(out_shape) if p == 0 else rng.standard_normal(out_shape)

            def scalar() -> float:
                out = fn(*inputs)
                return float(np.sum(out.data * w))

            for t in inputs:
                t.zero_grad()
            out = fn(*inputs)
            out.backward(w)

            for i, t in enumerate(inputs):
                if not t.requires_grad:
                    continue
                analytic = t.grad
                if analytic is None:
                    analytic = np.zeros_like(t.data)
                numeric = numerical_gradient(scalar, t.data, h=h)
                err = relative_error(analytic, numeric)
                worst[i] = max(worst.get(i, 0.0), err)

                a64 = np.asarray(analytic, dtype=np.float64)
                slack = atol + rtol * np.maximum(np.abs(a64), np.abs(numeric))
                bad = np.abs(a64 - numeric) > slack
                if raise_on_fail and bad.any():
                    where = np.argwhere(bad)[:5]
                    detail = "\n".join(
                        f"    at {tuple(int(k) for k in ix)}: "
                        f"analytic={a64[tuple(ix)]:+.12e}  numeric={numeric[tuple(ix)]:+.12e}"
                        for ix in where
                    )
                    raise AssertionError(
                        f"gradient check failed for input {i} on projection {p}: "
                        f"{int(bad.sum())}/{bad.size} entries outside "
                        f"atol={atol:.1e} + rtol={rtol:.1e}; "
                        f"max symmetric relative error {err:.3e}\n{detail}"
                    )
        return worst
    finally:
        set_default_dtype(prev_dtype)
