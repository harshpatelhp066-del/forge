"""Optimizers, operating directly on ``Parameter.grad`` buffers.

Only Adam is implemented (plus plain SGD, mostly as a control for the tests).
Both read the ``.grad`` arrays that Phase 1's ``backward()`` populated and write
into ``.data`` in place.
"""

from __future__ import annotations

import numpy as np

from .nn import Parameter

__all__ = ["Optimizer", "SGD", "Adam", "clip_grad_norm"]


class Optimizer:
    def __init__(self, parameters):
        self.parameters: list[Parameter] = list(parameters)
        if not self.parameters:
            raise ValueError("optimizer got an empty parameter list")

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Vanilla SGD with optional classical momentum. Used as a baseline in tests."""

    def __init__(self, parameters, lr: float = 1e-2, momentum: float = 0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._buf = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            if self.momentum:
                self._buf[i] *= self.momentum
                self._buf[i] += p.grad
                p.data -= self.lr * self._buf[i]
            else:
                p.data -= self.lr * p.grad


class Adam(Optimizer):
    """Adam with bias correction, and optional decoupled weight decay (AdamW).

    Update, per parameter, at step ``t`` (1-based):

        m_t = β₁·m_{t-1} + (1-β₁)·g
        v_t = β₂·v_{t-1} + (1-β₂)·g²
        m̂  = m_t / (1 - β₁ᵗ)
        v̂  = v_t / (1 - β₂ᵗ)
        θ  ← θ - lr · m̂ / (√v̂ + ε)

    **The bias correction is not optional.** ``m`` and ``v`` start at zero, so
    early estimates are biased towards zero, at ``t=1`` with β₂=0.999, the raw
    ``v`` is a thousand times smaller than the true second moment. Without the
    correction, the very first steps take an enormous effective learning rate
    (``m/√v`` is roughly ``1/√(1-β₂) ≈ 31``× too large), which is exactly when a
    transformer is least able to survive it. The naive version usually shows up
    as a loss spike or an immediate NaN in the first dozen steps.

    ``eps`` sits *outside* the square root, matching the original paper. Inside,
    it would be a floor on the variance rather than on the divisor and would not
    bound the step size when ``v̂`` is genuinely zero (a parameter whose gradient
    has been exactly zero so far, e.g. an embedding row for a token that has
    not appeared yet).

    Weight decay is decoupled (AdamW): applied straight to the parameter rather
    than added into the gradient. Folding it into ``g`` would let Adam's per-
    parameter ``1/√v̂`` scaling shrink the decay for exactly the large-gradient
    parameters that most need regularising.
    """

    def __init__(self, parameters, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(parameters)
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0 or not 0.0 <= b2 < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        self.lr = lr
        self.beta1, self.beta2 = b1, b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.t += 1
        # Fold the bias correction into the step size instead of correcting m and
        # v separately: algebraically identical, but it avoids allocating two
        # full-size temporaries per parameter per step.
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        step_size = self.lr * np.sqrt(bc2) / bc1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad

            self.m[i] *= self.beta1
            self.m[i] += (1.0 - self.beta1) * g

            self.v[i] *= self.beta2
            self.v[i] += (1.0 - self.beta2) * (g * g)

            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data

            p.data -= step_size * self.m[i] / (np.sqrt(self.v[i]) + self.eps * np.sqrt(bc2))

    def state_dict(self) -> dict:
        return {
            "t": self.t,
            "m": [a.copy() for a in self.m],
            "v": [a.copy() for a in self.v],
            "lr": self.lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.t = int(state["t"])
        self.lr = float(state.get("lr", self.lr))
        for i, a in enumerate(state["m"]):
            self.m[i][...] = a
        for i, a in enumerate(state["v"]):
            self.v[i][...] = a


def clip_grad_norm(parameters, max_norm: float) -> float:
    """Rescale gradients in place so their **global** L2 norm is at most ``max_norm``.

    Global, not per-parameter: the quantity that matters is the length of the
    single update vector formed by concatenating every gradient, and clipping
    each tensor separately would change the update's *direction*, not just its
    length. Scaling everything by one shared factor preserves direction exactly.

    Returns the pre-clip norm, which is worth logging, a sudden spike is the
    earliest visible symptom of a training run about to diverge.
    """
    total_sq = 0.0
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)
            total_sq += float(np.sum(p.grad.astype(np.float64) ** 2))
    total_norm = float(np.sqrt(total_sq))

    if not np.isfinite(total_norm):
        # An inf or NaN gradient cannot be rescaled into a useful one: scaling by
        # max_norm/inf = 0 would turn every inf into NaN and quietly corrupt the
        # buffers. Leave them exactly as they are and report the non-finite norm,
        # so the caller can skip the step, which the training loop does.
        return total_norm

    if max_norm is not None and total_norm > max_norm:
        # The 1e-6 keeps the divisor away from zero for a norm that is huge but
        # finite, where max_norm/total_norm could otherwise underflow.
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g *= scale
    return total_norm
