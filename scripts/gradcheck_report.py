"""Certify every Forge op against finite differences and print a summary table.

This is the human-readable companion to ``tests/test_phase1_gradcheck.py``: the
test suite asserts, this script reports how much headroom each op actually has.

    python scripts/gradcheck_report.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.gradcheck import check_gradient  # noqa: E402
from forge.tensor import Tensor  # noqa: E402

RTOL = 1e-6
ATOL = 1e-9


def t(shape, seed, low=None, high=None, away=0.0):
    rng = np.random.default_rng(seed)
    if low is not None:
        x = low + (high - low) * rng.random(shape)
    else:
        x = rng.standard_normal(shape)
        if away:
            x = np.sign(x) * (np.abs(x) + away)
    return Tensor(x, requires_grad=True)


def separated(shape, seed, axis, min_gap=1e-3):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        x = rng.standard_normal(shape)
        srt = np.sort(x, axis=axis) if axis is not None else np.sort(x.ravel())
        gaps = np.diff(srt, axis=(axis if axis is not None else 0))
        if gaps.size == 0 or gaps.min() > min_gap:
            return Tensor(x, requires_grad=True)
    raise RuntimeError("no well-separated draw")


CASES = [
    ("add (broadcast)",   lambda x, y: x + y,                     [t((4, 5), 0), t((5,), 1)]),
    ("sub (broadcast)",   lambda x, y: x - y,                     [t((4, 5), 0), t((4, 1), 1)]),
    ("mul (broadcast)",   lambda x, y: x * y,                     [t((2, 3, 4), 0), t((3, 4), 1)]),
    ("neg",               lambda x: -x,                           [t((3, 4), 0)]),
    ("divide",            lambda x, y: x / y,                     [t((4, 5), 0), t((5,), 1, away=0.5)]),
    ("power (scalar exp)", lambda x: x ** 1.5,                    [t((3, 4), 0, low=0.5, high=2.5)]),
    ("power (tensor exp)", lambda x, y: x ** y,                   [t((3, 4), 0, low=0.5, high=2.5),
                                                                   t((3, 4), 1, low=0.5, high=2.0)]),
    ("matmul 2-D",        lambda x, y: x @ y,                     [t((4, 5), 0), t((5, 3), 1)]),
    ("matmul batched",    lambda x, y: x @ y,                     [t((3, 2, 4, 5), 0), t((3, 2, 5, 6), 1)]),
    ("matmul 1-D lhs",    lambda x, y: x @ y,                     [t((5,), 0), t((5, 3), 1)]),
    ("exp",               lambda x: x.exp(),                      [t((3, 4), 0)]),
    ("log",               lambda x: x.log(),                      [t((3, 4), 0, low=0.3, high=3.0)]),
    ("sqrt",              lambda x: x.sqrt(),                     [t((3, 4), 0, low=0.3, high=3.0)]),
    ("tanh",              lambda x: x.tanh(),                     [t((3, 4), 0)]),
    ("sigmoid",           lambda x: x.sigmoid(),                  [t((3, 4), 0)]),
    ("relu",              lambda x: x.relu(),                     [t((4, 5), 0, away=0.1)]),
    ("abs",               lambda x: x.abs(),                      [t((4, 5), 0, away=0.1)]),
    ("sum (axis)",        lambda x: x.sum(axis=(0, 2)),           [t((2, 3, 4), 0)]),
    ("mean (axis)",       lambda x: x.mean(axis=1, keepdims=True), [t((2, 3, 4), 0)]),
    ("max (axis)",        lambda x: x.max(axis=1),                [separated((3, 4, 5), 0, 1)]),
    ("min (axis)",        lambda x: x.min(axis=-1),               [separated((3, 4, 5), 0, -1)]),
    ("var",               lambda x: x.var(axis=-1),               [t((3, 4), 0)]),
    ("reshape",           lambda x: x.reshape(6, 4),              [t((2, 3, 4), 0)]),
    ("transpose",         lambda x: x.transpose((2, 0, 1)),       [t((2, 3, 4), 0)]),
    ("broadcast_to",      lambda x: x.broadcast_to((2, 3, 4)),    [t((3, 1), 0)]),
    ("concat",            lambda x, y: Tensor.concat([x, y], 1),  [t((2, 3), 0), t((2, 5), 1)]),
    ("getitem (slice)",   lambda x: x[1:4],                       [t((5, 6), 0)]),
    ("getitem (repeats)", lambda x: x[np.array([0, 3, 3, 1, 0])], [t((6, 4), 0)]),
    ("where",             lambda x, y: Tensor.where(
                              np.arange(20).reshape(4, 5) % 2 == 0, x, y),
                                                                  [t((4, 5), 0), t((4, 5), 1)]),
    ("masked_fill",       lambda x: x.masked_fill(
                              np.arange(20).reshape(4, 5) % 3 == 0, -3.0),
                                                                  [t((4, 5), 0)]),
    ("composite chain",   lambda x, y, z: ((((x @ y) * 0.3 + z).tanh() ** 2 + 1.0).log()),
                                                                  [t((4, 5), 0), t((5, 3), 1), t((3,), 2)]),
]


def main() -> int:
    print(f"Forge gradient certification  (rtol={RTOL:.0e}, atol={ATOL:.0e}, "
          f"central differences at h=1e-5 in float64)\n")
    print(f"{'operation':<22} {'inputs':<8} {'max rel. error':>16}   status")
    print("-" * 62)
    failures = 0
    for name, fn, inputs in CASES:
        try:
            worst = check_gradient(fn, inputs, rtol=RTOL, atol=ATOL, n_projections=3)
            err = max(worst.values()) if worst else 0.0
            print(f"{name:<22} {len(inputs):<8} {err:>16.3e}   PASS")
        except AssertionError as exc:
            failures += 1
            print(f"{name:<22} {len(inputs):<8} {'--':>16}   FAIL\n    {exc}")
    print("-" * 62)
    print(f"{len(CASES) - failures}/{len(CASES)} operations certified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
