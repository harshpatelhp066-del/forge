"""Phase 1 verification: every op's analytical gradient vs. finite differences.

Run with:  python -m pytest tests/test_phase1_gradcheck.py -v

Each gradient test is repeated across several random input draws and several
random output projections (see :mod:`forge.gradcheck`), so a wrong Jacobian
cannot hide behind a lucky symmetry.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.gradcheck import check_gradient, relative_error  # noqa: E402
from forge.tensor import Tensor, no_grad, set_default_dtype  # noqa: E402

RTOL = 1e-6
ATOL = 1e-9
SEEDS = [0, 1, 2]


def rand(shape, seed, low=None, high=None, away_from_zero=0.0):
    """Well-conditioned random input.

    ``away_from_zero`` pushes samples out of a band around the origin.  Ops with
    a kink at zero (relu, abs) or a pole there (log, div) would otherwise have
    the finite-difference step straddle the singularity, which reports a huge
    "error" that says nothing about the analytical gradient.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    if away_from_zero:
        x = np.sign(x) * (np.abs(x) + away_from_zero)
    if low is not None:
        x = low + (high - low) * rng.random(shape)
    return x


def T(shape, seed, requires_grad=True, **kw):
    return Tensor(rand(shape, seed, **kw), requires_grad=requires_grad)


def rand_separated(shape, seed, axis, min_gap=1e-3):
    """Random data with no near-ties along ``axis``.

    ``max`` is piecewise linear, with kinks exactly where two entries are equal.
    If the top two entries sit closer together than the finite-difference step
    (2h = 2e-5), perturbing the runner-up flips the argmax between the ``+h`` and
    ``-h`` evaluations and the difference quotient measures the kink rather than
    a derivative.  Resampling until the gap exceeds ``min_gap`` keeps the test
    about the gradient rather than about the kink; the tie case is covered
    separately and exactly by ``test_max_splits_gradient_across_ties``.
    """
    rng = np.random.default_rng(seed)
    for _ in range(200):
        x = rng.standard_normal(shape)
        srt = np.sort(x, axis=axis) if axis is not None else np.sort(x.ravel())
        gaps = np.diff(srt, axis=(axis if axis is not None else 0))
        if gaps.size == 0 or gaps.min() > min_gap:
            return x
    raise RuntimeError(f"could not draw well-separated data for shape={shape}, axis={axis}")


# --------------------------------------------------------------------------- #
# Binary arithmetic, including broadcasting
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "shape_a,shape_b",
    [
        ((4, 5), (4, 5)),      # same shape
        ((4, 5), (5,)),        # rank-raising broadcast
        ((4, 5), (4, 1)),      # stretched axis
        ((1, 5), (4, 1)),      # mutual broadcast
        ((2, 3, 4), (3, 4)),   # batched
        ((3,), ()),            # scalar
    ],
)
def test_add_sub_mul(shape_a, shape_b, seed):
    a, b = T(shape_a, seed), T(shape_b, seed + 100)
    check_gradient(lambda x, y: x + y, [a, b], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x, y: x - y, [a, b], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x, y: x * y, [a, b], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape_a,shape_b", [((4, 5), (4, 5)), ((4, 5), (5,)), ((3, 1), (3, 4))])
def test_divide(shape_a, shape_b, seed):
    a = T(shape_a, seed)
    # Denominator kept away from zero: near a pole the true derivative blows up
    # and no finite step can track it.
    b = T(shape_b, seed + 100, away_from_zero=0.5)
    check_gradient(lambda x, y: x / y, [a, b], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_neg(seed):
    a = T((3, 4), seed)
    check_gradient(lambda x: -x, [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("exponent", [2.0, 3.0, 0.5, -1.0, 1.5])
def test_power_scalar_exponent(exponent, seed):
    # Positive base so fractional and negative exponents stay real and finite.
    a = Tensor(rand((3, 4), seed, low=0.5, high=2.5), requires_grad=True)
    check_gradient(lambda x: x ** exponent, [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_power_tensor_exponent(seed):
    a = Tensor(rand((3, 4), seed, low=0.5, high=2.5), requires_grad=True)
    b = Tensor(rand((3, 4), seed + 7, low=0.5, high=2.0), requires_grad=True)
    check_gradient(lambda x, y: x ** y, [a, b], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "shape_a,shape_b",
    [
        ((4, 5), (5, 3)),            # plain 2-D
        ((2, 4, 5), (2, 5, 3)),      # batched
        ((2, 4, 5), (5, 3)),         # broadcast rhs
        ((3, 2, 4, 5), (3, 2, 5, 6)),  # two batch dims
        ((5,), (5, 3)),              # 1-D lhs
        ((4, 5), (5,)),              # 1-D rhs
    ],
)
def test_matmul(shape_a, shape_b, seed):
    a, b = T(shape_a, seed), T(shape_b, seed + 100)
    check_gradient(lambda x, y: x @ y, [a, b], rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# Unary maths
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_exp(seed):
    a = T((3, 4), seed)
    check_gradient(lambda x: x.exp(), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_log(seed):
    a = Tensor(rand((3, 4), seed, low=0.3, high=3.0), requires_grad=True)
    check_gradient(lambda x: x.log(), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_sqrt(seed):
    a = Tensor(rand((3, 4), seed, low=0.3, high=3.0), requires_grad=True)
    check_gradient(lambda x: x.sqrt(), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_tanh_sigmoid(seed):
    a = T((3, 4), seed)
    check_gradient(lambda x: x.tanh(), [a], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: x.sigmoid(), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_relu_and_abs(seed):
    # Kept clear of the kink at zero -- see rand()'s docstring.
    a = T((4, 5), seed, away_from_zero=0.1)
    check_gradient(lambda x: x.relu(), [a], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: x.abs(), [a], rtol=RTOL, atol=ATOL)


def test_sigmoid_is_overflow_safe():
    """The logistic must not overflow on large-magnitude logits."""
    x = Tensor([-800.0, -50.0, 0.0, 50.0, 800.0], requires_grad=True)
    with np.errstate(over="raise", invalid="raise"):
        y = x.sigmoid()
    assert np.all(np.isfinite(y.data))
    assert y.data[0] == pytest.approx(0.0, abs=1e-12)
    assert y.data[-1] == pytest.approx(1.0, abs=1e-12)
    y.sum().backward()
    assert np.all(np.isfinite(x.grad))


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum(axis, keepdims, seed):
    a = T((2, 3, 4), seed)
    check_gradient(lambda x: x.sum(axis=axis, keepdims=keepdims), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis", [None, 0, 1, -1, (1, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_mean(axis, keepdims, seed):
    a = T((2, 3, 4), seed)
    check_gradient(lambda x: x.mean(axis=axis, keepdims=keepdims), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_max_min(axis, keepdims, seed):
    a = Tensor(rand_separated((3, 4, 5), seed, axis), requires_grad=True)
    check_gradient(lambda x: x.max(axis=axis, keepdims=keepdims), [a], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: x.min(axis=axis, keepdims=keepdims), [a], rtol=RTOL, atol=ATOL)


def test_max_splits_gradient_across_ties():
    """A tied maximum should share gradient, not hand it all to index 0."""
    x = Tensor([2.0, 2.0, 1.0, 2.0], requires_grad=True)
    x.max().backward()
    assert x.grad == pytest.approx([1 / 3, 1 / 3, 0.0, 1 / 3])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_var(axis, seed):
    a = T((3, 4), seed)
    check_gradient(lambda x: x.var(axis=axis), [a], rtol=RTOL, atol=ATOL)
    # Value must agree with NumPy's population variance.
    got = Tensor(rand((3, 4), seed)).var(axis=axis).data
    assert got == pytest.approx(np.var(rand((3, 4), seed), axis=axis))


# --------------------------------------------------------------------------- #
# Shape manipulation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_reshape_and_flatten(seed):
    a = T((2, 3, 4), seed)
    check_gradient(lambda x: x.reshape(6, 4), [a], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: x.reshape(24), [a], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: x.flatten(start_dim=1), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("axes", [(0, 1, 2), (2, 0, 1), (0, 2, 1), (2, 1, 0)])
def test_transpose(axes, seed):
    a = T((2, 3, 4), seed)
    check_gradient(lambda x: x.transpose(axes), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_transpose_default_and_swapaxes(seed):
    a = T((3, 4), seed)
    check_gradient(lambda x: x.T, [a], rtol=RTOL, atol=ATOL)
    b = T((2, 3, 4), seed)
    check_gradient(lambda x: x.swapaxes(1, 2), [b], rtol=RTOL, atol=ATOL)


def test_transpose_permutation_is_inverted_correctly():
    """A non-involutive permutation catches an inverse-permutation bug."""
    x = Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4), requires_grad=True)
    y = x.transpose((2, 0, 1))
    assert y.shape == (4, 2, 3)
    g = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
    y.backward(g)
    assert np.array_equal(x.grad, g.transpose((1, 2, 0)))


@pytest.mark.parametrize("seed", SEEDS)
def test_broadcast_to(seed):
    a = T((3, 1), seed)
    check_gradient(lambda x: x.broadcast_to((2, 3, 4)), [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_concat(seed):
    a, b, c = T((2, 3), seed), T((4, 3), seed + 1), T((1, 3), seed + 2)
    check_gradient(lambda x, y, z: Tensor.concat([x, y, z], axis=0), [a, b, c], rtol=RTOL, atol=ATOL)
    d, e = T((2, 3), seed), T((2, 5), seed + 1)
    check_gradient(lambda x, y: Tensor.concat([x, y], axis=1), [d, e], rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# Indexing / gather
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_getitem_slice(seed):
    a = T((5, 6), seed)
    check_gradient(lambda x: x[1:4], [a], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: x[:, 2:5], [a], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: x[2], [a], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_getitem_fancy_with_repeats(seed):
    """Gather with a repeated index -- the embedding-layer case."""
    a = T((6, 4), seed)
    idx = np.array([0, 3, 3, 1, 0, 5])
    check_gradient(lambda x: x[idx], [a], rtol=RTOL, atol=ATOL)


def test_gather_backward_is_scatter_add_not_overwrite():
    """Repeated indices must accumulate; a plain assignment would keep one."""
    w = Tensor(np.zeros((3, 2)), requires_grad=True)
    out = w[np.array([1, 1, 1])]
    out.backward(np.ones((3, 2)))
    assert w.grad == pytest.approx(np.array([[0.0, 0.0], [3.0, 3.0], [0.0, 0.0]]))


# --------------------------------------------------------------------------- #
# Comparison ops and masking
# --------------------------------------------------------------------------- #

def test_comparison_ops_produce_constant_masks():
    a = Tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
    b = Tensor([[2.0, 2.0], [3.0, 9.0]])
    assert np.array_equal((a > b).data, [[0.0, 1.0], [0.0, 0.0]])
    assert np.array_equal((a >= b).data, [[0.0, 1.0], [1.0, 0.0]])
    assert np.array_equal((a < b).data, [[1.0, 0.0], [0.0, 1.0]])
    assert np.array_equal((a <= b).data, [[1.0, 0.0], [1.0, 1.0]])
    assert np.array_equal((a == b).data, [[0.0, 0.0], [1.0, 0.0]])
    assert np.array_equal((a != b).data, [[1.0, 1.0], [0.0, 1.0]])
    # A mask is a constant: it must not drag gradient into the graph.
    for mask in (a > b, a == b, a <= b):
        assert mask.requires_grad is False
        assert mask._prev == ()


@pytest.mark.parametrize("seed", SEEDS)
def test_where(seed):
    a, b = T((4, 5), seed), T((4, 5), seed + 1)
    cond = rand((4, 5), seed + 2) > 0
    check_gradient(lambda x, y: Tensor.where(cond, x, y), [a, b], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_masked_fill(seed):
    a = T((4, 5), seed)
    mask = rand((4, 5), seed + 2) > 0
    # A moderate fill value keeps the finite difference well conditioned: with
    # the -1e9 used in real attention masking, the projected loss is O(1e9) while
    # the difference quotient's numerator is O(1e-5), so cancellation eats most
    # of float64's mantissa.  The large-value behaviour is exact, not
    # approximate, so it is asserted directly in the test below instead.
    check_gradient(lambda x: x.masked_fill(mask, -3.0), [a], rtol=RTOL, atol=ATOL)


def test_masked_fill_blocks_gradient_at_masked_positions():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    mask = np.array([[False, True, False], [True, True, False]])
    x.masked_fill(mask, -1e9).sum().backward()
    assert x.grad == pytest.approx(np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))


# --------------------------------------------------------------------------- #
# Graph mechanics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", SEEDS)
def test_composite_expressions(seed):
    """Longer chains, where a single wrong adjoint would be diluted but not hidden."""
    a = T((4, 5), seed)
    b = T((5, 3), seed + 1)
    c = T((3,), seed + 2)

    def f(x, y, z):
        # The 0.3 keeps the pre-activation off tanh's saturated tails.  Saturated,
        # every downstream gradient collapses to ~1e-7 and the test degenerates
        # into a measurement of finite-difference round-off.
        h = ((x @ y) * 0.3 + z).tanh()
        h = (h * h + 1.0).log()
        return (h / (h.sum(axis=-1, keepdims=True) + 1.0)).mean(axis=0)

    check_gradient(f, [a, b, c], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_reused_tensor_accumulates(seed):
    """A diamond graph: gradient must sum over every path, not overwrite."""
    a = T((3, 4), seed)
    check_gradient(lambda x: x * x + x.exp() * x, [a], rtol=RTOL, atol=ATOL)

    x = Tensor([2.0], requires_grad=True)
    (x * x * x).backward()          # d/dx x^3 = 3x^2 = 12
    assert x.grad == pytest.approx([12.0])


def test_backward_reaches_deep_chains_without_recursion_limit():
    """The graph is walked iteratively; 5000 nodes must not overflow the stack."""
    x = Tensor([1.0], requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert x.grad == pytest.approx([1.0])


def test_grad_accumulates_across_backward_calls_until_zeroed():
    x = Tensor([3.0], requires_grad=True)
    (x * 2.0).backward()
    (x * 2.0).backward()
    assert x.grad == pytest.approx([4.0])
    x.zero_grad()
    (x * 2.0).backward()
    assert x.grad == pytest.approx([2.0])


def test_requires_grad_propagates_and_stops():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=False)
    assert (a + b).requires_grad is True
    assert (b * b).requires_grad is False
    assert (b * b)._prev == ()          # no tape built for constants
    assert (a.detach() + b).requires_grad is False


def test_no_grad_context_builds_no_tape():
    a = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        out = (a * a).sum()
    assert out.requires_grad is False
    assert out._prev == ()
    # ...and the flag is restored on exit.
    assert (a * a).requires_grad is True


def test_non_scalar_backward_requires_explicit_seed():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    with pytest.raises(RuntimeError, match="non-scalar"):
        (a * 2.0).backward()
    with pytest.raises(ValueError, match="does not match"):
        (a * 2.0).backward(np.ones((5, 5)))


def test_dtype_default_is_restored_by_gradcheck():
    set_default_dtype(np.float32)
    a = Tensor([1.0, 2.0], requires_grad=True)
    check_gradient(lambda x: x * x, [a], rtol=RTOL, atol=ATOL)
    assert Tensor([1.0]).dtype == np.float32


# --------------------------------------------------------------------------- #
# The checker itself must be able to fail
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scale", [0.5, 1.001, 1.00001])
def test_gradient_checker_detects_a_deliberately_wrong_gradient(scale):
    """Guard against a checker that passes everything.

    A checker that never fires would silently bless the entire project, so it is
    pointed at a knowingly broken op and must reject it.  ``scale`` sweeps from a
    blatant 2x error down to a 1e-5 relative error -- ten times the configured
    rtol -- which fixes how much sensitivity the suite actually has.
    """
    def bad_square(x: Tensor) -> Tensor:
        val = x.data * x.data

        def bw(g):
            x._accumulate(g * 2.0 * x.data * scale)

        return Tensor._make(val, (x,), "bad_square", bw)

    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    with pytest.raises(AssertionError, match="gradient check failed"):
        check_gradient(bad_square, [a], rtol=RTOL, atol=ATOL)

    # And a correct version of the same op passes, so the failure above was
    # about the gradient and not about the harness.
    check_gradient(lambda x: x * x, [a], rtol=RTOL, atol=ATOL)


def test_gradient_checker_detects_a_wrong_gradient_on_a_single_entry():
    """A bug confined to one element of a large tensor must not average away."""
    def bad_sum(x: Tensor) -> Tensor:
        def bw(g):
            gx = np.broadcast_to(g, x.data.shape).copy()
            gx.flat[17] *= 1.01              # one entry, 1% off
            x._accumulate(gx)

        return Tensor._make(x.data.sum(), (x,), "bad_sum", bw)

    a = T((5, 8), 0)
    with pytest.raises(AssertionError, match="1/40 entries outside"):
        check_gradient(bad_sum, [a], rtol=RTOL, atol=ATOL)


def test_relative_error_metric():
    assert relative_error(np.array([1.0]), np.array([1.0])) == pytest.approx(0.0)
    assert relative_error(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0)
    assert relative_error(np.array([1.0]), np.array([-1.0])) == pytest.approx(1.0)
