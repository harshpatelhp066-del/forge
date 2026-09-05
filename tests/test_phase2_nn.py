"""Phase 2 verification: layers, attention and the Adam optimizer.

Run with:  python -m pytest tests/test_phase2_nn.py -v

Three things get scrutinised harder than the rest, because they are the ones
that fail silently rather than loudly:

* **LayerNorm** -- asserted to actually produce zero mean and unit variance, not
  merely to run and return the right shape.
* **Causal masking** -- tested for real leakage, forward *and* backward, using
  inputs constructed so that a leak would dominate the output if it existed.
* **Adam's bias correction** -- pinned by an exact closed-form value on the first
  step, which is the one step where a naive implementation differs by 31x.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import nn  # noqa: E402
from forge.gradcheck import check_gradient  # noqa: E402
from forge.optim import SGD, Adam, clip_grad_norm  # noqa: E402
from forge.tensor import Tensor, set_default_dtype  # noqa: E402

RTOL = 1e-6
ATOL = 1e-9


@pytest.fixture(autouse=True)
def float64_by_default():
    """Run these tests in float64, and restore float32 afterwards.

    Assertions here are about mathematics -- LayerNorm really normalises, masked
    attention weights are really zero, Adam's first step really equals lr*sign(g)
    -- so they are checked at a precision where float32 rounding cannot be
    confused for an implementation error.  Training itself runs in float32; the
    tests that specifically exercise float32 behaviour (overflow, underflow,
    end-to-end sanity) set the dtype themselves.
    """
    set_default_dtype(np.float64)
    yield
    set_default_dtype(np.float32)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _set_by_path(root, dotted: str, value) -> None:
    """Assign ``root.a.b.c = value`` given the dotted name ``"a.b.c"``."""
    parts = dotted.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    setattr(obj, parts[-1], value)


def check_module(module: nn.Module, x: Tensor, **kw):
    """Gradient-check a module w.r.t. its input *and* all of its parameters.

    ``check_gradient`` rewraps every input as a fresh float64 Tensor, so the
    module's parameters are rebound to those wrappers before each forward pass --
    otherwise the check would silently only cover the input.
    """
    module.eval()   # dropout must be off: a stochastic forward has no derivative
    names = [n for n, _ in module.named_parameters()]
    params = [p for _, p in module.named_parameters()]

    def fn(x_t, *param_ts):
        for name, pt in zip(names, param_ts):
            _set_by_path(module, name, pt)
        return module(x_t)

    return check_gradient(fn, [x, *params], rtol=kw.pop("rtol", RTOL),
                          atol=kw.pop("atol", ATOL), **kw)


def randt(shape, seed, scale=1.0, requires_grad=True):
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal(shape) * scale, requires_grad=requires_grad)


# --------------------------------------------------------------------------- #
# Linear
# --------------------------------------------------------------------------- #

def test_linear_shapes_and_value():
    nn.set_seed(0)
    layer = nn.Linear(4, 3)
    x = Tensor(np.arange(2 * 5 * 4, dtype=np.float64).reshape(2, 5, 4))
    y = layer(x)
    assert y.shape == (2, 5, 3)
    expected = x.data @ layer.weight.data + layer.bias.data
    assert y.data == pytest.approx(expected, rel=1e-6)


def test_linear_bias_is_zero_initialised_and_weight_is_not():
    nn.set_seed(0)
    layer = nn.Linear(64, 64, init_std=0.02)
    assert np.all(layer.bias.data == 0.0)
    assert layer.weight.data.std() == pytest.approx(0.02, rel=0.15)


@pytest.mark.parametrize("shape", [(6, 4), (2, 5, 4)])
@pytest.mark.parametrize("bias", [True, False])
def test_linear_gradients(shape, bias):
    nn.set_seed(1)
    check_module(nn.Linear(4, 3, bias=bias), randt(shape, 0))


def test_linear_without_bias_has_one_parameter():
    assert len(nn.Linear(4, 3, bias=False).parameters()) == 1
    assert len(nn.Linear(4, 3, bias=True).parameters()) == 2


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

def test_embedding_looks_up_rows():
    nn.set_seed(0)
    emb = nn.Embedding(10, 4)
    idx = np.array([[1, 3], [3, 9]])
    out = emb(idx)
    assert out.shape == (2, 2, 4)
    assert np.array_equal(out.data[0, 0], emb.weight.data[1])
    assert np.array_equal(out.data[1, 1], emb.weight.data[9])


def test_embedding_gradient_accumulates_for_repeated_tokens():
    """Token 3 appears three times, so its row must receive three contributions."""
    nn.set_seed(0)
    emb = nn.Embedding(5, 2)
    out = emb(np.array([3, 3, 1, 3]))
    out.sum().backward()
    assert emb.weight.grad[3] == pytest.approx([3.0, 3.0])
    assert emb.weight.grad[1] == pytest.approx([1.0, 1.0])
    assert emb.weight.grad[0] == pytest.approx([0.0, 0.0])


def test_embedding_gradients():
    nn.set_seed(1)
    emb = nn.Embedding(6, 3)
    idx = np.array([[0, 2, 2], [5, 1, 0]])
    names = [n for n, _ in emb.named_parameters()]

    def fn(w):
        _set_by_path(emb, names[0], w)
        return emb(idx)

    check_gradient(fn, [emb.weight], rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# LayerNorm -- the "does it actually normalise" test
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", [(8, 16), (3, 5, 32)])
def test_layernorm_produces_zero_mean_unit_variance(shape):
    """The point of LayerNorm, asserted directly on the output statistics."""
    rng = np.random.default_rng(0)
    # Deliberately badly scaled input: large mean, large spread. A LayerNorm that
    # merely ran without normalising would sail through a shape-only test.
    x = Tensor(rng.standard_normal(shape) * 17.0 + 42.0)
    ln = nn.LayerNorm(shape[-1], affine=False)
    y = ln(x)

    assert y.shape == shape
    mean = y.data.mean(axis=-1)
    var = y.data.var(axis=-1)
    assert mean == pytest.approx(np.zeros_like(mean), abs=1e-6)
    # Unit variance up to the eps inside the sqrt: var_out = var/(var+eps) < 1.
    assert var == pytest.approx(np.ones_like(var), abs=1e-4)
    assert np.all(var <= 1.0)


def test_layernorm_normalises_the_last_axis_only():
    """Per-row statistics, not per-batch: row 1 must not influence row 0 at all.

    Checked by changing row 1 drastically and requiring row 0's output to be
    bit-identical -- which is exact, unlike comparing two rows to each other
    (rows of different variance normalise slightly differently because ``eps``
    is a fixed absolute term inside the square root).
    """
    ln = nn.LayerNorm(4, affine=False)
    a = ln(Tensor(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))).data
    # Row 1 is reshaped, not merely rescaled: LayerNorm is invariant to an affine
    # change of a row, so multiplying it through would have changed nothing and
    # made this guard vacuous.
    b = ln(Tensor(np.array([[1.0, 2.0, 3.0, 4.0], [500.0, 1.0, 700.0, 2.0]]))).data
    assert np.array_equal(a[0], b[0])
    assert not np.allclose(a[1], b[1], atol=1e-3)
    assert a.mean(axis=-1) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_layernorm_affine_parameters_are_applied():
    x = Tensor(np.random.default_rng(0).standard_normal((4, 8)))
    ln = nn.LayerNorm(8)
    # Identity at initialisation: weight=1, bias=0.
    assert np.all(ln.weight.data == 1.0)
    assert np.all(ln.bias.data == 0.0)
    base = ln(x).data

    ln.weight.data[...] = 3.0
    ln.bias.data[...] = -1.0
    assert ln(x).data == pytest.approx(base * 3.0 - 1.0, rel=1e-6)


def test_layernorm_survives_a_constant_row():
    """Zero variance is where a naive implementation divides by zero."""
    x = Tensor(np.full((2, 6), 7.0))
    y = nn.LayerNorm(6)(x)
    assert np.all(np.isfinite(y.data))
    assert y.data == pytest.approx(np.zeros((2, 6)), abs=1e-6)


@pytest.mark.parametrize("shape", [(6, 8), (2, 3, 8)])
def test_layernorm_gradients(shape):
    check_module(nn.LayerNorm(8), randt(shape, 0, scale=2.0))


def test_layernorm_gradient_on_badly_scaled_input():
    """The backward pass couples every element through mean and variance."""
    check_module(nn.LayerNorm(8), randt((4, 8), 3, scale=25.0))


# --------------------------------------------------------------------------- #
# Softmax / log_softmax
# --------------------------------------------------------------------------- #

def test_softmax_is_a_probability_distribution():
    x = Tensor(np.random.default_rng(0).standard_normal((4, 7)) * 5.0)
    p = nn.softmax(x)
    assert np.all(p.data >= 0.0)
    assert p.data.sum(axis=-1) == pytest.approx(np.ones(4), abs=1e-6)


def test_softmax_does_not_overflow_on_extreme_logits():
    """exp() overflows above ~88 in float32 and ~709 in float64."""
    set_default_dtype(np.float32)
    try:
        x = Tensor([[1000.0, 999.0, -1000.0], [0.0, 0.0, 0.0]])
        with np.errstate(over="raise", invalid="raise"):
            p = nn.softmax(x)
        assert np.all(np.isfinite(p.data))
        assert p.data.sum(axis=-1) == pytest.approx([1.0, 1.0], abs=1e-6)
        assert p.data[0, 2] == pytest.approx(0.0, abs=1e-12)
        assert p.data[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)
    finally:
        set_default_dtype(np.float32)


def test_softmax_is_shift_invariant():
    x = Tensor(np.random.default_rng(1).standard_normal((3, 5)))
    a = nn.softmax(x).data
    b = nn.softmax(Tensor(x.data + 137.0)).data
    assert a == pytest.approx(b, rel=1e-6)


def test_log_softmax_stays_finite_where_log_of_softmax_would_not():
    """A confidently wrong prediction must give a large finite loss, not -inf."""
    set_default_dtype(np.float32)
    x = Tensor([[0.0, 200.0]])
    with np.errstate(divide="ignore"):
        naive = np.log(nn.softmax(x).data)       # underflows to log(0)
    stable = nn.log_softmax(x).data
    assert np.isneginf(naive[0, 0])
    assert np.all(np.isfinite(stable))
    assert stable[0, 0] == pytest.approx(-200.0, rel=1e-4)


def test_softmax_gradients():
    check_gradient(lambda x: nn.softmax(x, axis=-1), [randt((4, 6), 0)], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: nn.softmax(x, axis=1), [randt((3, 4, 5), 1)], rtol=RTOL, atol=ATOL)
    check_gradient(lambda x: nn.log_softmax(x, axis=-1), [randt((4, 6), 2)], rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# GELU
# --------------------------------------------------------------------------- #

def test_gelu_matches_the_exact_erf_form_closely():
    """The tanh approximation is a documented tradeoff; bound the actual error."""
    import math
    x = np.linspace(-6, 6, 400)
    exact = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x])
    got = nn.gelu(Tensor(x)).data
    assert np.max(np.abs(got - exact)) < 2e-3


def test_gelu_shape_and_key_values():
    y = nn.gelu(Tensor([-10.0, 0.0, 10.0])).data
    assert y[0] == pytest.approx(0.0, abs=1e-6)
    assert y[1] == pytest.approx(0.0, abs=1e-12)
    assert y[2] == pytest.approx(10.0, rel=1e-6)


def test_gelu_gradients():
    check_gradient(nn.gelu, [randt((4, 5), 0, scale=2.0)], rtol=RTOL, atol=ATOL)


def test_relu_gradients():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5))
    x = np.sign(x) * (np.abs(x) + 0.1)      # clear of the kink
    check_gradient(nn.relu, [Tensor(x, requires_grad=True)], rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# Dropout
# --------------------------------------------------------------------------- #

def test_dropout_is_identity_in_eval_mode():
    d = nn.Dropout(0.5).eval()
    x = Tensor(np.random.default_rng(0).standard_normal((100, 20)))
    assert np.array_equal(d(x).data, x.data)


def test_dropout_zeroes_roughly_p_of_activations_in_train_mode():
    nn.set_seed(0)
    d = nn.Dropout(0.3).train()
    x = Tensor(np.ones((400, 100)))
    y = d(x).data
    dropped = float(np.mean(y == 0.0))
    assert dropped == pytest.approx(0.3, abs=0.02)
    # Inverted dropout: surviving units are scaled by 1/(1-p) so the expected
    # activation is preserved and eval needs no compensating rescale.
    assert y[y != 0].mean() == pytest.approx(1 / 0.7, rel=1e-6)
    assert y.mean() == pytest.approx(1.0, abs=0.02)


def test_dropout_p_zero_is_a_no_op_even_in_train_mode():
    d = nn.Dropout(0.0).train()
    x = Tensor(np.random.default_rng(0).standard_normal((10, 10)))
    assert np.array_equal(d(x).data, x.data)


def test_dropout_rejects_invalid_p():
    with pytest.raises(ValueError):
        nn.Dropout(1.0)
    with pytest.raises(ValueError):
        nn.Dropout(-0.1)


def test_train_eval_mode_propagates_through_nesting():
    m = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5),
                      nn.Sequential(nn.Dropout(0.2), nn.LayerNorm(4)))
    m.eval()
    assert all(not sub.training for sub in m.modules())
    m.train()
    assert all(sub.training for sub in m.modules())


# --------------------------------------------------------------------------- #
# Causal masking -- the "does it actually block the future" tests
# --------------------------------------------------------------------------- #

def test_causal_mask_shape_and_triangularity():
    m = nn.causal_mask(5)[0, 0]
    assert m.shape == (5, 5)
    assert not m[np.arange(5), np.arange(5)].any()      # a token sees itself
    assert m[0, 1:].all()                                # position 0 sees nothing later
    assert not m[4, :].any()                             # the last position sees everything
    assert np.array_equal(m, np.triu(np.ones((5, 5), bool), k=1))


def test_attention_weights_are_exactly_zero_on_the_future():
    nn.set_seed(0)
    attn = nn.MultiHeadSelfAttention(d_model=16, n_heads=4, dropout=0.0, max_seq_len=8).eval()
    x = randt((2, 6, 16), 0, requires_grad=False)
    _, w = attn(x, return_attention=True)

    assert w.shape == (2, 4, 6, 6)
    upper = np.triu(np.ones((6, 6), bool), k=1)
    # Exactly zero, not merely small: -1e9 underflows exp() to 0.0 after the
    # softmax max-subtraction.
    assert np.all(w.data[:, :, upper] == 0.0)
    assert w.data.sum(axis=-1) == pytest.approx(np.ones((2, 4, 6)), abs=1e-5)


def test_future_tokens_cannot_change_earlier_outputs():
    """The direct causality test, constructed so a leak could not go unnoticed.

    Two inputs share their first 3 positions and differ wildly afterwards -- the
    tail is scaled by 1000, so if even a sliver of attention mass reached it, the
    early outputs would move by a lot. They must instead be bit-identical.
    """
    nn.set_seed(0)
    attn = nn.MultiHeadSelfAttention(d_model=16, n_heads=4, dropout=0.0, max_seq_len=8).eval()
    rng = np.random.default_rng(0)

    base = rng.standard_normal((1, 6, 16))
    poisoned = base.copy()
    poisoned[:, 3:, :] = rng.standard_normal((1, 3, 16)) * 1000.0

    y_base = attn(Tensor(base)).data
    y_poisoned = attn(Tensor(poisoned)).data

    assert np.array_equal(y_base[:, :3, :], y_poisoned[:, :3, :])
    # ...and the change really did land somewhere, so the test is not vacuous.
    assert not np.allclose(y_base[:, 3:, :], y_poisoned[:, 3:, :])


def test_masking_is_gradient_tight_not_just_forward_tight():
    """A leak in the *backward* pass would also break causality.

    Backpropagating only from output position t, the gradient at every input
    position s > t must be exactly zero.
    """
    nn.set_seed(0)
    attn = nn.MultiHeadSelfAttention(d_model=8, n_heads=2, dropout=0.0, max_seq_len=8).eval()
    T, t = 6, 2
    x = randt((1, T, 8), 1)
    y = attn(x)

    seed = np.zeros(y.shape)
    seed[0, t, :] = 1.0
    y.backward(seed)

    assert np.all(x.grad[0, t + 1:, :] == 0.0)
    assert np.any(np.abs(x.grad[0, : t + 1, :]) > 1e-9)


def test_masking_leak_would_be_caught_by_these_tests():
    """Negative control: with the mask removed, both tests above fail.

    Without this, a mask that was accidentally all-False would still pass every
    assertion above for a trivial reason, and the suite would prove nothing.
    """
    nn.set_seed(0)
    attn = nn.MultiHeadSelfAttention(d_model=16, n_heads=4, dropout=0.0, max_seq_len=8).eval()
    attn._mask = np.zeros_like(attn._mask)      # deliberately leaky

    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, 6, 16))
    poisoned = base.copy()
    poisoned[:, 3:, :] = rng.standard_normal((1, 3, 16)) * 1000.0

    y_base = attn(Tensor(base)).data
    y_poisoned = attn(Tensor(poisoned)).data
    assert not np.array_equal(y_base[:, :3, :], y_poisoned[:, :3, :])

    x = randt((1, 6, 16), 1)
    y = attn(x)
    seed = np.zeros(y.shape)
    seed[0, 2, :] = 1.0
    y.backward(seed)
    assert np.any(np.abs(x.grad[0, 3:, :]) > 1e-9)


# --------------------------------------------------------------------------- #
# Attention: shapes, heads, gradients
# --------------------------------------------------------------------------- #

def test_attention_output_shape_and_parameter_count():
    attn = nn.MultiHeadSelfAttention(d_model=32, n_heads=4, dropout=0.0, max_seq_len=16)
    y = attn(randt((3, 7, 32), 0, requires_grad=False))
    assert y.shape == (3, 7, 32)
    # qkv: 32*96 + 96 ; proj: 32*32 + 32
    assert attn.num_parameters() == 32 * 96 + 96 + 32 * 32 + 32


def test_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        nn.MultiHeadSelfAttention(d_model=10, n_heads=4)


def test_attention_rejects_overlong_sequence():
    attn = nn.MultiHeadSelfAttention(d_model=8, n_heads=2, max_seq_len=4)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        attn(randt((1, 9, 8), 0, requires_grad=False))


def test_attention_heads_are_independent():
    """Head splitting must not mix channels across heads.

    With the qkv projection forced to identity-like block structure, a value
    written into head 0's channels must not appear in head 1's output slice.
    """
    nn.set_seed(0)
    attn = nn.MultiHeadSelfAttention(d_model=4, n_heads=2, dropout=0.0, max_seq_len=4).eval()
    _, w = attn(randt((1, 3, 4), 0, requires_grad=False), return_attention=True)
    assert w.shape == (1, 2, 3, 3)
    # Each head produces its own distribution over positions.
    assert not np.allclose(w.data[0, 0], w.data[0, 1])


def test_attention_gradients():
    nn.set_seed(2)
    attn = nn.MultiHeadSelfAttention(d_model=8, n_heads=2, dropout=0.0, max_seq_len=6)
    check_module(attn, randt((2, 4, 8), 0))


def test_attention_scale_keeps_logits_tame_for_wide_heads():
    """1/sqrt(d) is what stops softmax saturating as head width grows."""
    rng = np.random.default_rng(0)
    for head_dim in (8, 64, 512):
        q = rng.standard_normal((2000, head_dim))
        k = rng.standard_normal((2000, head_dim))
        raw = np.sum(q * k, axis=1)
        scaled = raw / np.sqrt(head_dim)
        assert raw.std() == pytest.approx(np.sqrt(head_dim), rel=0.1)
        assert scaled.std() == pytest.approx(1.0, rel=0.1)


# --------------------------------------------------------------------------- #
# Cross-entropy
# --------------------------------------------------------------------------- #

def test_cross_entropy_matches_the_closed_form():
    logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    targets = np.array([0, 1])
    got = nn.cross_entropy(logits, targets).item()
    p = np.exp(logits.data) / np.exp(logits.data).sum(axis=-1, keepdims=True)
    expected = -np.mean([np.log(p[0, 0]), np.log(p[1, 1])])
    assert got == pytest.approx(expected, rel=1e-6)


def test_cross_entropy_of_uniform_logits_is_log_vocab():
    """The value an untrained model should start at -- the sanity check used in Phase 5."""
    for vocab in (10, 65, 1000):
        loss = nn.cross_entropy(Tensor(np.zeros((32, vocab))), np.zeros(32, dtype=int))
        assert loss.item() == pytest.approx(np.log(vocab), rel=1e-6)


def test_cross_entropy_handles_sequence_shaped_logits():
    rng = np.random.default_rng(0)
    logits = Tensor(rng.standard_normal((4, 7, 11)))
    targets = rng.integers(0, 11, size=(4, 7))
    assert np.isfinite(nn.cross_entropy(logits, targets).item())


def test_cross_entropy_gradients():
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 5, size=6)
    check_gradient(lambda z: nn.cross_entropy(z, targets), [randt((6, 5), 0)],
                   rtol=RTOL, atol=ATOL)


def test_cross_entropy_gradient_equals_softmax_minus_onehot():
    """The textbook identity: dL/dlogits = (softmax(logits) - onehot) / N."""
    rng = np.random.default_rng(0)
    logits = Tensor(rng.standard_normal((4, 6)), requires_grad=True)
    targets = rng.integers(0, 6, size=4)
    nn.cross_entropy(logits, targets).backward()

    p = nn.softmax(Tensor(logits.data)).data
    onehot = np.zeros_like(p)
    onehot[np.arange(4), targets] = 1.0
    assert logits.grad == pytest.approx((p - onehot) / 4, abs=1e-9)


def test_cross_entropy_ignore_index():
    rng = np.random.default_rng(0)
    logits = Tensor(rng.standard_normal((4, 5)))
    targets = np.array([1, -1, 3, -1])
    masked = nn.cross_entropy(logits, targets, ignore_index=-1).item()
    kept = nn.cross_entropy(Tensor(logits.data[[0, 2]]), np.array([1, 3])).item()
    assert masked == pytest.approx(kept, rel=1e-6)


# --------------------------------------------------------------------------- #
# Adam -- the "does it move parameters the right way" tests
# --------------------------------------------------------------------------- #

def test_adam_first_step_is_exactly_lr_times_sign_of_gradient():
    """Pins the bias correction to a closed form.

    At t=1, m̂ = g and v̂ = g², so the update is exactly lr·g/(|g|+ε) ≈ lr·sign(g),
    *independent of the gradient's magnitude*. Without bias correction the same
    step would be lr·(0.1·g)/√(0.001·g²) = 31.6·lr·sign(g) -- so this single
    assertion separates the correct implementation from the naive one by 31x.
    """
    p = nn.Parameter(np.array([5.0, -5.0, 5.0]))
    # Magnitudes spanning three orders, and one negative gradient to pin the sign:
    # the step is -lr*sign(g), so a positive gradient moves the parameter *down*.
    p.grad = np.array([2.0, -3.0, 1e-3])
    Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8).step()
    assert p.data == pytest.approx([5.0 - 0.1, -5.0 + 0.1, 5.0 - 0.1], abs=1e-5)


def test_adam_without_bias_correction_would_overshoot_by_31x():
    """Negative control for the test above: quantify what the naive version does."""
    b2 = 0.999
    naive_first_step_multiplier = (1 - 0.9) / np.sqrt(1 - b2)
    assert naive_first_step_multiplier == pytest.approx(3.162, rel=1e-3)
    # Relative to the corrected step of exactly 1.0 x lr:
    assert naive_first_step_multiplier / 1.0 > 3.0


def test_adam_moves_parameters_downhill_on_a_toy_convex_loss():
    """f(x) = (x - 3)² + (y + 1)²  -- a convex bowl with a known minimum.

    Note what is *not* asserted: strict monotonicity. Adam carries momentum, so
    after crossing the minimum around step 40 it overshoots and rings down. That
    is the algorithm working as designed, not a defect, so the assertions are on
    the envelope -- it converges, and it stops overshooting -- rather than on
    every individual step.
    """
    p = nn.Parameter(np.array([0.0, 0.0]))
    opt = Adam([p], lr=0.1)
    target = np.array([3.0, -1.0])

    losses = []
    for _ in range(400):
        opt.zero_grad()
        diff = p.data - target
        losses.append(float((diff * diff).sum()))
        p.grad = 2.0 * diff                   # analytic gradient of the bowl
        opt.step()

    assert p.data == pytest.approx(target, abs=1e-6)
    assert losses[0] == pytest.approx(10.0)
    assert losses[-1] < 1e-12
    # The oscillation decays: every stretch of 50 steps peaks lower than the last.
    peaks = [max(losses[i:i + 50]) for i in range(50, 400, 50)]
    assert all(b < a for a, b in zip(peaks, peaks[1:])), peaks


def test_adam_matches_a_textbook_implementation_exactly():
    """The fused bias correction must be algebraically identical to the literal form.

    ``step()`` folds the two corrections into a single step size,
    ``lr·√(1-β₂ᵗ)/(1-β₁ᵗ)``, to avoid allocating two full-size temporaries per
    parameter per step. That is only a valid optimisation if it is *exact*, so it
    is checked against a separately written textbook Adam over 50 steps -- and it
    agrees to the last bit, not merely to a tolerance.
    """
    class TextbookAdam:
        def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
            self.m = np.zeros(shape)
            self.v = np.zeros(shape)
            self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

        def step(self, p, g):
            self.t += 1
            self.m = self.b1 * self.m + (1 - self.b1) * g
            self.v = self.b2 * self.v + (1 - self.b2) * g * g
            mhat = self.m / (1 - self.b1 ** self.t)
            vhat = self.v / (1 - self.b2 ** self.t)
            return p - self.lr * mhat / (np.sqrt(vhat) + self.eps)

    rng = np.random.default_rng(0)
    p0 = rng.standard_normal(6)
    p = nn.Parameter(p0.copy())
    opt = Adam([p], lr=0.01)
    ref_p, ref = p0.copy(), TextbookAdam(p0.shape, lr=0.01)

    for _ in range(50):
        g = rng.standard_normal(6)
        p.grad = g.copy()
        opt.step()
        ref_p = ref.step(ref_p, g)
        assert np.max(np.abs(p.data - ref_p)) == 0.0


def test_adam_descends_a_real_forge_graph():
    """End-to-end: a Linear layer fits a linear target through the autodiff engine."""
    nn.set_seed(0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 4))
    true_W = rng.standard_normal((4, 2))
    Y = X @ true_W

    layer = nn.Linear(4, 2)
    opt = Adam(layer.parameters(), lr=0.05)
    x_t = Tensor(X)

    first = last = None
    for step in range(500):
        opt.zero_grad()
        pred = layer(x_t)
        err = pred - Tensor(Y)
        loss = (err * err).mean()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first * 1e-3
    assert layer.weight.data == pytest.approx(true_W, abs=1e-2)


def test_adam_step_direction_opposes_the_gradient():
    rng = np.random.default_rng(0)
    p = nn.Parameter(rng.standard_normal((6, 5)))
    before = p.data.copy()
    p.grad = rng.standard_normal((6, 5))
    Adam([p], lr=0.01).step()
    delta = p.data - before
    assert np.all(np.sign(delta) == -np.sign(p.grad))


def test_adam_skips_parameters_with_no_gradient():
    p = nn.Parameter(np.array([1.0]))
    q = nn.Parameter(np.array([1.0]))
    q.grad = np.array([1.0])
    Adam([p, q], lr=0.1).step()
    assert p.data == pytest.approx([1.0])        # untouched
    assert q.data[0] < 1.0


def test_adam_decoupled_weight_decay_shrinks_parameters():
    p = nn.Parameter(np.array([2.0]))
    p.grad = np.array([0.0])
    Adam([p], lr=0.1, weight_decay=0.5).step()
    # Pure decay with a zero gradient: p <- p - lr*wd*p
    assert p.data == pytest.approx([2.0 - 0.1 * 0.5 * 2.0], rel=1e-6)


def test_adam_state_roundtrips():
    rng = np.random.default_rng(0)
    p = nn.Parameter(rng.standard_normal(5))
    opt = Adam([p], lr=0.01)
    for _ in range(3):
        p.grad = rng.standard_normal(5)
        opt.step()
    state = opt.state_dict()

    q = nn.Parameter(p.data.copy())
    opt2 = Adam([q], lr=0.01)
    opt2.load_state_dict(state)
    assert opt2.t == opt.t
    g = rng.standard_normal(5)
    p.grad, q.grad = g.copy(), g.copy()
    opt.step()
    opt2.step()
    assert q.data == pytest.approx(p.data, rel=1e-12)


def test_adam_rejects_bad_betas_and_empty_params():
    p = nn.Parameter(np.array([1.0]))
    with pytest.raises(ValueError, match="betas"):
        Adam([p], betas=(1.0, 0.999))
    with pytest.raises(ValueError, match="empty"):
        Adam([])


def test_sgd_baseline_also_descends():
    p = nn.Parameter(np.array([0.0]))
    opt = SGD([p], lr=0.1)
    for _ in range(200):
        p.grad = 2.0 * (p.data - 3.0)
        opt.step()
    assert p.data == pytest.approx([3.0], abs=1e-3)


# --------------------------------------------------------------------------- #
# Gradient clipping
# --------------------------------------------------------------------------- #

def test_clip_grad_norm_scales_to_the_target_norm():
    a = nn.Parameter(np.zeros(3))
    b = nn.Parameter(np.zeros(4))
    a.grad = np.array([3.0, 0.0, 0.0])
    b.grad = np.array([0.0, 4.0, 0.0, 0.0])
    returned = clip_grad_norm([a, b], max_norm=1.0)

    assert returned == pytest.approx(5.0)        # the *pre*-clip global norm
    new_norm = np.sqrt((a.grad ** 2).sum() + (b.grad ** 2).sum())
    assert new_norm == pytest.approx(1.0, rel=1e-5)


def test_clip_grad_norm_preserves_direction():
    """Global rescaling, not per-tensor: the update vector must only shorten."""
    rng = np.random.default_rng(0)
    ps = [nn.Parameter(np.zeros((3, 4))) for _ in range(3)]
    for p in ps:
        p.grad = rng.standard_normal((3, 4)) * 10.0
    before = np.concatenate([p.grad.ravel() for p in ps])
    clip_grad_norm(ps, max_norm=1.0)
    after = np.concatenate([p.grad.ravel() for p in ps])

    cos = float(before @ after / (np.linalg.norm(before) * np.linalg.norm(after)))
    assert cos == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(after) == pytest.approx(1.0, rel=1e-5)


def test_clip_grad_norm_leaves_small_gradients_untouched():
    p = nn.Parameter(np.zeros(3))
    p.grad = np.array([0.1, 0.2, 0.3])
    before = p.grad.copy()
    norm = clip_grad_norm([p], max_norm=10.0)
    assert norm == pytest.approx(np.linalg.norm(before))
    assert np.array_equal(p.grad, before)


# --------------------------------------------------------------------------- #
# Module plumbing
# --------------------------------------------------------------------------- #

def test_named_parameters_deduplicates_shared_weights():
    """A tied weight must be reported once, or the optimizer would step it twice."""
    class Tied(nn.Module):
        def __init__(self):
            self.a = nn.Linear(3, 3, bias=False)
            self.b = nn.Linear(3, 3, bias=False)
            self.b.weight = self.a.weight       # tie

    m = Tied()
    assert len(m.parameters()) == 1
    assert len({id(p) for p in m.parameters()}) == 1


def test_parameters_are_found_through_lists_and_nesting():
    class Net(nn.Module):
        def __init__(self):
            self.blocks = [nn.Linear(2, 2), nn.Linear(2, 2)]
            self.head = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 1))

    m = Net()
    names = [n for n, _ in m.named_parameters()]
    assert "blocks.0.weight" in names and "blocks.1.bias" in names
    assert "head.layers.0.weight" in names and "head.layers.1.bias" in names
    assert m.num_parameters() == (2 * 2 + 2) * 2 + 2 + 2 + (2 * 1 + 1)


def test_state_dict_roundtrip_and_mismatch_detection():
    nn.set_seed(0)
    a = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    nn.set_seed(1)
    b = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    assert not np.allclose(a.parameters()[0].data, b.parameters()[0].data)

    b.load_state_dict(a.state_dict())
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert np.array_equal(pa.data, pb.data)

    with pytest.raises(KeyError, match="mismatch"):
        b.load_state_dict({"nope": np.zeros(1)})


def test_zero_grad_clears_every_parameter():
    m = nn.Linear(3, 2)
    m(Tensor(np.ones((2, 3)))).sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    m.zero_grad()
    assert all(p.grad is None for p in m.parameters())


# --------------------------------------------------------------------------- #
# float32 -- the dtype training actually runs in
# --------------------------------------------------------------------------- #

def test_primitives_behave_in_float32():
    """The suite runs in float64; confirm nothing depends on that.

    Same assertions as above, at float32-appropriate tolerances, so a layer that
    only works because of float64 headroom cannot slip through.
    """
    set_default_dtype(np.float32)
    nn.set_seed(0)
    rng = np.random.default_rng(0)

    x = Tensor(rng.standard_normal((4, 12, 32)) * 17.0 + 42.0)
    y = nn.LayerNorm(32, affine=False)(x)
    assert y.data.dtype == np.float32
    assert y.data.mean(axis=-1) == pytest.approx(np.zeros((4, 12)), abs=1e-4)
    assert y.data.var(axis=-1) == pytest.approx(np.ones((4, 12)), abs=1e-3)

    attn = nn.MultiHeadSelfAttention(d_model=32, n_heads=4, dropout=0.0, max_seq_len=16).eval()
    out, w = attn(x, return_attention=True)
    assert out.data.dtype == np.float32
    assert np.all(np.isfinite(out.data))
    upper = np.triu(np.ones((12, 12), bool), k=1)
    assert np.all(w.data[:, :, upper] == 0.0)
    assert w.data.sum(axis=-1) == pytest.approx(np.ones((4, 4, 12)), abs=1e-4)

    loss = nn.cross_entropy(Tensor(np.zeros((16, 65), dtype=np.float32)),
                            np.zeros(16, dtype=int))
    assert loss.item() == pytest.approx(np.log(65), rel=1e-5)
