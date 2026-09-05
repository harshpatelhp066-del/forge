"""Phase 5 verification: schedule, clipping, checkpointing, evaluation, logging.

Run with:  python -m pytest tests/test_phase5_train.py -v

Also includes the fused-vs-composed equivalence checks for GELU and softmax,
since the fused versions are the memory optimisation that made a real training
run fit in RAM and their correctness is what everything downstream rests on.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import nn  # noqa: E402
from forge.data import DataLoader  # noqa: E402
from forge.gradcheck import check_gradient  # noqa: E402
from forge.model import GPT, GPTConfig  # noqa: E402
from forge.optim import Adam, clip_grad_norm  # noqa: E402
from forge.tensor import Tensor, set_default_dtype  # noqa: E402
from forge.train import (CosineWarmupSchedule, LossLogger,  # noqa: E402
                         estimate_loss, format_duration, load_checkpoint,
                         plot_loss_curve, save_checkpoint)


@pytest.fixture(autouse=True)
def float64_by_default():
    set_default_dtype(np.float64)
    yield
    set_default_dtype(np.float32)


# --------------------------------------------------------------------------- #
# Fused ops must equal the composed reference
# --------------------------------------------------------------------------- #

def test_fused_gelu_matches_the_composed_version_in_value_and_gradient():
    """The fused op is a memory optimisation; it must change nothing else."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 40)) * 3.0

    a = Tensor(data, requires_grad=True)
    b = Tensor(data.copy(), requires_grad=True)
    ya, yb = nn.gelu(a), nn.gelu_composed(b)
    assert ya.data == pytest.approx(yb.data, rel=1e-12, abs=1e-14)

    seed = rng.standard_normal(ya.shape)
    ya.backward(seed)
    yb.backward(seed)
    assert a.grad == pytest.approx(b.grad, rel=1e-10, abs=1e-14)


def test_fused_softmax_matches_the_composed_version_in_value_and_gradient():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, 7, 9)) * 4.0

    a = Tensor(data, requires_grad=True)
    b = Tensor(data.copy(), requires_grad=True)
    ya, yb = nn.softmax(a, axis=-1), nn.softmax_composed(b, axis=-1)
    assert ya.data == pytest.approx(yb.data, rel=1e-12, abs=1e-14)

    seed = rng.standard_normal(ya.shape)
    ya.backward(seed)
    yb.backward(seed)
    assert a.grad == pytest.approx(b.grad, rel=1e-9, abs=1e-14)


def test_fused_ops_pass_the_finite_difference_checker():
    """Hand-derived gradients get the same scrutiny as every Phase 1 primitive."""
    rng = np.random.default_rng(2)
    x = Tensor(rng.standard_normal((5, 6)) * 2.0, requires_grad=True)
    check_gradient(nn.gelu, [x], rtol=1e-6, atol=1e-9)
    check_gradient(lambda t: nn.softmax(t, axis=-1), [x], rtol=1e-6, atol=1e-9)
    check_gradient(lambda t: nn.softmax(t, axis=0), [x], rtol=1e-6, atol=1e-9)


def test_fused_softmax_is_still_overflow_safe():
    set_default_dtype(np.float32)
    x = Tensor([[1000.0, 999.0, -1000.0]])
    with np.errstate(over="raise", invalid="raise"):
        p = nn.softmax(x)
    assert np.all(np.isfinite(p.data))
    assert p.data.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Freeing intermediate gradients must not change any result
# --------------------------------------------------------------------------- #

def test_freeing_intermediate_grads_leaves_leaf_grads_identical():
    """The memory optimisation in backward() must be numerically invisible."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 5))

    def run(retain):
        x = Tensor(data.copy(), requires_grad=True)
        y = ((x * x).exp() / (x.abs() + 2.0)).sum()
        y.backward(retain_grads=retain)
        return x.grad

    assert np.array_equal(run(True), run(False))


def test_intermediates_release_their_grads_but_leaves_keep_theirs():
    x = Tensor([1.0, 2.0], requires_grad=True)
    mid = x * 3.0
    out = (mid * mid).sum()
    out.backward()
    assert x.grad is not None            # leaf keeps it
    assert mid.grad is None              # intermediate released

    x.zero_grad()
    mid2 = x * 3.0
    (mid2 * mid2).sum().backward(retain_grads=True)
    assert mid2.grad is not None         # ...unless asked to retain


def test_backward_closures_do_not_capture_their_own_output():
    """Regression guard for the bug that killed the first training run.

    A closure written as ``def bw(): ... out.grad ...`` captures the output
    tensor, so every node forms an ``out -> closure -> out`` reference cycle.
    Reference counting can never break a cycle, so the whole graph -- every
    activation buffer in it -- survives until the cyclic collector happens to
    run, and with multi-megabyte activations that is an out-of-memory crash.
    The contract is that a backward takes the gradient as an argument instead.
    """
    a = Tensor(np.ones(4), requires_grad=True)
    b = Tensor(np.ones(4), requires_grad=True)
    for out in (a + b, a * b, a / b, a @ b, (a * b).sum(), a.exp(),
                a.reshape(2, 2), a[0:2], nn.gelu(a), nn.softmax(a)):
        captured = [c.cell_contents for c in (out._backward.__closure__ or ())]
        assert not any(c is out for c in captured), f"{out._op} captures its output"


def test_a_graph_is_freed_by_refcounting_alone():
    """With cycles gone, dropping the loss must release the graph immediately."""
    import gc
    gc.collect()
    gc.disable()
    try:
        baseline = len(gc.get_objects())
        for _ in range(30):
            x = Tensor(np.ones((32, 32)), requires_grad=True)
            y = (nn.gelu(x * x) + x).sum()
            y.backward()
            del x, y
        leaked = len(gc.get_objects()) - baseline
    finally:
        gc.enable()
    # A handful of objects may linger from interpreter bookkeeping; hundreds
    # would mean the graphs are still cyclic.
    assert leaked < 50, f"{leaked} objects survived without the cyclic collector"


def test_model_gradients_are_unchanged_by_the_memory_optimisation():
    nn.set_seed(0)
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2,
                    d_model=16, dropout=0.0)
    idx = np.random.default_rng(0).integers(0, 16, size=(2, 8))

    grads = []
    for retain in (True, False):
        nn.set_seed(0)
        model = GPT(cfg).eval()
        _, loss = model(idx, targets=idx)
        loss.backward(retain_grads=retain)
        grads.append([p.grad.copy() for p in model.parameters()])

    for a, b in zip(*grads):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Learning-rate schedule
# --------------------------------------------------------------------------- #

def test_schedule_warms_up_linearly_then_decays_to_the_floor():
    s = CosineWarmupSchedule(base_lr=1e-3, warmup_steps=100, total_steps=1000,
                             min_lr_ratio=0.1)
    assert s(0) == pytest.approx(1e-5)              # (0+1)/100 of base
    assert s(49) == pytest.approx(5e-4)
    assert s(99) == pytest.approx(1e-3)             # peak at end of warmup
    # Cosine midpoint sits halfway between base and floor.
    assert s(550) == pytest.approx((1e-3 + 1e-4) / 2, rel=1e-6)
    assert s(999) == pytest.approx(1e-4, rel=1e-3)  # floor
    assert s(5000) == pytest.approx(1e-4)           # clamped past the end


def test_schedule_is_monotone_after_the_peak_and_never_negative():
    s = CosineWarmupSchedule(1e-3, 100, 1000, 0.1)
    rates = [s(i) for i in range(1000)]
    assert all(r > 0 for r in rates)
    assert rates[:100] == sorted(rates[:100])              # rising through warmup
    assert rates[99:] == sorted(rates[99:], reverse=True)  # falling after
    assert max(rates) == pytest.approx(1e-3)


def test_schedule_floor_is_a_ratio_of_base_not_zero():
    """A zero terminal rate wastes the final steps."""
    s = CosineWarmupSchedule(2e-3, 10, 100, min_lr_ratio=0.05)
    assert s(99) == pytest.approx(1e-4, rel=1e-2)
    assert s(99) > 0


def test_schedule_rejects_impossible_configurations():
    with pytest.raises(ValueError, match="must be <"):
        CosineWarmupSchedule(1e-3, warmup_steps=500, total_steps=100)
    with pytest.raises(ValueError):
        CosineWarmupSchedule(1e-3, warmup_steps=-1, total_steps=100)
    with pytest.raises(ValueError):
        CosineWarmupSchedule(1e-3, warmup_steps=0, total_steps=0)


def test_zero_warmup_starts_at_the_base_rate():
    s = CosineWarmupSchedule(1e-3, warmup_steps=0, total_steps=100)
    assert s(0) == pytest.approx(1e-3)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def _small_model(seed=0):
    nn.set_seed(seed)
    return GPT(GPTConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2,
                         d_model=16, dropout=0.0))


def test_checkpoint_roundtrip_restores_logits_exactly(tmp_path):
    a, b = _small_model(0).eval(), _small_model(1).eval()
    idx = np.random.default_rng(0).integers(0, 32, size=(2, 8))
    assert not np.allclose(a(idx).data, b(idx).data)

    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, a, step=42, meta={"val_loss": 1.25})
    meta = load_checkpoint(path, b)

    assert meta["step"] == 42
    assert meta["val_loss"] == 1.25
    assert meta["config"]["n_layer"] == 2
    assert np.array_equal(a(idx).data, b(idx).data)


def test_checkpoint_roundtrips_optimizer_state_so_training_resumes_identically(tmp_path):
    """A resumed run must take the same next step as an uninterrupted one."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 32, size=(2, 8))

    model = _small_model(0)
    opt = Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        _, loss = model(idx, targets=idx)
        loss.backward()
        opt.step()

    path = tmp_path / "resume.npz"
    save_checkpoint(path, model, opt, step=5)

    # Continue the original...
    opt.zero_grad()
    _, loss = model(idx, targets=idx)
    loss.backward()
    opt.step()
    expected = [p.data.copy() for p in model.parameters()]

    # ...and the restored copy.
    model2 = _small_model(9)
    opt2 = Adam(model2.parameters(), lr=1e-3)
    load_checkpoint(path, model2, opt2)
    assert opt2.t == 5
    opt2.zero_grad()
    _, loss2 = model2(idx, targets=idx)
    loss2.backward()
    opt2.step()

    for a, b in zip(expected, [p.data for p in model2.parameters()]):
        assert a == pytest.approx(b, rel=1e-12)


def test_checkpoint_write_is_atomic(tmp_path):
    """A save must never leave a half-written file where a good one was."""
    path = tmp_path / "ckpt.npz"
    model = _small_model(0)
    save_checkpoint(path, model, step=1)
    first = path.read_bytes()

    save_checkpoint(path, model, step=2)
    assert not (tmp_path / "ckpt.npz.tmp").exists()      # temp file cleaned up
    assert len(list(tmp_path.iterdir())) == 1
    assert load_checkpoint(path)["step"] == 2
    assert len(first) > 0


def test_checkpoint_carries_enough_config_to_rebuild_the_model(tmp_path):
    """Generation loads a checkpoint without being told the architecture."""
    path = tmp_path / "c.npz"
    original = _small_model(3).eval()
    save_checkpoint(path, original, step=7)

    meta = load_checkpoint(path)
    rebuilt = GPT(GPTConfig(**meta["config"]))
    load_checkpoint(path, rebuilt)
    rebuilt.eval()

    idx = np.random.default_rng(0).integers(0, 32, size=(1, 8))
    assert np.array_equal(original(idx).data, rebuilt(idx).data)


def test_load_checkpoint_rejects_a_mismatched_architecture(tmp_path):
    path = tmp_path / "c.npz"
    save_checkpoint(path, _small_model(0), step=0)
    nn.set_seed(0)
    wrong = GPT(GPTConfig(vocab_size=32, block_size=8, n_layer=3, n_head=2,
                          d_model=16, dropout=0.0))
    with pytest.raises(KeyError, match="mismatch"):
        load_checkpoint(path, wrong)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def test_estimate_loss_runs_in_eval_mode_and_restores_training_mode():
    model = _small_model(0)
    model.train()
    loader = DataLoader(np.arange(2000) % 32, block_size=8, batch_size=4)
    estimate_loss(model, loader, "val", n_batches=2)
    assert model.training is True

    model.eval()
    estimate_loss(model, loader, "val", n_batches=2)
    assert model.training is False


def test_estimate_loss_builds_no_graph_and_leaves_no_gradients():
    model = _small_model(0)
    model.zero_grad()
    loader = DataLoader(np.arange(2000) % 32, block_size=8, batch_size=4)
    estimate_loss(model, loader, "train", n_batches=3)
    assert all(p.grad is None for p in model.parameters())


def test_estimate_loss_is_reproducible_for_a_fixed_seed():
    """Comparable across checkpoints only if it evaluates the same batches."""
    model = _small_model(0).eval()
    loader = DataLoader(np.arange(4000) % 32, block_size=8, batch_size=4)
    a = estimate_loss(model, loader, "val", n_batches=5, seed=3)
    b = estimate_loss(model, loader, "val", n_batches=5, seed=3)
    c = estimate_loss(model, loader, "val", n_batches=5, seed=4)
    assert a == pytest.approx(b, rel=1e-12)
    assert a != pytest.approx(c, rel=1e-12)


def test_estimate_loss_does_not_disturb_the_loaders_rng_stream():
    """Evaluation must not perturb the sequence of training batches."""
    loader = DataLoader(np.arange(4000) % 32, block_size=8, batch_size=4, seed=0)
    model = _small_model(0).eval()
    before = loader.random_batch("train")[0].copy()

    loader2 = DataLoader(np.arange(4000) % 32, block_size=8, batch_size=4, seed=0)
    estimate_loss(model, loader2, "val", n_batches=3, seed=99)
    after = loader2.random_batch("train")[0]
    assert np.array_equal(before, after)


def test_untrained_model_evaluates_near_the_uniform_baseline():
    nn.set_seed(0)
    V = 64
    model = GPT(GPTConfig(vocab_size=V, block_size=8, n_layer=2, n_head=2,
                          d_model=32, dropout=0.0)).eval()
    rng = np.random.default_rng(0)
    loader = DataLoader(rng.integers(0, V, size=4000), block_size=8, batch_size=8)
    assert estimate_loss(model, loader, "val", 5) == pytest.approx(np.log(V), rel=0.05)


# --------------------------------------------------------------------------- #
# Logging and plotting
# --------------------------------------------------------------------------- #

def test_loss_logger_writes_a_readable_csv(tmp_path):
    path = tmp_path / "loss.csv"
    with LossLogger(path) as log:
        log.log(step=0, train_loss=6.9, lr=1e-4, grad_norm=2.0, elapsed_s=0.1, tokens=4096)
        log.log(step=1, train_loss=6.5, val_loss=6.4, lr=2e-4, grad_norm=1.5,
                elapsed_s=0.2, tokens=8192)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["step"] for r in rows] == ["0", "1"]
    assert rows[0]["val_loss"] == ""            # blank, not absent
    assert float(rows[1]["val_loss"]) == 6.4
    assert set(rows[0]) == set(LossLogger.FIELDS)


def test_loss_logger_flushes_each_row_so_an_interrupted_run_keeps_its_curve(tmp_path):
    path = tmp_path / "loss.csv"
    log = LossLogger(path)
    log.log(step=0, train_loss=6.9, lr=1e-4, grad_norm=1.0, elapsed_s=0.0, tokens=1)
    # Read without closing, as a crash would.
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
    log.close()


def test_plot_loss_curve_writes_a_png(tmp_path):
    csv_path = tmp_path / "loss.csv"
    with LossLogger(csv_path) as log:
        for i in range(40):
            log.log(step=i, train_loss=6.9 - i * 0.1,
                    val_loss=(6.9 - i * 0.09) if i % 10 == 0 else "",
                    lr=1e-3, grad_norm=1.0, elapsed_s=i, tokens=i * 4096)
    out = tmp_path / "curve.png"
    plot_loss_curve(csv_path, out, title="test")
    assert out.exists() and out.stat().st_size > 5000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_format_duration():
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m05s"
    assert format_duration(3725) == "1h02m05s"


# --------------------------------------------------------------------------- #
# The training step, end to end
# --------------------------------------------------------------------------- #

def test_one_training_step_updates_every_parameter():
    set_default_dtype(np.float32)
    nn.set_seed(0)
    model = _small_model(0)
    opt = Adam(model.parameters(), lr=1e-3)
    idx = np.random.default_rng(0).integers(0, 32, size=(2, 8))
    before = [p.data.copy() for p in model.parameters()]

    opt.zero_grad()
    _, loss = model(idx, targets=idx)
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()

    for name, (b, p) in zip([n for n, _ in model.named_parameters()],
                            zip(before, model.parameters())):
        assert not np.array_equal(b, p.data), f"{name} did not move"


def test_gradient_clipping_engages_on_a_real_model_and_preserves_direction():
    set_default_dtype(np.float32)
    nn.set_seed(0)
    model = _small_model(0)
    idx = np.random.default_rng(0).integers(0, 32, size=(4, 8))
    _, loss = model(idx, targets=idx)
    loss.backward()

    before = np.concatenate([p.grad.ravel() for p in model.parameters()])
    norm = clip_grad_norm(model.parameters(), 0.01)     # deliberately tiny
    after = np.concatenate([p.grad.ravel() for p in model.parameters()])

    assert norm > 0.01                                   # clipping really engaged
    assert np.linalg.norm(after) == pytest.approx(0.01, rel=1e-4)
    cos = float(before @ after / (np.linalg.norm(before) * np.linalg.norm(after)))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_short_training_run_decreases_loss_on_real_batches():
    """A miniature version of the real run: loss must fall well below ln(V)."""
    set_default_dtype(np.float32)
    nn.set_seed(0)
    V = 32
    rng = np.random.default_rng(0)
    # A learnable pattern: token t is always followed by (t*7+3) mod V.
    seq = np.zeros(6000, dtype=np.int64)
    seq[0] = 1
    for i in range(1, len(seq)):
        seq[i] = (seq[i - 1] * 7 + 3) % V
    loader = DataLoader(seq, block_size=16, batch_size=16, seed=0)

    model = GPT(GPTConfig(vocab_size=V, block_size=16, n_layer=2, n_head=2,
                          d_model=32, dropout=0.0))
    opt = Adam(model.parameters(), lr=3e-3)
    sched = CosineWarmupSchedule(3e-3, 20, 200)

    first = None
    for step in range(200):
        opt.lr = sched(step)
        x, y = loader.random_batch("train")
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()

    val = estimate_loss(model, loader, "val", 5)
    assert first == pytest.approx(np.log(V), rel=0.2)
    assert val < 0.1, f"did not learn the pattern: {first:.3f} -> {val:.3f}"


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_non_finite_gradient_is_reported_without_corrupting_the_buffers(bad):
    """The guard the training loop relies on to skip a bad step.

    Scaling is deliberately *not* attempted: max_norm/inf is 0, and inf*0 is NaN,
    so a naive clip would convert one bad entry into a whole tensor of NaN. The
    gradients are left exactly as they were and the caller skips the step.
    """
    p = nn.Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([bad, 1.0])
    with np.errstate(invalid="raise", over="raise"):
        norm = clip_grad_norm([p], 1.0)
    assert not np.isfinite(norm)
    assert np.all(np.isfinite(p.data))            # parameters untouched
    assert p.grad[1] == 1.0                       # the good entry is not NaN-ed
