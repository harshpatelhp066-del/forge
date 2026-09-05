"""Phase 3 verification: the assembled GPT.

Run with:  python -m pytest tests/test_phase3_model.py -v

Covers the forward-pass contract the prompt asks for -- output shapes, a valid
probability distribution after softmax, no NaNs or Infs on realistic inputs --
plus the structural properties that are easy to get wrong when wiring blocks
together: causality end-to-end, weight tying, residual paths, and configurability.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import nn  # noqa: E402
from forge.model import GPT, GPTConfig  # noqa: E402
from forge.optim import Adam  # noqa: E402
from forge.tensor import Tensor, set_default_dtype  # noqa: E402


def tiny(**over) -> GPTConfig:
    base = dict(vocab_size=17, block_size=12, n_layer=2, n_head=2,
                d_model=16, dropout=0.0)
    base.update(over)
    return GPTConfig(**base)


def make(seed=0, **over):
    nn.set_seed(seed)
    return GPT(tiny(**over))


def tokens(B, T, vocab, seed=0):
    return np.random.default_rng(seed).integers(0, vocab, size=(B, T))


# --------------------------------------------------------------------------- #
# Forward pass contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("B,T", [(1, 1), (2, 5), (4, 12), (3, 7)])
def test_forward_output_shape(B, T):
    cfg = tiny()
    model = make()
    logits = model(tokens(B, T, cfg.vocab_size))
    assert logits.shape == (B, T, cfg.vocab_size)


def test_logits_form_a_valid_probability_distribution_after_softmax():
    cfg = tiny()
    model = make().eval()
    logits = model(tokens(4, 9, cfg.vocab_size))
    probs = nn.softmax(logits, axis=-1).data

    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert probs.sum(axis=-1) == pytest.approx(np.ones((4, 9)), abs=1e-5)


def test_no_nan_or_inf_on_realistic_inputs():
    """Untrained model, several shapes, both train and eval mode."""
    cfg = tiny()
    model = make()
    for mode in ("train", "eval"):
        getattr(model, mode)()
        for B, T in [(1, 1), (2, 12), (8, 6)]:
            logits = model(tokens(B, T, cfg.vocab_size, seed=B * T))
            assert np.all(np.isfinite(logits.data)), f"non-finite in {mode} at {(B, T)}"


def test_no_nan_or_inf_with_a_deeper_wider_model():
    """Depth is where activations blow up if the residual init is wrong."""
    nn.set_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=8, n_head=8,
                          d_model=64, dropout=0.0)).eval()
    logits = model(tokens(2, 32, 64))
    assert np.all(np.isfinite(logits.data))
    # Logits should be in a sane range at init, not saturated.
    assert np.abs(logits.data).max() < 50.0


def test_untrained_loss_is_close_to_log_vocab():
    """An untrained model should be near-uniform, i.e. loss ~= ln(V).

    This is the single most useful sanity check on a fresh transformer: it fails
    loudly if the initialisation scale is wrong or the output head is mis-wired.

    The targets are the true next tokens.  They must *not* be the input tokens
    themselves -- see the test below for why that would measure something else.
    """
    cfg = tiny(vocab_size=65, block_size=16, d_model=32)
    nn.set_seed(0)
    model = GPT(cfg).eval()
    idx = tokens(8, 17, cfg.vocab_size)
    _, loss = model(idx[:, :-1], targets=idx[:, 1:])
    assert loss.item() == pytest.approx(np.log(cfg.vocab_size), rel=0.05)


def test_tied_embeddings_bias_the_model_towards_predicting_the_current_token():
    """A real consequence of weight tying, isolated and measured.

    With tying, the output projection dots the residual stream against the same
    matrix the token embedding came from.  The stream still carries a large
    component of the input token's own embedding, so `logit[current_token]` gets
    a systematic self-similarity boost: at initialisation the model is already
    better than chance at "predict the token you just saw", scoring ~3.79 against
    ln(65) = 4.17.

    This is not a bug, and it is arguably why tying helps -- but it means an
    untrained-loss sanity check must use genuine next-token targets, or it will
    measure this artefact instead of the initialisation.  Untying removes the
    effect entirely, which is what makes the cause unambiguous.
    """
    cfg_kw = dict(vocab_size=65, block_size=16, n_layer=2, n_head=2,
                  d_model=32, dropout=0.0)
    idx = tokens(8, 16, 65, seed=0)
    independent = tokens(8, 16, 65, seed=1)

    nn.set_seed(0)
    tied = GPT(GPTConfig(**cfg_kw, tie_embeddings=True)).eval()
    nn.set_seed(0)
    untied = GPT(GPTConfig(**cfg_kw, tie_embeddings=False)).eval()

    ln_v = np.log(65)
    # Tied: predicting the current token is easier than chance...
    assert tied(idx, targets=idx)[1].item() < ln_v - 0.25
    # ...but on unrelated targets it is exactly at chance.
    assert tied(idx, targets=independent)[1].item() == pytest.approx(ln_v, rel=0.05)
    # Untied: no self-similarity path, so no advantage either way.
    assert untied(idx, targets=idx)[1].item() == pytest.approx(ln_v, rel=0.05)


def test_forward_with_targets_returns_logits_and_scalar_loss():
    cfg = tiny()
    model = make()
    idx = tokens(2, 6, cfg.vocab_size)
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert loss.shape == ()
    assert np.isfinite(loss.item())


def test_forward_rejects_sequences_longer_than_the_context():
    cfg = tiny()
    model = make()
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(tokens(1, cfg.block_size + 1, cfg.vocab_size))


def test_forward_accepts_exactly_block_size():
    cfg = tiny()
    assert make()(tokens(1, cfg.block_size, cfg.vocab_size)).shape[1] == cfg.block_size


# --------------------------------------------------------------------------- #
# Causality, end to end through the whole stack
# --------------------------------------------------------------------------- #

def test_changing_a_future_token_cannot_change_earlier_logits():
    """The Phase 2 masking test, repeated through N stacked blocks.

    Masking that is airtight in one attention layer can still be defeated by a
    wiring mistake between layers, so this asserts the property of the assembled
    model rather than of the component.
    """
    cfg = tiny(n_layer=4)
    model = make(n_layer=4).eval()
    a = tokens(1, 10, cfg.vocab_size, seed=1)
    b = a.copy()
    b[0, 6:] = (b[0, 6:] + 5) % cfg.vocab_size      # rewrite the tail
    assert not np.array_equal(a, b)

    la = model(a).data
    lb = model(b).data
    assert np.array_equal(la[:, :6, :], lb[:, :6, :])
    assert not np.allclose(la[:, 6:, :], lb[:, 6:, :])


def test_gradient_from_an_early_position_does_not_reach_later_tokens():
    """Backward-direction causality for the full model.

    Uses the position embedding's gradient as the probe: back-propagating from
    output position t alone, positions after t must receive exactly zero.
    """
    cfg = tiny(n_layer=3)
    model = make(n_layer=3).eval()
    T, t = 9, 3
    logits = model(tokens(1, T, cfg.vocab_size))

    seed = np.zeros(logits.shape)
    seed[0, t, :] = 1.0
    logits.backward(seed)

    g = model.wpe.weight.grad
    assert np.all(g[t + 1:T] == 0.0)
    assert np.any(np.abs(g[: t + 1]) > 0.0)


# --------------------------------------------------------------------------- #
# Structure: tying, residuals, configurability
# --------------------------------------------------------------------------- #

def test_tied_embeddings_share_one_tensor_and_are_stepped_once():
    model = make(tie_embeddings=True)
    assert model.lm_head is None
    ids = [id(p) for p in model.parameters()]
    assert len(ids) == len(set(ids))
    assert any(p is model.wte.weight for p in model.parameters())


def test_untied_model_has_a_separate_output_head_and_more_parameters():
    tied = make(tie_embeddings=True)
    untied = make(tie_embeddings=False)
    assert untied.lm_head is not None
    extra = tied.cfg.vocab_size * tied.cfg.d_model
    assert untied.num_parameters() == tied.num_parameters() + extra


def test_tied_weight_accumulates_gradient_from_both_uses():
    """The tied tensor is used twice per forward pass, so it must get both gradients.

    Compared against a model whose head is untied but numerically identical: the
    tied gradient must equal the sum of the two separate contributions.
    """
    cfg = tiny(vocab_size=11, block_size=8, n_layer=1, d_model=16)
    idx = tokens(2, 5, cfg.vocab_size, seed=3)

    nn.set_seed(4)
    tied = GPT(GPTConfig(**{**cfg.to_dict(), "tie_embeddings": True})).eval()
    nn.set_seed(4)
    untied = GPT(GPTConfig(**{**cfg.to_dict(), "tie_embeddings": False})).eval()
    # Make the untied head numerically identical to the tied one.
    untied.lm_head.weight.data[...] = tied.wte.weight.data.T
    untied.wte.weight.data[...] = tied.wte.weight.data

    _, lt = tied(idx, targets=idx)
    lt.backward()
    _, lu = untied(idx, targets=idx)
    lu.backward()

    assert lt.item() == pytest.approx(lu.item(), rel=1e-9)
    combined = untied.wte.weight.grad + untied.lm_head.weight.grad.T
    assert tied.wte.weight.grad == pytest.approx(combined, rel=1e-7, abs=1e-12)


def test_parameter_count_matches_the_hand_derived_formula():
    cfg = tiny(vocab_size=100, block_size=64, n_layer=3, n_head=4, d_model=32)
    model = make(vocab_size=100, block_size=64, n_layer=3, n_head=4, d_model=32)
    V, C, L, F, P = cfg.vocab_size, cfg.d_model, cfg.n_layer, cfg.d_ff, cfg.block_size

    per_block = (
        2 * C                    # ln_1
        + C * 3 * C + 3 * C      # attn qkv
        + C * C + C              # attn proj
        + 2 * C                  # ln_2
        + C * F + F              # mlp fc
        + F * C + C              # mlp proj
    )
    expected = V * C + P * C + L * per_block + 2 * C     # + ln_f, head is tied
    assert model.num_parameters() == expected
    assert model.num_parameters(non_embedding=True) == expected - V * C - P * C


@pytest.mark.parametrize("n_layer", [1, 2, 6])
@pytest.mark.parametrize("n_head,d_model", [(1, 8), (2, 16), (8, 32)])
def test_depth_width_and_head_count_are_configurable(n_layer, n_head, d_model):
    nn.set_seed(0)
    cfg = GPTConfig(vocab_size=13, block_size=8, n_layer=n_layer,
                    n_head=n_head, d_model=d_model, dropout=0.0)
    model = GPT(cfg).eval()
    assert len(model.blocks) == n_layer
    assert model.blocks[0].attn.n_heads == n_head
    assert model.blocks[0].attn.head_dim == d_model // n_head
    logits = model(tokens(2, 8, 13))
    assert logits.shape == (2, 8, 13)
    assert np.all(np.isfinite(logits.data))


def test_context_length_is_configurable():
    for block_size in (4, 16, 64):
        nn.set_seed(0)
        model = GPT(tiny(block_size=block_size)).eval()
        assert model.wpe.weight.shape[0] == block_size
        assert model(tokens(1, block_size, 17)).shape[1] == block_size


def test_config_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(d_model=10, n_head=4)


def test_d_ff_defaults_to_four_times_d_model_and_can_be_overridden():
    assert GPTConfig(d_model=64).d_ff == 256
    assert GPTConfig(d_model=64, d_ff=100).d_ff == 100


def test_residual_stream_variance_stays_bounded_with_depth():
    """The scaled residual init is what keeps activations from compounding.

    With n_layer blocks each adding two branches, an unscaled init grows the
    stream's variance roughly linearly in depth. Measured here across depths.
    """
    stds = []
    for n_layer in (2, 4, 8):
        nn.set_seed(0)
        model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=n_layer,
                              n_head=4, d_model=64, dropout=0.0)).eval()
        idx = tokens(4, 16, 64)
        x = model.drop(model.wte(idx) + model.wpe(np.arange(16)))
        for block in model.blocks:
            x = block(x)
        stds.append(float(x.data.std()))
    # Growth is mild, not multiplicative: 4x the depth must not mean 4x the spread.
    assert stds[-1] < 2.5 * stds[0], stds


def test_positional_embedding_actually_distinguishes_positions():
    """Without it, the model is permutation-invariant and cannot learn order."""
    cfg = tiny()
    model = make().eval()
    idx = np.full((1, 6), 3)                   # the same token everywhere
    logits = model(idx).data
    # Identical tokens at different positions must still produce different states.
    assert not np.allclose(logits[0, 0], logits[0, 5])


def test_dropout_is_active_in_train_mode_and_off_in_eval():
    model = make(dropout=0.5)
    idx = tokens(2, 6, 17)
    model.train()
    a, b = model(idx).data, model(idx).data
    assert not np.allclose(a, b)              # stochastic
    model.eval()
    assert np.array_equal(model(idx).data, model(idx).data)   # deterministic


# --------------------------------------------------------------------------- #
# Gradients flow, and the model can actually learn
# --------------------------------------------------------------------------- #

def test_every_parameter_receives_gradient():
    """A parameter with no gradient is a wiring bug -- a layer left out of the graph."""
    cfg = tiny(n_layer=2)
    model = make(n_layer=2)
    idx = tokens(2, cfg.block_size, cfg.vocab_size)
    _, loss = model(idx, targets=idx)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert np.all(np.isfinite(p.grad)), f"{name} has non-finite gradient"
        assert np.abs(p.grad).max() > 0.0, f"{name} has an all-zero gradient"


def test_gradient_reaches_the_first_block_undiminished():
    """Residual connections should keep early-layer gradients on the same scale."""
    nn.set_seed(0)
    model = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=6, n_head=4,
                          d_model=64, dropout=0.0))
    idx = tokens(4, 16, 32)
    _, loss = model(idx, targets=idx)
    loss.backward()

    norms = [float(np.linalg.norm(b.attn.qkv.weight.grad)) for b in model.blocks]
    assert all(np.isfinite(norms))
    # No vanishing: the first block's gradient is within an order of magnitude
    # of the last block's.
    assert norms[0] > norms[-1] / 10.0, norms


def test_model_can_overfit_a_single_batch():
    """The strongest cheap end-to-end test: memorising one batch must be possible.

    If any gradient in the stack were wrong, the loss would stall rather than
    approach zero.  Trains a small model on one fixed batch until the loss is far
    below the ln(V) starting point.
    """
    set_default_dtype(np.float32)
    nn.set_seed(0)
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2,
                    d_model=32, dropout=0.0)
    model = GPT(cfg)
    rng = np.random.default_rng(0)
    x = rng.integers(0, cfg.vocab_size, size=(4, 8))
    y = rng.integers(0, cfg.vocab_size, size=(4, 8))

    opt = Adam(model.parameters(), lr=3e-3)
    first = None
    for step in range(300):
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()

    assert first == pytest.approx(np.log(cfg.vocab_size), rel=0.2)
    assert loss.item() < 0.05, f"failed to overfit: {first:.3f} -> {loss.item():.3f}"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def test_generate_returns_the_prompt_plus_new_tokens():
    cfg = tiny()
    model = make().eval()
    prompt = tokens(2, 3, cfg.vocab_size)
    out = model.generate(prompt, max_new_tokens=5, rng=np.random.default_rng(0))
    assert out.shape == (2, 8)
    assert np.array_equal(out[:, :3], prompt)
    assert out.min() >= 0 and out.max() < cfg.vocab_size


def test_generate_accepts_a_1d_prompt():
    model = make().eval()
    out = model.generate(np.array([1, 2, 3]), max_new_tokens=4,
                         rng=np.random.default_rng(0))
    assert out.shape == (1, 7)


def test_generate_slides_the_context_window():
    """Generating past block_size must crop, not crash or index out of range."""
    cfg = tiny(block_size=6)
    model = make(block_size=6).eval()
    out = model.generate(tokens(1, 6, cfg.vocab_size), max_new_tokens=10,
                         rng=np.random.default_rng(0))
    assert out.shape == (1, 16)
    assert np.all(np.isfinite(out.astype(float)))


def test_top_k_1_is_greedy_and_deterministic():
    model = make().eval()
    prompt = tokens(1, 4, 17)
    a = model.generate(prompt, 6, top_k=1, rng=np.random.default_rng(0))
    b = model.generate(prompt, 6, top_k=1, rng=np.random.default_rng(99))
    assert np.array_equal(a, b)


def test_top_k_restricts_the_sampled_vocabulary():
    """With k=2, only the two highest-probability tokens may ever be produced."""
    cfg = tiny()
    model = make().eval()
    prompt = tokens(1, 4, cfg.vocab_size)
    logits = model(prompt).data[0, -1]
    allowed = set(np.argsort(logits)[-2:].tolist())

    produced = {model.generate(prompt, 1, top_k=2,
                               rng=np.random.default_rng(s))[0, -1]
                for s in range(40)}
    assert produced <= allowed


def test_low_temperature_concentrates_on_the_argmax():
    cfg = tiny()
    model = make().eval()
    prompt = tokens(1, 4, cfg.vocab_size)
    argmax = int(np.argmax(model(prompt).data[0, -1]))

    cold = [int(model.generate(prompt, 1, temperature=0.01,
                               rng=np.random.default_rng(s))[0, -1]) for s in range(20)]
    hot = [int(model.generate(prompt, 1, temperature=100.0,
                              rng=np.random.default_rng(s))[0, -1]) for s in range(20)]
    assert all(c == argmax for c in cold)
    assert len(set(hot)) > 3          # high temperature spreads the mass out


def test_generate_rejects_non_positive_temperature():
    model = make().eval()
    with pytest.raises(ValueError, match="temperature"):
        model.generate(tokens(1, 2, 17), 1, temperature=0.0)


def test_generate_restores_training_mode():
    model = make()
    model.train()
    model.generate(tokens(1, 2, 17), 2, rng=np.random.default_rng(0))
    assert model.training is True
    model.eval()
    model.generate(tokens(1, 2, 17), 2, rng=np.random.default_rng(0))
    assert model.training is False


def test_generate_builds_no_graph():
    """Generation runs under no_grad; nothing should accumulate gradient."""
    model = make().eval()
    model.zero_grad()
    model.generate(tokens(1, 3, 17), 5, rng=np.random.default_rng(0))
    assert all(p.grad is None for p in model.parameters())


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def test_state_dict_roundtrip_reproduces_logits_exactly():
    cfg = tiny()
    a = make(seed=0).eval()
    b = make(seed=1).eval()
    idx = tokens(2, 7, cfg.vocab_size)
    assert not np.allclose(a(idx).data, b(idx).data)

    b.load_state_dict(a.state_dict())
    assert np.array_equal(a(idx).data, b(idx).data)


def test_config_roundtrips_through_a_dict():
    cfg = tiny(n_layer=3, d_ff=99)
    assert GPTConfig(**cfg.to_dict()) == cfg


def test_parameter_summary_reports_the_geometry():
    s = make(n_layer=3, n_head=2, d_model=16).parameter_summary()
    assert "layers          3" in s
    assert "heads           2" in s
    assert "parameters" in s
