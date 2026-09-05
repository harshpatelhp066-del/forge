# Phase 3 — Transformer architecture

Files: [`forge/model.py`](forge/model.py), [`tests/test_phase3_model.py`](tests/test_phase3_model.py)

## What was built

A GPT-style decoder-only Transformer assembled from Phase 2 primitives.

```
tokens ──► token embedding ──┐
                             ├──► + ──► dropout ──► [ Block ] × N ──► LayerNorm ──► logits
positions ──► pos embedding ─┘                                                        ▲
                                                              tied to the token embedding

Block(x):  x = x + Attention(LayerNorm(x))
           x = x + MLP(LayerNorm(x))
```

`GPTConfig` makes depth (`n_layer`), width (`d_model`), heads (`n_head`), feed-forward
width (`d_ff`, defaulting to `4·d_model`), context (`block_size`), vocabulary,
dropout and embedding tying all free parameters. `generate()` implements
autoregressive sampling with temperature and top-k.

## Verification status

**48/48 Phase 3 tests pass** (349 total).

The forward-pass contract asked for in the brief:

- **Shapes.** `(B, T)` in, `(B, T, vocab)` out, checked across `(1,1)`, `(2,5)`,
  `(4,12)`, `(3,7)` and at exactly `block_size`; a sequence one token longer
  raises rather than silently truncating.
- **Valid distribution after softmax.** All probabilities in `[0,1]`, rows summing
  to `1 ± 1e-5`.
- **No NaNs or Infs on realistic inputs.** Checked in both train and eval mode
  across several batch shapes, and separately on a deeper/wider model (8 layers,
  `d_model=64`), where the logits are additionally asserted to stay under
  magnitude 50 — finite but saturated would also be a failure.

Beyond the contract:

- **Untrained loss ≈ ln(V).** The most informative single number for a fresh
  transformer: `4.19` against `ln(65) = 4.17`.
- **Causality end-to-end.** Rewriting the tail of the input leaves the earlier
  logits *bit-identical* through 4 stacked blocks, and back-propagating from
  output position `t` alone leaves the positional-embedding gradient at every
  position after `t` exactly zero. Phase 2 proved one attention layer is airtight;
  this proves the wiring between layers did not undo it.
- **Every parameter receives a finite, non-zero gradient.** A parameter with no
  gradient means a layer fell out of the graph — a wiring bug that otherwise shows
  up only as slightly worse training.
- **Gradient reaches the first block undiminished.** Across a 6-layer model, the
  first block's gradient norm is within an order of magnitude of the last's.
- **The model can overfit a single batch** to a loss below 0.05 from a `ln(16)`
  start. This is the strongest cheap end-to-end test there is: if any gradient
  anywhere in the stack were wrong, the loss would stall rather than approach zero.
- **Parameter count matches a hand-derived formula**, so the geometry is what it
  claims to be.
- **Generation**: prompt is preserved, the context window slides correctly past
  `block_size`, `top_k=1` is deterministic and greedy, `top_k=2` provably never
  emits a token outside the top 2, low temperature collapses onto the argmax and
  high temperature spreads out, training mode is restored afterwards, and no graph
  is built.

## Real tradeoffs and design decisions

### Pre-norm, not post-norm

`x + f(LayerNorm(x))`, not `LayerNorm(x + f(x))`. This is the one structural
choice that decides whether the model trains at all at this depth. With post-norm,
every residual addition passes through a normalization on the way *out*, so the
shortcut from input to output is not an identity and the gradient is rescaled once
per block; the original Transformer needed a learning-rate warmup specifically to
survive it. With pre-norm, the residual path is a clean identity from the first
block to the last and gradient reaches the earliest layer undiminished — measured
in `test_gradient_reaches_the_first_block_undiminished`.

The cost is that the residual stream's variance grows with depth (nothing
renormalizes it), which is what the final `ln_f` before the output projection is
for.

### Residual projections are initialized at `0.02/√(2·n_layer)`

Each block adds *two* branches into the residual stream, so with `n_layer` blocks
the stream accumulates `2·n_layer` independent contributions and its variance
grows linearly in depth. Scaling the output projection of each branch by
`1/√(2·n_layer)` cancels that growth. `test_residual_stream_variance_stays_bounded_with_depth`
measures it: quadrupling depth from 2 to 8 layers grows the activation standard
deviation by well under 2.5×, rather than the ~4× an unscaled init would give.

### Weight tying, and a genuine surprise it produced

The output projection *is* the token embedding, transposed — not a separate
parameter. That saves `vocab × d_model` parameters (a large fraction of a small
model: 25% here, and the embedding is the single biggest tensor) and couples the
two representations of a token.

Two things had to be right for this to work:

1. **Gradient must accumulate from both uses.** The tied tensor appears twice per
   forward pass. Verified against a numerically identical *untied* model: the tied
   gradient equals the sum of the untied model's two separate contributions, to
   1e-7.
2. **The optimizer must step it once.** `named_parameters` de-duplicates by object
   identity (Phase 2), so it does.

The surprise came from the untrained-loss sanity check, which initially read
**3.79 against `ln(65) = 4.17`** — meaningfully *better* than chance on an
untrained model, which normally means something is leaking. It is not a leak. The
test was constructed with `targets = inputs`, i.e. "predict the token you just
saw". With tying, the residual stream still carries a large component of the input
token's own embedding, and the output projection dots the stream against that very
same matrix — so `logit[current_token]` gets a systematic self-similarity boost.

Isolating it confirmed the cause exactly:

| targets | tied | untied |
|---|---|---|
| the input token itself | **3.79** | 4.19 |
| independent random tokens | 4.16 | 4.18 |
| the true next token | 4.19 | 4.18 |

Untying removes the effect completely, which pins the cause on the shared matrix
rather than on the initialization. This is arguably part of *why* tying helps —
but it means an untrained-loss check must use genuine next-token targets or it
measures the artifact instead of the initialization. Both the corrected sanity
check and a test that pins the effect itself are now in the suite.

### Learned positional embeddings

A `block_size × d_model` lookup added to the token embedding, as in GPT-2, rather
than sinusoidal encodings or RoPE. Learned embeddings are the simplest thing that
works and need no extra machinery in the engine; the cost is that the model cannot
extrapolate beyond `block_size` at all — there is simply no embedding for position
`block_size + 1`. `generate()` therefore crops its input to the last `block_size`
tokens on every step, which is tested.

`test_positional_embedding_actually_distinguishes_positions` feeds the *same*
token at every position and asserts the outputs differ. Without positional
information a transformer is permutation-invariant, and that failure is invisible
in shape checks.

### GELU and the 4× feed-forward ratio

`d_ff = 4·d_model` is the standard ratio and is left configurable. The MLP holds
`8·d²` parameters per block against attention's `4·d²`, so most of the model's
capacity — and most of its FLOPs — is in the feedforward, not in attention.

### `generate()` samples in float64 on the NumPy side

Sampling is done on raw arrays under `no_grad()`, and the logits are cast to
float64 before the temperature division and softmax. Dividing float32 logits by a
small temperature (0.01 in the tests) then exponentiating is exactly the case
where float32 loses the ordering of near-tied candidates. The cast costs nothing —
it is one `(B, vocab)` row per step.

`top_k` is applied *before* the softmax by setting the excluded logits to `-inf`,
so the surviving probabilities renormalize to sum to 1 rather than being a
truncated, sub-normalized slice.

## Known limitations at this phase

- **No KV-cache.** Every generation step recomputes keys and values for the entire
  prefix, making generation O(T²) per token instead of O(T). This is the single
  largest optimization left on the table; see the README's limitations section.
- Attention materializes the full `(B, H, T, T)` score matrix, so memory is
  quadratic in context length.
- No gradient checkpointing, so activation memory is linear in depth.
- `generate()` samples one row at a time via a Python loop over the batch, because
  NumPy has no batched `choice`. Negligible at this scale, but it would matter for
  large batches.
