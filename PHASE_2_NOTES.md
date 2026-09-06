# Phase 2: Neural network primitives

Files: [`forge/nn.py`](forge/nn.py), [`forge/optim.py`](forge/optim.py),
[`tests/test_phase2_nn.py`](tests/test_phase2_nn.py)

## What was built

All on top of the Phase 1 `Tensor`. No layer hand-derives a gradient: each is
composed from certified ops and the engine assembles the backward pass.

- `Module` / `Parameter`, parameter discovery by walking `__dict__` (including
  through lists and nested `Sequential`s), `train()`/`eval()`, `state_dict()`.
- `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `Sequential`
- `softmax`, `log_softmax`, `gelu`, `relu`, `cross_entropy`
- `MultiHeadSelfAttention` with causal masking
- `Adam` with bias correction and decoupled weight decay; `SGD`; `clip_grad_norm`

## Verification status

68/68 Phase 2 tests pass (301 total with Phase 1). Every layer is
gradient-checked *including its parameters*, not just its input, the check
rebinds each parameter to the checker's float64 wrapper before each forward pass,
otherwise the check would silently only cover the input tensor.

### LayerNorm actually normalizes

Asserted on the output statistics, not on the shape. Fed a deliberately
badly-scaled input (`σ=17`, `μ=42`), the output has mean `0 ± 1e-6` and variance
`1 ± 1e-4` along the last axis. Also verified: it normalizes the *last axis only*
(changing row 1 drastically leaves row 0's output bit-identical); the affine
parameters initialize to exact identity and are applied as `γ·x̂ + β`; and a
constant row (zero variance) yields finite zeros rather than a division by zero.

Writing that last test surfaced a subtlety worth writing down. The first version
compared a normalized `[1,2,3,4]` against a normalized `[100,200,300,400]` and
expected them equal to 1e-6. They differ at the 6th digit, because `eps` is a
fixed *absolute* term inside the square root and so perturbs a low-variance row
proportionally more than a high-variance one. Then the fix over-corrected: a row
scaled by 100 normalizes *identically* (LayerNorm is invariant to an affine change
of a row), which made the "did anything change" guard vacuous. The test now
reshapes the other row rather than rescaling it. Both mistakes were in the test;
the layer was right throughout.

### Causal masking actually blocks the future

Four tests, deliberately overlapping, because leaky masking is the failure mode
that still trains, it just trains a model that cheats at validation time and
cannot generate.

1. **Attention weights are exactly zero** in the strict upper triangle, `== 0.0`,
   not `< 1e-6`. After softmax's max-subtraction, `exp(-1e9)` underflows to
   exactly `0.0`.
2. **Future tokens cannot change earlier outputs.** Two inputs share their first
   3 positions; the tail of one is replaced with values scaled by 1000, so any
   attention mass reaching it would move the early outputs enormously. Outputs at
   positions 0–2 are asserted bit-identical (`np.array_equal`), and the tail
   is asserted to have actually changed so the test cannot pass vacuously.
3. **Masking is gradient-tight, not just forward-tight.** Backpropagating from
   output position `t` alone, `∂output[t]/∂input[s]` is exactly `0.0` for every
   `s > t`, and non-zero for `s ≤ t`. A forward-only check would miss a leak
   introduced in a backward closure.
4. **Negative control.** The mask is zeroed out and tests 2 and 3 are re-run with
   inverted assertions. Both detect the leak. Without this, a mask that was
   accidentally all-`False` would satisfy every assertion above for a trivial
   reason and the suite would prove nothing.

### Adam updates in the right direction

- **First-step closed form.** At `t=1`, `m̂ = g` and `v̂ = g²`, so the update is
  exactly `lr·g/(|g|+ε) ≈ lr·sign(g)` — *independent of the gradient's magnitude*.
  Asserted on gradients spanning three orders of magnitude (`2.0`, `-3.0`, `1e-3`)
  with mixed signs, all producing a step of exactly `±0.1` at `lr=0.1`. Without
  bias correction the same step would be `(1-β₁)/√(1-β₂) = 3.162` times larger in
  the ratio, i.e. 3.16× the intended learning rate on step one — the
  assertion separates the two implementations by that factor.
- **Bit-exact against a textbook implementation.** The fused bias correction
  (folding both corrections into one step size `lr·√(1-β₂ᵗ)/(1-β₁ᵗ)` to avoid two
  full-size temporaries per parameter per step) is only a valid optimisation if it
  is *exact*. Checked against a separately written literal Adam over 50 steps:
  maximum deviation 0.0.
- **Convex descent.** On `f(x,y) = (x-3)² + (y+1)²` from the origin, converges to
  the minimum with final loss `< 1e-12`.
- **End-to-end through the real graph.** A `Linear` layer fits a random linear
  target through the full autodiff engine: loss drops by >1000×, recovered weights
  match ground truth to 1e-2.
- Direction (`sign(Δθ) = -sign(g)` elementwise), decoupled weight decay, skipping
  parameters with no gradient, and optimizer-state round-tripping are each tested.

### A wrong test assertion

The convex-descent test originally asserted that Adam's loss decreases
monotonically. It does not, and should not: momentum carries it past the minimum
around step 40, after which it rings down and converges to `9.7e-19`. The
assertion was wrong, not the optimizer, confirmed by the bit-exact textbook
comparison above. The test now asserts the *envelope* decays (each 50-step window
peaks lower than the previous) rather than every individual step, and says why.

## Design notes

### Softmax overflow: subtract the max, and treat it as a constant

`exp` overflows above ~88 in float32, and attention logits routinely exceed that,
so `softmax` subtracts the row max first. Softmax is invariant to that shift, so
the value is unchanged.

The max is taken on the raw NumPy buffer rather than through a `Tensor.max` node.
This is not an approximation. Because the shift cancels exactly in the forward
value, its derivative contribution is exactly zero; detaching only avoids putting
a `max` node and its scatter-backward on the tape. Verified by a test that pushes
logits to `±1000` under `np.errstate(over='raise')` and confirms the distribution
still sums to 1.

`log_softmax` is computed as `z - log(Σexp(z))` rather than `log(softmax(x))`.
Softmax underflows to exactly `0.0` for sufficiently negative logits and `log(0)`
is `-inf`; this form never materialises the small probability. Tested: for logits
`[0, 200]`, the naive route gives `-inf` and the stable one gives `-200.0`. This
matters because a confidently-wrong prediction early in training is common, and
an `inf` loss poisons every gradient in the batch.

### The mask fill value is -1e9, not -inf

`-inf` would make an entirely-masked row produce `0/0 = NaN` after softmax. Causal
masking never produces such a row (position `i` always sees itself), so `-inf`
would work *today* — but a padding mask added later could, and the failure would
be a silent NaN across the whole batch. `-1e9` degrades to a uniform distribution
instead, and still underflows `exp` to exactly `0.0` after the max subtraction, so
nothing is given up for the safety.

### GELU: the tanh approximation

The exact GELU needs `erf`, which NumPy does not expose as a ufunc. The options
were a `math.erf` Python loop (unusably slow in the inner loop of every forward
pass), a SciPy dependency (against the spirit of the project), or the tanh
approximation that GPT-2 itself shipped. Chose the approximation, and pinned it:
a test compares against the exact `erf` form over `[-6, 6]` and asserts the
maximum absolute error is below `2e-3`.

### 1/√d attention scaling

`q·k` over `d` dimensions has standard deviation `√d`, so without scaling, logits
grow with head width, softmax saturates, and the gradient through it collapses.
A test measures this directly across `d ∈ {8, 64, 512}`: raw dot products have
`σ ≈ √d` and scaled ones have `σ ≈ 1` in every case.

### Inverted dropout

Scaling by `1/(1-p)` at training time makes evaluation an exact identity, so there
is no rescale to remember at inference. Tested: ~30% of activations are zeroed at
`p=0.3`, survivors are scaled by exactly `1/0.7`, the mean activation is preserved
at 1.0, and `eval()` returns the input array unchanged.

Dropout is the one layer whose forward pass is not a function of its input alone,
so every gradient check runs the module in `eval()`, a stochastic forward has no
derivative for finite differences to measure.

### Adam: eps outside the sqrt, weight decay decoupled

`eps` sits outside the square root, as in the original paper. Inside, it would be
a floor on the *variance* rather than on the divisor, and would not bound the step
size when `v̂` is zero, which happens for real: an embedding row for a
token that has not yet appeared in any batch has had an exactly-zero gradient
throughout.

Weight decay is applied straight to the parameter (AdamW) rather than added into
the gradient. Folding it into `g` would let Adam's per-parameter `1/√v̂` scaling
*shrink* the decay for precisely the large-gradient parameters that most need
regularising.

### Gradient clipping is global

The quantity that matters is the length of the single update vector formed by
concatenating every gradient. Clipping each tensor separately would change the
update's direction, not just its length. `clip_grad_norm` computes one global
L2 norm and applies one shared scale factor; a test asserts the cosine similarity
between the pre- and post-clip update vectors is `1.0` to within `1e-9`. It
returns the pre-clip norm, which is worth logging, a spike there is the earliest
visible symptom of a run about to diverge.

The norm is accumulated in float64 even when parameters are float32: summing
millions of squared values in float32 loses precision exactly where the number
matters most.

### Weight tying is handled at the parameter level

`named_parameters` de-duplicates by object identity, so a weight shared between
two modules (the tied input/output embedding in Phase 3) is yielded once and
therefore stepped once. Stepping it twice would apply the learning rate twice to
that tensor. Tested directly.

### Tests run in float64, training runs in float32

An autouse fixture puts the Phase 2 suite in float64 so that assertions about
mathematics are not confused with float32 rounding. Because that could hide a
layer that only works with float64 headroom, a dedicated test re-runs LayerNorm,
attention and cross-entropy in float32 with appropriate tolerances and asserts
the outputs really are float32.

## Known limitations at this phase

- No KV-cache: attention recomputes keys and values for the whole prefix on every
  generation step. Deferred to the Phase 6 limitations section with reasoning.
- Attention materialises the full `(B, H, T, T)` score matrix, no flash-attention
  style tiling, so memory grows quadratically in context length.
- `Dropout` draws a fresh mask on every call from a module-global RNG. Fine for
  training; it means two forward passes are not reproducible without reseeding.
- No `Conv`, `BatchNorm`, or RNN layers, nothing the GPT needs.
