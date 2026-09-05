# Phase 1 — Autodiff engine

Files: [`forge/tensor.py`](forge/tensor.py), [`forge/gradcheck.py`](forge/gradcheck.py),
[`tests/test_phase1_gradcheck.py`](tests/test_phase1_gradcheck.py),
[`scripts/gradcheck_report.py`](scripts/gradcheck_report.py)

## What was built

A `Tensor` wrapping a NumPy array that records a computation graph, plus a
`backward()` that performs a reverse-mode sweep and populates `.grad` on every
node reachable from the output.

Each operation returns a new `Tensor` holding (a) its parents and (b) a closure
that pushes gradient from output to parents. `backward()` topologically sorts the
graph and calls each closure exactly once, in reverse order, so a node's `.grad`
is complete before it is read.

Ops implemented with gradients: `add`, `sub`, `mul`, `div`, `matmul`, `pow`
(scalar and tensor exponent), `neg`, `exp`, `log`, `sqrt`, `tanh`, `sigmoid`,
`relu`, `abs`, `sum`, `mean`, `max`, `min`, `var`, `reshape`, `flatten`,
`transpose`/`permute`/`swapaxes`, `broadcast_to`, `concat`, `__getitem__`
(slice and fancy indexing), `where`, `masked_fill`, and the six comparison
operators.

## Verification status

**233/233 tests pass.** `python scripts/gradcheck_report.py` certifies 31
operation configurations against central finite differences:

```
31/31 operations certified   (rtol=1e-06, atol=1e-09, h=1e-5, float64)
worst observed relative error: 7.06e-08 (batched matmul)
typical: ~2e-10
```

Beyond the per-op gradient checks, the suite asserts:

- **Diamond graphs accumulate.** `x * x * x` at `x=2` yields `grad=12`, so a
  tensor reused on several paths sums its contributions instead of the last
  writer winning.
- **Deep graphs do not overflow the stack.** A 5000-node chain backpropagates
  correctly. The topological sort is an explicit stack, not recursion — CPython's
  default 1000-frame limit is well below the depth of a real transformer graph.
- **Gather backward is a scatter-add.** Indexing with `[1, 1, 1]` accumulates
  `3.0`, not `1.0`. This is the embedding-layer case and the one place where the
  obvious implementation is silently wrong (see below).
- **Masks are constants.** Comparison ops return tensors with `requires_grad=False`
  and no parents, so a mask cannot drag gradient into the graph.
- **`masked_fill` blocks gradient** at filled positions exactly.
- **`no_grad()` builds no tape** and restores the previous state on exit.
- **The checker can fail.** This matters more than any individual pass. The
  checker is pointed at deliberately broken gradients and must reject them:
  a 2× error, a 1e-3 relative error, a 1e-5 relative error (10× the configured
  rtol), and a 1% error confined to *one entry of a 40-entry tensor*. All four
  are caught. A gradient checker that passes everything would have silently
  blessed the entire project.

## Real tradeoffs and numerical decisions

### float64 for checking, float32 for training

Central differences lose roughly half the available significant digits to
cancellation in the numerator. float32 has ~7 digits, leaving ~3 — not enough to
distinguish a correct gradient from a subtly wrong one. So `set_default_dtype`
is a global that `check_gradient` flips to float64 for the duration of a check
and restores in a `finally`. Training runs in float32 for the memory and BLAS
throughput.

### The pass criterion needed an absolute floor, and finding out why took a real debugging pass

The first version judged gradients by relative error alone, at `tol=1e-7`. Nine
tests failed. **None of them were engine bugs** — all four distinct causes were
artefacts of how the *test* was set up, and each is worth recording because each
is a trap a from-scratch implementation walks into:

1. **Saturated `tanh` in the composite test.** Inputs were drawn positive in
   `[0.4, 2.0]`, so `x @ y + z` landed in `[5.3, 12.1]` where `tanh'` is
   `9.7e-05`. Every downstream gradient collapsed to ~5e-8, at which point the
   finite-difference noise floor (`eps·|f|/h ≈ 2e-11`) is a 4e-4 *relative*
   error. The measurement was broken, not the gradient. Fixed by scaling the
   pre-activation by 0.3 to keep it off the tails.
2. **Near-ties in `max`.** Seed 2 happened to produce two values along the
   reduction axis separated by `6.08e-06` — smaller than the `2h = 2e-5` step.
   Perturbing the runner-up flips the argmax between the `+h` and `-h`
   evaluations, so the difference quotient measures the kink rather than a
   derivative. Fixed by resampling until the gap exceeds `1e-3`; the tie case is
   then covered *exactly* by a separate test rather than approximately by finite
   differences.
3. **`masked_fill` with `-1e4`.** The projected loss is O(1e4) while the
   difference quotient's numerator is O(1e-5); cancellation eats most of the
   mantissa. The gradcheck now uses a moderate fill value, and the real `-1e9`
   attention case is asserted exactly instead of numerically.
4. A `pytest.approx` misuse on nested lists.

The lasting fix in the engine's checker is the standard `allclose` criterion,
`|a − n| ≤ atol + rtol·max(|a|, |n|)`, with `atol = 1e-9` documented as the
finite-difference noise floor rather than as slack. The sensitivity tests above
exist to prove `atol` is not hiding anything: a 1e-5 relative error is still
caught.

### Softmax overflow — handled at the op level

`sigmoid` is the one place in Phase 1 where a naive formula overflows:
`1/(1+exp(-x))` overflows for `x ≈ -800`. It is computed branch-wise —
`1/(1+exp(-x))` for `x ≥ 0` and `e/(1+e)` for `x < 0` — so the argument to `exp`
is never positive. Asserted under `np.errstate(over='raise')` on inputs spanning
±800. The same max-subtraction trick for `softmax` proper lands in Phase 2.

### Gather backward must use `np.add.at`, not fancy-index assignment

`buf[idx] = grad` silently keeps only the *last* write when an index repeats.
Since a token appearing twice in a batch produces exactly that, the naive version
would under-count embedding gradients in a way no shape check catches and no
crash reveals — the model would just train slightly wrong. `np.add.at` is the
buffered, duplicate-safe form. It is ~10× slower than the unbuffered path, which
is a real cost accepted for correctness.

### `max` splits gradient across ties

At a tie the function is not differentiable. Routing the full gradient to the
first tied index (the common shortcut) is a valid subgradient but disagrees with
what a *symmetric* finite difference measures, which is the average. Splitting
evenly among ties keeps the analytical and numerical gradients consistent, so the
checker stays meaningful there.

### Broadcasting: the adjoint of a copy is a sum

Broadcasting is the only op whose backward pass changes shape. `_unbroadcast`
handles both cases NumPy can produce — prepended axes (summed away entirely) and
stretched size-1 axes (summed with `keepdims`) — and is applied inside every
binary op rather than being left to callers. Six broadcast shape combinations are
covered per binary op, including mutual broadcast (`(1,5)` with `(4,1)`).

### `__eq__` returns a mask, so hashing had to be pinned

Overloading `__eq__` to build masks breaks the default `__hash__`, and set
membership during the topological sort would then call `__eq__` on collisions and
get a Tensor back rather than a bool. Two guards: `__hash__ = object.__hash__`,
and the topo sort's `visited` set stores `id(node)` rather than the nodes
themselves.

### Random projections, not `.sum()`

Reducing an op's output with `out.sum()` only ever tests `Jᵀ·1`. A transposed or
mis-permuted Jacobian passes that by accident — and `transpose` is exactly the op
where an inverse-permutation bug is easy to write. Each check runs several
projections `(out * w).sum()` with random `w`, plus a dedicated test that
transposes by the non-involutive permutation `(2,0,1)` and checks the gradient
against the hand-computed inverse.

## Known limitations at this phase

- No in-place operations. Every op allocates; there is no version counter to
  detect a buffer mutated after being recorded.
- No second derivatives — the graph is not itself differentiable.
- `pow` with a tensor exponent requires a positive base (the `log(a)` term is
  otherwise undefined); the backward guards with `where` to avoid a NaN, but the
  gradient is only meaningful for `a > 0`.
- `.grad` accumulates until explicitly zeroed. This is deliberate (it makes
  gradient accumulation across micro-batches free) but it means a forgotten
  `zero_grad()` is a silent bug, not a loud one. The optimizer in Phase 2 owns
  the zeroing.
