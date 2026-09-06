# Phase 5 — Training and generation

Files: [`forge/train.py`](forge/train.py), [`scripts/train.py`](scripts/train.py),
[`scripts/generate.py`](scripts/generate.py),
[`tests/test_phase5_train.py`](tests/test_phase5_train.py)

## What was built

- Cross-entropy loss on the Forge engine (`forge/nn.py`), via a stable
  `log_softmax` rather than `log(softmax(x))`.
- Training loop with global gradient clipping, linear warmup + cosine decay,
  periodic validation, checkpointing, and CSV loss logging.
- `CosineWarmupSchedule`, `save_checkpoint`/`load_checkpoint`, `estimate_loss`,
  `LossLogger`, `plot_loss_curve`.
- Generation with temperature and top-k sampling, and a `--compare` mode that
  samples from every checkpoint to produce a before/after file.

## Results

| | |
|---|---|
| Parameters | 940,800 (793,344 non-embedding) |
| Architecture | 4 layers, 4 heads, d_model 128, d_ff 512, context 128 |
| Training | 2500 steps × 32 × 128 = 10.2M tokens (24.7 epochs) |
| Uniform baseline | ln(1024) = 6.9315 |
| Final train / val | **3.0967** / **3.5931** |
| Best val | **3.5931** |
| Wall clock | 1h59m on CPU (2.87 s/step) |

![loss curve](checkpoints/loss_curve.png)

## Verification status

**35/35 Phase 5 tests pass** (436 total across all phases).

- **Schedule**: warmup is linear and reaches exactly `base_lr` at its end; the
  cosine midpoint sits exactly halfway between base and floor; the rate is
  monotone after the peak, never negative, clamped past `total_steps`, and the
  floor is a ratio of base rather than zero. Impossible configurations raise.
- **Checkpointing**: a round-trip reproduces logits *exactly*; optimizer state
  round-trips such that a resumed run takes a **bit-identical** next step to an
  uninterrupted one; the write is atomic; and a checkpoint carries enough config
  to rebuild the model without being told the architecture — which is what
  `generate.py` relies on.
- **Evaluation**: runs in eval mode (dropout off) and restores the previous mode;
  builds no graph and leaves no gradients; is reproducible for a fixed seed; and
  does not disturb the loader's RNG stream, so evaluating cannot change the
  sequence of training batches.
- **Clipping** engages on a real model and preserves gradient direction to a
  cosine similarity of 1.0 within 1e-6.
- **End-to-end**: a miniature run learns a deterministic token pattern to a
  validation loss below 0.1 from a `ln(V)` start.
- **Fused ops** (see below) match their composed reference in both value and
  gradient, and pass the finite-difference checker.

## The bug that killed the first run

The first full training run died at **step 103** with
`_ArrayMemoryError: Unable to allocate 8.00 MiB`. The loss had been falling
cleanly (6.93 → 4.82), so this was purely a memory failure.

Three things were wrong, found in this order.

### 1. Profiling first: where does graph memory actually go?

Rather than guess, the graph was walked from the loss node and every retained
buffer totalled by operation. For a batch of 8×128 through a 4-layer, `d=192`
model:

```
graph nodes: 363   retained buffers: 381 MiB   (380.6 KiB per token)

op              count      MiB      %
mul                59    116.2   30.5
add                51     64.5   17.0
matmul             25     46.0   12.1
sub                15     26.8    7.0
```

`mul` dominating at 30% was the clue. GELU, composed from primitives, builds
**nine** full-width temporaries (`x*x`, `x*x*x`, `·0.044715`, `+x`, `·c`, `tanh`,
`+1`, `·0.5`, `·x`) — and it is applied to the `(B, T, 4·d_model)` feed-forward
expansion, the widest activation in the model. Softmax added four more at
`(B, H, T, T)`.

**Fix**: fused `gelu` and `softmax` into single ops with hand-derived backwards,
each retaining one buffer instead of nine and four. With
`u = c(x + kx³)` and `t = tanh(u)`:

```
d(gelu)/dx = 0.5(1 + t) + 0.5·x·(1 − t²)·c·(1 + 3kx²)
d(softmax)/dx = y ⊙ (g − Σ(g ⊙ y))
```

This is not a departure from "built on the engine" — it is the same thing `exp`
and `tanh` already are: a primitive with a hand-written adjoint. And it is held
to the same standard: both are certified against finite differences *and*
asserted equal in value and gradient to the composed versions, which are kept in
the codebase as the reference. Result: **381 → 260 MiB**, a 32% cut.

### 2. Intermediate gradients were never freed

`backward()` accumulated `.grad` on every node and kept it. But an intermediate's
adjoint is read exactly once — by its own backward closure, to push into its
parents — so holding it afterwards achieves nothing while roughly doubling peak
memory, since gradient buffers are the same size as the activations they shadow.

**Fix**: free each intermediate's `.grad` immediately after its closure runs.
Leaves (inputs and parameters) always keep theirs — they have no closure and are
what the optimizer reads. `retain_grads=True` restores the old behaviour for
debugging. A test asserts leaf gradients are bit-identical either way.

### 3. The actual cause: every node was in a reference cycle

Even after both fixes, memory climbed. The ops were written in the natural way:

```python
def make_bw(out):          # <-- captures the output tensor
    def bw():
        a._accumulate(out.grad * b.data)
    return bw
out._backward = make_bw(out)
```

Every node therefore held `tensor → closure → tensor`. **Reference counting can
never break a cycle.** So dropping `loss` at the end of a step freed nothing; the
entire graph — every activation buffer in it — survived until Python's cyclic
collector happened to run. With multi-megabyte activations, memory outran the
collector.

This was confirmed rather than assumed, by inspecting a closure's cells:

```
closure captures: ['Tensor', 'Tensor', 'Tensor']
closure captures the OUTPUT itself: True

with cyclic GC disabled: 1250 objects still alive after 50 graphs
after gc.collect():      -4 objects alive
```

**Fix**: backward functions take the gradient as an argument,
`node._backward(g)`, so a closure captures only its parents and cached forward
values — never its own output. No cycle, so a graph is released by reference
counting the moment the loss goes out of scope. As a bonus this deleted a layer
of nesting from all 26 ops.

Verified by re-running real training steps **with the cyclic collector disabled
entirely** — the harshest possible test, since nothing can bail the program out:

```
cyclic GC disabled; 940,800 params, batch 32x128
baseline RSS 42.5 MiB
  step   0  RSS   629.5 MiB   peak   725.5 MiB   2.19s/step
  step   5  RSS   630.3 MiB   peak  1252.4 MiB   1.66s/step
  step  10  RSS   628.0 MiB   peak  1253.1 MiB   1.57s/step
  step  20  RSS   628.0 MiB   peak  1253.1 MiB   1.51s/step
  step  29  RSS   630.7 MiB   peak  1253.1 MiB   1.54s/step
```

Flat. Speed also improved from 2.8 to 1.51 s/step, because the failing run had
been thrashing.

Two regression tests now guard this: one asserts no backward closure captures its
own output, and one runs 30 graphs with the collector disabled and asserts fewer
than 50 objects survive.

**The lesson worth keeping**: in a framework where nodes hold closures, the
closure's capture list is part of the memory design. This is why PyTorch's
`Function` receives `grad_output` as an argument instead of reading it off the
output node — a detail that looks like style until you build one yourself.

## Other real tradeoffs

### Warmup is not optional

Adam's second-moment estimate `v` is meaningless for the first few dozen steps —
an average over a handful of gradients from a randomly initialised model — so
`1/√v̂` is a badly scaled step in a direction that is mostly noise. Warming up
from ~0 keeps those steps small enough not to matter. 150 warmup steps out of
2500, then cosine decay to a **floor of 10% of base**, not to zero: a zero
terminal rate wastes the final steps entirely.

Cosine over step decay because a step introduces a discontinuity that shows up as
a visible kink in the loss curve; over linear because linear spends too long at
large rates.

### Gradient clipping is global, and non-finite norms are not clipped

Clipping rescales by one shared factor computed from the **global** L2 norm over
all parameters concatenated, so the update's direction is preserved exactly and
only its length changes. Per-tensor clipping would change the direction.

An edge case surfaced in testing: if the norm is `inf`, then
`scale = max_norm/inf = 0`, and `inf × 0 = NaN` — a naive clip converts one bad
entry into a whole tensor of NaN. `clip_grad_norm` now returns the non-finite
norm and leaves gradients **untouched**, and the training loop skips the step.
One `inf` reaching the optimizer would turn every parameter it touches into NaN,
and the run would never recover. Tested for `inf`, `-inf` and `nan`.

The norm is accumulated in float64 even though parameters are float32: summing a
million squared values in float32 loses precision exactly where the number
matters.

### The checkpoint format, and a bug in writing it

Checkpoints are a single `.npz` holding parameters under a `param/` prefix,
optimizer moments, and a JSON metadata blob including the full model config. The
config travelling with the weights is what lets `generate.py` rebuild the model
without being told the architecture.

Writes go to a temporary file and are then renamed, so an interrupted save cannot
leave a truncated checkpoint where a valid one used to be. The first version of
this was broken: `np.savez` **silently appends `.npz`** to a filename that lacks
it, so writing to `best.npz.tmp` actually produced `best.npz.tmp.npz` and the
rename failed with `FileNotFoundError`. Fixed by opening the file handle
explicitly. Caught immediately because it fired on the first checkpoint of the
pilot run — which is the argument for doing a short pilot before a long run.

### Validation is measured on fixed batches

`estimate_loss` takes a `seed` and uses it for a temporary RNG, restoring the
loader's own generator state afterwards. Two consequences, both wanted: the
validation number is comparable across checkpoints because it is computed on the
*same* batches every time, and evaluating never perturbs the sequence of training
batches. Both are tested.

### Sampling details

`generate()` casts logits to float64 before the temperature division and softmax.
Dividing float32 logits by a small temperature and exponentiating is exactly where
float32 loses the ordering of near-tied candidates; the cast costs one
`(B, vocab)` row per step.

`top_k` is applied *before* softmax by setting excluded logits to `-inf`, so the
surviving probabilities renormalise to sum to 1 rather than being a truncated,
sub-normalised slice.

### Small operational things that mattered

- **The CSV is flushed every row.** A run that dies at step 1800 of 2500 should
  still leave a usable loss curve.
- **stdout is forced to UTF-8.** An untrained model emits random bytes, which
  decode to U+FFFD, which the default Windows console codepage (cp1252) cannot
  encode — printing the first sample crashed the run before this was fixed.
- **A 60-step pilot run before the real one.** It caught the `np.savez` bug and
  the encoding bug in three minutes rather than an hour in.

## Known limitations at this phase

- No KV-cache, so generation recomputes the whole prefix each step.
- No gradient accumulation across micro-batches, so batch size is bounded by RAM
  (the engine supports it — `.grad` accumulates until zeroed — but the loop does
  not expose it).
- No early stopping; the best-validation checkpoint is saved separately instead.
- No prefetching: batches are gathered synchronously with the training step
  rather than on a background thread.

## `--resume`, and why it exists

The first attempt at the 2500-step run was killed at step 1884 when the session
it was running under was torn down. Nothing was corrupted — the CSV had flushed
every row and checkpoints existed every 250 steps — but `run_summary.json`, the
plot and `final.npz` are only written at the end, so none of them existed.

Rather than throw away 1884 steps, `scripts/train.py` gained `--resume`. Four
details make it a real resume rather than a restart:

1. **`--steps` stays the *total*.** The schedule is a pure function of the step
   index, so resuming at 1750 continues the cosine decay from `9.24e-4` instead of
   re-running warmup. Verified on resume: loss picked up at 3.42, not 6.93, and
   the learning rate was exactly the schedule's value for step 1750 of 2500.
2. **Optimizer state is restored**, not just weights. Adam's `m`, `v` and `t` are
   in the checkpoint; without `t`, bias correction would behave as if the run had
   just started and take a 3.16× step.
3. **The loader is seeded with `seed + start_step`**, so the second leg does not
   replay the exact batch sequence the first leg already trained on.
4. **Best-val and elapsed time are recovered from the CSV**, so a resumed run
   cannot overwrite a better checkpoint or report a wall clock covering only its
   own leg.

The 134 rows the first leg logged *after* its last checkpoint (steps 1751–1884)
were truncated from the CSV before resuming, so the committed curve describes one
coherent trajectory rather than a segment walked twice.
