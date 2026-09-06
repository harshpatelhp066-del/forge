# Phase 6 — Polish and delivery

Files: [`README.md`](README.md), [`LICENSE`](LICENSE), [`.gitignore`](.gitignore)

## What was done

- README with an architecture explanation, two Mermaid diagrams, the concrete
  meaning of "from scratch", reproduction instructions, the real loss curve, real
  before/after generated text, and an honest limitations section.
- MIT licence; the tiny-shakespeare corpus is public domain.
- Git history in logical stages: one commit per phase, plus two commits for
  cross-cutting fixes found late (the graph-memory bug and a factual correction).

## What is committed, and what is not

The decision that needed a reason either way is checkpoint weights.

**Not committed** — `checkpoints/*.npz`. Eleven checkpoints at ~11 MB each is
~130 MB, they change completely on every run, and a binary in git history is in
that history permanently. The training command plus the fixed seed reproduces
them, which is the thing that actually matters for a reader.

**Committed** — the evidence that the run happened and what it produced:

| File | Why |
|---|---|
| `checkpoints/loss_curve.csv` | The raw per-step record: 2500 rows of train loss, val loss, LR, grad norm, elapsed time, tokens seen |
| `checkpoints/loss_curve.png` | The figure the README shows |
| `checkpoints/run_summary.json` | Config, parameter counts, final/best losses, wall clock — machine-readable |
| `samples/before_after.txt` | Generations from every checkpoint, step 0 → final, same prompt and seed |
| `samples/{step_000000,step_001000,final}.txt` | The three excerpts quoted in the README |

The whole repository is **657 KB**. The reasoning is written into `.gitignore`
itself rather than only here, since that is where someone will be standing when
they wonder about it.

Also excluded: `data/*.txt` (1.1 MB corpus, downloaded by
`scripts/prepare_data.py` in ~1 s), `data/*.npy` and `data/tokenizer_*.json`
(derived artefacts, rebuilt in ~3 s), and the training console logs.

## Honesty in the limitations section

The brief asked for an honest "Known limitations" section, and the temptation
with a portfolio piece is to write limitations that are secretly boasts. The
section is organised so the genuinely unflattering facts come first:

- **Scale.** 940,800 parameters against GPT-3's 175 billion — about 186,000×
  smaller — trained on ~1 MB against ~570 GB. The README says plainly that the
  model learns Shakespeare's *surface form* and some local grammar, and does not
  learn meaning or hold a thought across a paragraph. Reading the final samples,
  that is exactly what they show: correct speaker labels, verse rhythm, archaic
  grammar, and invented words like "draitor" and "seasure".
- **Skipped optimizations**, each with what it actually costs rather than a
  hand-wave: no KV-cache (generation is O(T²) per token instead of O(T) — the
  largest single win left on the table), no mixed precision (it is a GPU
  throughput technique and there is no GPU), no multi-GPU (NumPy is single-device
  by definition), no flash attention (memory is quadratic in context), no
  gradient checkpointing.
- **Engine limitations**: no in-place ops and no version counter to detect a
  buffer mutated after being taped, no second derivatives, CPU/float32 only,
  no special tokens, BPE training holds the corpus in memory.

The point of listing what a KV-cache *would* buy, rather than just saying it was
skipped, is that it shows the omission was a decision rather than an oversight.

## The performance claim is measured, not asserted

The README states that a training step is dominated by matmul and that NumPy's
BLAS reaches ~179 GFLOP/s on this machine, so there is no large algorithmic win
left without changing hardware. That came from profiling a real step:

```
matmul 1024x1024: 178.7 GFLOP/s

forward  0.609s   backward 0.741s   opt.step 0.005s
   ncalls  tottime  filename:lineno(function)
       50    0.866   tensor.py: matmul backward
       50    0.658   tensor.py: __matmul__
```

Worth stating because "it is slow because it is NumPy" is the kind of claim that
sounds reasonable and is often wrong — the alternative explanation, that the
Python interpreter overhead dominates, would have implied a very different fix.

## What a reader should do first

The README puts `python -m pytest tests/ -q` and
`python scripts/gradcheck_report.py` as step 1 of reproduction, before
downloading any data, with the note that nothing else is meaningful if they fail.
For a project whose entire claim is "the gradients are correct", the verification
should be the cheapest and first thing a stranger can run — it takes about six
seconds.

## Known limitations of this phase

- The Mermaid diagrams render on GitHub but not in every Markdown viewer.
- `run_summary.json` records the config and results but not the git commit that
  produced them, so a reader matching a curve to a code state has to rely on
  commit order.
- No CI. The test suite is fast enough (~6 s) that a GitHub Actions workflow
  would be cheap and is the obvious next addition.
