# Phase 4: Tokenizer and data pipeline

Files: [`forge/tokenizer.py`](forge/tokenizer.py), [`forge/data.py`](forge/data.py),
[`scripts/prepare_data.py`](scripts/prepare_data.py),
[`tests/test_phase4_tokenizer.py`](tests/test_phase4_tokenizer.py)

## Which tokenizer, and why

Byte-level BPE, the preferred option in the brief. Character-level is
implemented too, as a working fallback and as the comparison that justifies the
choice, but it is not what the model trains on.

Measured on the 1,115,394-character tiny-shakespeare corpus:

| tokenizer | vocab | tokens | chars/token | train time |
|---|---:|---:|---:|---:|
| char | 65 | 1,115,394 | 1.00 | 0.0 s |
| bpe | 512 | 575,345 | 1.94 | 0.9 s |
| **bpe** | **1024** | **459,760** | **2.43** | **1.4 s** |
| bpe | 2048 | 388,551 | 2.87 | 4.1 s |
| bpe | 4096 | 344,129 | 3.24 | 9.9 s |
| bpe | 8192 | 317,307 | 3.52 | 20.9 s |

The argument is about what a fixed context window can *see*. At one character per
token, a 128-token window covers 128 characters, roughly two lines of
Shakespeare, and the model spends its capacity learning how words are spelled.
At vocab 1024 the same window covers ~311 characters, so the model can spend its
capacity on what follows what.

Training settles on vocab 1024. Past ~2048 the compression gains clearly
flatten while the embedding matrix keeps growing linearly, and in a model this
small the embedding is the single largest tensor, extra vocabulary would come
straight out of the budget for depth and width.

BPE is over bytes, not Unicode code points. The base vocabulary is then
exactly 256 symbols and can represent any input, so there is no out-of-vocabulary
case and no `<unk>` token: an unseen character falls back to its UTF-8 bytes. The
character tokenizer has no such property, and a test demonstrates the difference
concretely, a `CharTokenizer` trained on `"abc"` silently drops `x`, `y`, `z`
from `"abcxyz"`, while byte-level BPE round-trips Japanese, emoji and random
code points it has never seen.

## Verification status

52/52 Phase 4 tests pass (401 total).

The property that matters most is exact round-tripping, and it is tested hardest:

- `decode(encode(x)) == x` on the training corpus; on empty strings, lone spaces
  and newlines, runs of whitespace, punctuation, digits, mixed case and
  contractions; on Unicode never seen during training (accented Latin,
  Japanese, emoji, mathematical symbols); and on 30 rounds of random byte soup
  drawn from code points up to U+2000.
- Compression is real: fewer tokens than bytes, and a larger vocabulary never
  produces *more* tokens (checked monotone across four vocab sizes).
- Training is deterministic, same corpus and vocab size give byte-identical
  merges, because the `max` over pair counts breaks ties on the pair itself.
- Learned tokens never span a pre-token boundary, checked by re-splitting every
  multi-character token in the vocabulary and asserting it yields exactly one
  piece.
- Save/load round-trips, and `load_tokenizer` dispatches on the file's `type`.

Data loader:

- **Targets are inputs shifted by one**, asserted both structurally
  (`x[:, 1:] == y[:, :-1]`) and against the source array.
- **Train/val are disjoint**, and no training window ever reaches into the
  validation region, checked over 200 sampled batches.
- **Shuffling actually shuffles**: epoch order differs from sequential order, no
  window is served twice in an epoch, order is reproducible for a fixed seed and
  differs across seeds.
- Bounds are respected for both splits; ragged final batches are handled;
  too-short corpora, 2-D inputs and bad `val_fraction` all raise.
- End-to-end: text → tokens → batches → decoded text, with the decoded window
  asserted to be a genuine substring of the source.

## Design notes

### The merge loop needed an incremental index

The obvious BPE implementation recounts every adjacent pair after every merge.
That is O(corpus) per merge, and at 1.1 M characters × 768 merges it is roughly
10⁹ Python-level operations, so minutes rather than seconds.

Two changes bring it to 1.4 s:

1. **Merge over unique pre-tokens weighted by frequency.** The corpus has 1.1 M
   characters but far fewer distinct words, so the inner loop runs over the word
   vocabulary rather than the text.
2. **A `pair → set of words containing it` index.** After choosing a merge, only
   the words that actually contained that pair are touched: their old pairs are
   retracted from the global counter and their new pairs posted. Most words are
   untouched by most merges.

The bookkeeping is the fiddly part, a word must retract *all* its old pairs and
post *all* its new ones, because a pair can survive its own merge (merging `aa`
in `aaa` leaves an `aa`), and zero-count entries must be deleted or `max` will
eventually select a pair that no longer occurs.

### Pre-tokenization, and the vocabulary ceiling it creates

Text is split with a GPT-2-style pattern before merging, so BPE cannot learn a
token spanning a word boundary like `" the cat"`, which would waste vocabulary
on word pairs and generalize badly. The leading optional space in each
alternative attaches a word's preceding space to the word, so `"the"` and
`" the"` are distinct tokens and no token is ever spent on a bare space.

This has a consequence that surfaced while writing the tests, and it is worth
knowing before choosing a vocabulary size: **merges never cross a pre-token
boundary, so the achievable vocabulary is bounded by the number of distinct
pre-tokens, not by corpus size.** Once every distinct word has been merged into a
single symbol there are no adjacent pairs left anywhere and training stops.
Repeating a corpus 20× raises pair *counts* but not the ceiling. A test asserts
this directly, after an earlier version of the test wrongly assumed more text
would buy more vocabulary.

The training corpus is comfortably above the ceiling for vocab 1024 (it reaches
1024 exactly), but a smaller or more repetitive corpus would stop early, which
the tokenizer does deliberately rather than merging pairs that occur once and buy
no compression.

The pattern uses the standard library's `re`, not the third-party `regex` module,
so `\p{L}` is unavailable; `[^\W\d_]`, a word character that is neither a digit
nor an underscore, is the Unicode-aware equivalent for letters in Python 3.

### Encoding applies the earliest-learned applicable merge

At encode time, several merges may apply to a sequence. The rule is to always
apply the lowest-ranked (earliest-learned) one first. Merge order *is* the
tokenizer's definition; applying a later merge first would produce a different
segmentation from the one training implied, and the model would see token
sequences it was never trained on.

### `decode` uses `errors="replace"`

A generated sequence can end in the middle of a multi-byte UTF-8 character,
since byte-level BPE has no notion of character boundaries. Raising there would make
generation fail on a truncation that is entirely expected; a replacement
character is the honest rendering.

### The train/val split is contiguous, not random

A single cut point, not randomly held-out windows. With random windows, a
validation window could overlap a training window by up to `block_size - 1`
tokens, so validation loss would partly measure memorization of text the model
had already trained on, and the val curve would look better than the model is.
A contiguous split makes the two sets genuinely disjoint, which
`test_no_training_window_overlaps_the_validation_split` verifies over 200 batches.

The cost is that validation comes from one specific stretch of the corpus (the
last 10%), so it is a slightly different distribution than the training text.
For a single-author corpus that is an acceptable trade for a clean measurement.

### Two samplers, for two purposes

`random_batch` samples window offsets uniformly *with replacement*. This is the
standard sampler for language-model training: there are 413,656 distinct training
windows against a few thousand training steps, so sampling with replacement is
statistically indistinguishable from an epoch schedule and needs no bookkeeping.

`epoch()` iterates non-overlapping windows in shuffled order, so one pass sees
each token exactly once. It is used for validation, where exact coverage makes
the number comparable across checkpoints. Shuffling matters there and in training:
consecutive windows are consecutive text, so unshuffled batches produce correlated
gradient estimates, and Adam's second-moment estimate ends up tracking a moving
target rather than the true gradient scale.

Window gathering is vectorized, `offsets[:, None] + arange(T+1)` produces the
whole `(B, T+1)` batch in one fancy-index, rather than looping in Python.

### Tokens are cached as `uint16`

Vocab 1024 fits in 16 bits, halving the memory of the token array against `int32`
and making it cheap to keep the whole corpus resident. `encode_to_array`
automatically widens to `int32` if the vocabulary would not fit.

## Known limitations at this phase

- No special tokens (`<bos>`, `<eos>`, `<pad>`). The model trains on a continuous
  token stream with no document boundaries, which is fine for one continuous work
  but would need `<eos>` for a multi-document corpus.
- No `ignore_index` padding path is exercised, since all sequences are exactly
  `block_size`; `cross_entropy` supports it but the loader never produces ragged
  sequences.
- BPE training holds the whole corpus in memory as Python objects. Fine at 1 MB;
  a multi-gigabyte corpus would need chunked counting.
- The encode cache is unbounded. On a corpus with millions of distinct
  pre-tokens it would grow without limit.
