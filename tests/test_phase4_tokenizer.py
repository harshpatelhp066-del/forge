"""Phase 4 verification: tokenizer and data pipeline.

Run with:  python -m pytest tests/test_phase4_tokenizer.py -v

The property that matters most for a tokenizer is that ``decode(encode(x)) == x``
exactly, for every input, including ones the tokenizer never saw in training.
Everything else -- compression ratio, merge quality -- is a performance question;
that one is a correctness question.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.data import DataLoader  # noqa: E402
from forge.tokenizer import (BPETokenizer, CharTokenizer,  # noqa: E402
                             load_tokenizer, _SPLIT_PATTERN)

CORPUS = """First Citizen:
Before we proceed any further, hear me speak.
All: Speak, speak.
First Citizen: You are all resolved rather to die than to famish?
All: Resolved. resolved.
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles.
"""


@pytest.fixture(scope="module")
def bpe():
    return BPETokenizer().train(CORPUS, vocab_size=350)


# --------------------------------------------------------------------------- #
# BPE: the round-trip property
# --------------------------------------------------------------------------- #

def test_bpe_roundtrips_the_training_corpus_exactly(bpe):
    assert bpe.decode(bpe.encode(CORPUS)) == CORPUS


@pytest.mark.parametrize("text", [
    "",
    " ",
    "\n",
    "a",
    "hello world",
    "First Citizen:",
    "   multiple   spaces   ",
    "\n\n\ttabs and newlines\n",
    "Punctuation!? Yes... (really).",
    "1234567890",
    "MiXeD CaSe WoRdS",
    "'tis 'twas don't",
])
def test_bpe_roundtrips_arbitrary_ascii(bpe, text):
    assert bpe.decode(bpe.encode(text)) == text


@pytest.mark.parametrize("text", [
    "café naïve résumé",
    "日本語のテキスト",
    "emoji: 🔥🚀✨",
    "mixed: café 日本 🔥 end",
    "Ω≈ç√∫˜µ≤≥÷",
])
def test_bpe_roundtrips_unicode_never_seen_in_training(bpe, text):
    """Byte-level BPE has no out-of-vocabulary case, by construction."""
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_roundtrips_random_byte_soup(bpe):
    """Adversarial: random text drawn from a much wider character range."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        text = "".join(chr(int(c)) for c in rng.integers(1, 0x2000, size=60))
        assert bpe.decode(bpe.encode(text)) == text


def test_bpe_vocab_size_never_exceeds_the_request(bpe):
    """The fixture corpus is ~450 bytes, so it exhausts repeating pairs early.

    Stopping there is deliberate -- merging a pair that occurs once buys no
    compression and just burns an id -- so the invariant is an upper bound plus
    internal consistency, not an exact size.  Exact sizing on a corpus large
    enough to support it is asserted in the next test.
    """
    assert bpe.vocab_size <= 350
    assert len(bpe.merges) == bpe.vocab_size - 256
    assert len(set(bpe.merges.values())) == len(bpe.merges)   # ids are unique


def test_bpe_reaches_the_exact_vocab_size_on_a_corpus_that_supports_it():
    rng = np.random.default_rng(0)
    alphabet = list("abcdefghij")
    words = ["".join(rng.choice(alphabet, size=6)) for _ in range(2000)]
    text = " ".join(rng.choice(words, size=20000))

    tok = BPETokenizer().train(text, vocab_size=1000)
    assert tok.vocab_size == 1000
    assert len(tok.merges) == 1000 - 256
    assert tok.decode(tok.encode(text[:5000])) == text[:5000]


def test_vocabulary_is_bounded_by_the_number_of_distinct_pretokens():
    """A direct consequence of pre-tokenisation, worth knowing before sizing a vocab.

    Merges never cross a pre-token boundary, so once every distinct word has been
    merged down to a single symbol there are no adjacent pairs left anywhere and
    training stops -- no matter how much text is fed in or how large a vocabulary
    is requested.  Repeating a corpus therefore raises pair *counts* but not the
    achievable vocabulary size.
    """
    small = BPETokenizer().train(CORPUS, vocab_size=5000)
    repeated = BPETokenizer().train(CORPUS * 20, vocab_size=5000)
    n_distinct = len(set(_SPLIT_PATTERN.findall(CORPUS)))

    assert small.vocab_size < 5000
    assert repeated.vocab_size < 5000
    # The ceiling scales with the distinct-word count, not with corpus length.
    assert repeated.vocab_size - 256 < 3 * n_distinct


def test_bpe_base_vocabulary_is_the_256_bytes(bpe):
    for i in range(256):
        assert bpe.vocab[i] == bytes([i])


def test_bpe_actually_compresses_relative_to_bytes(bpe):
    """The whole point of BPE: fewer tokens than characters."""
    n_tokens = len(bpe.encode(CORPUS))
    n_bytes = len(CORPUS.encode("utf-8"))
    ratio = n_bytes / n_tokens
    assert ratio > 1.5, f"only {ratio:.2f} bytes/token -- BPE learned nothing useful"
    assert n_tokens < n_bytes


def test_more_merges_compress_better():
    """A larger vocabulary must not produce more tokens."""
    lengths = []
    for vocab_size in (256, 300, 400, 600):
        tok = BPETokenizer().train(CORPUS, vocab_size=vocab_size)
        lengths.append(len(tok.encode(CORPUS)))
    assert lengths == sorted(lengths, reverse=True), lengths
    assert lengths[-1] < lengths[0] * 0.6


def test_bpe_training_is_deterministic():
    """Same corpus and vocab size must give byte-identical merges."""
    a = BPETokenizer().train(CORPUS, vocab_size=320)
    b = BPETokenizer().train(CORPUS, vocab_size=320)
    assert a.merges == b.merges
    assert a.encode(CORPUS) == b.encode(CORPUS)


def test_bpe_learns_frequent_multi_character_tokens(bpe):
    """Merges should capture recurring substrings, not arbitrary byte pairs."""
    learned = {v.decode("utf-8", errors="replace") for v in bpe.vocab.values()}
    multi = [s for s in learned if len(s) > 1]
    assert len(multi) > 40
    # The corpus is dominated by these, so they should have been merged.
    assert any("to" in s for s in multi)
    assert any(len(s) >= 4 for s in multi), "no token longer than 3 characters"


def test_bpe_merges_never_cross_a_pretoken_boundary(bpe):
    """A learned token must not span the split pattern's boundaries.

    Without pre-tokenisation, BPE would happily learn " the cat" as one symbol,
    which wastes vocabulary on word pairs and generalises badly.
    """
    for token in bpe.vocab.values():
        s = token.decode("utf-8", errors="replace")
        if len(s) > 1 and "�" not in s:
            pieces = _SPLIT_PATTERN.findall(s)
            assert len(pieces) == 1, f"token {s!r} spans multiple pre-tokens: {pieces}"


def test_bpe_encoding_is_stable_under_caching(bpe):
    """The chunk cache must not change results on a second call."""
    first = bpe.encode(CORPUS)
    second = bpe.encode(CORPUS)
    assert first == second


def test_bpe_rejects_a_vocab_smaller_than_the_byte_alphabet():
    with pytest.raises(ValueError, match="at least 256"):
        BPETokenizer().train(CORPUS, vocab_size=100)


def test_bpe_stops_early_when_no_pair_repeats():
    """Asking for more merges than the corpus supports must not crash."""
    tok = BPETokenizer().train("abcdef", vocab_size=1000)
    assert tok.vocab_size < 1000
    assert tok.decode(tok.encode("abcdef")) == "abcdef"


def test_bpe_decode_rejects_an_unknown_id(bpe):
    with pytest.raises(KeyError, match="not in the vocabulary"):
        bpe.decode([99999])


def test_bpe_encode_to_array_dtype(bpe):
    arr = bpe.encode_to_array(CORPUS)
    assert arr.dtype == np.uint16
    assert arr.max() < bpe.vocab_size
    assert bpe.decode(arr) == CORPUS


def test_bpe_save_and_load_roundtrip(bpe, tmp_path):
    path = tmp_path / "tok.json"
    bpe.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == bpe.vocab_size
    assert loaded.merges == bpe.merges
    assert loaded.encode(CORPUS) == bpe.encode(CORPUS)
    assert loaded.decode(loaded.encode(CORPUS)) == CORPUS


def test_load_tokenizer_dispatches_on_type(bpe, tmp_path):
    bpath, cpath = tmp_path / "b.json", tmp_path / "c.json"
    bpe.save(bpath)
    CharTokenizer().train(CORPUS).save(cpath)
    assert isinstance(load_tokenizer(bpath), BPETokenizer)
    assert isinstance(load_tokenizer(cpath), CharTokenizer)
    with pytest.raises(ValueError, match="not a BPE"):
        BPETokenizer.load(cpath)


# --------------------------------------------------------------------------- #
# Character tokenizer (the documented fallback)
# --------------------------------------------------------------------------- #

def test_char_tokenizer_roundtrips_and_sizes():
    tok = CharTokenizer().train(CORPUS)
    assert tok.vocab_size == len(set(CORPUS))
    assert tok.decode(tok.encode(CORPUS)) == CORPUS
    assert len(tok.encode(CORPUS)) == len(CORPUS)   # one token per character


def test_char_tokenizer_has_an_out_of_vocabulary_case():
    """The concrete reason byte-level BPE was preferred."""
    tok = CharTokenizer().train("abc")
    assert tok.encode("abcxyz") == tok.encode("abc")     # x, y, z silently dropped
    assert tok.decode(tok.encode("abcxyz")) == "abc"


def test_char_tokenizer_save_load(tmp_path):
    tok = CharTokenizer().train(CORPUS)
    path = tmp_path / "c.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.itos == tok.itos
    assert loaded.decode(loaded.encode(CORPUS)) == CORPUS


def test_bpe_compresses_better_than_character_level():
    """Quantifies the tradeoff documented in the tokenizer module."""
    bpe = BPETokenizer().train(CORPUS, vocab_size=512)
    char = CharTokenizer().train(CORPUS)
    n_bpe, n_char = len(bpe.encode(CORPUS)), len(char.encode(CORPUS))
    assert n_bpe < n_char * 0.6, f"bpe={n_bpe} char={n_char}"


# --------------------------------------------------------------------------- #
# Data loader
# --------------------------------------------------------------------------- #

@pytest.fixture
def tokens():
    return np.arange(1000, dtype=np.int32)


def test_targets_are_inputs_shifted_by_one(tokens):
    """The definition of next-token prediction, asserted directly."""
    dl = DataLoader(tokens, block_size=8, batch_size=4)
    x, y = dl.random_batch("train")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert np.array_equal(x[:, 1:], y[:, :-1])
    # And against the source array: each row is a contiguous window.
    for row_x, row_y in zip(x, y):
        start = int(row_x[0])
        assert np.array_equal(row_x, np.arange(start, start + 8))
        assert np.array_equal(row_y, np.arange(start + 1, start + 9))


def test_train_and_val_splits_are_disjoint_and_contiguous(tokens):
    dl = DataLoader(tokens, block_size=8, batch_size=4, val_fraction=0.1)
    assert len(dl.train_tokens) == 900
    assert len(dl.val_tokens) == 100
    assert dl.train_tokens[-1] + 1 == dl.val_tokens[0]
    assert not (set(dl.train_tokens.tolist()) & set(dl.val_tokens.tolist()))


def test_no_training_window_overlaps_the_validation_split(tokens):
    """The reason for a contiguous split rather than random held-out windows.

    Every training window must lie entirely inside the training tokens, or
    validation loss would partly measure memorisation of text already trained on.
    """
    dl = DataLoader(tokens, block_size=16, batch_size=8, val_fraction=0.2)
    boundary = len(dl.train_tokens)
    for _ in range(200):
        x, y = dl.random_batch("train")
        assert y.max() < boundary


def test_batches_stay_in_bounds_for_both_splits(tokens):
    dl = DataLoader(tokens, block_size=16, batch_size=8, val_fraction=0.25)
    for split in ("train", "val"):
        for _ in range(100):
            x, y = dl.random_batch(split)
            assert x.shape == (8, 16) and y.shape == (8, 16)
            assert x.min() >= 0 and y.max() <= tokens.max()


def test_shuffling_actually_shuffles(tokens):
    """Consecutive epoch batches must not be consecutive text."""
    dl = DataLoader(tokens, block_size=10, batch_size=4, seed=0)
    starts = [int(x[0, 0]) for x, _ in dl.epoch("train", shuffle=True)]
    ordered = sorted(starts)
    assert starts != ordered
    # ...and no window is served twice in one epoch.
    assert len(set(starts)) == len(starts)


def test_epoch_without_shuffle_is_sequential(tokens):
    dl = DataLoader(tokens, block_size=10, batch_size=4, seed=0)
    starts = [int(x[0, 0]) for x, _ in dl.epoch("train", shuffle=False)]
    assert starts == sorted(starts)


def test_epoch_covers_each_token_once_with_default_stride(tokens):
    """Non-overlapping windows: one pass sees the split exactly once."""
    dl = DataLoader(tokens, block_size=10, batch_size=1, seed=0)
    seen = np.concatenate([x.ravel() for x, _ in dl.epoch("train", drop_last=False)])
    assert len(seen) == len(set(seen.tolist()))
    assert len(seen) >= len(dl.train_tokens) - 10


def test_epoch_is_reproducible_for_a_fixed_seed(tokens):
    a = [int(x[0, 0]) for x, _ in DataLoader(tokens, 10, 4, seed=7).epoch("train")]
    b = [int(x[0, 0]) for x, _ in DataLoader(tokens, 10, 4, seed=7).epoch("train")]
    c = [int(x[0, 0]) for x, _ in DataLoader(tokens, 10, 4, seed=8).epoch("train")]
    assert a == b
    assert a != c


def test_epoch_drop_last_controls_the_ragged_final_batch(tokens):
    dl = DataLoader(tokens, block_size=10, batch_size=7, seed=0)
    dropped = list(dl.epoch("train", drop_last=True))
    kept = list(dl.epoch("train", drop_last=False))
    assert all(x.shape[0] == 7 for x, _ in dropped)
    assert len(kept) >= len(dropped)
    assert kept[-1][0].shape[0] <= 7


def test_dataloader_rejects_a_corpus_too_short_for_the_context():
    with pytest.raises(ValueError, match="fewer than"):
        DataLoader(np.arange(20), block_size=64, batch_size=2)


def test_dataloader_rejects_bad_arguments():
    with pytest.raises(ValueError, match="1-D"):
        DataLoader(np.zeros((10, 10)), block_size=2, batch_size=2)
    with pytest.raises(ValueError, match="val_fraction"):
        DataLoader(np.arange(100), block_size=2, batch_size=2, val_fraction=0.0)
    with pytest.raises(ValueError, match="unknown split"):
        DataLoader(np.arange(100), block_size=2, batch_size=2)._split("test")


def test_dataloader_summary_reports_sizes(tokens):
    s = DataLoader(tokens, block_size=8, batch_size=4).summary()
    assert "train tokens 900" in s
    assert "val tokens 100" in s


# --------------------------------------------------------------------------- #
# Tokenizer + loader together
# --------------------------------------------------------------------------- #

def test_end_to_end_text_to_batches_and_back():
    """Text -> tokens -> batches -> decoded text must round-trip."""
    tok = BPETokenizer().train(CORPUS, vocab_size=400)
    ids = tok.encode_to_array(CORPUS * 6)
    dl = DataLoader(ids, block_size=16, batch_size=4, seed=0)
    x, y = dl.random_batch("train")

    assert x.dtype.kind in "iu"
    assert x.max() < tok.vocab_size
    text = tok.decode(x[0])
    assert isinstance(text, str) and len(text) > 0
    # The decoded window is a genuine substring of the source.
    assert text in CORPUS * 6
