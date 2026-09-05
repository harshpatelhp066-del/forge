"""Tokenizers: byte-level BPE (default) and a character-level fallback.

**Which one, and why.** The default is byte-pair encoding. Character-level is
simpler and trains instantly, but it spends the model's whole context budget on
spelling: at exactly 1 character per token a 128-token window sees 128
characters, about two lines of Shakespeare, so the model can never observe
enough text to learn structure above the word.

Measured on the 1,115,394-character tiny-shakespeare corpus:

    tokenizer   vocab    tokens   chars/token   train time
    char           65 1,115,394          1.00        0.0 s
    bpe           512   575,345          1.94        0.9 s
    bpe          1024   459,760          2.43        1.4 s
    bpe          2048   388,551          2.87        4.1 s
    bpe          4096   344,129          3.24        9.9 s
    bpe          8192   317,307          3.52       20.9 s

The project trains at **vocab 1024**: 2.43 characters per token, so a 128-token
window covers ~311 characters rather than 128. Beyond ~2048 the returns clearly
diminish while the embedding matrix -- the single largest tensor in a model this
size -- keeps growing linearly, so the extra vocabulary would come straight out
of the budget for depth and width.

BPE is implemented over **bytes**, not Unicode code points. A byte-level base
vocabulary is exactly 256 symbols and can represent any input, so there is no
out-of-vocabulary case and no `<unk>` token to reason about -- an unseen
character simply falls back to its UTF-8 bytes.

Both tokenizers implement the same interface: ``encode``, ``decode``,
``vocab_size``, ``save``, ``load``.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

__all__ = ["BPETokenizer", "CharTokenizer", "load_tokenizer"]


# GPT-2's pre-tokenization split, expressed with the standard library's `re`
# rather than the third-party `regex` module: `[^\W\d_]` is "word character that
# is neither a digit nor an underscore", i.e. a letter, and it is Unicode-aware
# by default in Python 3.
#
# Splitting before merging is what keeps BPE from learning tokens that straddle
# a word boundary (" the cat" as one symbol). The leading optional space in each
# alternative is deliberate: it attaches a word's preceding space to the word, so
# "the" and " the" are distinct tokens and the model never has to spend a token
# on a bare space.
_SPLIT_PATTERN = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    """Byte-level byte-pair encoding.

    Training repeatedly merges the most frequent adjacent symbol pair into a new
    symbol. Starting vocabulary is the 256 byte values; each merge adds one id.
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # pair -> new id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Learn merges until the vocabulary reaches ``vocab_size``.

        Two things make this fast enough to run on a 1 MB corpus in seconds
        rather than minutes:

        1. **Merging over unique pre-tokens, weighted by frequency.** The corpus
           has ~1.1 M characters but only ~30 k distinct words, so the inner loop
           runs over the vocabulary of words rather than over the text.
        2. **An incremental pair index.** Recounting every pair after every merge
           is O(corpus) per merge. Instead, a ``pair -> set of words containing
           it`` index means each merge only touches the words that actually
           contained the merged pair, which is a small fraction of them.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256 (the byte alphabet), got {vocab_size}")
        n_merges = vocab_size - 256

        # Pre-tokenize, then work on the *unique* chunks weighted by frequency.
        freqs = Counter(_SPLIT_PATTERN.findall(text))
        if not freqs:
            raise ValueError("corpus produced no tokens")
        splits = {w: list(w.encode("utf-8")) for w in freqs}

        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[str]] = defaultdict(set)
        for w, f in freqs.items():
            s = splits[w]
            for pair in zip(s, s[1:]):
                pair_counts[pair] += f
                where[pair].add(w)

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(n_merges):
            if not pair_counts:
                if verbose:
                    print(f"  stopped early at {len(self.vocab)} tokens: no pairs left")
                break
            # Ties broken by the pair itself, so training is deterministic.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] < 2:
                if verbose:
                    print(f"  stopped early at {len(self.vocab)} tokens: no pair repeats")
                break

            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            for w in list(where[best]):
                old = splits[w]
                f = freqs[w]
                for pair in zip(old, old[1:]):          # retract this word's old pairs
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(w)

                new = _apply_merge(old, best, new_id)
                splits[w] = new
                for pair in zip(new, new[1:]):          # and post its new ones
                    pair_counts[pair] += f
                    where[pair].add(w)

            if verbose and (i + 1) % 200 == 0:
                print(f"  merge {i + 1}/{n_merges}: "
                      f"{self.vocab[new_id]!r} (vocab {len(self.vocab)})")

        self._cache.clear()
        return self

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the *earliest-learned* applicable merge. Merge order is the
            # tokenizer's definition; applying a later merge first would produce
            # a different segmentation from the one training implied.
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            ids = _apply_merge(ids, best, best_rank)

        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in _SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids) -> str:
        parts = []
        for i in ids:
            i = int(i)
            if i not in self.vocab:
                raise KeyError(f"token id {i} is not in the vocabulary")
            parts.append(self.vocab[i])
        # errors="replace": a partial generation can end mid-multi-byte
        # character, and refusing to render it would be worse than a placeholder.
        return b"".join(parts).decode("utf-8", errors="replace")

    def encode_to_array(self, text: str, dtype=np.uint16) -> np.ndarray:
        ids = self.encode(text)
        if self.vocab_size > np.iinfo(dtype).max + 1:
            dtype = np.int32
        return np.array(ids, dtype=dtype)

    # ------------------------------------------------------------------ #

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            # JSON has no tuple keys, so merges are stored as [a, b, new_id]
            # triples in learned order -- which also makes the file readable.
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "bpe":
            raise ValueError(f"not a BPE tokenizer file: {path}")
        tok = cls()
        for a, b, i in payload["merges"]:
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        return tok


class CharTokenizer:
    """Character-level fallback: one token per distinct character in the corpus.

    Kept for comparison rather than as the default -- see the module docstring.
    Unlike the byte-level BPE it *does* have an out-of-vocabulary case, since a
    character absent from the training corpus has no id; ``encode`` skips such
    characters rather than inventing an `<unk>`.
    """

    def __init__(self, chars: str = ""):
        self.itos: list[str] = sorted(set(chars))
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def train(self, text: str, vocab_size: int | None = None) -> "CharTokenizer":
        self.itos = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        return self

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def encode_to_array(self, text: str, dtype=np.uint16) -> np.ndarray:
        return np.array(self.encode(text), dtype=dtype)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "char", "chars": "".join(self.itos)}),
                        encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "char":
            raise ValueError(f"not a character tokenizer file: {path}")
        return cls(payload["chars"])


def load_tokenizer(path):
    """Load either tokenizer type, dispatching on the file's ``type`` field."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = payload.get("type")
    if kind == "bpe":
        return BPETokenizer.load(path)
    if kind == "char":
        return CharTokenizer.load(path)
    raise ValueError(f"unknown tokenizer type {kind!r} in {path}")


def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
