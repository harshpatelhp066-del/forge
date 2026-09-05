"""Download the corpus, train the tokenizer, and cache the encoded token stream.

    python scripts/prepare_data.py --vocab-size 1024

Corpus: tiny-shakespeare (1.1 MB, public domain -- the works of Shakespeare).
Downloaded on demand rather than committed; see .gitignore for the reasoning.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.tokenizer import BPETokenizer, CharTokenizer  # noqa: E402

CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def download(path: Path) -> None:
    if path.exists():
        print(f"corpus already present: {path} ({path.stat().st_size:,} bytes)")
        return
    print(f"downloading {CORPUS_URL}")
    DATA.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CORPUS_URL, timeout=60) as resp:
        path.write_bytes(resp.read())
    print(f"saved {path} ({path.stat().st_size:,} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
    ap.add_argument("--corpus", type=Path, default=DATA / "tinyshakespeare.txt")
    args = ap.parse_args()

    download(args.corpus)
    text = args.corpus.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} characters, {len(set(text))} distinct")

    if args.tokenizer == "bpe":
        tok = BPETokenizer()
        t0 = time.time()
        tok.train(text, vocab_size=args.vocab_size, verbose=True)
        print(f"trained BPE in {time.time() - t0:.1f}s -> vocab {tok.vocab_size}")
    else:
        tok = CharTokenizer().train(text)
        print(f"character vocab: {tok.vocab_size}")

    t0 = time.time()
    ids = tok.encode_to_array(text)
    print(f"encoded in {time.time() - t0:.1f}s")

    assert tok.decode(ids) == text, "round-trip failed"
    print("round-trip verified: decode(encode(corpus)) == corpus")

    ratio = len(text) / len(ids)
    print(f"tokens: {len(ids):,}  ({ratio:.2f} characters per token)")

    tok_path = DATA / f"tokenizer_{args.tokenizer}_{tok.vocab_size}.json"
    ids_path = DATA / f"tokens_{args.tokenizer}_{tok.vocab_size}.npy"
    tok.save(tok_path)
    np.save(ids_path, ids)
    print(f"wrote {tok_path.name} and {ids_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
