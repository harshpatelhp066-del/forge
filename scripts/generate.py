"""Generate text from a trained Forge checkpoint.

    python scripts/generate.py --checkpoint checkpoints/final.npz --prompt "ROMEO:"
    python scripts/generate.py --compare        # early vs. fully-trained samples

``--compare`` walks every ``step_*.npz`` checkpoint plus ``final.npz`` and writes
a single before/after file, which is the evidence that the model learned rather
than merely ran.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from forge.model import GPT, GPTConfig  # noqa: E402
from forge.tokenizer import load_tokenizer  # noqa: E402
from forge.train import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_model(path: Path):
    """Rebuild the model from the config stored inside the checkpoint."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    model = GPT(GPTConfig(**meta["config"]))
    load_checkpoint(path, model)
    model.eval()
    return model, meta


def sample(model, tok, prompt: str, n: int, temperature: float, top_k: int | None,
           seed: int) -> str:
    ids = tok.encode(prompt)
    if not ids:
        ids = tok.encode("\n") or [0]
    out = model.generate(np.array(ids)[None, :], max_new_tokens=n,
                         temperature=temperature, top_k=top_k,
                         rng=np.random.default_rng(seed))
    return tok.decode(out[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/final.npz")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer_bpe_1024.json")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--compare", action="store_true",
                   help="sample from every checkpoint, early to final")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    tok = load_tokenizer(args.tokenizer)

    if args.compare:
        ckpt_dir = args.checkpoint.parent
        paths = sorted(ckpt_dir.glob("step_*.npz"))
        final = ckpt_dir / "final.npz"
        if final.exists():
            paths.append(final)
        if not paths:
            print(f"no checkpoints found in {ckpt_dir}", file=sys.stderr)
            return 1

        chunks = []
        for path in paths:
            model, meta = load_model(path)
            text = sample(model, tok, args.prompt, args.tokens,
                          args.temperature, args.top_k, args.seed)
            loss = meta.get("val_loss") or meta.get("train_loss")
            header = (f"checkpoint: {path.name}   step {meta.get('step', '?')}"
                      + (f"   loss {loss:.4f}" if isinstance(loss, float) else ""))
            chunks.append(f"{'=' * 78}\n{header}\n{'=' * 78}\n{text}\n")
            print(chunks[-1])

        out = args.out or (ckpt_dir.parent / "samples" / "before_after.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "Forge — generated samples across training\n"
            f"prompt={args.prompt!r}  temperature={args.temperature}  "
            f"top_k={args.top_k}  seed={args.seed}\n\n" + "\n".join(chunks),
            encoding="utf-8")
        print(f"wrote {out}")
        return 0

    model, meta = load_model(args.checkpoint)
    print(f"{args.checkpoint.name}: step {meta.get('step')}, "
          f"{model.num_parameters():,} parameters\n")
    chunks = []
    for i in range(args.n_samples):
        text = sample(model, tok, args.prompt, args.tokens,
                      args.temperature, args.top_k, args.seed + i)
        chunks.append(text)
        print(f"--- sample {i + 1}/{args.n_samples} " + "-" * 44)
        print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n\n".join(chunks), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
