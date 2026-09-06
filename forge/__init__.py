"""Forge, a deep learning framework written from scratch on NumPy.

Phase 1 gives you a reverse-mode autodiff engine (:mod:`forge.tensor`) and the
finite-difference checker that certifies it (:mod:`forge.gradcheck`). Every
later phase, layers, optimizer, transformer, tokenizer, training loop, is
built on those two files and nothing else.
"""

from .tensor import Tensor, default_dtype, is_grad_enabled, no_grad, set_default_dtype
from .gradcheck import check_gradient, numerical_gradient, relative_error

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "set_default_dtype",
    "default_dtype",
    "no_grad",
    "is_grad_enabled",
    "check_gradient",
    "numerical_gradient",
    "relative_error",
]
