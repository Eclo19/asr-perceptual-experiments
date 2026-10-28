"""
Define the Tokenizer abstract base class and any child tokenizers. Add tokenizers with the 
register decorator to register them.

Author: Eric Oliveira 
Date: 10/27/2026

* Written with the help of Claude *

"""

from abc import ABC, abstractmethod
from typing import Iterable

from .registry import register

# ==============================
# Tokenizer ABC
# ==============================

class Tokenizer(ABC):
    """Text ↔ ids. Id 0 is the CTC blank (and the padding id).
    
    Enforce base behavior for all tokenizers.
    
    """
    blank_id: int = 0

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def normalize(self, text: str) -> str: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> str: ...

    @abstractmethod
    def state_dict(self) -> dict: ...


@register("tokenizer", "char")
class CharTokenizer(Tokenizer):
    """Lower-case letters, space and apostrophe, plus <blank>, <bos>, <eos>: 31 symbols."""

    # <blank> is the CTC blank and the padding id (0); <bos>/<eos> are reserved for seq2seq decoders
    SPECIALS = ("<blank>", "<bos>", "<eos>")
    CHARS = " 'abcdefghijklmnopqrstuvwxyz"                  # space, apostrophe, a-z

    def __init__(self):
        # Specials first, so their ids are 0, 1, 2
        self.vocab = list(self.SPECIALS) + list(self.CHARS)

        # Character → id lookup
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)                              # 31

    def normalize(self, text: str) -> str:
        """Lower-case, map unknown characters to spaces, collapse runs of spaces, trim the ends."""
        # Keep known characters; anything else becomes a space so words don't merge
        kept = "".join(c if c in self.stoi else " " for c in text.lower())

        # split() drops empty pieces, so this collapses repeated spaces and trims the ends
        return " ".join(kept.split())

    def encode(self, text: str) -> list[int]:
        """Text -> ids, always through normalize(), so decode(encode(t)) == normalize(t)."""
        return [self.stoi[c] for c in self.normalize(text)]

    def decode(self, ids: Iterable[int]) -> str:
        """Ids -> text, dropping the special tokens."""
        n_special = len(self.SPECIALS)
        return "".join(self.vocab[int(i)] for i in ids if int(i) >= n_special)

    def state_dict(self) -> dict:
        return {"kind": "char", "vocab": self.vocab}