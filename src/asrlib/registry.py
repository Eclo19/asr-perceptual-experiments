"""
This file contains the work to convert config files into working objects and the register
of all frontends, models, corpora, and tokenizers the ASR lib should know about. 

Author: Eric Oliveira 
Date: 10/27/2026

* Written with the help of Claude *

"""

from typing import Callable, Any
import warnings


# Register all known modules that ASR must know
_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "frontend": {},     # "logmel", "mfcc", "carfac", later "gammatone"
    "model": {},        # "transformer_ctc", later "conformer_ctc", "transformer_aed"
    "corpus": {},       # "librispeech", "local_folder", later multilingual corpora
    "tokenizer": {},    # "char", later "bpe"
}



def register(kind: str, name: str):
    """Class decorator that makes a component buildable by name.

    Example:
        @register("frontend", "carfac")
        class CARFAC(FrontEnd): ...

    Registering a name twice replaces the old entry with a warning. That keeps
    notebooks re-runnable: re-running a cell that defines a class just redefines it.
    """
    def deco(cls):
        if name in _REGISTRY[kind]:
            warnings.warn(f"Re-registering {kind} '{name}' (replacing the previous definition)")
        _REGISTRY[kind][name] = cls
        return cls
    return deco


def build(kind: str, name: str, **kwargs) -> Any:
    """Instantiate a registered component; fails loudly and lists the valid names."""
    if name not in _REGISTRY[kind]:
        raise KeyError(f"Unknown {kind} '{name}'. Registered: {sorted(_REGISTRY[kind])}")
    return _REGISTRY[kind][name](**kwargs)
