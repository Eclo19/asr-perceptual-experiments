"""
Define ASR model architectures allowed for experiments. It is vital that different front-ends
are benchmarked with the same model architecture. 

Author: Eric Oliveira
Date: 10/28/2026

* Written with the help of Claude *

"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .registry import register

class AcousticModel(nn.Module, ABC):
    """Features → encoder states → (loss | token ids).

    Every architecture (Transformer-CTC now; Conformer or AED later) has the same
    three methods, so Trainer and ASR never change when the architecture does.
    """

    @abstractmethod
    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, F), (B,) → model output (CTC: log-probs (B, T', V)) and output lengths (B,)."""

    @abstractmethod
    def loss(self, feats, feat_lens, tokens, tok_lens) -> torch.Tensor:
        """Scalar training loss for one batch. CTC loss is computed in fp32."""

    @abstractmethod
    def decode(self, feats, feat_lens) -> list[list[int]]:
        """Token ids per utterance (CTC: greedy argmax, merge repeats, drop blanks)."""


@register("model", "transformer_ctc")
class TransformerCTC(AcousticModel):
    """ConvSubsample(x4) → sinusoidal positions → Pre-LN Transformer → Linear → log-softmax.

    The notebook's model, with n_feats and vocab_size passed in instead of read from globals.
    `arch` is accepted and ignored so _build_model can pass every ModelConfig field at once.
    """

    def __init__(self, n_feats: int, vocab_size: int, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, dropout=0.1, subsample_channels=128, subsample_factor=4, arch=None):
        super().__init__()
        self.n_feats = n_feats
        self.vocab_size = vocab_size
        # TODO(M1): build ConvSubsample, PositionalEncoding, TransformerEncoder, output Linear

    def forward(self, feats, feat_lens):
        raise NotImplementedError("TODO(M1)")

    def loss(self, feats, feat_lens, tokens, tok_lens):
        raise NotImplementedError("TODO(M1)")

    def decode(self, feats, feat_lens):
        raise NotImplementedError("TODO(M1)")