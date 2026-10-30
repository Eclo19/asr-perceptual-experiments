"""
Define ASR model architectures allowed for experiments. It is vital that different front ends
are benchmarked with the same model architecture.

Author: Eric Oliveira
Date: 10/28/2026

* Written with the help of Claude *

"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .registry import register


# ==============================
# Helpers
# ==============================

def conv_out_len(lengths, n_convs: int = 2):
    """Lengths after n stride-2 convs (kernel 3, padding 1): (n - 1) // 2 + 1 each time.
    Works on ints (for sizing layers) and on int64 tensors (for the true lengths)."""
    for _ in range(n_convs):
        lengths = (lengths - 1) // 2 + 1
    return lengths


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) true lengths → (B, max_len) bool. PyTorch's convention: True = padding, ignore it."""
    positions = torch.arange(max_len, device=lengths.device)                    # (max_len,)
    return positions[None, :] >= lengths[:, None]                                # (B, max_len)


# ==============================
# Building blocks
# ==============================

class ConvSubsampleAndProject(nn.Module):
    """Stride-2 convs over the (time, freq) 'image', then flatten + project to d_model.

    (B, T, F) → (B, T / factor, d_model). The true lengths go through the same arithmetic.
    """

    def __init__(self, n_feats: int, d_model: int, channels: int = 128, factor: int = 4):
        super().__init__()

        # One stride-2 conv per halving: factor 4 → 2 convs, factor 8 → 3 convs
        self.n_convs = int(math.log2(factor))
        if 2 ** self.n_convs != factor:
            raise ValueError(f"subsample_factor must be a power of 2, not {factor}")

        # Conv -> ReLU, repeated; the first conv sees 1 input channel (the feature 'image')
        layers, c_in = [], 1
        for _ in range(self.n_convs):
            layers += [nn.Conv2d(c_in, channels, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            c_in = channels
        self.conv = nn.Sequential(*layers)

        # The frequency axis shrinks too, so the projection's input size depends on n_feats
        f_out = conv_out_len(n_feats, self.n_convs)

        # Project to d_model for embedding vector of each frame
        self.proj = nn.Linear(channels * f_out, d_model)                         

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x.unsqueeze(1))                                            # (B, 1, T, F) → (B, C, T', F')
        x = x.permute(0, 2, 1, 3).flatten(2)                                     # (B, T', C·F'): one vector per frame
        return self.proj(x), conv_out_len(lengths, self.n_convs)                 # (B, T', d_model), (B,)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions (Vaswani et al., 2017), added to every frame, then dropout."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        # Row k of the table is the code for position k: sin on even dims, cos on odd dims
        pos = torch.arange(max_len).unsqueeze(1)                                 # (max_len, 1)

        # Frequency term
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))   # (d_model / 2,)

        # Init pe vector (agnostic of the data)
        pe = torch.zeros(max_len, d_model)

        # Even indices sin, odd indices cos
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # A buffer moves with .to(device) but isn't trained; persistent=False keeps it out of checkpoints
        self.register_buffer("pe", pe, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[: x.size(1)])                               # (B, T', d) + (T', d), broadcast


# ==============================
# Model ABC
# ==============================

class AcousticModel(nn.Module, ABC):
    """Features -> encoder states -> (loss | token ids).

    Every architecture (Transformer-CTC now; Conformer or AED later) has the same
    three methods, so Trainer and ASR never change when the architecture does.
    """

    @abstractmethod
    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, F), (B,) -> model output (CTC: log-probs (B, T', V)) and output lengths (B,)."""

    @abstractmethod
    def loss(self, feats, feat_lens, tokens, tok_lens) -> torch.Tensor:
        """Scalar training loss for one batch. CTC loss is computed in fp32."""

    @abstractmethod
    def decode(self, feats, feat_lens) -> list[list[int]]:
        """Token ids per utterance (CTC: greedy argmax, merge repeats, drop blanks)."""


# ==============================
# Transformer + CTC
# ==============================

@register("model", "transformer_ctc")
class TransformerCTC(AcousticModel):
    """ConvSubsample(x4) -> sinusoidal positions -> Pre-LN Transformer encoder -> Linear -> log-softmax.

    `arch` is accepted and ignored so _build_model can pass every ModelConfig field at once.
    """

    def __init__(self, n_feats: int, vocab_size: int, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, dropout=0.1, subsample_channels=128, subsample_factor=4,
                 blank_id: int = 0, arch=None):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_feats = n_feats
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        # 1) Time / subsample_factor, and features -> d_model
        self.subsample = ConvSubsampleAndProject(n_feats, d_model, subsample_channels, subsample_factor)

        # 2) Position codes
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        # 3) Encoder only (CTC has no decoder). batch_first: tensors are (B, T, d_model).
        #    norm_first = Pre-LN, much more stable to train than the original Post-LN.
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)

        # Pre-LN leaves the last layer's output unnormalized, so add a final LayerNorm.
        # TransformerEncoder deep-copies `layer` n_layers times: independent weights, not shared.
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)          # not supported with norm_first

        # 4) One score per vocabulary symbol per frame
        self.head = nn.Linear(d_model, vocab_size)

        # Loss: zero_infinity so an impossible alignment (too few frames) gives loss 0 instead of inf
        self.ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, F), (B,) → log-probs (B, T', V), out_lens (B,)."""
        x, out_lens = self.subsample(feats, feat_lens)                           # (B, T', d), (B,)

        # True = padding; attention gives those frames zero weight
        pad_mask = lengths_to_padding_mask(out_lens, x.size(1))                  # (B, T')

        x = self.encoder(self.pos(x), src_key_padding_mask=pad_mask)             # (B, T', d)
        return self.head(x).log_softmax(dim=-1), out_lens                        # (B, T', V), (B,)

    def loss(self, feats, feat_lens, tokens, tok_lens) -> torch.Tensor:
        """Mean CTC loss over the batch. Padding frames and padding tokens are excluded by the lengths."""
        # PyTorch has no CTC loss for Apple's MPS (pytorch issue 141287): fail clearly, before any work
        if feats.device.type == "mps":
            raise RuntimeError(
                "CTC loss isn't implemented on Apple MPS, so training and evaluation can't run there. "
                "Use device='cpu' on the Mac (e.g. run_e2e.py --device cpu) and CUDA on Quest. "
                "transcribe() works on MPS: it only decodes.")

        log_probs, out_lens = self(feats, feat_lens)

        # CTCLoss wants (T', B, V), and fp32: it sums many tiny probabilities, which bf16 can't hold
        return self.ctc(log_probs.transpose(0, 1).float(), tokens, out_lens, tok_lens)

    @torch.no_grad()
    def decode(self, feats, feat_lens) -> list[list[int]]:
        """Greedy CTC: argmax per frame → merge repeats → drop blanks, on the first out_len frames only.

        Call it with the model in eval() mode (ASR.transcribe/evaluate do this), so dropout is off.
        """
        log_probs, out_lens = self(feats, feat_lens)
        best = log_probs.argmax(dim=-1).cpu()                                    # (B, T')

        # Hypothesis lists for the models transcription prediction
        hyps = []
        for seq, n in zip(best, out_lens.cpu()):
            # "hh__ee_ll_lo" → "h_e_l_lo" → "hello"
            ids = torch.unique_consecutive(seq[:n])
            hyps.append(ids[ids != self.blank_id].tolist())
        return hyps