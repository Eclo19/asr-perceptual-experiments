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


class Conv1dSubsampleAndProject(nn.Module):
    """Whisper-style stem: the features are input CHANNELS and the convs slide over time only.

    It makes no assumption about the feature axis, so it suits any front end: tonotopic ones
    (log-mel, CARFAC) and non-tonotopic ones (MFCC, whose coefficients have no "neighbours").
    Its size barely depends on n_feats (only the first conv's input width), unlike the Conv2d
    stem, whose projection grows with n_feats.

    (B, T, F) → (B, T / factor, d_model). Stride 2 in every conv (Whisper strides only its second
    conv) to keep the same x4 subsampling as the Conv2d stem. `channels` is unused: every conv is d_model wide.
    """

    def __init__(self, n_feats: int, d_model: int, channels: int | None = None, factor: int = 4):
        super().__init__()

        # One stride-2 conv per halving: factor 4 → 2 convs
        self.n_convs = int(math.log2(factor))
        if 2 ** self.n_convs != factor:
            raise ValueError(f"subsample_factor must be a power of 2, not {factor}")

        # Conv -> GELU, repeated; the first conv maps n_feats channels to d_model
        layers, c_in = [], n_feats
        for _ in range(self.n_convs):
            layers += [nn.Conv1d(c_in, d_model, kernel_size=3, stride=2, padding=1), nn.GELU()]
            c_in = d_model
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)                         # (B, F, T) → (B, T', d_model)
        return x, conv_out_len(lengths, self.n_convs)                            # (B, T', d_model), (B,)


# Stems by name: TransformerCTC picks one through its STEM class attribute
STEMS = {"conv2d": ConvSubsampleAndProject, "conv1d": Conv1dSubsampleAndProject}


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

    STEM selects the subsampling stem. It's a class attribute, not a ModelConfig field, so configs
    (and the hashes of models already trained) don't change: "transformer_ctc" keeps the Conv2d stem,
    "transformer_ctc_conv1d" (below) is the same model with the Conv1d stem.
    """
    STEM = "conv2d"

    def __init__(self, n_feats: int, vocab_size: int, d_model=384, n_heads=6, n_layers=6,
                 d_ff=1536, dropout=0.05, subsample_channels=128, subsample_factor=4,
                 blank_id: int = 0, arch=None):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_feats = n_feats
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        # 1) Time / subsample_factor, and features -> d_model
        self.subsample = STEMS[self.STEM](n_feats, d_model, subsample_channels, subsample_factor)

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


@register("model", "transformer_ctc_conv1d")
class TransformerCTCConv1d(TransformerCTC):
    """TransformerCTC with the Whisper-style Conv1d stem: the fair choice when MFCC is in the comparison.

        ModelConfig(arch="transformer_ctc_conv1d", ...)     # every other field works the same
    """
    STEM = "conv1d"


# ==============================
# Conformer + CTC
# ==============================

class FeedForward(nn.Module):
    """Pre-LN feed-forward: LayerNorm → Linear(d → d_ff) → Swish → Dropout → Linear(d_ff → d) → Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                                                       # (B, T, d)


class ConvModule(nn.Module):
    """The Conformer's convolution module: local patterns over ±kernel/2 frames.

    LayerNorm → pointwise conv (d → 2d) → GLU → depthwise conv (kernel k, one filter per channel)
    → LayerNorm → Swish → pointwise conv (d → d) → Dropout.

    Two choices that differ from the paper, both so that an utterance's output never depends on
    what else is in its batch:
      * padding frames are zeroed before the depthwise conv, so they can't leak into real frames
      * LayerNorm instead of BatchNorm after the depthwise conv (BatchNorm's statistics would
        include padding and change with batch composition)
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.mid_norm = nn.LayerNorm(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x (B, T, d), pad_mask (B, T) with True = padding → (B, T, d)."""
        y = self.norm(x).transpose(1, 2)                                         # (B, d, T)
        y = nn.functional.glu(self.pointwise_in(y), dim=1)                       # (B, 2d, T) → (B, d, T)
        y = y.masked_fill(pad_mask[:, None, :], 0.0)                             # Padding contributes nothing
        y = self.depthwise(y)                                                    # (B, d, T)
        y = self.mid_norm(y.transpose(1, 2)).transpose(1, 2)                     # Normalize over channels
        y = self.pointwise_out(nn.functional.silu(y))                            # (B, d, T)
        return self.drop(y.transpose(1, 2))                                      # (B, T, d)


class ConformerBlock(nn.Module):
    """Gulati et al. (2020): the "macaron" block, two half-step feed-forwards around attention + convolution.

        x = x + ½·FF(x)                       first half of the feed-forward
        x = x + MHSA(LayerNorm(x))            global context (attention over the whole utterance)
        x = x + Conv(x)                       local context (~kernel frames around each frame)
        x = x + ½·FF(x)                       second half of the feed-forward
        x = LayerNorm(x)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, kernel_size: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.attn_drop(y)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)                                                  # (B, T, d)


@register("model", "conformer_ctc")
class ConformerCTC(TransformerCTC):
    """Conv1d stem (×4 in time) → sinusoidal positions → Conformer blocks → Linear → log-softmax, trained with CTC.

    Uses the same ModelConfig fields as TransformerCTC (d_model, n_heads, n_layers, d_ff, dropout,
    subsample_factor), so configs and hashes work unchanged; loss() and decode() are inherited.
    The convolution kernel is fixed at 31 frames after subsampling (31 × 40 ms ≈ 1.2 s), the
    paper's value. Positions: absolute sinusoidal codes added once, as in TransformerCTC; the
    convolution modules add relative position information on their own (the paper uses relative
    positional attention instead, a refinement left out here for simplicity).

        ModelConfig(arch="conformer_ctc", d_model=256, n_heads=4, n_layers=12, d_ff=1024, dropout=0.1)
    """
    STEM = "conv1d"
    CONV_KERNEL = 31

    def __init__(self, n_feats: int, vocab_size: int, d_model=256, n_heads=4, n_layers=12,
                 d_ff=1024, dropout=0.1, subsample_channels=128, subsample_factor=4,
                 blank_id: int = 0, arch=None):
        # Skip TransformerCTC.__init__ (it would build a Transformer encoder): initialize nn.Module directly
        AcousticModel.__init__(self)
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_feats = n_feats
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        # 1) Time / subsample_factor, and features → d_model
        self.subsample = STEMS[self.STEM](n_feats, d_model, subsample_channels, subsample_factor)

        # 2) Position codes (+ dropout)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        # 3) Conformer blocks
        self.blocks = nn.ModuleList(ConformerBlock(d_model, n_heads, d_ff, self.CONV_KERNEL, dropout)
                                    for _ in range(n_layers))

        # 4) One score per vocabulary symbol per frame, and the loss (as in TransformerCTC)
        self.head = nn.Linear(d_model, vocab_size)
        self.ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, F), (B,) → log-probs (B, T', V), out_lens (B,)."""
        x, out_lens = self.subsample(feats, feat_lens)                           # (B, T', d), (B,)
        pad_mask = lengths_to_padding_mask(out_lens, x.size(1))                  # (B, T'), True = padding
        x = self.pos(x)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.head(x).log_softmax(dim=-1), out_lens                        # (B, T', V), (B,)