"""
Tests for multi-level training: Condition.level_range, mix="sample", and unchanged hashes.

    pytest tests/test_multilevel.py

Author: Eric Oliveira
Date: 11/19/2026

* Written with the help of Claude *

"""

import json
import math

import numpy as np
import pytest
import torch

from asrlib import ASRConfig, CLEAN, CarfacConfig, Condition, DataConfig, LogMelConfig, MFCCConfig, ModelConfig, TrainConfig
from asrlib.conditions import rms_to_db_spl, rms
from asrlib.data import LengthBucketSampler


# =============================================================================
# Condition.level_range
# =============================================================================

def test_level_is_drawn_in_range_and_reproducible():
    c = Condition(level_range=(45, 85))
    wave = torch.randn(16000)
    levels = [c.level_db_spl(f"utt{i}", v) for i in range(200) for v in range(3)]
    assert all(45 <= x <= 85 for x in levels)
    # Roughly uniform over the range, not stuck at one value
    assert np.std(levels) > 8
    # Same utterance and variant → same level, and apply() really produces it
    assert c.level_db_spl("utt7", 2) == c.level_db_spl("utt7", 2)
    assert rms_to_db_spl(rms(c.apply(wave, "utt7", variant=2))) == pytest.approx(c.level_db_spl("utt7", 2), abs=1e-4)
    # Different variants of one utterance get different levels
    assert len({c.level_db_spl("utt7", v) for v in range(3)}) == 3


def test_level_range_properties_and_name():
    c = Condition(level_range=(45, 85))
    assert c.is_random and not CLEAN.is_random and Condition("white", 5).is_random
    assert c.reference_db_spl == 65 and CLEAN.reference_db_spl == 65
    assert c.name == "clean@45-85dBSPL"
    assert Condition("white", 5, level_range=(45, 85)).name == "white@5dB@45-85dBSPL"


def test_level_range_rejects_bad_input():
    with pytest.raises(ValueError):
        Condition(level_range=(85, 45))
    with pytest.raises(ValueError):
        Condition(speech_db_spl=45, level_range=(45, 85))


def test_level_draw_doesnt_change_the_noise_segment():
    # Same noise samples with or without a level range: only their scale differs
    wave = torch.randn(16000)
    fixed = Condition("white", 0)
    ranged = Condition("white", 0, level_range=(45, 85))
    lvl = ranged.level_db_spl("u1")
    # Recover each noise by subtracting the speech at the level that was applied
    from asrlib.conditions import set_level
    n1 = fixed.apply(wave, "u1") - set_level(wave, 65)
    n2 = ranged.apply(wave, "u1") - set_level(wave, lvl)
    assert torch.allclose(n1 / n1.std(), n2 / n2.std(), atol=1e-4)


def test_json_round_trip_keeps_conditions_equal():
    cfg = ASRConfig(train=TrainConfig(train_conditions=(Condition(level_range=(45, 85)),),
                                      n_noisy_variants=3, mix="sample"))
    back = ASRConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back == cfg and back.hash() == cfg.hash()
    assert back.train.train_conditions[0].level_range == (45.0, 85.0)


# =============================================================================
# Hashes of existing models must not change
# =============================================================================

def _reference_configs():
    car = CarfacConfig(car_params={
        "velocity_scale": 0.1, "v_offset": 0.04, "min_zeta": 0.1, "max_zeta": 0.35,
        "first_pole_theta": 0.95 * math.pi, "zero_ratio": math.sqrt(2), "high_f_damping_compression": 0.5,
        "erb_per_step": 0.42, "min_pole_hz": 30, "erb_break_freq": 165.3, "erb_q": 1000 / (24.7 * 4.37),
        "ac_corner_hz": 20, "high_f_factor": 0.0},
        agc_params={"decimation": (6, 2, 2, 2)}, ihc_params={}, open_loop=False, post="log", pooling="hann25")
    data = DataConfig(corpus="librispeech", train_split="train.100", n_train_shards=None, val_split="validation",
                      test_split="test", source="auto", max_train_seconds=None)
    model = ModelConfig(arch="conformer_ctc", d_model=256, n_heads=4, n_layers=12, d_ff=1024, dropout=0.1)
    train = TrainConfig(epochs=100, peak_lr=5e-4, warmup_steps=10_000, weight_decay=0.01, max_frames_per_batch=35_000,
                        grad_clip=1.0, amp="auto",
                        specaugment={"freq_masks": 2, "freq_width": 27, "time_masks": 2, "time_width": 50,
                                     "scale_freq_to_n_feats": True},
                        train_conditions=(CLEAN,), n_noisy_variants=0, early_stop_patience=None, num_workers=6)
    kw = dict(model=model, data=data, train=train, norm="global", tokenizer="char", seed=30)
    return {"0b1a55": ASRConfig(frontend=car, **kw), "44568e": ASRConfig(frontend=LogMelConfig(n_mels=80), **kw),
            "9c4af7": ASRConfig(frontend=MFCCConfig(), **kw)}


@pytest.mark.parametrize("expected", ["0b1a55", "44568e", "9c4af7"])
def test_trained_model_hashes_unchanged(expected):
    cfg = _reference_configs()[expected]
    assert cfg.hash() == expected and cfg.controlled_hash() == "a8c0bf"


def test_new_fields_change_the_hash_only_when_used():
    cfg = _reference_configs()["0b1a55"]
    sampled = cfg.replace(train=TrainConfig(**{**cfg.train.__dict__, "mix": "sample"}))
    assert sampled.hash() != cfg.hash()
    assert "mix" not in cfg.to_dict()["train"] and "level_range" not in cfg.to_dict()["train"]["train_conditions"][0]


# =============================================================================
# mix="sample": LengthBucketSampler(groups=K)
# =============================================================================

def _frames(n=300):
    return np.random.default_rng(1).integers(50, 1500, n)


def test_sampling_uses_each_utterance_once_per_epoch():
    n, groups = 300, 3
    s = LengthBucketSampler(_frames(n), max_frames=8000, seed=30, groups=groups)
    for epoch in range(3):
        s.set_epoch(epoch)
        items = np.concatenate([np.array(b) for b in s])
        assert len(items) == n and set(items % n) == set(range(n))       # every utterance exactly once
        assert items.max() < n * groups                                 # valid dataset items


def test_sampling_varies_copies_across_epochs_and_is_reproducible():
    n, groups = 300, 3
    s = LengthBucketSampler(_frames(n), max_frames=8000, seed=30, groups=groups)
    s.set_epoch(0); c0 = s.copies()
    s.set_epoch(1); c1 = s.copies()
    assert set(c0) == {0, 1, 2} and not np.array_equal(c0, c1)
    # Same seed and epoch → same choice and same batches
    t = LengthBucketSampler(_frames(n), max_frames=8000, seed=30, groups=groups)
    t.set_epoch(1); s.set_epoch(1)
    assert [list(b) for b in s] == [list(b) for b in t]


def test_sampling_batches_respect_the_frame_budget():
    frames = _frames()
    s = LengthBucketSampler(frames, max_frames=8000, seed=30, groups=4)
    n = len(frames)
    for batch in s:
        lengths = frames[np.array(batch) % n]
        assert lengths.max() * len(batch) <= 8000 or len(batch) == 1


def test_one_group_is_the_old_sampler():
    frames = _frames()
    a = LengthBucketSampler(frames, max_frames=8000, seed=30)
    b = LengthBucketSampler(frames, max_frames=8000, seed=30, groups=1)
    a.set_epoch(2); b.set_epoch(2)
    assert [list(x) for x in a] == [list(x) for x in b]