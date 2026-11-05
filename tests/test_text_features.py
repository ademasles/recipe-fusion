from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from transformers import BertConfig, BertModel, BertTokenizer

from recipe_fusion.features import text as T
from recipe_fusion.features.cache import read_features

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "features"
VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "chocolate", "cake",
    "tomato", "soup", "apple", "pie", "easy", "best", "&", "##s",
]  # fmt: skip
TEXTS = ["Chocolate Cake", "easy tomato soup", "apple pie", "zzz", ""]


@pytest.fixture
def tokenizer(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("\n".join(VOCAB) + "\n")
    return BertTokenizer(vocab_file=str(p), do_lower_case=True)


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    return BertModel(cfg, add_pooling_layer=False).eval()


def test_shapes_mask_no_nan(tokenizer, model):
    pooled, tokens, mask = T.encode(TEXTS, tokenizer, model, max_len=8, batch_size=2)
    assert pooled.shape == (5, 32) and pooled.dtype == np.float16
    assert tokens.shape == (5, 8, 32) and tokens.dtype == np.float16
    assert mask.shape == (5, 8) and mask.dtype == np.bool_
    assert mask.sum(axis=1).min() >= 1  # even the empty title has [CLS] [SEP]
    assert np.isfinite(pooled).all() and np.isfinite(tokens).all()


def test_padding_does_not_change_pooled(tokenizer, model):
    p8, t8, _ = T.encode(TEXTS, tokenizer, model, max_len=8)
    p16, t16, _ = T.encode(TEXTS, tokenizer, model, max_len=16)
    np.testing.assert_allclose(p8.astype(np.float32), p16.astype(np.float32), atol=5e-3)
    np.testing.assert_allclose(t8.astype(np.float32), t16[:, :8].astype(np.float32), atol=5e-3)


def test_masked_mean_ignores_padded_positions():
    hidden = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    ref = T.masked_mean(hidden, mask)
    noisy = hidden.clone()
    noisy[mask == 0] = 1e4
    assert torch.allclose(T.masked_mean(noisy, mask), ref)
    assert torch.allclose(ref[0], hidden[0, :2].mean(0))


def test_truncation_keeps_max_len(tokenizer, model):
    _, _, mask = T.encode([" ".join(["cake"] * 20)], tokenizer, model, max_len=8)
    assert mask.sum() == 8


def test_length_helpers(tokenizer):
    lengths = T.token_lengths(["chocolate cake", "apple"], tokenizer)
    assert lengths.tolist() == [4, 3]  # + [CLS] [SEP]
    assert T.next_pow2(32) == 32 and T.next_pow2(32.2) == 64 and T.next_pow2(1) == 1
    assert T.truncated_pct(np.array([3, 5, 9, 10]), 8) == 50.0


def test_preprocess_is_unescape_and_strip_only():
    assert T.preprocess("  Bob&#039;s Red Mill ") == "Bob's Red Mill"
    assert T.preprocess("Raw Apple Pie &amp; Crumble") == "Raw Apple Pie & Crumble"
    # known and intended: a single unescape pass
    assert T.preprocess("Mom&amp;rsquo;s Apple Pie") == "Mom&rsquo;s Apple Pie"
    assert T.preprocess("Know \\xe2\\x84\\xa2") == "Know \\xe2\\x84\\xa2"


def test_residual_escape_pct():
    assert T.residual_escape_pct(["ok", "Mom&rsquo;s", "a \\xe2 b", "fine"]) == 50.0


def test_git_commit_returns_string():
    assert isinstance(T.git_commit(), str)


def test_load_model_is_frozen_and_eval(tmp_path, tokenizer, model):
    model.save_pretrained(tmp_path / "ckpt")
    tokenizer.save_pretrained(tmp_path / "ckpt")
    _, loaded = T.load_model(str(tmp_path / "ckpt"), torch.device("cpu"))
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())
    assert loaded.config.hidden_size == 32


def test_run_from_composed_hydra_config(tmp_path, tokenizer, model, monkeypatch):
    fake = {
        "train": [
            SimpleNamespace(image_name=f"tr{i}", text=t)
            for i, t in enumerate(["Chocolate &amp; Cake ", "easy soup", "apple pie"])
        ],
        "test": [SimpleNamespace(image_name="te0", text="best pie")],
    }
    monkeypatch.setattr(T, "read_split", lambda root, split: iter(fake[split]))

    out = tmp_path / "out"
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="text", overrides=[f"out_dir={out}", "batch_size=2"])
    assert cfg.model_name == "bert-base-uncased"  # defaults come from text.yaml

    metas = T.run(Path(cfg.root), cfg, tokenizer, model)

    for split in ("train", "test"):
        features = read_features(out, split, mmap=False)  # validated by write_features
        assert features.ids == [s.image_name for s in fake[split]]
        assert features.pooled.shape == (len(features), 32)
        assert features.tokens is not None and features.mask is not None
        assert features.tokens.shape == (len(features), metas[split]["L"], 32)
        meta = features.meta
        assert meta["extraction_seconds"] >= 0
        assert meta["d"] == 32
        assert meta["L"] & (meta["L"] - 1) == 0  # power of two
        assert "git_commit" in meta and "truncated_pct" in meta
