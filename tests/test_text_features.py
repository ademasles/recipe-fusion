import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from transformers import BertConfig, BertModel, BertTokenizer

from recipe_fusion.features import text as T

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


@pytest.mark.parametrize("case", ["nan", "inf", "empty_mask_row"])
def test_check_outputs_rejects_invalid(case):
    pooled = np.zeros((2, 4), dtype=np.float16)
    tokens = np.zeros((2, 3, 4), dtype=np.float16)
    mask = np.ones((2, 3), dtype=np.bool_)
    if case == "nan":
        pooled[0, 0] = np.nan
    elif case == "inf":
        tokens[1, 0, 0] = np.inf
    else:
        mask[1] = False
    with pytest.raises(ValueError):
        T.check_outputs(pooled, tokens, mask)


def test_check_outputs_accepts_valid():
    T.check_outputs(
        np.zeros((2, 4), dtype=np.float16),
        np.zeros((2, 3, 4), dtype=np.float16),
        np.ones((2, 3), dtype=np.bool_),
    )


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

    meta = T.run(Path(cfg.root), cfg, tokenizer, model)

    for split in ("train", "test"):
        ids = (out / f"{split}_ids.txt").read_text(encoding="utf-8").splitlines()
        assert ids == [s.image_name for s in fake[split]]
        pooled = np.load(out / f"{split}_pooled.npy")
        tokens = np.load(out / f"{split}_tokens.npy")
        mask = np.load(out / f"{split}_mask.npy")
        assert pooled.shape == (len(ids), 32) and pooled.dtype == np.float16
        assert tokens.shape == (len(ids), meta["L"], 32) and tokens.dtype == np.float16
        assert mask.shape == tokens.shape[:2] and mask.dtype == np.bool_
    saved = json.loads((out / "meta.json").read_text())
    assert saved["extraction_seconds"] >= 0
    assert saved["d"] == 32
    assert saved["L"] & (saved["L"] - 1) == 0  # power of two
    assert "git_commit" in saved and "truncated_pct" in saved["splits"]["train"]
