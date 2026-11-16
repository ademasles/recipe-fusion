import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from recipe_fusion.data.loading import Splits
from recipe_fusion.data.upmc import Sample
from recipe_fusion.features.cache import write_features
from recipe_fusion.models import bert_finetune as B
from recipe_fusion.models.heads import MLPHead
from recipe_fusion.preds import ckpt_path, preds_path
from recipe_fusion.training import ArrayBatches, TrainConfig, fit

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "apple", "pie", "tomato", "soup",
    "chocolate", "cake", "easy", "best",
]  # fmt: skip
CLASSES = ["apple_pie", "tomato_soup", "chocolate_cake"]
WORDS = ["apple pie", "tomato soup", "chocolate cake"]
TEXTS = [
    "apple pie", "tomato soup", "chocolate cake",
    "easy apple pie", "best tomato soup", "easy chocolate cake",
]  # fmt: skip


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", VOCAB.index("[CLS]")), ("[SEP]", VOCAB.index("[SEP]"))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def make_encoder() -> BertModel:
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return BertModel(config, add_pooling_layer=False)


def test_tokenize_pads_and_truncates(tokenizer) -> None:
    ids, mask = B.tokenize(["apple pie", " ".join(["cake"] * 20), ""], tokenizer, max_len=8)
    assert ids.shape == (3, 8) and mask.shape == (3, 8)
    assert mask.sum(axis=1).tolist() == [4, 8, 2]  # [CLS] .. [SEP], truncated, empty title
    assert ids[0, 0] == tokenizer.cls_token_id
    assert (ids[:, 1:3] != tokenizer.unk_token_id).all()  # words are in the vocabulary


def test_cache_max_len_reads_the_b1_meta(tmp_path) -> None:
    pooled = np.zeros((2, 4), dtype=np.float16)
    names = ["a.jpg", "b.jpg"]
    write_features(
        tmp_path, "train", names, pooled, meta={"L": 40, "preprocessing": B.PREPROCESSING}
    )
    assert B.cache_max_len(tmp_path) == 40
    write_features(tmp_path, "train", names, pooled, meta={"L": 40, "preprocessing": "lowercase"})
    with pytest.raises(ValueError, match="preprocessing"):
        B.cache_max_len(tmp_path)
    write_features(tmp_path, "train", names, pooled, meta={"preprocessing": B.PREPROCESSING})
    with pytest.raises(ValueError, match="no L"):
        B.cache_max_len(tmp_path)


def test_param_groups_exclude_biases_and_layer_norm_from_decay() -> None:
    model = B.BertClassifier(make_encoder(), n_classes=3)
    decay, no_decay = B.param_groups(model, 0.01)
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    assert no_decay["params"] and all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


def test_warmup_then_linear_decay_to_zero() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = B.warmup_linear_decay(optimizer, total_steps=20, warmup_frac=0.1)
    rates = []
    for _ in range(20):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert rates[0] == 0.0 and rates[2] == pytest.approx(1.0)  # 2 warmup steps
    assert rates[:3] == sorted(rates[:3]) and rates[2:] == sorted(rates[2:], reverse=True)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)


def test_two_optimizer_steps_lower_the_loss(tokenizer) -> None:
    torch.manual_seed(0)
    model = B.BertClassifier(make_encoder(), n_classes=3, dropout=0.0)
    ids, mask = B.tokenize(TEXTS, tokenizer, max_len=8)
    batch = {
        "input_ids": torch.from_numpy(ids).long(),
        "attention_mask": torch.from_numpy(mask).long(),
    }
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    optimizer = torch.optim.AdamW(B.param_groups(model, 0.01), lr=1e-3)

    before = F.cross_entropy(model(**batch), labels).item()
    for _ in range(5):
        optimizer.zero_grad()
        F.cross_entropy(model(**batch), labels).backward()
        optimizer.step()
    assert F.cross_entropy(model(**batch), labels).item() < before


def test_fit_steps_the_scheduler_after_every_batch() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 20)
    x = rng.standard_normal((60, 8)).astype(np.float32)
    x[np.arange(60), y] += 4.0
    model = MLPHead(dim_in=8, hidden=8, n_classes=3, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = B.warmup_linear_decay(optimizer, total_steps=6)
    cfg = TrainConfig(lr=1e-2, weight_decay=0.0, batch_size=30, max_epochs=3, patience=3)
    result = fit(
        model, ArrayBatches(x), y, ArrayBatches(x), y, cfg, torch.device("cpu"), 0,
        optimizer=optimizer, scheduler=scheduler,
    )  # fmt: skip
    assert result.epochs_run == 3
    assert scheduler.last_epoch == 6  # 2 batches x 3 epochs


def make_splits() -> Splits:
    def samples(split: str, indices: range) -> list[Sample]:
        return [
            Sample(f"{label}_{i}.jpg", f"{'easy' if i % 2 else 'best'} {words}", label, split)
            for label, words in zip(CLASSES, WORDS, strict=True)
            for i in indices
        ]

    train, val = samples("train", range(3, 11)), samples("train", range(3))
    return Splits(train + val, train, val, samples("test", range(11, 14)), CLASSES)


def make_cfg(tmp_path, **overrides) -> DictConfig:
    cfg = {
        "method": "text_bert_ft",
        "condition": "text_only",
        "model_name": "tiny",
        "outputs_dir": str(tmp_path),
        "seeds": [0, 1],
        "lr": None,
        "search": {"seed": 0, "lr": [0.001, 0.003]},
        "recipe": {
            "weight_decay": 0.01,
            "warmup_frac": 0.1,
            "batch_size": 8,
            "max_epochs": 2,
            "patience": 2,
            "dropout": 0.0,
        },
    }
    cfg.update(overrides)
    return DictConfig(cfg)


def test_run_searches_lr_then_writes_every_output(tmp_path, tokenizer) -> None:
    splits = make_splits()
    data = B.load_token_data(splits, tokenizer, max_len=8)
    summary = B.run(make_cfg(tmp_path), data, make_encoder, torch.device("cpu"))

    assert [t["lr"] for t in summary["trials"]] == [0.001, 0.003]
    assert summary["chosen_lr"] in (0.001, 0.003)
    for seed in (0, 1):
        for split, n in (("val", len(splits.val)), ("test", len(splits.test))):
            with np.load(preds_path(tmp_path, "text_bert_ft", seed, split, "text_only")) as saved:
                assert saved["logits"].shape == (n, 3)
        directory = ckpt_path(tmp_path, "text_bert_ft", seed).with_suffix("")
        assert B.load_classifier(directory).classifier.out_features == 3
    meta = json.loads((tmp_path / "preds" / "text_bert_ft" / "meta.json").read_text())
    assert meta["L"] == 8 and meta["seconds_per_epoch_mean"] > 0
    assert all(f["seconds_per_epoch"] > 0 for f in meta["finals"])


def test_run_skips_the_search_when_lr_is_given(tmp_path, tokenizer) -> None:
    data = B.load_token_data(make_splits(), tokenizer, max_len=8)
    cfg = make_cfg(tmp_path, lr=0.003, seeds=[0])
    summary = B.run(cfg, data, make_encoder, torch.device("cpu"))
    assert summary["trials"] == [] and summary["chosen_lr"] == 0.003
    assert preds_path(tmp_path, "text_bert_ft", 0, "test", "text_only").is_file()
