import torch

from recipe_fusion.seed import pick_device, seed_everything


def test_seed_makes_torch_reproducible() -> None:
    seed_everything(123)
    a = torch.randn(8)
    seed_everything(123)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_pick_device_respects_explicit_choice() -> None:
    assert pick_device("cpu").type == "cpu"
