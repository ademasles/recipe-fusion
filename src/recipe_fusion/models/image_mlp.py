"""MLP head for frozen ViT image features (protocol 2.4)."""

from torch import Tensor, nn

FEATURE_DIM = 768
HIDDEN_DIM = 512
NUM_CLASSES = 101


class ImageMLP(nn.Module):
    """Map pooled ViT features to class logits."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Linear(FEATURE_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)
