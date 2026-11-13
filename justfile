default:
    @just --list

# Create or update the virtual environment from uv.lock
sync:
    uv sync

test:
    uv run pytest

lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff format .
    uv run ruff check --fix .

typecheck:
    uv run basedpyright

# Text references (sheet B2): TF-IDF + logistic regression, and the MLP head on frozen BERT
train-tfidf *ARGS:
    uv run python -m recipe_fusion.models.tfidf_logreg {{ARGS}}

train-text-mlp *ARGS:
    uv run python -m recipe_fusion.models.pooled_mlp {{ARGS}}

extract-text *overrides:
    uv run python -m recipe_fusion.features.text {{overrides}}

# All checks run before a commit or a PR
check: lint typecheck test

# GPU availability + micro-benchmark
gpu:
    uv run python scripts/gpu_check.py

# Register this project's venv as a Jupyter kernel
kernel:
    uv run python -m ipykernel install --user --name recipe-fusion

# Download UPMC Food-101 from Kaggle (8.3 GB); Hydra overrides go after the recipe name,
# e.g. `just download data.texts_only=true`
download *ARGS:
    uv run python -m recipe_fusion.data.download {{ARGS}}

# Draw the validation split once (splits/val_ids.txt, committed); needs the title files
make-split *ARGS:
    uv run python -m recipe_fusion.data.split {{ARGS}}

# Evaluation subsets (title seen/unseen in train, no class name) into splits/, plus the
# duplicate report under outputs/subsets/
make-subsets *ARGS:
    uv run python -m recipe_fusion.data.subsets {{ARGS}}

# Fully decode every dataset image; the report goes under outputs/check-images/
check-images *ARGS:
    uv run python -m recipe_fusion.features.image {{ARGS}}

# Extract ViT-B/16 pooled and 8x8 grid features; Hydra overrides go after the recipe name
extract-image *ARGS:
    uv run python -m recipe_fusion.features.extract_image {{ARGS}}

# Image reference: MLP on frozen ViT pooled features (same protocol 2.4 code as train-text-mlp)
train-image-mlp *ARGS:
    uv run python -m recipe_fusion.models.pooled_mlp --config-name train/image_mlp {{ARGS}}

# Score every prediction file with the shared metrics; `just evaluate split=test` for the final table
evaluate *ARGS:
    uv run python -m recipe_fusion.evaluate {{ARGS}}

# Cross-attention fusion: search (lr, weight decay) on seed 0, then train seeds 0-2 and write
# val/test predictions with the ablations; `lr=1e-3 weight_decay=1e-2` skips the search
train-cross-attention *ARGS:
    uv run python -m recipe_fusion.models.cross_attention {{ARGS}}

# Late fusion of text_mlp and image_mlp predictions (calibrated on val, nothing trained);
# `text.method=tfidf_logreg name=late_calibrated_tfidf sum_name=null` for another text reference
train-late-fusion *ARGS:
    uv run python -m recipe_fusion.models.late_fusion {{ARGS}}

# Early fusion of the BERT and ViT pooled vectors (sheet B3) with the 4 ablations; same search
# and `lr=... weight_decay=...` shortcut as train-cross-attention
train-early-fusion *ARGS:
    uv run python -m recipe_fusion.models.early_fusion {{ARGS}}
