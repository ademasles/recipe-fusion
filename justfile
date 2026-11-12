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
    uv run python -m recipe_fusion.models.text_mlp {{ARGS}}

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

# Fully decode every dataset image; the report goes under outputs/check-images/
check-images *ARGS:
    uv run python -m recipe_fusion.features.image {{ARGS}}

# Extract ViT-B/16 pooled and 8x8 grid features; Hydra overrides go after the recipe name
extract-image *ARGS:
    uv run python -m recipe_fusion.features.extract_image {{ARGS}}
