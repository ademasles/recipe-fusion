# recipe-fusion

[![CI](https://github.com/ademasles/recipe-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/ademasles/recipe-fusion/actions/workflows/ci.yml)

**Recipe Fusion — Comparing Image–Text Fusion Strategies for Food Recognition**

Classify a recipe into one of 101 dishes from its photo and the title of its web page.

Project 1 of the Advanced Deep Learning course at Mines Nancy: multimodal text and image
classification, covering preprocessing, per-modality feature extraction, fusion, training and
comparative evaluation. The data is UPMC Food-101 (Wang et al., 2015): 90,688 image and title
pairs across 101 classes.

The work plan is in [docs/plan.md](docs/plan.md).

## Getting started

Requirements: Linux, macOS on Apple silicon or WSL2, with git, [uv](https://docs.astral.sh/uv/)
and [just](https://just.systems). An NVIDIA GPU speeds things up; 8 GB is enough for development.

```bash
just sync        # create .venv from uv.lock
just gpu         # check that PyTorch sees the GPU
just check       # ruff + basedpyright + pytest
just download    # UPMC Food-101 from Kaggle (8.3 GB)
just make-split  # validation split, once (already committed in splits/)
just check-images  # fully decode every image; report under outputs/check-images/
just extract-image # ViT-B/16 pooled + 8x8 grid features
just train-image-mlp # image baseline: MLP on frozen ViT
just evaluate      # score every prediction file (outputs/results/)
just train-cross-attention  # search lr/wd, then 3 seeds + test ablations
just train-late-fusion      # combine existing image/text logits; no new model training
just train-tfidf   # text baseline: TF-IDF + logistic regression
just train-text-mlp # text baseline: MLP on frozen BERT
uv run pre-commit install   # once after cloning
```

To explore the text alone, `just download data.texts_only=true` fetches the two title files
(7.6 MB) in a few seconds.

## Installing the tools

Once per machine, no admin rights needed. Python does not need to be installed: uv downloads the
version pinned in `.python-version`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, macOS and Linux
uv tool install rust-just                         # just, distributed through uv
```

## Layout

| Path | Contents |
|---|---|
| `src/recipe_fusion/data/` | Dataset index (`upmc.py`) and download (`download.py`) |
| `src/recipe_fusion/features/` | Image integrity check and ViT extraction, plus feature cache |
| `src/recipe_fusion/seed.py` | Seeding and device selection |
| `configs/` | Hydra configuration: paths, data |
| `scripts/` | One-off scripts (`gpu_check.py`) |
| `notebooks/` | Exploration; `just kernel` registers the environment with Jupyter |
| `tests/` | pytest |
| `docs/` | Project plan |
| `data/`, `outputs/` | Data and experiment outputs, ignored by git |

## Data

- Data goes to `data/upmc-food101/`. On another machine, `export F101_DATA_DIR=/path/to/data`.
- Source: the Kaggle copy `gianmarco96/upmcfood101`, version 1, downloaded with `kagglehub`
  without an account.
- `texts/{train,test}_titles.csv`: no header, three columns (image file, title, class). Images
  in `images/{train,test}/<class>/<file>.jpg`.
- 67,972 training pairs and 22,716 test pairs. There is no official validation split: ours is
  drawn once from the training set, stratified by class, and listed in `splits/val_ids.txt`.
- Titles are kept as is, HTML entities included (`&#8217;`).

## Conventions

- Dependencies: `uv add <package>`, dev tools: `uv add --dev <package>`. Never `pip install`.
- PyTorch comes from the CUDA index declared in `pyproject.toml` on Linux, from PyPI on macOS.
- Every experiment is seeded and configured through Hydra.
- GPU-free code: `pick_device()`, never a hard-coded `"cuda"`, tests run on CPU.
- No data, checkpoints or secrets in git; `gitleaks` runs as a pre-commit hook.
- Code, comments and commit messages in English, following Conventional Commits.

Late fusion reads the common prediction files under `outputs/preds/<method>/seed<k>/` and does
not train a network. Run `just train-late-fusion text_method=tfidf_logreg` to select another text
baseline. It writes raw-logit sums (`late_sum`) and validation-calibrated combinations
(`late_calibrated`); configure `text_ft_method` to also write `late_calibrated_ft`. Calibration
uses only validation logits, with a temperature grid in log-space from -2 to 2 and a weight grid
from 0 to 1 in steps of 0.05. Each seed's selected temperatures and image weight are recorded in
the corresponding method `meta.json`. The normal input predictions use their configured base
conditions; ablation runs require modality-specific `text_mean`/`text_shuffled` or
`image_mean`/`image_shuffled` prediction files from the unimodal model.
To run the normal-condition comparison before ablation logits are ready, use
`just train-late-fusion conditions=[both]`.

## Licences

The code is released under the MIT licence (see [LICENSE](LICENSE)).

The Kaggle copy of UPMC Food-101 states no licence. The images and titles were collected from
the web by the dataset authors: research and teaching use only, no redistribution. Cite Wang,
Kumar, Thome, Cord and Precioso (2015), *Recipe recognition with large multimodal food dataset*,
ICME Workshops.
