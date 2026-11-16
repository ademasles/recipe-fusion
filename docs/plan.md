# recipe-fusion project plan

Version of 20 October 2026, decisions filled in on 16 November 2026. This file set the frame and
the requirements before any compute was launched; the outcome is in [results.md](results.md).

## Assignment

Project 1 of the Advanced Deep Learning course (Mines Nancy): multimodal classification.
Preprocessing, per-modality feature extraction, fusion, training, comparative evaluation.

Deadline: 19 November 2026, with an 8-minute defence and no written report.

## Task

Predict the class (101 dishes) of a (photo, web page title) pair from UPMC Food-101
(Wang et al., 2015). Main metric: top-1 accuracy on the official test split (22,716 pairs).

## Minimum protocol requirements

1. **Unimodal baselines**: an image-only and a text-only model, trained and evaluated under the
   same conditions as the multimodal models.
2. **At least three fusion strategies**, compared at a comparable budget.
3. **Several seeds** (at least three) per configuration; report mean and standard deviation.
4. **Modality ablations**: what happens to a multimodal model when one modality is removed or
   degraded at test time?
5. **Validation split** drawn from the training set, stratified by class and frozen once and for
   all; the test split is only used for the final table.

## Data facts that constrain the plan

- Titles are short: 7 words median, 15 at the 95th percentile.
- 73% of titles spell out the class name ("Apple Pie Recipe"). The text-only baseline will be
  strong, and the contribution of the image has to be measured against it.
- 39% of test titles (lowercased) appear verbatim in the training set. Duplicate pages or generic
  titles? To be examined before drawing any conclusion.
- 14% of titles contain HTML entities.

## Decisions

| Item | Choice | Rationale |
|---|---|---|
| Text preprocessing | `clean_title`: HTML entities and byte escapes decoded, whitespace collapsed | 14 % of titles carry entities; the same cleaning for every text model |
| Image preprocessing | The weights' own transforms: bicubic resize, 384 px centre crop, ImageNet normalisation | Native resolution of the SWAG weights |
| Image encoder | ViT-B/16 SWAG (torchvision), CLS token and 8×8 patch grid | Strong frozen features, a grid for cross-attention |
| Text encoder | BERT-base-uncased, 40 tokens, masked mean and token sequence | Covers the 95th percentile of title length |
| Frozen or fine-tuned encoders | Frozen and cached for every head; BERT also fine-tuned as a text reference | Fair budget across fusions; fine-tuning required by the course |
| Fusion strategies | Early (concatenation), cross-attention (both directions), late (calibrated log-probability sum) | One per family, one per team member |
| Ablation protocol | One modality replaced by its training mean, or shuffled, at test time | Measures reliance on each modality without retraining |
| Hyperparameter selection | lr × weight decay 3×2 grid on validation with seed 0, then seeds 0–2 | Same budget for every head; test never used for a choice |
| Metrics and statistical tests | Top-1, top-5 and macro-F1, mean ± std over 3 seeds | Top-1 is the benchmark metric; seeds give the noise level |

## Team split

Same group as scene-to-score: `ademasles`, `Argodm`, `OriGiNaLGet9`. The split covers both
projects at once: everyone holds a "data and evaluation" role in one project and a "model" role in
the other, and everyone writes one fusion here, since fusion is the core of the course.

| Member | recipe-fusion (project 1) | scene-to-score (project 2) |
|---|---|---|
| ademasles | Data, evaluation and ablations; intermediate fusion (cross-attention) | Encoders and extraction; generation and demo |
| Argodm | Text branch and its baseline; early fusion | Data and evaluation |
| OriGiNaLGet9 | Image branch and its baseline; late fusion | Projection and training |

Project 1 roles:

- **Data, evaluation and ablations**: frozen validation split, analysis of duplicate titles
  between train and test, shared evaluation script, multi-seed runs, modality ablations, final
  table.
- **Text branch**: title preprocessing, text encoder, text-only baseline, cached features.
- **Image branch**: image preprocessing, image encoder, image-only baseline, cached features.
- **Fusions**: each member designs, trains and justifies their own, on the cached features and
  with the shared evaluation script. The fusion families above are a proposal; the exact
  architecture is up to each member.

To settle first, before any parallel work (data and evaluation role):

1. The validation split file, committed.
2. The feature cache format: one file per modality and split, rows in the order of the title
   files, with the image ids.
3. The evaluation script interface: predictions in, metrics out.

Neither collaborator has an NVIDIA GPU. Frozen encoders run once, on the tech lab or on
ademasles's laptop; heads and fusions then train on CPU from the cache. Encoder fine-tuning runs
on the tech lab.

## Compute

Laptop: RTX 4070, 8 GB, for everything: feature extraction once, heads on cached features in
minutes, BERT fine-tuning at about 2.5 minutes per epoch.

## Done when

- [x] The unimodal baselines × fusions table exists, with three seeds, on the test split.
- [x] Modality ablations are done.
- [x] The repository runs from scratch with the README commands.
- [x] The defence slides are written ([slides/](slides/)).
