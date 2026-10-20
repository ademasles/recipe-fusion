# recipe-fusion project plan

Version of 20 October 2026. This file sets the frame and the requirements. Model, fusion and
protocol choices are written here by the project authors, then reviewed **before** any compute is
launched.

## Assignment

Project 1 of the Advanced Deep Learning course (Mines Nancy): multimodal classification.
Preprocessing, per-modality feature extraction, fusion, training, comparative evaluation.

Deadline, report format, defence: **[TBD]**.

## Task

Predict the class (101 dishes) of a (photo, web page title) pair from UPMC Food-101
(Wang et al., 2015). Main metric: top-1 accuracy on the official test split (22,716 pairs)
**[to confirm]**.

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

## To write before launching compute

| Item | Choice | Rationale |
|---|---|---|
| Text preprocessing | **[TBD]** | |
| Image preprocessing | **[TBD]** | |
| Image encoder | **[TBD]** | |
| Text encoder | **[TBD]** | |
| Frozen or fine-tuned encoders | **[TBD]** | |
| Fusion strategies (≥ 3) | **[TBD]** | |
| Ablation protocol | **[TBD]** | |
| Hyperparameter selection | **[TBD]** | |
| Metrics and statistical tests | **[TBD]** | |

## Team split (proposal, to be agreed by the group)

Same group as scene-to-score: `ademasles`, `Argodm`, `OriGiNaLGet9`. The split covers both
projects at once: everyone holds a "data and evaluation" role in one project and a "model" role in
the other, and everyone writes one fusion here, since fusion is the core of the course.

| Member | recipe-fusion (project 1) | scene-to-score (project 2) |
|---|---|---|
| ademasles | Data, evaluation and ablations; intermediate fusion (cross-attention) | Encoders and extraction; generation and demo |
| Argodm **[to confirm]** | Text branch and its baseline; early fusion | Data and evaluation |
| OriGiNaLGet9 **[to confirm]** | Image branch and its baseline; late fusion | Projection and training |

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

Laptop: RTX 4070, 8 GB. Anything larger goes to the tech lab. Time per configuration: **[TBD]**
after a first measurement.

## Done when

- The unimodal baselines × fusions table exists, with three seeds, on the test split.
- Modality ablations are done.
- The repository runs from scratch with the README commands.
- The report is submitted.
