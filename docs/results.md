# Results

Final numbers on the official test split (22,716 pairs), produced by `just evaluate split=test`
and scored once. Top-1 accuracy in %, mean ± sample standard deviation over seeds 0, 1 and 2.
All choices (hyperparameters, fusion weights, temperatures) were made on the validation split.

## Methods by subset

| Method | All | Unseen | Title unseen | Image unseen | No dish name |
|---|---|---|---|---|---|
| Image only, MLP on frozen ViT-B/16 | 74.4 ± 0.1 | 75.6 ± 0.2 | 75.6 ± 0.2 | 74.2 ± 0.1 | 70.5 ± 0.1 |
| Text only, MLP on frozen BERT | 83.7 ± 0.1 | 79.3 ± 0.1 | 79.2 ± 0.1 | 83.5 ± 0.1 | 46.0 ± 0.5 |
| Text only, TF-IDF + logistic regression | 86.4 | 82.8 | 82.8 | 86.3 | 51.1 |
| Text only, fine-tuned BERT | 86.5 ± 0.3 | | | | |
| Early fusion | 91.5 ± 0.1 | | | | |
| Cross-attention | 92.6 ± 0.2 | 91.5 ± 0.2 | 91.5 ± 0.2 | 92.7 ± 0.2 | 75.3 ± 0.5 |
| Late, raw logit sum (frozen BERT + image) | 93.4 ± 0.1 | 91.9 ± 0.1 | | | 80.1 ± 0.1 |
| Late, calibrated (frozen BERT + image) | 93.3 ± 0.1 | 91.9 ± 0.2 | | | 80.2 ± 0.4 |
| Late, calibrated (TF-IDF + image) | 94.5 ± 0.0 | 93.4 ± 0.0 | | | 82.3 ± 0.3 |
| Late, calibrated (fine-tuned BERT + image) | **94.7 ± 0.0** | | | | |

Empty cells were not extracted from the summary for that subset. Subsets, defined in
[splits/](../splits/) by `just make-subsets`:

- *Title unseen*: the cleaned title does not occur in the training set (62.3 % of test).
- *Image unseen*: the image file is not byte-identical to a training image (94.9 %).
- *Unseen*: both (61.5 %).
- *No dish name*: the title does not contain the class name, ignoring case and accents (26.4 %).

## Modality ablations

The same trained models, with one modality degraded at test time: replaced by its training mean,
or shuffled across test examples (one fixed permutation). The late fusion weights are fitted once
on the full validation inputs and kept frozen.

| Test condition | Early | Cross-attention | Late (frozen BERT) | Late (TF-IDF) | Late (fine-tuned BERT) |
|---|---|---|---|---|---|
| Text + image | 91.5 | 92.6 | 93.3 | 94.5 | 94.7 |
| Image replaced by its mean | 74.4 | 81.6 ± 1.4 | 82.0 | 85.8 | 86.1 |
| Image shuffled | 37.8 | 73.5 ± 1.1 | 56.6 | 60.0 | 59.0 |
| Text replaced by its mean | 67.8 | 53.6 ± 1.9 | 73.1 | | |
| Text shuffled | 40.3 | 14.5 ± 0.6 | 29.5 | 32.1 | 34.1 |

Single-modality references: text 83.7 (frozen BERT), 86.4 (TF-IDF), 86.5 (fine-tuned BERT);
image 74.4. TF-IDF and fine-tuned BERT have no feature vector to average, hence no
"text replaced by its mean" row for those fusions.

## Benchmark leak

Measured by `just make-subsets` on the test split:

- 37.7 % of test titles occur verbatim (after cleaning) in the training set. On them, predicting
  the majority training label of the title scores 92.6 %.
- 1,165 test images (5.1 %) are byte-identical to a training image; 1,025 of them carry the same
  label.
- The ranking of methods is the same on the *unseen* subset as on the full test split.

## Selected hyperparameters

| Model | Search | Selected |
|---|---|---|
| Heads on cached features (text MLP, image MLP, early fusion, cross-attention) | lr ∈ {3e-4, 1e-3, 3e-3} × weight decay ∈ {1e-4, 1e-2} | lr 3e-4 for all four |
| TF-IDF + logistic regression | C ∈ {1, 10, 100} | C = 10 |
| Fine-tuned BERT | lr ∈ {2e-5, 3e-5, 5e-5}, 4 epochs, 10 % warm-up | lr 5e-5 |
| Late fusion | temperatures on a log grid, weight α on a 0.05 grid | per pair, in `meta.json` |
