# Shared Prediction Errors Limit Ensemble Uncertainty Quantification in Pretrained Universal Interatomic Potentials

This repository contains the analysis code and reproducibility documentation
associated with a study of ensemble-based uncertainty quantification for
pretrained universal machine-learning interatomic potentials.

The analysis examines the extent to which disagreement among heterogeneous
pretrained models represents total prediction error when the models share
systematic error components.

## Repository structure

- `src/`
  Core estimators used for common-mode and model-specific error decomposition.

- `scripts/`
  Executed analysis and inference scripts for the primary MatPES-PBE study,
  sensitivity analyses, MatPES-r2SCAN control, and WBM control.

- `environment/`
  Notes on the separate execution environments required by the pretrained
  model families.

- `docs/`
  Information about upstream datasets and pretrained model checkpoints.

## Data

Raw upstream datasets are not redistributed in this repository.

The study uses:

- MatPES-PBE for the primary analysis
- MatPES-r2SCAN for the reference-functional control
- WBM for the chemical-novelty control

Processed reproducibility artifacts, frozen analysis outputs, bootstrap
results, sensitivity analyses, and publication source data are maintained
separately in the accompanying reproducibility archive.

## Pretrained model weights

Third-party pretrained model checkpoints are not redistributed.

Relative checkpoint paths appearing in executed scripts are retained for
provenance. Users should obtain the corresponding checkpoints from their
original providers.

## Reproducibility

The repository preserves the executed scientific scripts rather than
rewriting them into a new unified software environment. Different pretrained
model families required separate dependency environments.

The accompanying reproducibility archive contains the frozen processed
outputs needed to trace the numerical results reported in the manuscript.


## Upstream resources

Upstream datasets, software packages, and pretrained model checkpoints remain
subject to the licenses and terms of their respective providers.
