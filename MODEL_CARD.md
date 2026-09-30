# Model Card

## Model

The current model is a scikit-learn pipeline:

- `StandardScaler`
- `RandomForestClassifier`

Random Forests do not require feature scaling for tree-based splitting. The
scaler is kept so preprocessing remains explicitly fold-local and the same
pipeline interface can support future non-tree baselines.

Configuration:

- 400 trees
- balanced class weights
- minimum leaf size of 2
- fixed random seed

## Inputs

The model consumes aggregated clip-level features:

- 28 audio features
- 170 skeleton features
- 198 multimodal features in total

Duration, raw count, and path-length features are excluded from training.

## Outputs

The model predicts one dominant violin technique label. It can also output raw
Random Forest probabilities, but these should not be interpreted as calibrated
confidence scores.

## Intended Use

This model is intended as a research prototype for violin technique recognition
and as a component for future educational feedback systems.

## Not Intended For

The model should not be used as a complete automated grading system. It does not
evaluate intonation, rhythm, tone quality, musical expression, or teacher rubric
scores.

## Evaluation Summary

The best current result is multimodal fusion:

- Stratified clip CV: 96.60% accuracy, 96.43% macro-F1
- Source-video-group CV: 95.92% accuracy, 96.08% macro-F1

These results should be interpreted as within-dataset performance because
verified performer and piece metadata are not yet available.
