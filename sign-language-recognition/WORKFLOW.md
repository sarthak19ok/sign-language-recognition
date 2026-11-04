# Project workflow

Run notebooks in order when needed. The current feature and model artifacts are already saved, so model notebooks can run directly.

## Main steps

1. `01_dataset_exploration_polished_v2.ipynb` checks the dataset and split.
2. `05_feature_engineering.ipynb` creates fixed-length, normalized landmark features.
3. `06_classical_ml_baselines.ipynb` tunes and compares six classical model families.
4. `07_temporal_models.ipynb` compares LSTM and GRU with signer-grouped folds.
5. `08_error_analysis.ipynb` examines validation confusions.

The extraction and video-acquisition notebooks document earlier data preparation. Use Python 3.12 with the packages in `requirements.txt`.

## Results

Each model has one `results.csv` in its own folder under `out/`. `out/comparison.csv` gives the cross-model comparison. The frozen split is signer-independent: 868 train, 294 validation, and 304 test clips. Model settings are selected with signer-grouped training folds; test metrics are reported only after selection.

Landmark extraction overlays and feature ablations are saved under `data/processed/WLASL100/`.
