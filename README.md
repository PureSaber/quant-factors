# quant-factors

Shared factor computation library for PureSaber quant research.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
quant-factors list
quant-factors compute --config configs/example.yaml
```

## Factors

- `momentum_20d`
- `volatility_20d`
- `mean_reversion_z_20d`
- `volume_surge_5d`

## Research validation

The Python API provides ordered expanding/rolling walk-forward splits, interval-overlap purged
K-fold, embargo windows, feature-availability leakage audits, Benjamini-Hochberg FDR correction,
probabilistic Sharpe ratio, and fold-stability summaries. Splitters never shuffle observations.

```python
from quant_factors.validation import purged_kfold_splits, walk_forward_splits

folds = walk_forward_splits(dates, train_size=504, test_size=63, embargo_size=5)
purged = purged_kfold_splits(sample_times, label_end_times, n_splits=5, embargo_size=5)
```
