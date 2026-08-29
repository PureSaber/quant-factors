# quant-factors

Shared factor computation library for PureSaber quant research.

## Install

```bash
python -m pip install --requirement requirements.lock
python -m pip install --no-deps --no-build-isolation --editable .
python -m pip check
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

## 契约治理与依赖锁定

本仓库属于`strategy`层，声明消费`standard/v2@2.0.0`。治理元数据位于
`pyproject.toml`的`[tool.quant-workspace]`，由全栈清单校验；因子输出必须保留
时间来源和可用时间，PIT审计、walk-forward、purged和embargo验证不能被绕过。

`requirements.lock`精确覆盖运行时、开发和editable构建依赖。CI先按锁文件安装，再以
`--no-deps --no-build-isolation`安装本仓库editable包；更新锁文件必须同时运行三版本矩阵、ruff、全量测试、branch
coverage、`pip check`和跨仓清单校验。若发布验证失败，回滚到上一个默认分支提交及
对应锁文件；不移动旧tag、不改写历史因子产物，也不降低既有PIT测试门禁。

锁文件使用`pip-compile --extra dev --build-deps-for editable --allow-unsafe --strip-extras`
生成，禁止在锁之外临时解析构建后端。
