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

## M8认证FactorFrame

`compute_factor_frame`是跨资产、全频率认证入口。它不接受调用方直接传入DataFrame或Arrow
表，只接受`FactorInputRef`，并按layer调用`quant-data-kit v0.8.1`冻结的
`load_verified_curated_bars`或`load_verified_normalized_events`工厂。频率、年化周期、因子窗口、
逐行`as_of`和辅助数据快照都必须显式声明。

```python
from quant_factors import AsOfSpec, FactorInputRef, compute_factor_frame

frame = compute_factor_frame(
    input_ref,
    frequency,
    factor_specs,
    as_of=AsOfSpec("source_available_at", None),
)
```

冻结测试数据必须使用`compute_factor_frame_from_fixture`，其结果只能标记为
`fixture-certified`；固定时点重述只能标记为`research-restated`。输出携带完整manifest、
规范typed-cell envelope、RFC8785逻辑SHA-256和独立Parquet物理SHA-256。认证对象只能由受控
工厂创建，并会反解Parquet字节与Arrow逻辑表逐字段比对，防止拼接不同运行的物理文件和
逻辑manifest。正式与fixture认证级别只由已验证输入的真实类型推导，不接受调用方传入scope
或可导入的令牌。`code_version`只来自干净Git工作树或pip生成的VCS安装元数据中的40位实际
commit，不预填尚未存在的tag；工作树有未提交修改或无法证明代码来源时拒绝生成认证产物。

外部基本面、FX或参考数据先通过`load_verified_auxiliary_source`从单次读取的Parquet字节
校验物理hash、Arrow逻辑hash和PIT区间，再参与确定性的双时间选择。

## Factors

- `momentum_20d`
- `volatility_20d`
- `mean_reversion_z_20d`
- `volume_surge_5d`

以上`*_d`名称属于永久保留的`legacy-daily`兼容入口，不能产生M8认证声明。认证因子使用
`*_p`周期语义，例如`momentum_20p`和`volatility_20p`；`20p`表示同一标的20个已完成period，
不表示自然日。

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

锁文件使用`pip-compile --extra dev --build-deps-for editable --allow-unsafe --strip-extras --constraint requirements-constraints.txt`生成，禁止在锁之外临时解析构建后端。
`requirements-constraints.txt`只保存Python3.10—3.12共同解析所需的兼容性上界；依赖声明、
约束和重新生成的锁文件必须作为同一审查单元提交，禁止只手改某个传递依赖版本。
