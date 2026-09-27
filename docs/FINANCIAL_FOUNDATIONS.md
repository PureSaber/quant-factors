# 01 金融单位修正

保留旧默认 17 个因子及历史名字，避免旧实验静默改变。旧 turnover_20d 是成交量均值，旧 amihud 仍为兼容代理；新研究应显式使用以下名称：

- average_volume_20d：过去 20 个观测的平均成交股数。
- turnover_rate_20d_v2：mean(volume / free_float_shares)，输入 volume_unit="shares"、share_basis="raw"；free_float_shares 必须为当时已知原始流通股数。
- amihud_illiq_20d_v2：mean(abs(return_close.pct_change()) / amount)，amount_unit="currency"、明确 currency；return_close 为一致股东收益口径，amount 为实际成交金额，不以复权价×量替代。

`compute_factors(frame, factors=["turnover_rate_20d_v2", "amihud_illiq_20d_v2"])` 显式启用。使用 QDK financial.units 转换手/股与金额单位。不同币种 Amihud 不直接排序；如需比较，先按已知 FX 统一金额口径。零金额/缺失不伪造流动性。

验证：tests/test_financial_units.py；A 股共享实现逐因子数值一致性测试。全部新字段是来源契约，不意味着已有真实历史覆盖。
