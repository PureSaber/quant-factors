# 完整因子目录

```text
quant-factors list --all
quant-factors list --all --json
```

两种输出来自同一个`factor_catalog()`结果，覆盖core、价量、具名研究、中低频、基本面特征和分钟因子。相同名称合并成一条记录并保留所有来源目录，不能将目录数量相加当成独立因子数。历史`list --json`和计算默认选择保持兼容。

机器接口`quant_factors.catalog.factor_catalog()`返回`quant.factor-catalog/v1`：

- 唯一名称、描述来源、因子家族和历史17个默认组标记。
- 可调用的日频面板或分钟计算入口及输入要求；日频要求直接复用`expression_requirements`，不另维护一套参数规则。
- 分钟聚合要求排序后的日内数据；缺少`time`列时保持调用者的原始组内次序。最少bar数只是计算下限，不保证方差、成交量或完整交易日条件满足；不声称PIT认证。
- 源文件SHA-256和整个目录的规范JSON哈希，识别实际实现和目录内容变化。
- 缓存真正支持的六个FactorSpec算法及单市场输入、无拟合状态、全表读取边界。

持久缓存不通过这个名称目录自动选择算法。名称相近不代表日频函数与显式频率/PIT引擎数值等价，应建立带版本、依赖、日历和as-of的FactorSpec，并分别验证研究用途。

兼容语义会显式提示：`turnover_20d`是平均成交量；真实换手率使用`turnover_rate_20d_v2`及对应单位/PIT流通股输入。旧Amihud采用价格乘成交量代理，v2要求实际成交金额。目录不修改任何历史因子结果，也不把公式数量解释为经过验证的有效alpha。
