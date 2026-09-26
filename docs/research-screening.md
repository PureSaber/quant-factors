# 因子研究筛选

`quant_factors.research.factor_requirements(names)`返回输入字段、已完成行情预热长度和基本面PIT需求。现有17个共享因子均可由研究配方引用；未知ID和重复symbol/date会抛错，不再静默忽略。

`factor_report(prices, names, cutoff=..., start=..., end=...)`返回覆盖率、1/5/20日IC衰减、RankIC、描述性ICIR、正IC比例、逐年/可选行业和regime分段，以及两两截面秩相关。标签端点必须严格早于cutoff。不足3个证券或常数截面的相关性返回null，不能当作零。

## 受限因子表达式

研究配方可声明`factor_expressions`，然后在普通`factors`方向字典中引用自定义名称。表达式先解析为受限AST，再由内部解释器计算；不使用`eval`。属性访问、导入、索引、推导式和任意函数调用都会被拒绝。单个表达式最多500字符、100个AST节点、深度12；一次最多定义32个表达式。

```yaml
factors:
  risk_adjusted_momentum: 1
factor_expressions:
  risk_adjusted_momentum: clip(momentum_20d / volatility_20d, -10, 10)
```

输入名称限于`open`、`high`、`low`、`close`、`volume`、`amount`、`turnover_rate`、`market_cap`、`pe_ratio`、`pb_ratio`、内置因子名和无环的其他自定义因子名。函数限于`abs`、`log`、`sqrt`、`sign`、`rank`、`minimum`、`maximum`、`clip`、`where`、`lag`、`shift`、`delta`、`pct_change`、`rolling_mean`、`rolling_std`、`rolling_min`、`rolling_max`、`rolling_sum`和`zscore`。`lag`和`shift`只接受非负整数字面量；变化周期必须为正；滚动窗口必须是2至252的整数字面量。时间操作按symbol分组并按date稳定排序，`rank`按date做截面排名。

```bash
quant-factors validate-expressions --expressions expressions.yaml
```

公开API为`validate_expressions(mapping)`、`expression_requirements(names, expressions)`和`compute_research_factors(frame, names, expressions)`。需求结果包含传递后的源字段、保守的已完成行情预热长度、PIT依赖和直接表达式依赖。

`factor_report(..., expressions=...)`复用同一校验与计算入口。传入`baseline_names=(...)`会并列报告候选因子的原始RankIC，以及每日截面对基准因子残差化后的RankIC。传入`neutralize_by=("industry", "market_cap")`会比较原始与中性化RankIC。文件式筛选可通过`quant-factors screen --config screening.yaml`运行，配置支持`input`、`output`、`cutoff`、`factors`、`factor_expressions`、`baseline_factors`和`neutralize_by`。

这些都是描述性研究证据，重叠标签下的ICIR不是显著性检验；因子方向不得通过全样本报告自动反选。增量组合价值仍需工作台中预登记、成对的样本外回放验证。
