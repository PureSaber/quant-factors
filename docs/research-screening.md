# 因子研究筛选

`quant_factors.research.factor_requirements(names)`返回输入字段、已完成行情预热长度和基本面PIT需求。现有17个共享因子均可由研究配方引用；未知ID和重复symbol/date会抛错，不再静默忽略。

`factor_report(prices, names, cutoff=..., start=..., end=...)`返回覆盖率、1/5/20日IC衰减、RankIC、描述性ICIR、正IC比例、逐年/可选行业和regime分段，以及两两截面秩相关。标签端点必须严格早于cutoff。不足3个证券或常数截面的相关性返回null，不能当作零。

这些是描述性研究证据，重叠标签下的ICIR不是显著性检验；因子方向不得通过全样本报告自动反选。组合价值通过工作台的预登记单因子、消融和成本后账本验证。新公式仍需显式实现、测试并注册，不执行配方中的Python表达式。
