# 持久滚动因子缓存v1

`quant_factors.cache.FactorCache`复用`engine_v2.compute_factor_table`及既有`FactorSpec`、`FrequencySpec`、`AsOfSpec`。原有因子、PIT及FactorFrame接口保持原样。

```python
from pathlib import Path
from quant_factors.cache import FactorCache
from quant_factors.contracts_v2 import AsOfSpec, FactorDependency, FactorSpec, FrequencySpec

frequency = FrequencySpec(
    "daily", "session_bar", "365", "synthetic-daily-v1", "v1",
    None, "trading_day", None, None, None,
)
factor = FactorSpec(
    "momentum", "1.0.0", "momentum", "bar",
    [FactorDependency("market", "close", "available_at")],
    5, "float64", False, "null",
)
result = FactorCache(Path("state/factors.sqlite3")).compute(
    verified_arrow_table, frequency, [factor],
    as_of=AsOfSpec("source_available_at", None),
    input_identities=[snapshot_artifact_ref_json, member_artifact_ref_json],
    universe=["SYNTH-A", "SYNTH-B"], run_id="window-5",
)
print(result.computation_id, result.metadata)
```

输入由可信resolver验证；缓存不打开`ArtifactRef.uri`，也不把传入身份当作已验证字节。实际投影后的Arrow内容摘要同时绑定到计算身份。因此同一ref意外返回不同内容不会命中旧结果；外部真实性仍由resolver保证。Lab的`QdkResearchAdapter`使用真实QDK验证清单、成员、角色、schema和摘要。

## 算法和增量语义

支持`momentum`、`volatility`、`downside_volatility`、`mean_reversion_z`、`rolling_mean`、`rolling_sum`，每个因子只允许一个`market`依赖。窗口单位是按证券排序的**观测数**，不补齐缺失交易日，不推断日历。波动率年化沿用显式`FrequencySpec.periods_per_year`。其他算法、多依赖及辅助源请求明确拒绝。

行情按`instrument_id/event_time/sequence/event_id`规范排序；重复完整事件身份、空身份、空选择被拒绝。键不同的同时间事件是不同观测。缺失数值、可得性、固定点及时间顺序沿用现有引擎；`missing_policy=null`保留null，`error`报错。

首个版本全量执行。相同语义族的新输入逐证券比较最长相同前缀和后缀。重算从第一个变化处开始，延伸到最后变化处之后一个完整依赖窗口；输入切片再向前补充回看窗口。未变前缀及影响范围之外的后缀直接使用旧结果。插入、删除会改变观测位置，因此后缀必须先经过完整窗口才恢复复用。追加只执行新增行及回看区间。数值运算仍完全由同一个既有引擎完成，验收使用Arrow整表`equals`，比较字段、键、null及数值，**不使用数值容差**。

`execution_mode`区分`full`、`incremental`、`cache_hit`。`engine_rows`是实际送入引擎的行数，包含回看；`recomputed_rows`是实际替换输出行数；`reused_rows`是保留输出行数；`input_rows`是选中行情行数。参数或源码变化走全量；新输入内容相同而不可变版本变化时可零计算复用，生成新的计算身份。

## 身份和持久化

完整计算键绑定操作/版本、因子参数、源码文件字节、Python/pandas/PyArrow版本、输入内容身份、投影Arrow摘要、频率/日历、PIT、证券池及训练声明。源码来自`cache.py`、`engine_v2.py`、`pit_v2.py`、`contracts_v2.py`，摘要不包含机器绝对路径。输入ref的URI和媒体标签不构成内容身份。

增量候选族不包含输入版本，但包含上述其他语义。即使上一条候选来自不同快照，也必须逐行实际比较，不能凭版本前缀猜测追加。所有旧结果按完整身份保留，历史修订不覆盖旧结果。

SQLite表`factor_cache_inputs_v1`按投影输入字节摘要只存一份Arrow IPC；`factor_cache_v1`的各参数结果引用`input_hash`。输入、输出和元数据在一个事务中提交。读取校验BLOB摘要；并发同身份发布以唯一键归并，并核对输出摘要一致。没有自动删除、迁移旧业务库或把缓存作为任务状态权威。

**资源边界：**QDK保留原始共享字节，缓存另外保留一份去重投影和每组因子输出。读取、比较和序列化仍为O(N)，本版本显式全表物化；只承诺因子计算增量，不能视为端到端流式。多个稀疏修订之间的区间可能一并重算。本机SQLite/可信进程适用；不承诺多机共享文件数据库或恶意缓存防护。

## PIT与训练边界

引擎沿用每个依赖的`available_at`检查；迟到依赖在相应as-of之前输出null，窗口位置保留。固定as-of不会让未来行情产生可用值；后续追加不修改旧观测结果。每个输入快照必须已是期望市场版本的一行一个事件，**不在本缓存内选择多vintage市场修订**。如果以晚可得修订替换旧行，旧值不再包含在该快照里，既有引擎会产生null，而不会从其他缓存版本猜回旧值；要重现旧时点必须使用包含正确历史vintage的不可变快照。已有辅助源双时态选择器及其回归保留，新缓存不支持辅助源。

可选`training={"start_ns":"...","end_ns":"...","fold_id":"fold-1","state_sha256":null}`要求固定as-of不晚于end；event_time晚于end的行情不参与引擎。start之前的行只供回看，Lab摘要仅统计start之后的区间。折、起止边界均进入身份；本版没有拟合，非null拟合状态明确拒绝，不能将声明隔离测试称为已验证拟合无泄漏。

## 验证

`python -m pytest tests/test_cache.py -q --no-cov`覆盖六算法的追加、修改、插入、删除、迟到、窗口1、多窗口、多证券、重启命中、并发、损坏拒绝、时间与源码身份、训练截止。运行完整`python -m pytest -q`同时验证原有PIT及统计回归。
