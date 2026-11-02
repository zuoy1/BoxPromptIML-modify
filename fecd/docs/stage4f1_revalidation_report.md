# FECD 阶段 4F1-V：正式训练结果的最终工程复验报告

**最终补充工程验收：PASS。科研结论：NO CLEAR GAIN。**

> 原运行的最终验收因数值校验误报而失败；经独立版本化验收器复核，全部训练和评估产物通过补充工程验收。

原始执行器的 `FAILED` 与这次补充 `PASS` 是两个不同时间、不同验收器的结果。没有修改原 `status.json`、事件日志、训练结果、统计或 checkpoint，不能将此报告表述为原运行当时就已 PASS。

## 1. 身份、时间与执行范围

- 输入运行：`formal20_seed20261029_noshutdown_20261029a`。
- 原始运行路径：`/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a`。
- 验收器：`fecd/tools/check_stage4f1_v2.py`。
- 验收器版本：`stage4f1-v2-float32-exact-1`。
- 新验收开始：`2026-10-30T11:04:26.703173+08:00`。
- 新验收结束：`2026-10-30T11:06:15.909329+08:00`。
- 独立输出目录：`/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2`。
- 独立结果：`validation_v2.json`，`engineering=PASS`。
- 运行环境：`CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`；所有 checkpoint 以 `map_location='cpu'` 读取。

本次没有构造 BoxPromptIML 或任何待训模型，没有执行模型 forward、backward、optimizer step、seed 初始化或 GPU 训练；没有重新打开任何数据集 RGB/GT 文件，没有重新做 new_dev 推理，也没有打开 new_confirm、teacher_diag 或 IND/OOD 测试。单元测试使用内存数值、tensor、临时合成文件及不完整运行夹具，不训练模型。

## 2. 原校验为什么失败

原 `fecd/tools/check_stage4f1.py:72` 在已保存 JSON 的 Python float 数值上执行：

```python
assert abs(step['loss'] - step['pos_contribution'] - step['neg_contribution']) < 1e-7
```

训练端 `fecd/training/losses.py` 使用 float32 tensor 计算：

```python
pos_contribution = .5 * pos_mean
neg_contribution = .5 * neg_mean
loss = pos_contribution + neg_contribution
```

`fecd/training/f0_engine.py` 随后将各 tensor 分别通过 `.item()` 和 `float()` 保存到 JSON。训练相加时先发生 float32 舍入；JSON 读回后在 Python float 上重算残差，使用的是不同精度的相加路径。因此 loss 与两个贡献在 Python float 下的和可以相差一个合法的 float32 舍入误差。

当 loss≈2 时，float32 ULP 约为 `2.384185791015625e-7`，半个 ULP 约为 `1.1920928955078125e-7`。原固定阈值 `1e-7` 小于该半 ULP，合法记录可能被拒绝。

本次完整检查发现且仅发现以下三条原断言误报：

| 组 | Epoch | 轮内update | 累计update | loss | 正贡献 | 负贡献 | 原残差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 17 | 15 | 14143 | 2.0073587894439697 | 1.8338468074798584 | 0.17351208627223969 | 1.043081283569336e-07 |
| P3F | 12 | 728 | 10441 | 2.1430554389953613 | 1.5717717409133911 | 0.5712838172912598 | 1.192092895507812e-07 |
| P3F | 16 | 534 | 13779 | 2.018709182739258 | 1.3345842361450195 | 0.6841250658035278 | 1.192092895507812e-07 |

原执行器首先在 P0/Epoch17/update15（累计14143）中止；P3F 两条误报是本次完整检查额外识别到的同类数值情形。没有通过排除 P0/P3F 或修改任何日志来绕过问题。

## 3. float32 修复依据与实现范围

v2 先拒绝缺字段、非 Python float（包括 bool、字符串、整数伪装的训练 scalar）、NaN/Inf、负值/负零、溢出以及不能精确表示为 float32 的记录；再核对每项半权重贡献，最后使用以下固定语义：

```python
expected = struct.unpack(
    '<f', struct.pack('<f', pos_contribution + neg_contribution)
)[0]
assert loss == expected
```

数学口径为 `loss = round_binary32(pos_contribution + neg_contribution)`，同时要求 `pos_contribution = round_binary32(0.5 * pos_mean)`、负贡献同理。JSON scalar 必须是训练 tensor `.item()` 可精确输出的有限非负 float32 值。本实现没有放宽为更大的绝对容差，也没有将 NaN 当作比较成功。

`check_stage4f1_v2.check_training` 是原训练验收函数的独立版本化副本，仅将那一条 loss 断言替换为 `check_loss(step)`。其余预算、身份、顺序、几何、选择、梯度、访问、mean loss 和 checkpoint 断言保留。原两条贡献均值容差断言也保留；新数值函数补充了与训练 float32 半权重语义一致的精确核验。

AST 契约测试和正式验收都将新旧训练函数作结构比较：除指定 loss 断言替换之外，必须完全一致。评估允许名单、summary hash、聚合、memory gate 检查直接调用原 `check_evaluation`，没有重新执行 evaluator。

新入口还追加了用户要求的更强检查：严格 JSON/JSONL 解析、逐图 TP/FP/FN 到 F1/IoU 的复算、直接从 checkpoint 保存的 tensor 和 memory 重建 model hash、相对于前一报告的 final.pt SHA256 核对，以及独立输入快照和写入边界。所有新增内容均在新文件中，原验收器及冻结代码未改动。

## 4. 正反例单元测试及静态分析

| 测试行为 | 真实结果 |
| --- | --- |
| 三条实际误报记录 | 全部通过；同时确认原残差≥1e-7 |
| 正常float32与零值记录 | 通过；另用NumPy float32作为独立数值来源检查500组数值 |
| 篡改贡献及其对应均值而保留旧loss | 拒绝 |
| 正负贡献互相抵消但均值不变 | 拒绝 |
| loss改变1 ULP，偏差仍小于原1e-7 | 拒绝；防止将真错误当合法舍入 |
| 保存Python float和而没有float32舍入 | 拒绝 |
| NaN/±Inf、溢出、非float32数值、负值及负零 | 拒绝 |
| 各必需字段缺失、字符串、bool、None、容器 | 拒绝 |
| 逐图F1/IoU或混淆计数损坏 | 拒绝；正常与空分割约定通过 |
| checkpoint权重或memory tensor改变 | 内容哈希改变；非有限tensor拒绝 |
| 打开数据集或写到独立输出目录之外 | 边界测试拒绝 |
| 不完整运行 | CLI退出非零，独立写出FAIL，保留夹具原FAILED |
| JSON损坏、重复字段、NaN/Infinity、指数溢出 | 严格解析拒绝 |
| 原非数值逻辑保留 | AST对比通过 |

新增 `test_stage4f1_v2.py` 共 **15 项测试通过**。结合已有的配对统计和固定选择测试，最终 **28 项 CPU 验证测试通过，用时 11.784 秒**。

TDD 证据保留在 `logs/tests_red.log`：首次因尚无 v2 实现而失败。补充“不完整输入必须写出 FAIL”的集成测试也先失败（`logs/failure_delivery_red.log`），修复后通过。最终通过日志为 `logs/validation_test_suite.log`。

为遵守不初始化模型、不训练模型的限制，没有重跑含模型构造、恢复和参数更新行为的原始整套训练测试。本轮完整执行了新验收器全部测试、既有纯统计与固定选择测试，并真实执行完整产物验收；原运行自身174项测试通过的历史记录保持原样。

新源码及测试使用已有 AST/符号表检查器，语法和未解析全局名检查均通过；正式复验共检查 **50 个 Python 文件，issues=0**。未安装依赖。该检查不冒充完整类型检查器或第三方 lint。所有新增源码与测试文件各不超过300行。

## 5. 逐项正式复验结果

| 验收项 | 结果 | 实际核验内容 |
| --- | --- | --- |
| 四组20 Epoch | PASS | 80份summary齐全；逐轮3530图、883步；四组保留全部结果 |
| 每组17,660次正式更新 | PASS | 逐步累计update连续、最后batch为2；共70,640条 |
| 完整checkpoint可读 | PASS | 4个final.pt均在CPU读出；position={epoch:20,next_batch:0}；Adam状态及四类RNG存在 |
| 模型及checkpoint哈希 | PASS | 直接hash保存tensor+memory，不构造模型；与记录及原报告SHA256一致；final=last |
| 初始化和成员身份 | PASS | 3530个训练成员、530个new_dev成员；共同初始化hash；四组初始化及原恢复探针记录一致 |
| P2F/P3F监督预算 | PASS | 每图每类均为floor(n/2)，同一冻结候选计数；不排除任何组 |
| 顺序、几何与选择策略 | PASS | 20轮schedule在CPU按冻结算法重建一致；实际逐图顺序/几何hash、选择hash匹配 |
| 训练GT读取为零 | PASS | 80份访问日志gt_reads=0且denied为空；每轮文件集合精确匹配允许的RGB+缓存 |
| new_dev访问allowlist | PASS | 每组530个身份与路径、role、operation、prepared_mask归一化路径精确一致 |
| 锁定集合无记录访问 | PASS | 全部实际训练访问仅允许集合；全部评估访问均为new_dev；new_confirm计数0，teacher_diag/正式测试不在访问集合 |
| 每组530个唯一评估身份 | PASS | 逐图结果和GT访问日志与已验证new_dev排序身份完全一致 |
| F1/IoU符合TP/FP/FN | PASS | 2120条逐图记录逐项核对，四组全部域汇总与域等权值精确重聚合 |
| 三项配对统计 | PASS | 原seed=20261029、2000次、域内来源组配对bootstrap，完整JSON逐项精确一致 |
| 非有限、损坏、缺失与哈希 | PASS | 372个JSON/JSONL严格解析；模型及优化器状态有限；输入快照18330文件无变化 |
| 新loss数值核验全量覆盖 | PASS | 70,640条全部通过精确float32分解；合法旧误报3条，真正分解不一致0条 |

此外，400条训练曲线数据完整；原memory gate四组均为PASS且全部检查值为真。本轮只是检查已生成memory gate证据，没有用图像重跑该gate，也没有把两张RGB样本的原gate证据表述为全部530图的全排列实测。

历史“未访问锁定集合”的结论以既有受审训练/评估日志及其允许名单为证据；不是对任意未受审外部进程的全局证明。本轮 CPython `open` 边界另行阻止所有 `/home/datasets`、`/hy-tmp/datasets` 数据集文件读取和独立输出目录之外的写入，实际违规计数为0。

## 6. 输入不变性与历史状态保留

| 边界 | 结果 |
| --- | --- |
| 原运行文件 | 395个文件，文件集合无新增/缺失，全部SHA256与大小前后一致 |
| 完整基线 | 18,330个原项目/历史/运行文件，全部SHA256与大小前后一致 |
| 原protected_before | 17,716个历史保护文件SHA256符合原记录 |
| 原runtime_hashes | 54个冻结运行代码/配置文件SHA256符合原记录 |
| 原check_stage4f1.py | 未改动；包含原失败断言的版本仍保留 |
| 原status.json | 仍为FAILED；原字节及SHA256完全保留 |
| 原事件、训练日志与统计 | 未覆盖、未编辑；包含在运行快照中 |
| 新输出 | 独立保存于stage4f1_v/revalidation_20261030_v2，不写入原运行目录 |
| 实际读取证据 | 18,337个已存在文件的消费哈希，含历史保护集合及新验证源；非GT数据集读取数0 |

原 `status.json` SHA256：`5a83dda9acc1f7b96e149bab04956fb1a661fb0a4f505ad8153d1624cd349904`。

本次原运行仍没有成功的原版 `artifacts/validation.json`，也没有被伪造为成功。独立 `validation_v2.json` 的时间、版本、输入运行与 `historical_status_relationship` 字段清楚记录本次是后续补充验收。

CPython 审计是本程序的文件访问约束和可追踪证据，不是操作系统沙箱；输入文件哈希比对则独立验证原始内容未被改变。没有改写历史hash基准来接受新文件。

## 7. 冻结科研结果

| 组 | new_dev macro F1 | 工程复验 |
| --- | --- | --- |
| P0 | 0.345859 | PASS |
| P1 | 0.336740 | PASS |
| P2F | 0.346021 | PASS |
| P3F | 0.342369 | PASS |

| 预设总体主对照 | F1差值（0–1） | F1 95% CI（0–1） | 跨零 |
| --- | --- | --- | --- |
| P1-P0 | -0.009119 | [-0.023742, +0.004350] | 是 |
| P2F-P1 | +0.009281 | [-0.004743, +0.023603] | 是 |
| P3F-P2F | -0.003652 | [-0.021000, +0.015543] | 是 |

科研结论继续为 **NO CLEAR GAIN**。三项总体主对照的主口径F1/IoU区间以及来源组等权、数据域等权敏感性区间均跨零。工程PASS意味着这些既有结果完整且符合冻结协议，不意味着方法被证明有效，也不意味着方法等效。

P3F相对P2F，在NIST16、Columbia、Coverage的macro F1点估计分别增加约2.094、6.619、5.745个百分点，同时在CASIAv2下降约1.497个百分点。这些异质性同时保留，没有挑选小域优势来改变主指标结论。

仍只有一个训练seed；new_confirm未打开；来源代理组不保证真实独立来源；小域组数、域不均衡及224×224评估网格等既有科研限制依然存在。没有增加多seed、调阈值或新增创新模块。

## 8. 交付文件、输入输出哈希与复现

| 文件 | SHA256 |
| --- | --- |
| [check_stage4f1_v2.py](/home/projects/BoxPromtIML/fecd/tools/check_stage4f1_v2.py) | `19d6a92fccbbb49a0191586c98a0e7efb05703ec0f5a757ed60a8d6aaa087395` |
| [stage4f1_v2_checks.py](/home/projects/BoxPromtIML/fecd/tools/stage4f1_v2_checks.py) | `79218fc2480821e7c338b2c71023448dd3a9d067224a4353a2e7b568ec8c3332` |
| [stage4f1_v2_io.py](/home/projects/BoxPromtIML/fecd/tools/stage4f1_v2_io.py) | `957f6def97af5a8f8f5384393a90577fccfd32d7977c75549110e74bbf4b8b36` |
| [stage4f1_v2_run.py](/home/projects/BoxPromtIML/fecd/tools/stage4f1_v2_run.py) | `1cbb6958b83599c93e8d3a1090f67b0b9082f24352e5ca4e29deca55c66c8ae4` |
| [test_stage4f1_v2.py](/home/projects/BoxPromtIML/fecd/tests/test_stage4f1_v2.py) | `2cc9a08eccd7959153ff787ff9a8705c0deab075bbaf974e28c3936ee2262c1e` |
| [validation_v2.json](/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2/validation_v2.json) | `97af9005e6f611bfdc7df31f859babfe32dea8b2b256fa3c8c9828e4b35d7a42` |
| [inputs_before.json](/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2/inputs_before.json) | `4d5924445de40913b4791c848aca7418cd54c3e2ad6abdbf3acdedd103942f46` |
| [access_and_inputs.json](/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2/access_and_inputs.json) | `58faf57975bbba411d866af0be7583dd903971292a0a14d91c12450a62c582b8` |
| [revalidation.log](/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2/logs/revalidation.log) | `585fdc22b876cc2d5397b9c7142029c4dfc07ce524b1a596f5637eb37864dc2a` |
| [validation_test_suite.log](/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2/logs/validation_test_suite.log) | `e34f8a6a1f4a65afc10634e2c5e7fab84cd5b80cd64648e72564c48fe958e55d` |
| [tests_red.log](/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2/logs/tests_red.log) | `ab1a07b4864377b91e294261ccd0519c48cf2d3665ec4f7b8cd07b1d95e5a6d7` |
| [failure_delivery_red.log](/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2/logs/failure_delivery_red.log) | `c4810b7ae10e88e5fb33015f3001b7c6cf7eb2cf2a2fa43362b2178bd599f4b9` |

完整交付清单将写入同目录 `delivery_manifest.json`，收录源码、测试、报告、JSON、输入清单及全部日志；它自身的SHA256保存在 `delivery_manifest.sha256`，以避免自引用哈希。`inputs_before.json` 列出全部18,330个原文件的路径、字节数及哈希，`access_and_inputs.json` 列出复验实际读取文件的哈希及边界结果。

| 组 | 原final.pt SHA256 | 直接重建model+memory hash |
| --- | --- | --- |
| P0 | `9ea55e01e5e354d252fb6470c7a3aa04712840eb657917e694c19f56650bd1ea` | `434086f339dcaecbee82bd55c5db0e9389d3e48ed95def6256a4e47204b4db1d` |
| P1 | `b0e56a4eb4ffc361f80476af580b30c2351ad885bea10f251b08ae119cdf4d89` | `6cc9abfbc0fcf1bf9dd5388fed1e120ff25c2d466cac9ee44095efd3008620c5` |
| P2F | `1d1997cd71b8ff82b5e42debdf5e1e9e8ea659b8c1194c231481c3e0088a9404` | `36cc7e0ee4ce48636d2ff818e9bf777a31f729e3e7efc0fd19ef03b8b892d4b2` |
| P3F | `8c6dc3512bdebaf0885b85dbe9b65af737b804f90e21238405073cbe4fa11ea6` | `1c46889105401c9d848b65c1971fe33e5befc15a8d7a9d3d36b7a9bdd7046cce` |

独立JSON的canonical `result_hash`：`ed39e8d6cf278ccbcedba5b8caa95915dd7a4070a77328453599be0376bc7093`。这是去掉 `result_hash` 字段后的规范JSON内容哈希，与整份文件的SHA256不同。

本次实际执行命令（项目根目录 `/home/projects/BoxPromtIML`）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest fecd.tests.test_stage4f1_v2 fecd.tests.test_stage4f1_statistics fecd.tests.test_stage4f0_selection -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_stage4f1_v2 --input-run /hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a --output-dir /home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2
```

上面第二条是本次已执行命令的记录。验收器拒绝覆盖已有 `validation_v2.json`，不能直接重跑覆盖已签发结果；新复验必须使用新的独立目录，并将原 `inputs_before.json` 作为基准带入，保留旧输出。正常命令不启动子进程，CLI本身的CPython边界拒绝其启动子进程。

## 9. 未解决问题与停止状态

本阶段工程范围内没有未解释的NaN/Inf、损坏日志、缺失文件、模型哈希改变、预算偏差或访问边界异常；所有正式验收项已实际执行，最终补充工程验收为 **PASS**。

保留的限制是科研证据强度及日志审计边界：单训练seed、开发集分析、来源代理依赖、小域样本量与未做独立确认。它们不会因工程复验成功而自动消失。

本阶段至此停止：不重新训练、不重新评估GT、不运行多seed、不开放new_confirm，不继续增加模块。

