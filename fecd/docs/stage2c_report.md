# FECD 阶段 2C：缓存预检与全量 SAM 缓存报告

日期：2026-10-23。状态：**PASS**。仅使用锁定的 5,737 条训练数据生成 K=1/K=8 cache；没有读取测试 split/GT，没有改动 split，没有训练学生模型。

## 预检和冻结协议

增强后的 schema 为 `fecd-sam-cache-v2`。检查覆盖 K、schema/config hash、sample key 推导、dataset/split/sample ID、image SHA1/SHA256、box hash、native 尺寸、box count/empty flag/box masks 交叉一致性、mu/var/agreement dtype/shape/range/finite、K1 方差为零及 K8 Base 辅助图。K8 单独保存 `base_mu` 供全量 Base 对照。配置 hash 包含 K、SAM checkpoint SHA256、prompt 列表、native probability resize、统计网格/阈值和 max-box 策略，因此不同 K、权重或空间协议不能复用缓存。

cache 写入仍使用每路径排他锁和临时文件原子替换。新增 PID 锁检查：只有锁内 PID 已不存在时才回收；活跃 PID 锁保留并等待超时；无法解析的锁不会被擅自删除。单元测试验证活跃锁不删、陈旧锁可回收、关键 metadata 篡改报错、不同 K/config 不能混用、损坏缓存可安全重建。

六张真实训练样本的统计顺序比较结果见 `fecd/assets/stage2c_preflight_comparison/preflight_report.json` 和 `comparison_statistics.csv`。边界区域定义为方案 B 的 `mu>=0.5` mask 轮廓外扩 5 像素，未使用 GT。

| 统计图 | 最大绝对差（六样本逐图最大值中的最大值） | 平均绝对差（像素 MAE 再对六图平均） | 边界最大差 | 边界平均差（六图平均） |
|---|---:|---:|---:|---:|
| mu | 2.38e-7 | 8.91e-9 | 1.79e-7 | 1.82e-8 |
| var | 0.03281 | 2.93e-5 | 0.03281 | 1.58e-4 |
| threshold vote | 0.51637 | 0.001542 | 0.51637 | 0.007501 |

选定**方案 B**：先将每个 prompt 的 sigmoid probability resize 到 224×224，再在学生实际接收的网格计算 population mean、variance 和 `[0.4,0.5,0.6]` threshold vote。mean 几乎不变，但 var/vote 在边界插值处明显不具交换性；方案 B 令不确定性对应学生坐标。该定义写入 config hash 并冻结，阶段 2B 的旧 `v1` caches 不会被复用。六张预检样本的方案 A/B 对照图位于 `fecd/assets/stage2c_preflight_comparison/`。

Float16 量化对比见 `float16_quantization.csv`。六张样本、K1/K8 汇总：

| 图 | float32 vs 磁盘 float16 MAE | 最大绝对误差 | 0.25 / 0.5 / 0.75 阈值翻转像素数 |
|---|---:|---:|---:|
| mu | 1.68e-5 | 2.4414e-4 | 3 / 6 / 3 |
| var | 2.34e-7 | 6.1035e-5 | 0 / 0 / 0 |
| agreement | 1.45e-6 | 1.6278e-4 | 0 / 0 / 0 |

mu 的 12 次阈值翻转全部在对应阈值 ±0.002 范围内（六样本总计 233、161、60 个像素分别落在这三个近阈值带）。这属于 float16 边界量化影响；缓存保留 float16 以控制体积，涉及阈值消融或硬 mask 比较时应将近阈值像素作为量化敏感区域报告，或从 `base_mu`/概率 cache 读回后按明示阈值计算。量化前/磁盘 min/max 分列在预检报告中，没有将 float32 范围误标为磁盘值域。

预检测试命令：

```bash
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tests.run_stage2b_tests
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.stage2c_preflight
```

测试 harness 输出 9 checks passed；包含 Box 的 HFlip、VFlip、Rot90 坐标同步检查。六张样本 K1/K8 Dataset 读取、source/config 校验成功，K1 与 K8 Base 的 float16 cache 逐像素完全一致。

## 全量生成结果

固定使用 `fecd/assets/boxes_train.json`，没有重导 Box。SAM checkpoint 为 `/home/weights/sam_vit_h_4b8939.pth`，SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`，RTX 4090 / `cuda:0`。

| 数据集 | train 样本 | K1 成功 | K8 成功 | K1 平均 SAM 秒/张 | K8 平均 SAM 秒/张 | K1 cache | K8 cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| CASIAv2 | 5,123 | 5,123 | 5,123 | 0.2294 | 0.2763 | 147,424,889 B | 353,357,437 B |
| NIST16 | 414 | 414 | 414 | 0.5268 | 1.8465 | 13,354,914 B | 31,421,556 B |
| Columbia | 130 | 130 | 130 | 0.2400 | 0.3316 | 9,852,273 B | 26,305,281 B |
| Coverage | 70 | 70 | 70 | 0.2305 | 0.2762 | 3,501,471 B | 8,041,151 B |
| **合计** | **5,737** | **5,737** | **5,737** | — | — | **174,133,547 B** | **419,125,425 B** |

生成总壁钟约 **4,231 秒（70.5 分钟）**：K1 阶段 1,686 秒（28.1 分钟），K8 阶段约 2,546 秒（42.4 分钟）。K1 SAM 推理累计 1,441 秒，图像 I/O 44 秒，cache 写入 115 秒；K8 SAM 推理累计 2,242 秒，图像 I/O 39 秒，cache 写入 132 秒。余下为模型与逐样本 Python/校验等开销。峰值 GPU 显存为运行期间 `nvidia-smi` 采样观察约 **6,731 MiB（6.57 GiB）**；模型运行显存约 6.0 GiB allocated 的阶段 2B 实测作为参考，未把它标为阶段 2C 的 allocator peak。

K1 manifest：`fecd/assets/stage2c_cache/k1/d7ccb769414f/manifest.json`  
K8 manifest：`fecd/assets/stage2c_cache/k8/2bff688546a7/manifest.json`  
全量生成逐样本 JSONL：各目录 `generation.jsonl`；成功/失败清单和无 GT Dataset index：各目录 `manifest.json`、`dataset_index.json`。顶层运行摘要：`fecd/assets/stage2c_cache/stage2c_run_summary.json`；标准输出/逐样本日志：`fecd/assets/stage2c_cache_run.log`。

## 全量验收与异常诊断

完整只读审计结果：`fecd/assets/stage2c_cache_audit.json`；逐样本统计：`fecd/assets/stage2c_cache_statistics.csv`。审计逐条打开 11,474 个 NPZ 并校验字段、形状、dtype、有限值、值域、config/source/box 身份；K1/K8 身份集合均为同一批 5,737 个唯一训练 key，cache key 共 11,474 个且无冲突；所有 `box_masks` 与冻结 Box 一致，非空 Box 未在 224 rasterize 后消失。所有 cache 可加载，未解决错误 0，生成重试 0。

修正后的两套 `FECDDataset` 分别重新完整遍历 **5,737/5,737**，所有 image/cache shape、finite 和 identity 校验通过。图像、mu、var、agreement、box_masks 使用同一离散几何操作；Box 从 native XYXY 映射到 224 网格后也使用相同操作变换，并同时保留 native boxes。索引 allowlist 仅含图像路径、缓存路径、boxes 和训练身份/hash 元数据；未含 GT 路径，Dataset 未读取 GT。K1 `var` 全部为 0；K8 `mu/agreement∈[0,1]`、`var∈[0,0.25]`。K1 与 K8 `base_mu` 在 5,737 张上逐像素最大差 **0.0**。修正后重跑全量审计状态 `passed`，未解决错误 0。

前景覆盖率定义为 `mean(mu >= 0.5)`，不是准确率。全训练样本分布：均值 0.07782，中位数 0.03069，p99 0.55451，最大 0.94976。出现 1 张 K8 全零（CASIAv2 `Tp_S_CNN_M_N_sec00091_sec00091_10355`），全图前景 0 张；前景覆盖率最高为 CASIAv2 `Tp_D_NNN_L_N_ani00010_nat00099_10217` 的 0.94976；没有全图前景样本。均值 var 的样本间中位数 `1.54e-5`、p99 0.03704、最大 0.19299（NIST16 `NC2016_2255`）。`u_threshold=4*agreement*(1-agreement)` 的样本均值中位数 0.001013、p99 0.27617、最大 0.95998。K1/K8 mean 的逐样本平均绝对差中位数 `9.34e-5`、p99 0.07289、最大 0.46325（NIST16 `NC2016_3660`）。以上异常保留并显式记录，没有依据主观判断剔除。

阶段 2B 指定的 CASIAv2 动物样本 `Tp_D_CND_M_N_ani00018_sec00096_00138` 已保留为重点诊断图。其他最低/最高前景覆盖率、最高/最低 mean variance、最大 K1/K8 差异及各 dataset 典型样本均在 `fecd/assets/stage2c_anomaly_visualizations/`。这些只展示图像、Box、K1 hard mask、K8 mean、K8 variance 和 threshold uncertainty；没有 GT，不用于判断教师准确率。

## 新增工具与文件

- `fecd/pseudo/cache_schema.py`：v2 schema、source/K cross-check、safe lock recovery。
- `fecd/pseudo/sam_ensemble.py`、`generate_cache.py`：方案 B 统计与带身份校验的可续跑生成。
- `fecd/datasets/fecd_dataset.py`、`transforms.py`：无 GT 学生 Dataset；图像、缓存图及 Box 同步几何变换。
- `fecd/tools/stage2c_preflight.py`、`run_stage2c_cache.py`、`audit_stage2c_cache.py`：预检、两套全量运行、全量审计/统计/诊断图。
- `fecd/tests/test_cache_schema.py`、`run_stage2b_tests.py`：篡改、锁与旧测试回归。
- 产物：`fecd/assets/stage2c_cache/`、`stage2c_cache_audit.json`、`stage2c_cache_statistics.csv`、`stage2c_preflight_comparison/`、`stage2c_anomaly_visualizations/`、`stage2c_cache_run.log`。

原始 BoxPromptIML 文件、checkpoint 和既有工作区修改未被覆盖或还原。阶段 2C 只生成训练集 SAM/Box cache；NIST16 `world/NC2016_0359.jpg` 跨 train/test source-group 风险仍须在后续报告中披露。

## 阶段结论

阶段 2C 最终验收为 **PASS**。cache 生产、schema、身份、修正后的 Dataset 读取及 Box/图像几何同步、无 GT 边界均通过，零失败、零损坏、零重复 key。最终回归命令 `/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tests.run_stage2b_tests` 输出 `9 checks passed`；最终全量审计日志位于 `fecd/assets/stage2c_audit_run.log`。

**阶段 3 的数据输入条件已具备，可在用户验收后启动阶段 3 的独立教师诊断与新模块开发。** 本报告不证明 SAM 伪标签准确性；全零/高覆盖案例及 NIST16 同源组风险应在阶段 3 使用隔离流程进一步审阅。阶段 3、取证特征、学生训练和 B1–B5 实验均未开始。
