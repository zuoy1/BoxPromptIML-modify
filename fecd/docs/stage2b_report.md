# FECD 阶段 2B：多提示 SAM 教师与离线缓存报告

日期：2026-10-23  
状态：**PASS（本阶段范围内）**。本阶段仅生成 6 张训练样本的 K=1/K=8 测试缓存；没有生成 5,737 张正式缓存，没有训练学生模型，也没有使用测试集选择参数。

## 实现内容

新增实现均位于 `fecd/`：

- `fecd/tools/export_boxes.py`：读取 2A 锁定的 `train_manifest.csv`，用 `(dataset,id)` 连接既有根 manifest；GT 只读取 manifest 指定的 `prepared_mask`，图像和 GT 路径均按显式 `/hy-tmp/datasets` → `/home/datasets` 映射解析。图像 SHA256、尺寸和 GT 尺寸逐记录核验。
- `fecd/pseudo/box_ops.py`：固定 8 个 prompt variant，native XYXY 半开区间；K=1 是 K=8 的同一 Base Box。无随机 jitter。
- `fecd/pseudo/sam_ensemble.py`：加载本地 SAM ViT-H；每张图只调用一次 `set_image`，每个 prompt 下最多 3 个 box 分开 decode，概率图按像素 maximum 合并，然后计算均值、总体方差和阈值投票比例。SAM 返回顺序按 `(masks/logits, iou_predictions, low_res_masks/logits)` 处理。
- `fecd/pseudo/cache_schema.py`：`fecd-sam-cache-v1` NPZ schema，配置及源身份哈希、字段校验、原子替换、排他锁、断点复用和损坏缓存安全重建。cache key 含 dataset、split、sample ID、image SHA1、box hash、config hash 与 schema version。
- `fecd/pseudo/generate_cache.py`：按 K/config 分目录生成 cache，写 JSONL 运行日志、成功/失败 manifest 和不含 GT 的 Dataset index；支持按 ID/数量选择和续跑。
- `fecd/datasets/fecd_dataset.py`、`fecd/datasets/transforms.py`：学生 Dataset 只接收 image、boxes、cache 和源身份 metadata；不接受任何 GT 路径/字段。使用 ImageNet mean/std 与原学生预处理一致；HFlip、VFlip、Rot90 同步变换 image、mu、var、agreement、box_masks。K=1 hard baseline 可通过 resized `mu >= 0.5` 生成。
- `fecd/tools/benchmark_runtime.py`、`fecd/tools/validate_stage2b_smoke.py`：100 张实际运行测量、K1/K8 对照、Dataset/缓存验证与诊断图。
- `fecd/tests/test_box_ops.py`、`test_sam_ensemble.py`、`test_cache_schema.py`、`test_fecd_dataset.py`、`run_stage2b_tests.py`：固定 prompt、空框、多框融合、统计、缓存及空间变换测试。

本阶段没有增加 evidence 字段，也没有缓存 tri-state mask。

## Box 导出协议与结果

Boxes 保存于 `fecd/assets/boxes_train.json`，包含全部 **5,737** 个 train 身份、image SHA256/SHA1、native 尺寸、sample key、box hash、坐标及版本元数据。读取了 5,737/5,737 张 manifest 指定的 native prepared GT；导出失败数为 0。框数统计：1 框 5,069 张、2 框 457 张、3 框 211 张；无空 GT 样本。

导出复现 `train.py::mask_to_boxes()` 的 `>127` 二值阈值、`RETR_EXTERNAL` 轮廓、轮廓面积降序、最多保留 3 框、原 1%–10% 扩框比例及概率分布、整数扩张和边界裁剪。原函数虽然接收 `min_area=20`，实际没有过滤 contour；本实现也不应用此过滤。为确定性与断点顺序无关，每个样本用固定全局 seed 与稳定 sample key 派生独立随机种子。重复完整导出到独立文件，SHA256 均为 `2461ca0b6cb8a16496947773797c71954b61ccf7540ef1549e68afb69b38493c`。

本导出在 native GT 上计算；原训练代码在随机增强后的图像/mask 上导框，逐样本 box 未必相同。之后 B1/B5 必须共用本次导出文件，避免教师 box 来源不一致。

## SAM 与缓存闭环测试

SAM checkpoint：`/home/weights/sam_vit_h_4b8939.pth`，实际 SHA256 为 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`，设备 `cuda:0`（RTX 4090）。没有联网下载。

六张训练样本分别覆盖单 Box、多个 Box、小组件、边界 Box、多数据集及不同长宽比：CASIAv2 两张、Coverage 一张、Columbia 一张、NIST16 一张，另有 CASIAv2 单 Box 样本。两套 cache 均 **6/6 成功**、失败 0。每条 SAM decoder 输出 native logits `[1,H,W]`、quality `[1]`、low-resolution logits `[1,256,256]`；例如 NIST16 样本 logits 为 `[1,3024,4032]`。实现先将 native logits 做 sigmoid，再将 probability 统计图 bilinear resize 到 `[224,224]`；未插值 raw logits。K=1 var 为零符合总体方差定义；K=8 输出均有限，mu/agreement 在 `[0,1]`、var 在 `[0,0.25]`；六张样本 cache 中 K=8 实测 `var.max` 范围为 0.0083–0.1984，SAM quality mean 的六样本均值为 0.8495，quality std 为逐 cache 记录的教师 mask quality 离散度（不解释为像素准确率）。

K=8 首个 Base prompt 与独立 K=1 预测逐样本比较，经过 float16 cache 回读后的最大绝对误差为 **0.0002442**；两者使用完全相同的训练清单、SAM checkpoint、Base boxes 和学生输入预处理。K=1 Dataset target 已验证等于 `mu >= 0.5`。K=1/K=8 的同一样本学生图像 tensor 完全一致。

cache 示例目录：

- `fecd/assets/stage2b_smoke/cache/k1/e7a53a9439d8/`
- `fecd/assets/stage2b_smoke/cache/k8/e96554760a5e/`

K=1 与 K=8 配置哈希不同且缓存键不同；六条身份各自 key 唯一。真实 cache 经关闭后重新读取并完成字段、形状、dtype、有限值和范围校验。重复生成时已有 5/6 条 cache 被校验并复用；新单框样本正常补写。单元检查也确认同名 sample ID 在不同 dataset 会获得不同 cache key；错误 config 不接受旧缓存。使用两个独立进程同时写同一记录，结果为一个写入、另一个安全复用。损坏 NPZ 可在锁内安全重建。

空 Box 合成输入跳过 SAM decode，mu/var/agreement 全零，不会产生 0.5 前景。三组件 box 最大值融合有合成预测测试。cache 只保存 `mu/var/agreement` float16 `[224,224]`、box masks uint8 `[3,224,224]`、quality 汇总、native 尺寸、K 与 schema/config/source 身份字段；没有 GT 像素 mask 或虚构 evidence。

## Dataset、增强和可视化

学生 Dataset 的 allowlist 拒绝 GT/额外字段，返回中没有 GT、GT 路径或完整 mask。缓存 K1/K8 都可加载。boxes 被 pad 到 `[3,4]` 以便 DataLoader batch；`num_boxes` 单独返回。测试了 HFlip、VFlip、Rot90 对图像和所有空间 cache 图的同步变换，并测试最小/边缘 Box 扰动不会越界且宽高为正。

六张真实样本的六宫格诊断图（原图、box overlay、K=1 hard mask、K=8 mean、K=8 variance、K=8 threshold vote）位于：`fecd/assets/stage2b_smoke/visualizations/`。这些图只用于人工检查坐标和输出，不表示性能提升。

## 单元测试和实际命令

环境未安装 `pytest`，因此使用标准 Python 直接运行同一组测试函数的 harness；未为此安装额外包。测试命令：

```bash
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tests.run_stage2b_tests
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m compileall -q fecd
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.export_boxes --output fecd/assets/boxes_train.json
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.pseudo.generate_cache --k 1 --output-root fecd/assets/stage2b_smoke/cache --id Tp_D_CND_M_N_ani00018_sec00096_00138 --id Tp_D_CND_M_N_art00077_art00076_10290 --id Tp_D_CNN_M_N_art00052_arc00030_11853 --id 089 --id canong3_nikond70_sub_17 --id NC2016_5131
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.pseudo.generate_cache --k 8 --output-root fecd/assets/stage2b_smoke/cache --id Tp_D_CND_M_N_ani00018_sec00096_00138 --id Tp_D_CND_M_N_art00077_art00076_10290 --id Tp_D_CNN_M_N_art00052_arc00030_11853 --id 089 --id canong3_nikond70_sub_17 --id NC2016_5131
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.validate_stage2b_smoke
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.benchmark_runtime --limit 100 --output fecd/assets/stage2b_smoke/benchmark
```

K8 命令中的 CASIAv2 ID 实际正确值为 `Tp_D_CND_M_N_art00077_art00076_10290`；具体最终缓存与成功记录见对应 `manifest.json`。运行时报告存于 `fecd/assets/stage2b_smoke/benchmark/benchmark.json`。

## 100 张实际 SAM 性能测量

在 4090 上对训练 manifest 前 100 张图像逐一测量，平均组件框数 1.1。分项计时包含 SAM image encoder 的 `set_image` 与总 prompt decode；未计图像读取和日志写入。数字为实测，5,737 张耗时仅按测得均值线性估算：

| K | 单图均值 | encoder | decoder 总计 | 峰值 GPU 显存 | 平均压缩 cache | 5,737 张推理估算 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2256 s | 0.2194 s | 0.0063 s | 5.60 GiB | 42.5 KB | 0.360 h |
| 3 | 0.2310 s | 0.2181 s | 0.0129 s | 5.60 GiB | 87.5 KB | 0.368 h |
| 5 | 0.2404 s | 0.2191 s | 0.0213 s | 5.60 GiB | 90.8 KB | 0.383 h |
| 8 | 0.2545 s | 0.2205 s | 0.0340 s | 5.60 GiB | 93.3 KB | 0.406 h |

按 K=8 实测均值估算全训练集压缩缓存约 **0.50 GiB**。100 张是 manifest 顺序前缀，主要代表 CASIAv2；估时和空间大小用于资源规划，正式生成仍需监控 I/O 与失败重试。

## 数据隔离、风险和保护检查

仅使用现有 train manifest。NIST16 原有 source-group 审计风险保持不变：`world/NC2016_0359.jpg` 同源组有 train `NC2016_5290`/`NC2016_7882` 和 test `NC2016_0796`/`NC2016_9257`。本阶段没有读取或使用测试集图像、GT、参数或结果；未改变 split，也没有擅自做 source-disjoint 重划分。后续正式研究需继续标注该风险。

没有修改 `train.py`、`eval.py`、`boxpromptiml.py`、原始 `utils/` 或原始 checkpoint；没有执行训练。工作区原有修改均保留。

## 验收结论与边界

阶段 2B 验收项在小规模真实样本和 100 张实测范围内全部通过。K=1/K=8 cache、Box 变换、同步 Dataset、原子缓存与无 GT Dataset 已可用于下一步准备。

**阶段 2C 具备开始正式全量缓存的技术条件，但本次没有启动。** 依照任务要求，停在 2B 等待用户验收；全量 5,737 张缓存和学生训练仍未执行。
