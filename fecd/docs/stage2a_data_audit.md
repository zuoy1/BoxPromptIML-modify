# FECD 阶段 2A：正式数据可读性与隔离审计

审计日期：2026-10-23。原始正式 split 与成员关系未改动；只读核验并通过 `fecd/configs/stage2a_path_mapping.json` 映射历史根路径。

## 数据来源和协议边界

- `/home/datasets/processed/boxprompt_paper_scale_20261016/README.md` 与原有 manifest 记录本地重建的 5,737 train / 230 paper-scale test，身份清单不是作者 414/150 NIST16 列表的精确复刻。
- CASIAv1 `C1` 是 2026-10-19 单独添加的 920 张锁定测试集，不包含在根 `train_manifest.csv`/`test_manifest.csv` 的 5,737/230 历史统计中；本报告将其独立计数。
- 原始预处理 manifest 存有 dataset、id、原图/GT 来源路径、SHA256、尺寸、group 和处理后路径。本次按 `(dataset,id)` 连接 CSV 与 JSON manifest，不按排序位置配对；图片只允许将 manifest 声明路径中的 `/hy-tmp/datasets` 前缀映射到 `/home/datasets`，并核验记录 SHA256。处理后 GT 读取 manifest 声明的 `prepared_mask` 路径并逐像素检查二值性/尺寸。C1 使用独立 manifest 中的 raw image 与 source GT 路径，并逐张核对输出 GT 是否等于灰度 `>127` 的二值结果。

## 可读数量和检查结果

| Split / dataset | Manifest 数量 | 图像可读 | Source GT 可读 | Prepared GT 可读且二值 | 图像/GT 尺寸匹配 | 缺失/错误 |
|---|---:|---:|---:|---:|---:|---:|
| Train / CASIAv2 | 5,123 | 5,123 | 5,123 | 5,123 | 5,123 | 0 |
| Train / NIST16 | 414 | 414 | 414 | 414 | 414 | 0 |
| Train / Columbia | 130 | 130 | 130 | 130 | 130 | 0 |
| Train / Coverage | 70 | 70 | 70 | 70 | 70 | 0 |
| **Train total** | **5,737** | **5,737** | **5,737** | **5,737** | **5,737** | **0** |
| Test / NIST16 | 150 | 150 | 150 | 150 | 150 | 0 |
| Test / Columbia | 50 | 50 | 50 | 50 | 50 | 0 |
| Test / Coverage | 30 | 30 | 30 | 30 | 30 | 0 |
| **Historical test total** | **230** | **230** | **230** | **230** | **230** | **0** |
| Locked test / CASIAv1 C1 | 920 | 920 | 920 | 920 | 920 | 0 |

所有 5,967 个主 split 记录的原始图像和原始 GT 文件均真实读取并核对源 SHA；所有 prepared GT 均可解码，值域限于 `{0,255}`。另外 920 个 C1 图像、源 GT、输出 GT 全量解码；输出 mask 与 manifest 声明的阈值转换逐像素相等。每个 split 图像/GT 目录的文件名成员集合也和 manifest 完全一致（train 各 5,737、主 test 各 230、C1 各 920，无缺失、无额外文件）。实际核验错误数为 0。

## 已修复路径与尚未修复路径

已通过新配置和验证器建立只读映射：`/hy-tmp/datasets/...` → `/home/datasets/...`。5,737 train、230 主 test 和 920 C1 test 的图像/GT 都可根据既有 manifest 身份在当前数据盘读出并完成校验；没有按文件名猜 raw 资源。

原数据树中的历史软链接仍然悬空，未替换或覆盖。后续工具必须显式应用该 mapping，或使用单独生成并验证的路径清单；直接把旧 split 目录交给 loader 仍不可用。C1 `Gt` 目录本身可读，C1 `Tp` 的旧软链接不可读，但其 manifest 指向的原始图像可读且 SHA 匹配。

## 重复样本和 NIST16 source group

- 主 train/test 之间：精确文件 SHA-256 重叠 0；manifest 的解码像素 SHA-256 重叠 0。
- C1 test 与 train：精确文件 SHA-256 重叠 0；解码像素 SHA-256 重叠 0。
- 全部 split 内另有 NIST16 重复项：2 组完全相同图像文件（train: `NC2016_8067`/`NC2016_5809`；test: `NC2016_9790`/`NC2016_6889`），4 组相同解码像素（另有 train: `NC2016_0484`/`NC2016_7504`；test: `NC2016_1572`/`NC2016_5937`）。它们分别位于同一个 split、同一个 base group 内，没有跨 train/test。重复样本会造成该 NIST source 在样本级训练或评估中重复计权，后续统计应按既定协议说明。
- 主 split 的 NIST16 有 414 train、150 test。没有相同图像文件或像素，但有一个 `group=world/NC2016_0359.jpg` 跨 split：训练 `NC2016_5290`、`NC2016_7882`；测试 `NC2016_0796`、`NC2016_9257`。这是 source/base 级相关性风险，不是完全重复图像。
- 本地作者 414/150 样本身份列表不可用，当前协议是本地重建协议；不能称作作者身份协议复现。

建议保留此 split 作为与既有 BoxPromptIML 结果对照的锁定版本，并报告该同源组风险。若需要严格 source-disjoint 研究结果，应另行生成明确命名、版本化的新 manifest，按 NIST16 base/source group 整组切分；不得覆盖当前 split，也不得混称两种协议。

## GT 隔离

本阶段只读访问 GT 用于审计和确认离线 box 来源。没有学生 Dataset、训练 loop 或 confidence/loss 实现；没有启动训练。新 path mapping 明确标记为 `contains_ground_truth_paths=true`、`do_not_use_as_student_dataset=true`。阶段 2B 的学生 Dataset 必须只拿 train-only image/cache 索引，不能读取本审计映射或任何 test GT。

## 可复现检查

```bash
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_data_manifest
```

真实输出摘要：`status=passed`，train `5737/5737`、main test `230/230`、C1 test `920/920`；duplicate image/pixel overlap 为 0；NIST16 跨 split source group 为 1；`error_count=0`。机器可读完整结果位于 `fecd/assets/stage2a_data_audit.json`。
