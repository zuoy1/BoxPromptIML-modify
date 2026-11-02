# FECD 阶段 1：原仓库审计

审计日期：2026-10-23  
项目目录：`/home/projects/BoxPromtIML`  
范围：只读审计；本阶段唯一新增文件为本文档。未开始 FECD 实现或完整训练。

## 1. 审计结论

原始 BoxPromptIML 主体是 `boxpromptiml.py` 中的 TinyViT 学生网络；`train.py` 在训练循环内读取 GT、从 GT 提取最多三个外接框，再逐框调用冻结 SAM，最后以 SAM 二值 mask 通过 `structure_loss` 监督学生。训练 loader 还会在取框前对图像和 GT 同步做随机增强。FECD 应单独实现数据协议与训练入口，不能直接复用这段会读取 GT 的训练循环。

代码、模型权重文件和 GPU 硬件曾有实验配置，但当前容器不能按仓库现有路径直接运行正式流程：Python 环境缺少 `timm` 等包；`/hy-tmp` 当前为空目录，正式划分中的大多数图像/GT软链接目标及仓库内 SAM checkpoint 链接均无法解析。可见的替代权重位于 `/home/weights`，不能据此假设仓库默认路径有效。

## 2. 真实项目结构与工作区状态

核心跟踪文件：

- `train.py`：原始学生训练、GT 到 box 转换、SAM 教师调用、structure loss。
- `eval.py`：学生推理和数据集评估；当前工作区版本有本地修改。
- `boxpromptiml.py`：TinyViT backbone、MGFM 门控融合、ExternalAttention 和输出头。
- `utils/data_val.py`：训练/测试数据集、图像与 GT 同步变换。
- `utils/utils.py`：梯度裁剪、学习率调整等。
- `TinyViT/`：仓库内 TinyViT 实现。
- `segment-anything/`：目录内仅有 SAM checkpoint 软链接。

`git status --short` 在审计前显示 `train.py`、`eval.py` 和两个 `utils/__pycache__` 文件已修改，`TinyViT/`、`segment-anything/`、`__pycache__/` 未跟踪。没有恢复或修改这些既有内容。当前提交为 `2b5b7c8 Update README.md`。未发现 `AGENTS.md`。

## 3. 模型接口

`BoxPromptIML()` 不接收 box 或 GT，输入为归一化 RGB tensor `[B,3,H,W]`，输出为单通道 logits `[B,1,H,W]`，空间尺寸插值回输入大小。网络使用 `tiny_vit_5m_224(pretrained=True)`，TinyViT 有 5 组中间特征，经投影、门控融合、外部注意力后输出 mask logits。训练图像尺寸默认 224×224。

仓库存在 TinyViT checkpoint `/home/weights/tiny_vit_5m_22kto1k_distill.pth`，但构造函数的 `pretrained=True` 是否能在本环境自动命中该文件，尚未验证；实际 import 当前先被缺少 `timm` 阻断。

## 4. GT 到 coarse box 与 SAM 协议

`train.py::mask_to_boxes(mask, min_area=20)` 对阈值化 GT 调用 `cv2.findContours(RETR_EXTERNAL)`，按轮廓面积降序排列外接矩形，并随机扩大每个框 1%–10%。参数 `min_area` 实际没有过滤轮廓。训练循环最多使用前三个框；空 mask 生成全零教师 mask。框随机性来自 Python `random`，而审计到的循环未显式设置统一随机种子。

每个训练样本通过 `SamPredictor.set_image(image_np)` 设置图像，再对框逐个调用：`multimask_output=False, return_logits=False`。因此现有教师伪标签是二值 mask，多个 box 的结果以像素最大值并集组合。同一图像的 embedding 可在多个 box 预测间复用。

本地 `segment-anything-main/segment_anything/predictor.py` 确认 `predict` 支持 `return_logits=True`，返回的主 mask 会保留未阈值 logits；SAM Python 源码可在加入该源码路径后导入。训练代码默认 checkpoint 路径 `./segment-anything/sam_vit_h_4b8939.pth` 是指向 `/hy-tmp/weights/...` 的软链接，目前目标不可见；实际权重文件存在于 `/home/weights/sam_vit_h_4b8939.pth`。

## 5. 数据协议、增强和隔离现状

原训练入口默认 `train_root=/home/xidongdong/Dataset/mai_Train/`，实际拼接 `Tp/`、`Gt/`、`Edge/`。`get_loader` 的 `PolypObjDataset` 按排序后的文件名列表配对 image/GT，并只检查尺寸相同；没有通过文件名 stem 显式配对，也没有读取或使用 Edge。训练 dataset 同时返回图像、GT tensor、NumPy 图像和 NumPy GT。FECD 不能沿用此 dataset 接口，否则 GT 会进入训练 loop。

训练变换由 Albumentations 同步作用于图像和 GT：`RandomScale`、随机 copy-move、随机 inpainting、水平/垂直翻转、亮度对比度、JPEG 压缩、90 度旋转和模糊；此外还 resize 到 224。SAM 当前在增强后的 NumPy 图像和增强后的 GT 框上运行。因此离线缓存必须明确选定缓存图像坐标系，并保证增强时图像、box、SAM 缓存和证据图严格同步；直接使用原图缓存后再套原始随机增强会坐标/内容不一致。

已发现的正式 split 清单位于 `/home/datasets/processed/boxprompt_paper_scale_20261016/`，报告记录训练 5,737、论文规模 IND 测试 230，并补入独立 CASIAv1 `C1` 测试 920 张。该 split 报告特别说明 NIST16 本地重建划分与作者身份清单不等价，且一个 base 同源组跨 train/test；后续应将此列为现有限制，并将测试集保持锁定。train 和主要 test 目录中的图像/GT软链接目标都指向 `/hy-tmp/datasets/...`，当前无法解析；例外是 `C1/Gt` 的 920 个文件可直接读取。需恢复数据位置或从 raw 数据重建可读 split 后才能运行正式训练/评估。`/home/datasets/raw` 有原始数据目录；smoke 数据 `/home/datasets/boxprompt-smoke/{Tp,Gt,Edge}` 存在少量文件。

当前 `eval.py` 工作区版本将 sigmoid map 逐图 min-max 后再保存，因此不符合 FECD 阶段 7 所要求的 raw sigmoid probability 严格评估；FECD 将使用新评估入口，不应沿用其概率后处理。原始评估 pairing 也依赖数据集名称推断 GT 文件名。

## 6. 损失、优化和 checkpoint

原训练 loss 是 `structure_loss`：以 31×31 平均池化差产生边界权重，对教师二值 mask 做 weighted BCE 与 weighted IoU。另定义 `weighted_cross_entropy_loss`，但训练调用的是 `structure_loss`。优化器 AdamW，默认学习率 `1e-4`、weight decay `1e-4`、梯度逐参数裁剪阈值 `0.5`。默认 `epoch=21`，主循环 `range(1, epoch)`，实际 20 个 epoch；学习率按 epoch 10 衰减十倍。

可见已有 checkpoint：`/home/runs/boxprompt_formal_20261016_122447/checkpoints/checkpoint-{1..20}.pth`，另有 smoke checkpoint。`eval.py` 当前硬编码 checkpoint `/hy-tmp/runs/...`，此路径在当前容器不可见，但对应 checkpoint 在 `/home/runs/...` 可见。它们是既有 BoxPromptIML 产物，不是 FECD 结果。

## 7. 依赖与运行条件

审计命令确认 Python 3.11.8、PyTorch 2.2.1+cu121、CUDA 可用，设备为 NVIDIA GeForce RTX 4090；`torchvision`、OpenCV、NumPy、PIL 可导入。项目声明的 `BoxPromptIML.yml` 固定 Python 3.10，但不是完整 pip 依赖清单。当前缺少 `timm`、`albumentations`、`tensorboardX`、`scikit-learn`、`thop`。`segment_anything` 在临时加入 `/home/projects/segment-anything-main` 后可导入；`from boxpromptiml import BoxPromptIML` 因缺少 `timm` 失败。因此没有运行 forward smoke test，也没有启动训练。GPU 可用不代表现有代码和数据路径可运行。

## 8. 后续阶段文件规划（本阶段不创建）

所有新增实现置于 `fecd/`，实验输出置于独立 `/home/runs/fecd_*` 目录。预期按阶段逐步创建，不提前实现：

| 阶段 | 规划文件 | 职责 |
|---|---|---|
| 2 | `fecd/tools/export_boxes.py`、`fecd/pseudo/box_ops.py`、`fecd/pseudo/sam_ensemble.py`、`fecd/pseudo/cache_schema.py`、`fecd/pseudo/generate_cache.py`、`fecd/datasets/fecd_dataset.py`、`fecd/datasets/transforms.py`、`fecd/tests/` | 从既定训练 split 离线导出框；K=1/K=8 SAM 缓存及校验；缓存/图像/框同步增强；不把 GT 接入学生训练 dataset。 |
| 3 | `fecd/evidence/extractors.py`、`fecd/evidence/normalize.py`、`fecd/tools/fit_evidence_stats.py`、`fecd/tools/audit_pseudo.py`、`fecd/diagnostics/` | HP/noise/JPEG/FFT 局部证据、仅训练集统计量、锁定诊断集伪标签错误分析、Risk-Coverage 与案例可视化。 |
| 4 | `fecd/confidence/rules.py`、`fecd/losses/fecd_loss.py`、`fecd/train_fecd.py`、`fecd/configs/fecd_rule.yaml`、`fecd/tests/` | Rule-FECD confidence、互斥 Tri-State、soft-target pixel loss、弱 Box loss、独立无 GT 训练入口；consistency 关闭。 |
| 5 | `fecd/configs/`、`fecd/scripts/`、`fecd/eval_fecd.py`、`fecd/metrics.py` | B1–B5 统一配置、seed=0 受控训练、raw probability 评估和独立产物记录。 |
| 6 | `fecd/confidence/head.py`、`fecd/synthetic/`、`fecd/train_confidence.py`、对应 configs/scripts | 仅在阶段 5 达到继续条件后实现 B6/B7。 |
| 7 | `fecd/tools/bootstrap_metrics.py`、`fecd/tools/benchmark_runtime.py`、实验配置与报告 | 多 seed、IND/OOD、配对 bootstrap、失败案例和效率汇总。 |

建议共享基础文件：`fecd/__init__.py`、`fecd/configs/`、`fecd/metrics.py`、`fecd/docs/`。任何需要 GT 的 box 导出、诊断和评估入口应与无 GT 学生训练入口分开，配置也不得向学生训练暴露 `gt_root`。

## 9. 实际执行命令与结果

- `sed`/标题索引检查 `/home/deep-research-report.md`：核对研究问题、推荐文件结构、缓存与增强约束、取证证据和规则置信度、三态损失、实验对照、严格评估与 Go/No-Go 条件。报告强调原始 `train.py` 已硬编码 GT→box→SAM→学生监督，要求旁路新增代码；具体实现仍以本仓库接口为准。
- `find`、`sed`、`git status --short`、`git diff`：完成目录、核心代码、评估逻辑和工作区状态检查。
- Python 依赖导入检查：PyTorch/CUDA 正常，SAM 源码路径导入成功；`BoxPromptIML` import 因 `ModuleNotFoundError: timm` 失败。
- 数据路径检查：正式 split 的统计/manifest 存在，但图片软链接目标指向当前不可见的 `/hy-tmp`。
- forward smoke test：未运行（模型 import 被缺少依赖阻断）。
- 完整训练：未运行。

## 10. 阶段 1 验收

**结果：PASS（阶段 1 审计验收）；后续运行条件：BLOCKED。** 已完成代码结构、模型接口、GT-to-box/SAM、loss、增强、评估、checkpoint、数据 split 和依赖审计，并制定后续文件规划。阶段 1 不要求启动训练，smoke test 因依赖缺失而未运行。开始阶段 2 的缓存生成/测试前，需要恢复可读数据与 SAM checkpoint 路径并提供项目依赖。未执行阶段 2。
