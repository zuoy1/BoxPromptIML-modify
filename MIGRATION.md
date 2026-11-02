# BoxPromptIML / FECD 科研项目迁移说明（A 服务器 → B 平台）

- 生成/复核日期：2026-11-02 14:45 CST（A 服务器）
- 项目根目录：`/home/projects/BoxPromtIML`
- 迁移资料目录：`/home/projects/BoxPromtIML/migration`
- 目标读者：B 平台上的 Codex / 运维人员
- 本次工作性质：**只读盘点 + 更新迁移资料**。没有修改核心代码，没有删除数据，没有执行 Git 提交/推送，没有安装依赖，没有实际迁移数据或模型，也没有中断任何训练任务。
- 与 2026-10-27 版本相比，代码主体和 Git HEAD 未变；**FECD 研究已从阶段 2C 推进到阶段 5A**，`fecd/assets` 从约 660M 增长到 6.43GB，并新增 `/hy-tmp/BoxPromtIML/fecd_runs` 约 2.27GB。旧版迁移清单中的“阶段 3 尚未实现”等描述已经过时，以本文件为准。

> 阅读顺序：先看 §0 快速结论，再按 §7 恢复；§8、§9 是不能想当然的兼容性和待确认项。

---

## 0. 快速结论

1. **代码状态没有变化**：Git 仍是 `main` @ `2b5b7c8baeb8b54d958a8ded022c1d34c420f257`，与 `origin/main` 一致；`eval.py` 仍是 +142/−51，`train.py` 仍是 1 行改动。未执行 `git add/commit/push`。变化集中在未跟踪的 `fecd/` 研究代码和资产。
2. **数据本体没有变化**：原始数据在 `/home/datasets/raw`，划分/掩码/清单在 `/home/datasets/processed`。`/home/datasets` 下仍有 30,787 个符号链接；当前 `/hy-tmp/datasets` 不存在，所以链接在 A 上显示为断链，但把 `/hy-tmp/datasets` 映射到 `/home/datasets` 后 **30,787 个链接全部可解析（本次逐跳复核 0 失败）**；`train_manifest.csv` 的 5,737 条与 `test_manifest.csv` 的 230 条，`image/mask` 也全部可解析。
3. **FECD 资产大幅增长**：`fecd/assets` 现在 **6,432,406,786 B（约 6.0 GiB）**，其中阶段 3 的 GT-free 证据缓存 4.15GB、阶段 4F1 正式运行的完整 checkpoint 不在项目内而在 `/hy-tmp/BoxPromtIML/fecd_runs`（1.08GB），阶段 5A 诊断 1.19GB。
4. **迁移总量约 21.1GB**（不含 conda 环境）：`/home/projects/BoxPromtIML` 6.45GB + `/home/datasets` 9.16GB + `/home/weights` 2.59GB + `/home/runs` 0.60GB + `/hy-tmp/BoxPromtIML` 2.27GB + 其他小文件。A 的系统盘 `/` 只剩 8.9GB，**不能把全部迁移包放在 `/home`**；可放 `/hy-tmp`（当前 48GB 可用，但它是数据盘/可能是临时盘）或外部对象存储。
5. **环境本身没有变化**：conda `boxpromptiml`，Python 3.10.22，torch 2.2.1+cu121，torchvision 0.17.1+cu121，CUDA runtime 12.1，`pip check` 通过。当前 `nvidia-smi` 仍报 “No devices were found”，`torch.cuda.is_available() == False`；`/proc/driver/nvidia` 能看到的硬件是 RTX 4090、驱动 535.230.02。**不能把 A 的 GPU 可见性当作 B 的依据。**
6. **当前实验状态**：原始 BoxPromptIML 正式训练（20 epoch、5,737 张）与旧 eval 结果仍在；FECD 阶段 1–2C、3、3R、3R-A、3R-B0、4P、4F0、4F1、5A 都已有产物和报告。
   - 阶段 3：**NO-GO**；阶段 3R-A：**STOP**；阶段 3R-B0：**STOP**。
   - 阶段 4F1：P0/P1/P2F/P3F 四组各完成 20 epoch、17,660 次更新；原始执行器最终 `status.json` 为 **FAILED**（`check_stage4f1.py` 的 float32 loss 分解断言误报），独立 4F1-V 复验为 **PASS（engineering）**，科研结论 **NO CLEAR GAIN**。
   - 阶段 5A：CPU 只读诊断 **PASS**，科研结论 **NO CLEAR GAIN**，交付后停止；没有执行报告建议的下一轮实验。
7. **当前没有训练/评估任务在运行**：`ps` 只有 `jupyter-lab`；`tmux` 无会话；本次工作没有中断任何任务。

---

## 1. 项目概况与目录结构

### 1.1 项目是什么

- 原论文：AAAI 2026 **BoxPromptIML**（弱监督图像篡改定位，基于粗粒度 box + SAM 伪标签蒸馏）。
- 原仓库：`https://github.com/vpsg-research/BoxPromtIML`（公开仓库，远端名 `origin`）。
- 当前研究：个人新增 **FECD — Forensic Evidence-Calibrated Distillation**（篡改证据感知的可信伪标签蒸馏），代码全部在 `fecd/`。原始 `train.py` / `eval.py` / `boxpromptiml.py` 的训练逻辑没有被 FECD 改写；`eval.py`、`train.py` 仅保留原有的本地修改（见 §2）。

### 1.2 目录树（迁移相关）

```text
/home                                  # overlay 系统盘 30G，已用 22G，可用 8.9G
├── projects/
│   ├── BoxPromtIML/                   # 6.45GB（含 .git 5.0M、fecd 6.43GB）
│   │   ├── train.py                   # 原始训练入口（本地改 1 行）
│   │   ├── eval.py                    # 原始评估入口（本地改 142/51 行）
│   │   ├── boxpromptiml.py            # 学生网络定义（未改）
│   │   ├── utils/                     # data_val.py、tp_au.py、utils.py
│   │   ├── TinyViT/                   # vendored TinyViT backbone（未跟踪，约 0.97M）
│   │   ├── Figures/ README.md LICENSE BoxPromptIML.yml
│   │   ├── segment-anything/
│   │   │   └── sam_vit_h_4b8939.pth -> /hy-tmp/weights/sam_vit_h_4b8939.pth
│   │   ├── fecd/                      # 个人 FECD 代码、文档、测试、assets
│   │   │   ├── assets/                # 6.43GB：见 §4、§6
│   │   │   ├── diagnostics/ evidence/ forensic_reference/ confidence/
│   │   │   ├── training/ evaluation/ execution/ pseudo/ datasets/
│   │   │   ├── tools/ tests/ configs/ docs/
│   │   │   ├── coordinates.py model_loading.py
│   │   │   └── ...
│   │   └── migration/                 # 本迁移资料（约 5.0MB）
│   ├── segment-anything-main/         # SAM 源码（22M，可选 pip 安装来源）
│   └── TinyViT-main.zip               # 642K，可选/冗余
├── datasets/                          # 9.16GB（见 §4）
│   ├── raw/                           # 9.09GB：CASIAv2、NIST16、Columbia、Coverage、CASIAv1
│   ├── processed/                     # 69.6MB + 30,787 个软链
│   ├── patches/ scripts/ boxprompt-smoke/ boxprompt-casia16/
├── weights/                           # 2.59GB：SAM ViT-H + TinyViT
├── runs/                              # 597.7MB：原始正式训练、旧 eval、smoke、NIST16 诊断
├── checkpoints/                       # 1.5K：旧 smoke 日志（没有正式 checkpoint）
├── note.md                            # FECD 总任务说明（重要）
├── deep-research-report.md            # FECD 研究设计参考
└── conversation_20261026.md           # 早期迁移/空间排查对话记录

/hy-tmp                                # /dev/nvme0n1，50G xfs；当前用 2.2G，可用 48G
└── BoxPromtIML/
    └── fecd_runs/
        ├── stage4f1/formal20_seed20261029_noshutdown_20261029a/   # 1.08GB
        └── stage5a/diagnosis_20261030/                            # 1.19GB
```

### 1.3 训练时的路径与当前路径

原始正式训练 `runs/boxprompt_formal_20261016_122447/console.log` 显示，训练时项目位于 `/hy-tmp/projects/BoxPromtIML`。后来数据盘被清空，项目/权重/数据恢复到了 `/home`。因此：

- **当前真实位置**：`/home/projects/BoxPromtIML`、`/home/datasets`、`/home/weights`、`/home/runs`。
- **历史代码/日志中的位置**：`/hy-tmp/projects/BoxPromtIML`、`/hy-tmp/datasets`、`/hy-tmp/weights`、`/hy-tmp/runs`。
- **新增 FECD 运行产物的真实位置**：`/hy-tmp/BoxPromtIML/fecd_runs`，不在 `/home`。
- 大量代码仍硬编码 `/hy-tmp`：`eval.py:229`（checkpoint）、`eval.py:243`（test root）、`eval.py:253`（输出）、`fecd/diagnostics/stage5a_io.py:16-17`、`fecd/configs/stage4f1/storage.json`、`train_manifest.csv`/`test_manifest.csv` 的 `image/mask` 列、`segment-anything` 软链、torch hub 的 TinyViT 缓存软链。B 上必须同时提供 `/home/...` 和 `/hy-tmp/...` 两套可解析路径（§7.2）。

---

## 2. 当前代码状态与个人修改

### 2.1 Git 状态（采集/复核于 2026-11-02）

- 分支：`main`
- HEAD：`2b5b7c8baeb8b54d958a8ded022c1d34c420f257`（`Update README.md`，与 `origin/main` 一致）
- 远端：`https://github.com/vpsg-research/BoxPromtIML.git`
- 历史只有 3 个提交：`2b5b7c8`、`aea2b40`、`5a29822`
- **未执行** `git add` / `git commit` / `git push`。
- `migration/BoxPromtIML-git.bundle` 仍是完整的 3 个提交 bundle，`git bundle verify` 通过；`migration/code-local-changes.patch` 与当前工作树逐字一致。

```text
 M eval.py
 M train.py
 M utils/__pycache__/data_val.cpython-310.pyc   # 运行时字节码，可忽略
 M utils/__pycache__/utils.cpython-310.pyc       # 运行时字节码，可忽略
?? MIGRATION.md
?? TinyViT/
?? __pycache__/
?? fecd/
?? migration/
?? segment-anything/
```

### 2.2 已跟踪文件的个人修改

#### `eval.py`（+142 / −51）

这是主要的个人修改，目的：让评估更严格、适配当前本地重建 split、支持 checkpoint/输出路径。

- GT 路径配对改为 `os.path.join` + `Path(file).stem`，避免字符串截断；支持 NC16 / C1 / Coverage / Columbia / Imd / Korus / Coco / In-the-Wild / DSO 的文件名规则。
- `process_single_image`：统一灰度读取；预测/GT 尺寸不一致时**直接报错**，不再静默 resize。
- 额外保存连续概率图：`scores/<dataset>/`；AUC 用连续分数计算，F1/IoU 仍用 0.5 阈值二值图。
- `evaluate`：只统计实际文件；有效预测数不等于输入数时直接报错，拒绝部分指标；`Pool(2)` 降低 NIST 大掩码的并发内存。
- 新增 `validate_dataset_layout`：推理前检查每张图/GT 存在、尺寸一致、GT 只含 `{0,255}`、目录文件集合与数量精确匹配。
- 主入口：GPU 不可用时回退 CPU；checkpoint 改为 `/hy-tmp/runs/boxprompt_formal_20261016_122447/checkpoints/checkpoint-20.pth`；测试集改为 `NC16, C1, Columbia, Coverage` 四个 IND 数据集；输出改到 `/hy-tmp/runs/boxprompt_eval_<timestamp>/`；每个数据集推理前 `model.ex_attn.memory_bank = None`；最后写 `f1_in` 汇总。
- 这些路径是 A 服务器历史布局；B 上要么建立 `/hy-tmp` 别名，要么改回 `/home` 路径。

#### `train.py`（1 行）

```diff
-        if  epoch >= 10:
+        if epoch >= 1:
             torch.save(model.state_dict(), save_path + 'checkpoint-{}.pth'.format(epoch))
```

作用：从第 1 个 epoch 开始每个 epoch 都保存 checkpoint（原仓库只从第 10 epoch 开始）。这也是 `/home/runs/boxprompt_formal_20261016_122447/checkpoints/checkpoint-{1..20}.pth` 齐全的原因。

#### `utils/__pycache__/*.pyc`

只是运行 `train.py`/`eval.py` 时 Python 自动重写的字节码，不是有意义的源码修改；建议迁移时忽略或在 B 上删除后自然重建。`migration/code-local-changes.patch` 已排除这些二进制文件。

### 2.3 未跟踪但必须迁移的内容

| 路径 | 大小 | 说明 |
|---|---:|---|
| `TinyViT/` | 970,721 B | vendored TinyViT 源码，`boxpromptiml.py` 直接 import；必须迁移 |
| `fecd/` | 6,434,213,674 B | 个人 FECD 研究代码 + docs + tests + assets；**代码必须迁移，assets 按 §6 分批** |
| `segment-anything/` | 0 | 目录里只有一个指向 `/hy-tmp/weights/sam_vit_h_4b8939.pth` 的软链；权重在 `/home/weights` |
| `migration/` | 5,064,333 B | 本迁移资料；建议随项目一起传 |
| `__pycache__/` | 约 44K | 可忽略/重建 |
| `MIGRATION.md` | 约 80K | 本文件 |

### 2.4 FECD 代码结构与阶段状态

`fecd/` 是当前科研创新的核心。代码目录（按职责）：

```text
fecd/
├── coordinates.py / model_loading.py
├── confidence/          # c_sam 等可靠性启发式
├── evidence/            # Noise/JPEG/FFT 取证证据与归一化
├── forensic_reference/  # 3R-B0 参考区域条件化
├── diagnostics/         # stage3、3R-A、5A 诊断、风险-覆盖、bootstrap
├── training/            # 4P/4F0/4F1 受控训练、数据、loss、checkpoint、storage
├── evaluation/          # 严格 new_dev 评估、allowlist、指标
├── execution/           # 4F1 执行器、锁、原子状态、关机证据
├── pseudo/              # SAM 多提示、K1/K8 cache
├── datasets/            # FECD Dataset（不读 GT）
├── tools/               # 各阶段入口/验收/导出
├── tests/               # 197 个单元/契约测试（CPU 可跑）
├── configs/             # stage2a/4p/4f0/4f1 冻结配置
├── docs/                # 阶段报告（23 个 md）
└── assets/              # 6.43GB 二进制/中间产物
```

阶段状态（以 `fecd/docs/` 的报告为准）：

- 阶段 1：原仓库审计，`stage1_audit.md`。
- 阶段 2A：坐标协议、数据隔离审计、环境 smoke，`stage2a_*.md`。
- 阶段 2B：多提示 SAM 教师、K=1/K=8 离线缓存 schema、Dataset/几何同步、测试，`stage2b_report.md`。
- 阶段 2C：全量 cache（5,737 张 × K=1/K=8）、审计、统计、预检，`stage2c_report.md`。
- 阶段 3：取证证据 + 教师可靠性诊断；工程完成，**研究判定 NO-GO**，`stage3_report.md`、`stage3_data_protocol.md`。
- 阶段 3R：NO-GO 失败复盘与研究路线，`stage3r_failure_analysis.md`、`stage3r_research_options.md`。
- 阶段 3R-A：路线 A 最小教师开发诊断；候选 q_A **STOP / INSUFFICIENT EVIDENCE**，`stage3ra_report.md`。
- 阶段 3R-B0：参考区域条件化机制预检；**STOP**，`stage3rb0_report.md`、`stage3rb0_interpretation_audit.md`。
- 阶段 4P：四组 8-update 受控学生蒸馏 smoke；工程 **PASS**，没有效果结论，`stage4p_report.md`。
- 阶段 4F0：协议冻结 + 四组各 1 个完整 epoch + 530 张 new_dev 严格评估；工程 **PASS**，`stage4f0_report.md`。
- 阶段 4F1：四组各 20 epoch 正式训练 + 530 张 new_dev 评估；原始验收 **FAILED**，独立 4F1-V 复验 **PASS（engineering）**，科研结论 **NO CLEAR GAIN**，`stage4f1_training_report_20261030.md`、`stage4f1_revalidation_report.md`。
- 阶段 5A：学生失败机制诊断；工程 **PASS**，科研结论 **NO CLEAR GAIN**，交付后停止，`stage5a_failure_diagnosis.md`。

**当前没有 FECD 训练进程**；阶段 5A 没有训练、没有优化器更新、没有新损失/网络/多 seed。

### 2.5 测试与代码快照

- 2026-11-02 实际运行：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s fecd/tests -p 'test_*.py'` → **197 tests OK**（21.7s，CPU）。
- `python -m fecd.tests.run_stage2b_tests` → **9 checks passed**（与旧版一致）。
- `migration/BoxPromtIML-git.bundle`：完整 3 个提交的 git bundle（5.0M），`git bundle verify` 通过。
- `migration/code-local-changes.patch`：`eval.py` + `train.py` 的文本 diff（14.6K），与当前工作树一致。
- `migration/git-state.txt`：HEAD、status、log、未跟踪目录大小。
- **未跟踪的 `fecd/`、`TinyViT/` 不在 bundle/patch 中**，必须从项目目录整体拷贝，或单独打包。

---

## 3. 运行环境

### 3.1 系统与硬件（A 服务器，2026-11-02 复核）

| 项目 | 值 |
|---|---|
| OS | Ubuntu 20.04.6 LTS (Focal Fossa)，容器（`/.dockerenv`） |
| Kernel | `5.4.0-204-generic` x86_64 |
| CPU | 64 vCPU，AMD EPYC 7542 32-Core |
| RAM | 251 GiB，无 Swap（本次 used 15Gi，available 232Gi） |
| 系统盘 | overlay 30G，已用 22G，可用 8.9G（71%） |
| 数据盘 | `/dev/nvme0n1` 50G xfs，挂载 `/hy-tmp`，当前用 2.2G，可用 48G |
| GPU 硬件 | NVIDIA GeForce RTX 4090，UUID `GPU-530e734d-...`，PCI `0000:c2:00.0`（`/proc/driver/nvidia`） |
| NVIDIA 驱动 | 535.230.02 |
| nvcc | CUDA 12.1，V12.1.105 |
| **当前 GPU 可用性** | **`nvidia-smi`: “No devices were found”；`torch.cuda.is_available() == False`**。原因未确认；历史训练使用过 RTX 4090 / `cuda:0`，但当前容器不能作为 B 的依据。 |

### 3.2 Python / Conda / PyTorch（已确认，2026-11-02 复核未变）

| 项目 | 值 |
|---|---|
| Conda | 24.1.2，base `/usr/local/miniconda3` |
| 环境名 | `boxpromptiml` |
| 环境路径 | `/usr/local/miniconda3/envs/boxpromptiml` |
| 环境大小 | 5.5G（5,761,765,260 B） |
| Python | 3.10.22 |
| PyTorch | 2.2.1+cu121 |
| torchvision | 0.17.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| cuDNN | 8.9.2（torch 报告 8902） |
| `pip check` | `No broken requirements found` |

关键依赖（完整列表见 `migration/pip-freeze.txt` / `conda-list.txt`）：

| 包 | 版本 |
|---|---|
| numpy | 1.26.4（**不要升到 2.x**） |
| opencv-python-headless | 4.9.0.80 |
| timm | 0.9.12 |
| albumentations | 1.3.1 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| Pillow | 12.3.0 |
| tensorboardX | 2.6.5 |
| thop | 0.1.1-2209072238 |
| segment-anything | 1.0（当时从 `/hy-tmp/projects/segment-anything-main` 本地安装） |
| tqdm | 4.70.1 |
| PyYAML | 6.0.3 |

### 3.3 环境配置文件（已更新/复核）

| 文件 | 用途 |
|---|---|
| `migration/environment.yml` | A 的 conda 导出（去掉 `prefix:` 和 SAM 的本地路径行）；用于版本对照，不保证一条命令装完 |
| `migration/environment-from-history.yml` | 从 shell 历史恢复的初始环境（只有 `python=3.10`/`pip`；仅作溯源，不完整） |
| `migration/pip-freeze.txt` | A 的精确 `pip freeze` 归档（含 `segment_anything @ file:///hy-tmp/...`，仅作记录） |
| `migration/pip-freeze.raw.txt` | 未整理的原始 `pip freeze` 输出（当前环境与 2026-10-27 一致） |
| `migration/pip-requirements.txt` | **可用的安装清单**：已排除 torch/torchvision/triton/nvidia-*/segment-anything |
| `migration/conda-list.txt` | `conda list -n boxpromptiml` 完整输出 |
| `migration/system-info.txt` | OS/CPU/内存/磁盘/GPU/驱动/版本/大小快照（2026-11-02） |
| `migration/install-env.sh` | 在 B 上重建环境的参考脚本（需人工确认 CUDA tag/GPU 兼容性） |
| `migration/git-state.txt` | Git 状态快照（2026-11-02） |
| `migration/code-local-changes.patch` | 已跟踪源码改动 |
| `migration/BoxPromtIML-git.bundle` | 原仓库 3 个提交的完整 bundle |
| `migration/checksums.sha256` | 权重、20 个旧 checkpoint、boxes、stage3/4 关键产物、bundle 的 SHA256（相对 `/home`） |
| `migration/checksums-hy-tmp.sha256` | 4F1 四组 final.pt、status、stage5a 关键文件的 SHA256（相对 `/hy-tmp`） |
| `migration/verify_paths.py` | B 恢复后的路径/数据/权重/缓存/新阶段产物自动检查 |
| `migration/transfer-manifest.tsv` | 迁移文件清单（优先级/大小/方式/GitHub） |

### 3.4 为什么不要直接复制 conda 环境

- A/B 的驱动、GPU 架构、CUDA 版本可能不同；直接拷贝 `envs/boxpromptiml` 可能因 torch 的 CUDA 运行时与 B 驱动不匹配而失败。
- 环境里有 `/hy-tmp` 本地安装的 `segment_anything` 元数据；换机器后路径不存在。
- 环境 5.5G，重建更干净；按 §7.4 安装，通常 10–20 分钟（取决于网络）。
- 新阶段 3–5A 没有引入新的第三方依赖；现有 `pip-requirements.txt` 已覆盖当前代码所需包。

---

## 4. 数据集、权重与 checkpoint

### 4.1 数据集总览（2026-11-02，数据未变）

| 数据集 | 原始目录 | 大小 | 关键数量 |
|---|---|---:|---|
| CASIAv2 | `/home/datasets/raw/CASIAv2/casia/{Tp,Gt,Au}` | 3.4G | Tp 5,123；Gt 5,123 |
| NIST16 | `/home/datasets/raw/NIST16/NC2016_Test0613/` | 3.8G | probe 1,124；split 使用 564 |
| Columbia | `/home/datasets/raw/Columbia/{extracted_splc,extracted_auth}` | 1.2G | splc 181；auth 184；split 使用 180 |
| Coverage | `/home/datasets/raw/Coverage/COVERAGE/{image,mask,label}` | 143M | image 200；mask 300 |
| CASIAv1 | `/home/datasets/raw/CASIAv1/CASIA1.0/{Modified Tp/Tp,GroundTruth}` | 144M | 920 图 + 920 GT |
| **合计** | `/home/datasets` | **9,163,709,381 B（8.54 GiB）** | 23,827 个 raw 文件 + 30,787 个软链 |

### 4.2 正式划分（paper scale）

`/home/datasets/processed/boxprompt_paper_scale_20261016/`：

| Split | 数量 | 说明 |
|---|---:|---|
| train | 5,737 | CASIAv2 5,123 + NIST16 414 + Columbia 130 + Coverage 70 |
| test / NC16 | 150 | NIST16 本地重建 test |
| test / C1 | 920 | 2026-10-19 补充的 CASIAv1 锁定量测试集 |
| test / Columbia | 50 | — |
| test / Coverage | 30 | — |
| **IND 测试合计** | **1,150** | 旧 `eval.py` 只评估 NC16/C1/Columbia/Coverage |

- 清单：`train_manifest.csv` / `test_manifest.csv`，字段包括 `dataset,id,split,split_image,split_mask,image,mask,rule,group,label`。
- `image` / `mask` 列指向 `/hy-tmp/datasets/raw/...`；按 `/hy-tmp → /home` 映射后 **5,737 + 230 条全部存在**（本次逐条检查 0 缺失）。
- `split_image` / `split_mask` 是 `train/Tp`、`train/Gt` 等目录中的软链，目标同样指向 `/hy-tmp/datasets/...`；把 `/hy-tmp/datasets` 映射到 `/home/datasets` 后，**30,787 个软链全部可解析，0 失败**（其中 11,934 个是 `split → pool → raw` 的两级链）。
- `reports/validation.json`：`moved_from_train_to_test = [NC2016_9257, NC2016_0796]`；`overlap.NIST16_base = ["world/NC2016_0359.jpg"]`；没有完全重复图像/像素，但**同一 base 源组跨 train/test**，这是本地重建 split 的已知限制。
- `README.md` 明确说明：本地重建的身份清单**不是作者原始 414/150 NIST16 清单的精确复刻**，不能把本 split 指标当作论文原表的严格复现。

### 4.3 旧 preflight 划分

`/home/datasets/processed/boxprompt_preflight_20261016/`：

- train 5,739；test 228；NIST16 416/148。
- 与 paper scale 的差异：paper scale 把 `NC2016_0796`、`NC2016_9257` 从 train 移到 test（5,739 → 5,737，228 → 230）。
- **原始正式 checkpoint 使用 5,737 的 paper scale**（console.log 打印 4×5737）；preflight 仅保留用于对照/历史，不要用它评估 checkpoint-20 的 NC16。

### 4.4 FECD 内部派生划分（阶段 3 / 3R-A，新增）

这些是**派生清单**，不替换上面锁定的 5,737 张训练清单：

| 集合 | 数量 | 说明 |
|---|---:|---|
| `train_fit` | 4,590 | 阶段 3 用于 GT-free 证据拟合，648 个来源代理组 |
| `teacher_diag` | 1,147 | 阶段 3 教师诊断集，490 个来源代理组 |
| `student_train` | 3,530 | 阶段 4F1 实际训练成员（CASIA 3,278 / NIST 172 / Columbia 52 / Coverage 28） |
| `new_dev` | 530 | 阶段 4F0/4F1/5A 开发评估（CASIA 410 / NIST 80 / Columbia 26 / Coverage 14） |
| `new_confirm` | 530 | **锁定确认集，未打开**（CASIA 410 / NIST 80 / Columbia 26 / Coverage 14） |

- 阶段 3 split 文件：`fecd/assets/stage3_split_manifest.json`（doc 中的内容哈希 `7f18b459...`；文件 SHA256 见 `checksums.sha256`）。
- 阶段 3R-A 派生 split：`fecd/assets/stage3ra_split_manifest.json` / `stage3ra_split_audit.json`，内容哈希 `022da143...`；审计 `status: passed`，counts 3,530/530/530。
- `fecd/docs/stage3_data_protocol.md`、`stage3ra_report.md` 有完整成员/来源代理/限制说明。**不要把 `new_confirm`、`teacher_diag`、正式 IND/OOD GT 混入训练索引**；FECD 的学生 Dataset 只接受 image/cache/boxes，不读取 GT。

### 4.5 软链接机制（迁移时最容易出错）

- `/home/datasets` 中约 **30,787 个符号链接**，多数是链式：
  `train/Tp/CASIAv2_X.tif -> /hy-tmp/datasets/processed/.../pool/CASIAv2/Tp/CASIAv2_X.tif -> /hy-tmp/datasets/raw/CASIAv2/casia/Tp/X.tif`
- 当前 `/hy-tmp` 是真实数据盘但**没有 `datasets` 目录**（只有 `BoxPromtIML/fecd_runs`）；因此直接 `readlink` 或 `os.path.exists` 会看到“断链”。这不是数据丢失，只是缺少 `/hy-tmp/datasets` 路径别名。
- **拷贝时必须保留符号链接**：
  - `rsync -a`（不要 `-L` / `--copy-links`）
  - `tar`（不要 `-h` / `--dereference`）
  - 不要用 `cp -rL`、不要用某些网盘的“跟随软链”选项。
- 如果 B 平台的网页上传/下载会把软链变成普通文件或丢失，恢复后必须用 `migration/verify_paths.py` 检查；必要时从 manifest 重建软链（但当前数据完整，优先修复路径别名，不要重建）。
- `processed/Coverage/mask_corrected/`（2.2M）和 `/home/datasets/patches/Coverage/` 是 Coverage 修正掩码的溯源，必须随 processed 一起迁移。

### 4.6 权重

| 文件 | 大小 | SHA256 | 用途 |
|---|---:|---|---|
| `/home/weights/sam_vit_h_4b8939.pth` | 2,564,550,879 B | `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e` | SAM ViT-H 教师 |
| `/home/weights/tiny_vit_5m_22kto1k_distill.pth` | 21,712,177 B | `e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d` | TinyViT 预训练 backbone |
| `~/.cache/torch/hub/checkpoints/tiny_vit_5m_22kto1k_distill.pth` | 软链 | — | 当前指向 `/hy-tmp/weights/...`，**断链**；B 上需重建（§7.3） |

### 4.7 Checkpoints 与训练产物

#### 原始 BoxPromptIML（旧流水线）

| 路径 | 内容 | 大小/数量 |
|---|---|---:|
| `/home/runs/boxprompt_formal_20261016_122447/checkpoints/checkpoint-{1..20}.pth` | 正式 BoxPromptIML 学生权重，每 epoch 一个，只有 `model.state_dict()` | 20 × 23.49M = 449M |
| `.../checkpoint-20.pth` | 旧 eval 使用的最终 checkpoint | 23,491,814 B；SHA256 `7e54d4c1...35f892` |
| `.../console.log` | 训练控制台（含 4×5737、loss、时间） | 约 32K |
| `.../checkpoints/2026-10-16 12:28:50.472745_log.log` | 训练配置与逐 epoch loss | 约 34K |
| `/home/runs/boxprompt_data_smoke_20261016/` | smoke 训练、manifest、`smoke_checkpoint.pth` | 23M |
| `/home/runs/boxprompt_eval_20261019_121727_145312/` | eval masks、scores、metrics、NIST16 分析图 | 106M |
| `/home/runs/nist16_diagnosis_20261019/` | NIST16 诊断脚本/JSON/报告 | 104K |

**旧 checkpoint 没有 optimizer state / scheduler state / epoch 计数**：`train.py --load` 只能加载模型权重，不能精确续训。见 §8.7。

#### FECD 新 checkpoint（项目内，完整训练 checkpoint）

| 路径 | 内容 | 大小 |
|---|---|---:|
| `fecd/assets/stage4p/{P0,P1,P2,P3}/final.pt` | 4P 的 8-update smoke，完整 checkpoint（含 optimizer/RNG/memory） | 4 × 70,991,246 B ≈ 271M |
| `fecd/assets/stage4f0/{P0,P1,P2F,P3F}/epoch1.pt` | 4F0 的单 epoch 预检 checkpoint | 4 × 70,991,054 B ≈ 271M |
| `fecd/assets/stage4f1/P0/last.pt` | **早期中断的 4F1 P0 部分训练**（epoch 1–5 日志），不是正式 final | 70,991,310 B |

#### FECD 4F1 正式运行（在 `/hy-tmp`，务必迁移）

| 路径 | 内容 | 大小 |
|---|---|---:|
| `/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a/artifacts/{P0,P1,P2F,P3F}/final.pt` | 四组 20 epoch 最终 checkpoint，完整 checkpoint（model + optimizer + memory + RNG + position） | 4 × 70,991,310 B ≈ 271M |
| 同目录 `last.pt` | 与 `final.pt` 内容/哈希相同（final 是 epoch20 复制），保留用于状态一致性 | 另 271M |
| 同目录 `status.json` / `events.jsonl` | 原始执行器状态：`FAILED` at validation；完整事件 | — |
| 同目录 `logs/`、`artifacts/evaluation/`、`artifacts/analysis/` | 逐 epoch 日志、530 图严格评估、配对 bootstrap、域指标、失败案例、训练曲线 | 约 1.08GB 总量 |

四个 `final.pt` 的 SHA256：

| 组 | SHA256 |
|---|---|
| P0 | `9ea55e01e5e354d252fb6470c7a3aa04712840eb657917e694c19f56650bd1ea` |
| P1 | `b0e56a4eb4ffc361f80476af580b30c2351ad885bea10f251b08ae119cdf4d89` |
| P2F | `1d1997cd71b8ff82b5e42debdf5e1e9e8ea659b8c1194c231481c3e0088a9404` |
| P3F | `8c6dc3512bdebaf0885b85dbe9b65af737b804f90e21238405073cbe4fa11ea6` |

注意：`status.json` 里的 `completed_epochs=19` / `total_update=17577` 是最后一轮进度快照，不是最终完成度；四组 `result.json`、epoch20 summary 和 checkpoint metadata 一致证明每组完成 **20 epoch / 17,660 update**。原始 `status.json` 仍保留 `FAILED`，不要为了“看起来完成”去改它；4F1-V 的独立 PASS 是另一个文件、另一个验收器。

#### 阶段 5A 诊断（在 `/hy-tmp`）

`/hy-tmp/BoxPromtIML/fecd_runs/stage5a/diagnosis_20261030/`：530 张 new_dev × 4 个 checkpoint 的 CPU 概率图（`maps/` 约 1.1GB）、逐图/来源组/阈值统计、12 个案例、`hash_manifest.json` + `.sha256`、`validation_final.json`。工程 **PASS**，科研 **NO CLEAR GAIN**。

### 4.8 已有评估结果

#### 旧 `eval.py`（checkpoint-20，paper scale，1,150 张 IND）

来自 `/home/runs/boxprompt_eval_20261019_121727_145312/metrics.txt`：

| 数据集 | AUC | F1 | IoU |
|---|---:|---:|---:|
| NC16 | 0.7984 | 0.2801 | 0.2117 |
| C1 | 0.8724 | 0.5604 | 0.4722 |
| Columbia | 0.9637 | 0.8773 | 0.8339 |
| Coverage | 0.8449 | 0.4046 | 0.2948 |
| **f1_in（四数据集 F1 算术平均）** | — | **0.5306** | — |

注意：这是**本地重建 split** 的结果，不是论文原表；NIST16 存在同源 base 跨 split（`world/NC2016_0359.jpg`），NC16 指标不能当作干净的 held-out 证据；AUC 使用逐图 min-max 后的连续分数，F1/IoU 用 0.5 固定阈值。

#### FECD 4F1 严格评估（530 张 new_dev，raw sigmoid，0.5）

| 组别 | macro F1 | macro IoU | micro F1 | 域等权 F1 | 域等权 IoU |
|---|---:|---:|---:|---:|---:|
| P0 | 0.345859 | 0.271365 | 0.411323 | 0.450691 | 0.379294 |
| P1 | 0.336740 | 0.264194 | 0.427836 | 0.430960 | 0.361902 |
| P2F | **0.346021** | 0.269402 | 0.401325 | 0.460241 | 0.381476 |
| P3F | 0.342369 | 0.263192 | **0.434270** | **0.492642** | **0.404194** |

- 主指标 P2F 最高，但只比 P0 高 **0.016pp**；三个预设主对照（P1−P0、P2F−P1、P3F−P2F）的 F1/IoU 95% source-group bootstrap 区间**均跨零**。
- P3F 在 NIST16/Columbia/Coverage 点估计更高，但 CASIAv2 下降约 1.50pp；没有全指标一致领先者。
- **结论：NO CLEAR GAIN**；不要把它改写成“FECD 有效”或“P3F 全面更好”。
- 原始 `status.json` 仍为 **FAILED**（float32 loss 分解断言误报）；`fecd/assets/stage4f1_v/revalidation_20261030_v2/validation_v2.json` 的独立复验为 **PASS（engineering）**，两者必须同时保留、不能互相覆盖。

#### 4F0 单 epoch（530 张 new_dev，供对照）

| 组别 | macro F1 | macro IoU | 域等权 F1 |
|---|---:|---:|---:|
| P0 | 0.186627 | 0.121262 | 0.284857 |
| P1 | 0.193589 | 0.127398 | 0.298708 |
| P2F | 0.188188 | 0.122896 | 0.294549 |
| P3F | 0.187117 | 0.121600 | 0.304197 |

4F0 只是工程预检；20 epoch 后各组约提升 14–16pp，但这是训练充分程度的描述性观察，不是方法因果结论。

---

## 5. 当前实验状态与主要运行命令

### 5.1 状态

- 原始 BoxPromptIML 正式训练：**已完成**（2026-10-16 12:29 → 2026-10-17 00:02，约 11.5 小时，20 epoch，5,737 张，batch 16，359 steps/epoch）。
- 4F1 正式运行：**训练/评估/统计均完成**，但原始最终验收 `FAILED`（validation 阶段）；4F1-V 独立复验 `PASS`；科研结论 `NO CLEAR GAIN`。
- 5A：**完成并停止**，没有新训练。
- 当前 `ps` 只有 `jupyter-lab`；`tmux` 无会话；**不要假设有后台训练**。

### 5.2 原始训练命令（历史/复现用）

`train.py` 默认 `--train_root=/home/xidongdong/Dataset/mai_Train/`（不存在），**必须显式传参**。正式 run 的完整命令行未被 shell history 完整记录，但日志确认了 5,737 样本、batch 16、save path。B 上建议使用：

```bash
cd /home/projects/BoxPromtIML
CUDA_VISIBLE_DEVICES=0 /usr/local/miniconda3/envs/boxpromptiml/bin/python -u train.py \
  --gpu_id 0 \
  --train_root /hy-tmp/datasets/processed/boxprompt_paper_scale_20261016/train/ \
  --save_path /hy-tmp/runs/boxprompt_formal_20261016_122447/checkpoints/ \
  --batchsize 16
```

注意：

- `--train_root` 结尾必须保留 `/`；代码会拼 `Tp/`、`Gt/`、`Edge/`。
- `Edge/` 可以为空；当前 `utils/data_val.py` 的 `get_loader` 实际未使用 edge（相关读取被注释）。
- `train.py` 内部直接 `images.cuda()`，**必须有可用 GPU**；CPU 不能训练。
- 这是原始仓库训练循环：在线从 GT 取 box、逐样本调用 SAM、`structure_loss` 蒸馏；不是 FECD 训练。

### 5.3 旧评估命令

```bash
cd /home/projects/BoxPromtIML
CUDA_VISIBLE_DEVICES=0 /usr/local/miniconda3/envs/boxpromptiml/bin/python eval.py
```

- `eval.py` 硬编码 checkpoint `/hy-tmp/runs/boxprompt_formal_20261016_122447/checkpoints/checkpoint-20.pth` 和测试根 `/hy-tmp/datasets/processed/boxprompt_paper_scale_20261016/test`；B 上需要 `/hy-tmp` 别名，或手动改这两处（`eval.py:229`、`eval.py:243`）。
- 输出目录：`/hy-tmp/runs/boxprompt_eval_<timestamp>/`（`eval.py:253`），包含 `masks/`、`scores/`、`metrics.txt`。
- 无 GPU 时会回退 CPU（`eval.py:227`），但 1,150 张图会非常慢；不建议。
- 评估前 `validate_dataset_layout` 会检查 150/920/50/30 的数量、GT 二值性和尺寸；任何断链会在这里报错，而不是静默算错。

### 5.4 FECD 测试与验证命令

```bash
cd /home/projects/BoxPromtIML
ENV=/usr/local/miniconda3/envs/boxpromptiml

# 全量单元/契约测试（CPU；2026-11-02 实测 197 tests OK）
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -m unittest discover -s fecd/tests -p 'test_*.py'

# 旧阶段 2B/2C 回归（9 checks passed）
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -m fecd.tests.run_stage2b_tests

# 环境/依赖/模型报告（需要 TinyViT 权重可解析；会构造 BoxPromptIML）
$ENV/bin/python -m fecd.tools.check_environment

# 审计全量 K1/K8 cache（不需要重新推理）
$ENV/bin/python -m fecd.tools.audit_stage2c_cache

# 阶段 2C 预检（需要 GPU + SAM）
$ENV/bin/python -m fecd.tools.stage2c_preflight

# 重新生成全量 cache（不必要；已有 cache；约 70 分钟 RTX 4090）
$ENV/bin/python -m fecd.tools.run_stage2c_cache --device cuda:0
```

### 5.5 4F1 正式训练（只有明确授权才运行）

历史实际命令（`fecd/assets/stage4f1_run_only_setup/launch_receipt.json`）：

```bash
cd /home/projects/BoxPromtIML
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python \
  -m fecd.tools.run_stage4f1_autoshutdown \
  --no-shutdown --detach \
  --run-id formal20_seed20261029_noshutdown_20261029a \
  --output-root /hy-tmp/BoxPromtIML/fecd_runs/stage4f1
```

B 上如果只是恢复/评估，**不要重跑**；如需新实验，使用新的 `run-id` 和新的输出目录：

```bash
cd /home/projects/BoxPromtIML
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python \
  -m fecd.tools.run_stage4f1_autoshutdown \
  --no-shutdown --detach \
  --run-id <新的_run_id> \
  --output-root /hy-tmp/BoxPromtIML/fecd_runs/stage4f1
```

硬性限制（代码实际行为）：

- 输出必须在 `/hy-tmp` 下，且 `/hy-tmp` 必须是**独立于 `/` 的挂载/文件系统**；`fecd/training/f1_storage.py` 和 `fecd/execution/storage.py` 会检查 `st_dev`，不满足直接拒绝（`DATA_MOUNT_LOST`）。
- 预检要求系统盘至少 5GiB、数据盘至少 8GiB 空闲。
- 同一基础目录用 flock 单实例锁；`run_id` 不能重复，重跑不会覆盖旧目录。
- `--no-shutdown` 禁用关机证据；默认 `shutdown_on_failure=False`。本次正式运行使用 `no_shutdown=true`，`shutdown_status=DISABLED`，不能据此认为云实例已关机/停止计费。
- 该入口会先做完整预检（3,530 成员、缓存身份、四组初始化、20 轮顺序/几何、CUDA 恢复等），预检 PASS 后才训练；预计四组约 5–6 小时（本机 5.087 小时）。

4F1 的配置/协议文件：`fecd/configs/stage4f1/{P0,P1,P2F,P3F}.json`、`fecd/configs/stage4f1/analysis_protocol.json`、`fecd/configs/stage4f1/storage.json`。

### 5.6 严格评估新 checkpoint

```bash
cd /home/projects/BoxPromtIML
$ENV/bin/python -m fecd.tools.evaluate_student_strict \
  --checkpoint /hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a/artifacts/P2F/final.pt \
  --output-dir /path/to/new/eval_P2F
```

- 该入口只读 530 张 `new_dev`，使用 raw sigmoid + 0.5，不做 min-max；输出 `summary.json`、`per_image.jsonl`、`gt_access.jsonl` 等。
- `new_confirm` 没有访问路径；不要为了“多评估一点”去改 allowlist。
- 旧 `eval.py` 与新严格 evaluator 口径不同，**不要拼表比较**。

### 5.7 4F1-V 复验命令（可选，会写新目录）

```bash
cd /home/projects/BoxPromtIML
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python \
  -m fecd.tools.check_stage4f1_v2 \
  --input-run /hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a \
  --output-dir /path/to/new/revalidation
```

现有结果在 `fecd/assets/stage4f1_v/revalidation_20261030_v2/`，`validation_v2.json` 已记录 `status: PASS`、`engineering: PASS`、`research_conclusion: NO CLEAR GAIN`。复验不训练、不构造模型、不重新读 RGB/GT。

### 5.8 阶段 4P / 4F0 / 5A

- 4P：`python -m fecd.tools.train_controlled_baseline --variant P0`（P1/P2/P3 同理）；需要 GPU；这是 8-update smoke，不是正式训练。
- 4F0：`python -m fecd.tools.train_epoch_preflight --variant P0`（P1/P2F/P3F 同理）；需要 GPU；单 epoch。
- 5A：现有产物是 CPU 只读诊断；实际命令记录在 `/hy-tmp/BoxPromtIML/fecd_runs/stage5a/diagnosis_20261030/commands_final.json`。不要在没有新协议的情况下重跑/覆盖；它只用于开发诊断，不是训练入口。

---

## 6. 需要迁移的文件清单

### 6.1 总表

| 优先级 | 内容 | 大小 | 迁移方式 | GitHub？ |
|---|---|---:|---:|---|
| P0 | 项目源码/文档/`.git`/`TinyViT`/`fecd` 代码与配置/`migration`（不含 `fecd/assets`） | 18.7MB | 目录拷贝，或 git bundle + patch + 未跟踪目录 | **是**（源码/文档/配置） |
| P0 | `fecd/assets/stage2c_cache` | 643MB | 单独传（OSS/数据盘） | 否 |
| P0 | `fecd/assets/boxes_train.json` | 4.4MB | 单独传 | 否 |
| P0 | `fecd/assets/stage3ra_split_manifest.json` + audit + acceptance | 3.3MB | 随代码/配置；4F1 成员重建依赖 | 否（含路径/身份） |
| P0 | `/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_...` | 1.08GB | 单独传；保持 run_id/目录 | 否 |
| P0 | `/home/datasets/raw` | 9.09GB | 单独传；保留目录名/大小写 | 否 |
| P0 | `/home/datasets/processed` | 69.6MB + 30,787 链接 | 与 raw 一起；**保留符号链接** | 否 |
| P0 | `/home/datasets/patches` + `/home/datasets/scripts` | 174KB | 随 datasets | 可选 |
| P0 | `/home/weights` | 2.59GB | 单独传；校验 SHA256 | 否 |
| P0 | `/home/runs` | 598MB | 单独传；checkpoint 目录不要改名 | 否 |
| P1 | `/home/datasets/boxprompt-smoke` + `boxprompt-casia16` | 24KB | smoke 数据与 CASIA16 软链样本 | 否 |
| P1 | `fecd/assets/stage3_evidence` | 4.15GB | 单独传；复现阶段 3/3R 用 | 否 |
| P1 | `fecd/assets/stage3_diagnostics` + `stage3ra_*` + `stage3rb0` | 约 302MB | 随 fecd assets | 否 |
| P1 | `fecd/assets/stage4p` | 571MB | 单独传；历史工程基线 | 否 |
| P1 | `fecd/assets/stage4f0` | 610MB | 单独传；历史开发基线 | 否 |
| P1 | `fecd/assets/stage4f1`（早期部分训练） | 105MB | 单独传；不要与正式 run 混淆 | 否 |
| P1 | `fecd/assets/stage4f1_v` | 8.6MB | 随项目；保存 FAILED→PASS 证据链 | 可选（文本） |
| P1 | `/hy-tmp/BoxPromtIML/fecd_runs/stage5a/diagnosis_20261030` | 1.19GB | 单独传；省略需重跑 CPU 诊断 | 否 |
| P1 | `migration/` | 5.0MB | 随项目 | 是（文本） |
| P1 | `segment-anything-main/` | 22MB | 可选；B 上也可重新 pip install | 否 |
| P1 | `note.md`、`deep-research-report.md`、`conversation_20261026.md` | 87KB | 随项目文档/小文件 | 否（研究隐私） |
| P1 | `/home/checkpoints` | 1.5KB | 可选（只有旧 smoke 日志） | 否 |
| P2 | `TinyViT-main.zip` | 642KB | 可省略（项目内已有解压版） | 否 |
| P2 | `stage2b_smoke`、`stage2c_anomaly_visualizations`、`stage2c_cache_statistics.csv`、`stage2c_preflight_comparison`、`stage2c_cache_run.log` | 约 20MB | 可选，溯源用 | 否 |
| P2 | conda 环境 `boxpromptiml` | 5.4GB | **不要拷贝**；重建 | 否 |
| 禁止 | `/root/.ssh`、`/root/.codex/auth.json`、`/root/.hycloud_ossutil_config`、`/root/.npmrc`、`/root/.vscode-server` | — | 禁止打包/上传 | 否 |

机器可读版本：`migration/transfer-manifest.tsv`。**不含 conda 的完整迁移约 21.1GB（19.6GiB）**；只迁移 P0 约 14.1GB。

### 6.2 打包建议

如果 B 与 A 的目录布局一致（推荐，能少改路径）：

```bash
# 在 A 上（示例；不要直接执行上传）
cd /home
tar -czf /hy-tmp/boxprompt_code.tar.gz projects/BoxPromtIML
tar -czf /hy-tmp/boxprompt_datasets.tar.gz datasets
tar -czf /hy-tmp/boxprompt_weights.tar.gz weights
tar -czf /hy-tmp/boxprompt_runs.tar.gz runs
tar -czf /hy-tmp/boxprompt_fecd_runs.tar.gz -C /hy-tmp BoxPromtIML/fecd_runs
sha256sum /hy-tmp/boxprompt_*.tar.gz
```

- `tar` 默认保留符号链接，不要加 `-h`/`--dereference`。
- 如果只用 `rsync`：`rsync -a`（archive 保留链接、权限、时间），不要 `-L`。
- `/hy-tmp` 只有 48GB 可用，压缩包 + 原文件可能超过；**不要把唯一副本留在 `/hy-tmp`**。可先流式上传，或分批（datasets / weights / runs / fecd assets）。
- 代码如果走 GitHub：只提交/推送源码、`fecd` 代码、`TinyViT`、`migration` 里的文本文件；**不要提交** `fecd/assets`、`runs`、`weights`、`datasets`、checkpoint。本次没有替你执行 commit/push。
- 数据/权重/checkpoint/cache 建议走 OSS、B 平台对象存储或私有传输；NIST16/CASIA 等数据集有各自许可，不要公开上传。

### 6.3 完整性校验

- `migration/checksums.sha256`（相对 `/home`）：SAM、TinyViT、20 个旧 checkpoint、`boxes_train.json`、bundle、K1/K8 cache config、stage3/4 关键产物、4F1-V validation。
- `migration/checksums-hy-tmp.sha256`（相对 `/hy-tmp`）：4F1 四组 `final.pt`、原始 `status.json`、stage5a `hash_manifest.json` / `validation_final.json`。
- 在 B 上恢复到相同布局后：
  ```bash
  cd /home
  sha256sum -c /home/projects/BoxPromtIML/migration/checksums.sha256
  cd /hy-tmp
  sha256sum -c /home/projects/BoxPromtIML/migration/checksums-hy-tmp.sha256
  ```
- 如果项目/数据路径不同，checksums 里的相对路径需要相应调整；至少手工核对 SAM、TinyViT、旧 checkpoint-20、boxes、4F1 四组 final.pt 六个关键 SHA。
- cache 的 11,474 个 NPZ 不建议逐个 SHA；用 `fecd/tools/audit_stage2c_cache.py` + 每个目录的 `manifest.json` 校验，或整目录 tar 的 SHA。
- stage3 evidence 的 5,737 个 NPZ 用 `fecd/assets/stage3_evidence/stage3_evidence_cache_audit.json` 和 `stage5a` 的 `hash_manifest.json` 校验；不要只数文件数。

---

## 7. B 平台恢复步骤

> 下面的路径以 `PROJECT_ROOT=/home/projects/BoxPromtIML`、`ENV=/usr/local/miniconda3/envs/boxpromptiml`、数据根 `/home/...`、运行根 `/hy-tmp/BoxPromtIML/fecd_runs` 为例。B 的盘符/用户名可能不同，按实际替换。

### 7.0 B 上先确认这些（不要假设）

```bash
uname -a
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 --version
df -hT
free -h
```

- 确认 B 有可用 GPU；确认驱动版本能支持你要安装的 PyTorch CUDA 版本。
- 确认 `/hy-tmp` 是否是**独立数据盘/临时盘**；`fecd/training/f1_storage.py` 要求它与 `/` 不是同一个 `st_dev`。如果是临时盘，不要把唯一一份数据放在那里。
- 确认 B 的系统盘剩余空间：P0 约 14.1GB，全部约 21.1GB；加环境 5.5GB 和临时空间，建议 ≥ 30GB 可用。
- 确认 B 能读 600 权限的 cache/NPZ；如果用非 root 用户，需要 `chown -R` 或 `chmod`。

### 7.1 恢复代码

**方式 A（推荐，最完整）**：把 A 的整个项目目录恢复到相同路径，保留 `.git`、未跟踪目录、软链。

```bash
# 在 B 上
mkdir -p /home/projects
# 从 A 传输/解压 BoxPromtIML 到 /home/projects/BoxPromtIML
cd /home/projects/BoxPromtIML
git status --short          # 应与 migration/git-state.txt 一致
git diff --stat
```

**方式 B（GitHub / bundle）**：

```bash
cd /home/projects
git clone https://github.com/vpsg-research/BoxPromtIML.git BoxPromtIML
cd BoxPromtIML
git bundle verify /path/to/migration/BoxPromtIML-git.bundle
git fetch /path/to/migration/BoxPromtIML-git.bundle main:refs/remotes/bundle/main
git checkout main
git apply --stat /path/to/migration/code-local-changes.patch
git apply /path/to/migration/code-local-changes.patch
# 然后单独恢复未跟踪的 TinyViT/、fecd/、segment-anything/ 软链、migration/
```

- `code-local-changes.patch` 只含 `eval.py`/`train.py` 的文本改动，不含 `fecd/`、`TinyViT/`。
- `code-local-changes.patch` 是相对**干净 HEAD** 的 diff。A 工作区已经应用了这些修改，所以在 A 的工作区直接 `git apply --check` 会失败（already applied），这是正常的；请在 bundle/远端 clone 出的干净仓库上应用。2026-10-27 已在 `/tmp` 干净 clone 验证；2026-11-02 复核 patch 与 A 工作树逐字一致。
- `segment-anything/` 目录只需恢复一个软链：
  ```bash
  mkdir -p /home/projects/BoxPromtIML/segment-anything
  ln -sf /hy-tmp/weights/sam_vit_h_4b8939.pth \
         /home/projects/BoxPromtIML/segment-anything/sam_vit_h_4b8939.pth
  ```
  （如果 `/hy-tmp/weights` 已按 §7.2 建好。）

### 7.2 恢复数据/权重/runs/4F1 运行产物，并建立两套路径

代码里同时存在 `/home/...` 和 `/hy-tmp/...` 两种绝对路径，**两者都必须能解析到同一份真实数据**；另外 `/hy-tmp/BoxPromtIML/fecd_runs` 必须是真实目录（4F1 训练入口会对 `--output-root` 做 `resolve()`，不接受解析到 `/hy-tmp` 之外的普通软链）。

#### 方案 1：数据放在 `/home`（与 A 当前布局一致，最省改代码）

```bash
# 假设 /home/datasets、/home/weights、/home/runs、/home/projects 已恢复为真实目录
mkdir -p /hy-tmp/BoxPromtIML
ln -s /home/datasets /hy-tmp/datasets
ln -s /home/weights  /hy-tmp/weights
ln -s /home/runs     /hy-tmp/runs
ln -s /home/projects /hy-tmp/projects   # 可选，兼容历史路径
# 把 A 的 /hy-tmp/BoxPromtIML/fecd_runs 恢复到 B 的同一路径
```

#### 方案 2：数据放在持久数据盘 `/data`（推荐给系统盘较小的 B）

```bash
# 真实数据：/data/datasets、/data/weights、/data/runs
mkdir -p /hy-tmp
ln -s /data/datasets /home/datasets
ln -s /data/datasets /hy-tmp/datasets
ln -s /data/weights  /home/weights
ln -s /data/weights  /hy-tmp/weights
ln -s /data/runs     /home/runs
ln -s /data/runs     /hy-tmp/runs

# 4F1 运行根：不能在 /hy-tmp 下用普通软链指向 /data（resolve 后会脱离 /hy-tmp，
# f1_storage.py 会拒绝）。二选一：
#   (a) 直接把 4F1/stage5a 产物恢复到 /hy-tmp 的真实目录，并定期备份到 /data：
mkdir -p /hy-tmp/BoxPromtIML/fecd_runs
#   (b) 或者用 bind mount（需要 root；重启后要重新挂载）：
# mkdir -p /hy-tmp/BoxPromtIML/fecd_runs
# mount --bind /data/fecd_runs /hy-tmp/BoxPromtIML/fecd_runs
```

#### 方案 3：数据直接放在 `/hy-tmp`（只有确认 `/hy-tmp` 持久时）

```bash
ln -s /hy-tmp/datasets /home/datasets
ln -s /hy-tmp/weights  /home/weights
ln -s /hy-tmp/runs     /home/runs
```

无论哪种方案，验收标准是：

```bash
python3 - <<'PY'
import os
for a,b in [('/hy-tmp/datasets','/home/datasets'),
            ('/hy-tmp/weights','/home/weights'),
            ('/hy-tmp/runs','/home/runs')]:
    print(a, '->', os.path.realpath(a), '| same as', b, ':', os.path.samefile(a,b))
PY
```

**特别注意**：

- 如果 `/home/datasets` 已经是真实目录，就不要再 `ln -s` 覆盖它；先确认内容，再决定用方案 1 还是移动。**不要删除任何现有数据来“腾位置”**。
- 4F1 运行器要求 `/hy-tmp` 是独立文件系统；不要把 `/hy-tmp` 做成指向 `/` 的普通目录。方案 2 的 `/data` 必须是真实数据盘；如果 B 只有系统盘，4F1 训练入口会拒绝运行（这是保护，不是 bug）。
- 4F1 的正式产物在 `/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/...`，不是 `/home/runs`；旧 `eval.py` 的 checkpoint-20 仍在 `/home/runs/...`。两套都要恢复。
- `stage5a_io.py` 硬编码 `/hy-tmp/BoxPromtIML/fecd_runs/...`；它只读现有诊断，用软链通常可以。但 **4F1 训练入口不能接受解析到 `/hy-tmp` 外的 `--output-root`**；4F1 训练必须用 `/hy-tmp` 下的真实目录，或对 `/data` 做 bind mount。若 B 把运行根放到别处，需改相应路径并记录（改代码属于 B 上的适配，不是本次迁移）。

### 7.3 修复 TinyViT / SAM 的软链

`boxpromptiml.py` 的 `BoxPromptIML()` 调用 `tiny_vit_5m_224(pretrained=True)`；TinyViT 代码会从 GitHub 下载 `tiny_vit_5m_22kto1k_distill.pth` 到 torch hub cache。A 上该缓存是一个指向 `/hy-tmp/weights` 的软链，当前断链。B 上必须让 cache 指向本地权重，否则 `train.py`/`eval.py` 在构造模型时就可能联网下载或失败。

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
ln -sf /home/weights/tiny_vit_5m_22kto1k_distill.pth \
       ~/.cache/torch/hub/checkpoints/tiny_vit_5m_22kto1k_distill.pth
ls -l ~/.cache/torch/hub/checkpoints/tiny_vit_5m_22kto1k_distill.pth
```

- 如果 B 的 `HOME` 不是 `/root`，把 `~` 展开到实际用户目录；也可以用 `torch.hub.get_dir()` 确认路径。
- 原始 `train.py`/`eval.py` 仍然走 `pretrained=True`，所以这个 cache 必须存在；FECD 的 `fecd/model_loading.py` 则显式加载本地权重，不依赖下载。
- SAM 权重软链 `segment-anything/sam_vit_h_4b8939.pth -> /hy-tmp/weights/...` 由 §7.2 的别名解决；如果路径不同，手动 `ln -sf` 到真实文件。

### 7.4 重建 Python 环境

不要复制 conda 环境目录。推荐：

```bash
# 1) 创建 Python 3.10 环境
conda create -y -n boxpromptiml python=3.10 pip
conda activate boxpromptiml

# 2) 安装与 B 驱动/GPU 匹配的 PyTorch
#    A 的原环境：torch 2.2.1+cu121 / torchvision 0.17.1+cu121（驱动 >= 525.60.13）
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 3) 安装其余固定版本依赖（已排除 torch/torchvision/triton/nvidia-*/SAM）
pip install -r /home/projects/BoxPromtIML/migration/pip-requirements.txt

# 4) 安装 SAM（推荐用已恢复的本地源码）
pip install /home/projects/segment-anything-main
#    或：pip install git+https://github.com/facebookresearch/segment-anything.git

# 5) 检查
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
pip check
```

也可以直接参考/运行 `migration/install-env.sh`：

```bash
bash /home/projects/BoxPromtIML/migration/install-env.sh
# 可通过环境变量覆盖：CUDA_TAG=cu118 TORCH_VERSION=2.2.1 SEGMENT_ANYTHING_SRC=/home/projects/segment-anything-main
```

**选择 PyTorch 版本的规则**：

| B 的 GPU/驱动 | 建议 |
|---|---|
| RTX 4090 / A100 / H100，驱动 ≥ 525 | `cu121` + torch 2.2.1（与 A 最接近） |
| 只有 CUDA 11.8 驱动（旧） | `cu118` + torch 2.2.1 / torchvision 0.17.1（代码不变） |
| RTX 5090 / B200 等 Blackwell（sm_100/120） | torch 2.2.1 可能不支持新架构；安装更新的 torch（如 2.7/2.8 + cu128），并重新验证；不要强行用 cu121 |
| 平台预装 torch | 先用平台版本做 import/forward smoke；如果版本差异大，记录并说明 |

**不要**：`pip install -U torch numpy`、混用 conda 的 torch 和 pip 的 torch、把 numpy 升到 2.x。这些都可能让 A 上已验证的代码失效。

### 7.5 验收：路径、数据、环境、测试

```bash
cd /home/projects/BoxPromtIML
ENV=/usr/local/miniconda3/envs/boxpromptiml

# 1) 路径/数据/权重/缓存/4F1/stage5A 检查（默认不做 2.4G 哈希；加 --hash 做关键 SHA）
$ENV/bin/python migration/verify_paths.py
$ENV/bin/python migration/verify_paths.py --hash
# 如果 4F1 运行根不在默认位置：
$ENV/bin/python migration/verify_paths.py --fecd-runs /path/to/fecd_runs

# 2) FECD 全量单元/契约测试（CPU 可跑；期望 197 tests OK）
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -m unittest discover -s fecd/tests -p 'test_*.py'

# 3) 旧回归（期望 9 checks passed）
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -m fecd.tests.run_stage2b_tests

# 4) 环境报告（构造模型；需要 TinyViT cache 已修好）
$ENV/bin/python -m fecd.tools.check_environment

# 5) 全量 cache 审计
$ENV/bin/python -m fecd.tools.audit_stage2c_cache
```

期望：

- `verify_paths.py` 输出 `RESULT: PASS`；没有 FAIL；最多有 `/hy-tmp/projects` 之类 WARN；若未迁移 P1 资产，会有对应 WARN，不影响 P0 训练。
- `unittest discover` 输出 `Ran 197 tests ... OK`。
- `run_stage2b_tests` 输出 `9 checks passed`。
- `check_environment` 能 import `boxpromptiml`，并且 `torch.cuda.available` 为 True（如果 B 有 GPU）。
- `audit_stage2c_cache` 的状态为 `passed`，K1/K8 各 5,737，0 重复/0 错误。

### 7.6 复现旧评估

```bash
cd /home/projects/BoxPromtIML
CUDA_VISIBLE_DEVICES=0 /usr/local/miniconda3/envs/boxpromptiml/bin/python eval.py
```

- 先确认 `eval.py:229` / `eval.py:243` 在 B 上可解析；推荐用 §7.2 的 `/hy-tmp` 别名，避免改代码。
- 新输出在 `/hy-tmp/runs/boxprompt_eval_<timestamp>/metrics.txt`。
- 与 A 的参考值对比（允许 GPU 非确定性带来的小波动）：NC16 F1 0.2801、C1 0.5604、Columbia 0.8773、Coverage 0.4046、f1_in 0.5306。
- 如果 `validate_dataset_layout` 报错，先修路径/软链，不要绕过检查；这通常意味着数据没有完整恢复。

### 7.7 评估/继续 4F1 checkpoint

**只做推理/严格评估**（推荐先做）：

```bash
cd /home/projects/BoxPromtIML
$ENV/bin/python -m fecd.tools.evaluate_student_strict \
  --checkpoint /hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a/artifacts/P2F/final.pt \
  --output-dir /hy-tmp/BoxPromtIML/fecd_runs/stage4f1/eval_recheck_P2F
```

期望与 `artifacts/evaluation/P2F/summary.json` 的数值一致（GPU 非确定性可能带来极小波动）；先比对 `summary.json` / `per_image.jsonl`，不要直接覆盖原评估目录。

**从 checkpoint 继续实验**：

- 4F1 `final.pt` / `last.pt` 是**完整 checkpoint**，包含 `model`、`optimizer`、`memory`、`rng`、`position: {epoch: 20, next_batch: 0}`、`model_hash`，与旧 `checkpoint-20.pth` 只有 `state_dict` 不同。
- 但 `run_stage4f1_autoshutdown` 的协议固定 20 epoch；直接用它“从 final.pt 再跑 20 epoch”不是现有入口的功能。若要严格续训，需要单独的、明确登记的新协议/入口，保存/恢复 optimizer/scheduler/RNG/epoch，不能声称是原 4F1 的自动延续。
- 如果只是把 4F1 权重当初始化做新实验，使用新的 run_id、新的输出目录，并在报告中说明“不是原 4F1 的严格续训”。
- 不要覆盖 `/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_...`；不要改 `status.json` 把 FAILED 改成 PASS。

### 7.8 FECD 的下一步（研究状态）

1. 先读 `/home/note.md`、`/home/deep-research-report.md`、`fecd/docs/stage1_audit.md`、`stage2a_*`、`stage2b_report.md`、`stage2c_report.md`、`stage3_report.md`、`stage3r_*`、`stage3ra_report.md`、`stage3rb0_*`、`stage4p_report.md`、`stage4f0_report.md`、`stage4f1_*`、`stage5a_failure_diagnosis.md`。
2. 确认阶段 2C cache 审计通过、`boxes_train.json` SHA 不变、SAM SHA 不变；确认 4F1 四组 `final.pt` SHA 与 `checksums-hy-tmp.sha256` 一致。
3. **当前科研结论是 NO CLEAR GAIN，且 5A 已 STOP**；不要假设存在已验证的 FECD 方法或下一阶段训练入口。5A 报告给出的“先 A 校准基线、再 B 区域误激活机制”只是建议，尚未授权/实现。
4. 若必须重跑 cache：`run_stage2c_cache` 有断点复用和锁；不要删除现有 cache；预计 RTX 4090 约 70 分钟。若 B 的 GPU 不同，先做小样本 benchmark，不要直接全量重跑。
5. FECD 的 Student Dataset 只接受 image/cache/boxes，不读取 GT；不要为了“方便”把 GT 路径塞进训练 index，这会破坏阶段 2A/2B 的隔离约束。

---

## 8. 兼容性问题与注意事项

### 8.1 路径依赖（最容易踩的坑）

| 路径 | 引用位置 | B 上怎么办 |
|---|---|---|
| `/hy-tmp/datasets` | `eval.py:243`；`train_manifest.csv`/`test_manifest.csv` 的 `image/mask/split_*`；`fecd` 的路径映射 | 建别名指向真实 datasets |
| `/hy-tmp/runs` | `eval.py:229`、`eval.py:253` | 建别名指向真实 runs |
| `/hy-tmp/weights` | `segment-anything/sam_vit_h_4b8939.pth` 软链；torch hub TinyViT 缓存软链 | 建别名指向真实 weights |
| `/hy-tmp/BoxPromtIML/fecd_runs` | `fecd/diagnostics/stage5a_io.py:16-17`；4F1 `storage.json` / 运行器 | 恢复真实运行目录；或改路径并记录 |
| `/home/datasets` | `fecd/assets/boxes_train.json` 的 `image_path`；`fecd/assets/stage2c_cache/**/dataset_index.json`；`fecd/tools/export_boxes.py:9` | 保持真实路径或建别名 |
| `/home/weights` | `fecd/pseudo/generate_cache.py:14`、`fecd/tools/*.py`、`fecd/tests/test_model_smoke.py`、4F1 配置 | 保持真实路径或建别名 |
| `/home/runs` | 旧 checkpoint 路径、`fecd` 文档、旧 eval 结果 | 保持真实路径或建别名 |
| `/home/xidongdong/Dataset/mai_Train/` | `train.py:193` 默认值 | **不存在**；必须传 `--train_root` |

建议在 B 上保持 `PROJECT_ROOT=/home/projects/BoxPromtIML`、`DATA_ROOT` 两套别名；这样不需要改任何核心代码。

### 8.2 符号链接

- 约 30,787 个数据集软链；**必须保留**。`tar`/`rsync -a` 可以；`cp -L`、`rsync -L`、`tar -h` 会把 raw 图像重复展开，体积爆增且可能破坏目录结构。
- A 上已逐跳验证：在 `/hy-tmp/datasets` → `/home/datasets` 的逐跳映射下，30,787 个链接全部可解析；其中 11,934 个是 `split → pool → raw` 的两级链，因此**只替换 manifest 的一层前缀不够**。B 上优先建立真实 `/hy-tmp` 别名，不要急着重写数据目录。
- 如果 B 的上传/下载工具不保留软链，恢复后 `verify_paths.py` 会报 broken；此时可用 manifest 的 `image`/`mask` 列重建链接，但必须先确认 Coverage 修正掩码和 C1 的真实 GT 没有被覆盖。
- 4F1 的 cache NPZ 权限是 `600`（root）；B 用非 root 用户时需要 `chown -R`/`chmod`，否则 Dataset 读取失败。

### 8.3 GPU / 驱动 / CUDA

- A：RTX 4090、驱动 535.230.02、CUDA runtime 12.1、torch cu121。B 可能完全不同。
- `torch==2.2.1+cu121` 的 CUDA 运行时是 wheel 自带的；系统 nvcc 版本不决定运行，驱动版本决定兼容性。CUDA 12.1 需要驱动 ≥ 525.60.13（Linux）。
- 如果 B 是 H100/A100，cu121 + sm_90/sm_80 通常可用；如果是 RTX 5090/B200（sm_100/120），2.2.1 可能没有对应 kernel，需要更新 torch。
- 不要用 A 的 `nvcc` 版本去推断 B 的运行时；以 `torch.version.cuda` + `torch.cuda.is_available()` 为准。
- A 当前容器 `nvidia-smi` 失败的原因未确认；如果 B 上 `nvidia-smi` 正常但 `torch.cuda.is_available()` 为 False，先排查驱动/容器 GPU 透传，而不是改代码。

### 8.4 TinyViT 与 SAM 的加载

- `BoxPromptIML()` 的 `tiny_vit_5m_224(pretrained=True)` 依赖 torch hub 缓存；cache 不在项目内，B 上必须按 §7.3 建软链或放入真实文件。若 B 无外网且 cache 缺失，`train.py`/`eval.py` 会在构造模型时失败。
- `fecd/model_loading.py` 是离线安全构造的参考实现：它用 `pretrained=False` 构造，再从 `/home/weights/tiny_vit_5m_22kto1k_distill.pth` 严格加载 296 个 tensor；4F1 预检记录 missing/unexpected keys 均为空，无网络下载。FECD 后续训练应优先用它。
- SAM 包当时从 `/hy-tmp/projects/segment-anything-main` 安装；B 上要重新安装（本地源码或 GitHub）。`segment_anything` 的版本必须能 `from segment_anything import SamPredictor, sam_model_registry`。
- SAM 权重 2.4G，不要用 `torch.load` 反复加载到 CPU 再复制；按 `sam_model_registry['vit_h'](checkpoint=...)` 的用法即可。

### 8.5 依赖版本

- `numpy==1.26.4`：不要升级到 2.x；albumentations 1.3.1 / opencv 4.9 / 旧 torch 生态对 numpy 2 的兼容性不稳。
- `albumentations==1.3.1`：与 `opencv-python-headless==4.9.0.80` 搭配；不要只装 `opencv-python` 造成重复/冲突。
- `timm==0.9.12`：TinyViT 代码会注册模型名并打印 “Overwriting ... in registry” 警告；这是正常的，不是错误。
- `thop` 的发行版版本字符串是 `0.1.1-2209072238`，pip freeze 显示 `0.1.1.post2209072238`；两者是同一个包，不要重复安装。
- 如果 B 的平台镜像自带 torch，先确认其版本/是否 CPU-only；不要直接 `pip install -r` 把平台 torch 覆盖成 A 的版本而不记录。
- 新阶段 3–5A 只使用已有依赖（numpy/PIL/torch/sklearn/scipy/opencv 等），没有新增第三方包；`pip-requirements.txt` 仍是可用清单。

### 8.6 训练随机性

- 原始 `train.py` **没有设置 `random.seed` / `np.random.seed` / `torch.manual_seed`**；`mask_to_boxes` 使用 Python `random`，数据增强使用 albumentations 的随机数。正式 run 的权重不能通过“同命令重跑”逐位复现。
- FECD 的 box 导出使用固定 seed `20261023` 和每个样本的 identity-derived seed，`boxes_train.json` 可复现；4F1 使用 seed=20261029、固定几何增强/顺序/初始化；但这些仍只覆盖一个训练 seed。
- 如果 B 上要从 checkpoint 继续，GPU 非确定性（cudnn benchmark/atomic）也会带来差异；对比实验应固定 seed 并报告。

### 8.7 Checkpoint 续训限制

- 旧 `checkpoint-{1..20}.pth` 只有 `model.state_dict()`；没有 optimizer/scheduler/epoch/RNG。`train.py --load` 只能恢复模型参数，不能恢复训练轨迹。
- 4F1 `final.pt`/`last.pt` 是完整 checkpoint（model/optimizer/memory/RNG/position），但 4F1 runner 固定 20 epoch，不能直接用它“自动再跑 20 epoch”；严格续训要新协议/新入口，并明确记录。
- `train.py` 在 `KeyboardInterrupt` 时会保存 `checkpoint-{epoch+1}.pth`；这也不是完整断点。
- 如果目标是“从已有 checkpoint 继续实验”：
  1. 只做推理/评估：用 `evaluate_student_strict` / 旧 `eval.py`；
  2. 做新实验：把 4F1 权重当初始化，使用新的 run 目录和明确的 epoch/lr 配置，并在报告中说明“不是严格续训”；
  3. 需要严格续训时，在 FECD 独立训练入口保存/恢复 optimizer/scheduler/epoch/RNG（后续代码工作，不要改原始 `train.py` 的核心逻辑）。

### 8.8 评估口径

- 旧 `eval.py` 对每个数据集做逐图 min-max，再保存 0–255 的概率图；AUC 用该量化图计算。FECD 严格评估要求 **raw sigmoid probability**，因此 `eval.py` 的结果只能作为原始流水线参考，不能替代 FECD 的严格评估。
- `f1_in` 是四个数据集 F1 的**未加权算术平均**；不是按像素/样本加权。
- NIST16 本地重建 split 有同源 base 跨 train/test；NC16 F1 0.2801 不能与论文表直接比较。
- 旧评估保持 `expected_counts = {NC16:150, C1:920, Columbia:50, Coverage:30}`；如果 B 上数据目录多了/少了文件，`validate_dataset_layout` 会直接报错。
- 4F1 严格评估使用 530 张 `new_dev`、raw sigmoid、0.5；不要把 4F1 的 530 图指标与旧 `eval.py` 的 1,150 图指标拼表。
- 4F1 原始 `status.json` 是 **FAILED**；4F1-V 独立复验才是 **PASS（engineering）**，科研结论仍是 **NO CLEAR GAIN**。引用时必须同时说明两者。

### 8.9 权限、磁盘与平台差异

- `fecd/assets/stage2c_cache/**/*.npz` 的权限是 `600`（owner root）；4F1 `final.pt` 是 `644`（root）。如果 B 用非 root 用户，需要 `chown -R <user>:<group>` 或 `chmod`；否则 Dataset 读取会失败。
- A 的 `/home` 在 overlay 系统盘（30G，已用 22G，仅剩 8.9G）；B 如果系统盘更小，务必把 datasets/weights/runs/4F1 runs 放到数据盘，用软链映射；不要把 21GB 全塞进系统盘。
- `/hy-tmp` 在 A 上是 50G 数据盘，当前只有 2.2G 运行产物；平台可能在实例释放/迁移时清空。B 上要确认其持久性；不要把唯一副本放临时盘。
- 如果 B 的平台上传播放器不支持软链/文件名大小写，先在小样本上验证 `train/Tp` 的软链和 `CASIAv1/CASIA1.0/Modified Tp`（含空格）路径。
- CASIAv1 原始目录名含空格（`Modified Tp`），所有 shell 命令要加引号；Python manifest 路径已按原样保存。
- 4F1 预检要求系统盘 ≥5GiB、数据盘 ≥8GiB 空闲；如果 B 的空间接近上限，训练会在预检被拒绝，不会静默写到系统盘。

### 8.10 安全

- **不要迁移/上传**：`/root/.ssh`、`/root/.codex/auth.json`、`/root/.hycloud_ossutil_config`、`/root/.npmrc`、`/root/.vscode-server`、任何 token/密码/私钥。
- 数据集（NIST16、CASIA 等）有各自许可，不要上传到公开 GitHub；`fecd/assets` 的 cache、stage3 evidence、4F1 checkpoint、stage5a maps 也属于研究中间产物，建议私有传输。
- 原仓库 remote 是公开的；本次没有 push。B 上如果要把个人代码推到公开仓库，先确认没有数据路径/隐私信息/大文件。
- `stage3_evidence`、`stage4f1`、`stage5a` 的报告/日志可能包含本机绝对路径；上传前检查是否要去掉 `/home/...`、`/hy-tmp/...` 路径。

---

## 9. 已确认 vs 待确认

### 9.1 已确认（有实际命令/文件依据，2026-11-02 复核）

- 项目根、Git HEAD、分支、远端、已跟踪修改（`eval.py` +142/−51、`train.py` 1 行）、未跟踪目录。
- conda 环境 `boxpromptiml`、Python 3.10.22、torch 2.2.1+cu121、torchvision 0.17.1+cu121、CUDA 12.1、驱动 535.230.02、关键依赖版本、`pip check` 通过；环境自 2026-10-27 未变。
- `/proc/driver/nvidia` 显示 RTX 4090；当前 `nvidia-smi` 无设备、`torch.cuda.is_available()==False`。
- `/home/datasets` 9,163,709,381 B；raw 9,093,879,098 B；processed 69,632,888 B；30,787 个 dataset 软链在 `/hy-tmp/datasets → /home/datasets` 映射下全部可解析；manifest 5,737 + 230 条 image/mask 全部存在。
- 权重大小/SHA256；旧 20 个 checkpoint；checkpoint-20 SHA；boxes SHA；K1/K8 cache 各 5,737、审计 passed、config hash。
- FECD 阶段 1–2C、3、3R、3R-A、3R-B0、4P、4F0、4F1、5A 产物和报告；4F1 四组 final.pt SHA；stage5a validation PASS；197 个测试通过；9 个旧回归通过。
- 4F1 原始 `status.json` 为 FAILED（validation），4F1-V `validation_v2.json` 为 PASS(engineering)/NO CLEAR GAIN；5A 为 PASS/NO CLEAR GAIN。
- `/hy-tmp/BoxPromtIML` 2,273,429,263 B；4F1 formal run 1,078,402,850 B；stage5a 1,190,146,305 B。
- 无运行中的训练/tmux 会话；只有 jupyter-lab。

### 9.2 待确认（不要凭空假设）

- **A 当前 GPU 不可见的原因**：`nvidia-smi` 报无设备，但 `/proc` 有 RTX 4090；可能是容器 GPU 透传/驱动/平台调度问题，未确认。
- **B 的 GPU 型号、驱动、CUDA 兼容性**：必须现场 `nvidia-smi` + `torch` 验证；不能假设可以复制 A 的 cu121 环境。
- **B 的 `/hy-tmp` 是否持久**、B 的系统盘/数据盘大小、是否有独立数据盘（4F1 训练硬要求）。
- **B 的上传/下载是否保留软链和文件名大小写**。
- **原始正式训练 run 的完整命令行**：shell history 未完整记录；日志确认了 5,737 样本/batch 16/save path，但 `--train_root` 的精确写法（paper_scale 路径）是按数据布局推断的。
- **B 是否有外网**（TinyViT 下载、pip、conda 镜像）；如果无外网，所有依赖和权重必须预先传好。
- **B 上 torch 版本是否能覆盖新 GPU 架构**（尤其 RTX 5090/B200）。
- **B 的用户/权限**是否可读 cache 的 600 文件和 `/hy-tmp` 软链。
- **5A 报告建议的下一轮实验尚未授权/实现**；不要把建议当成已有代码或已验证结论。

---

## 10. 关闭 A 服务器前的检查清单

- [ ] 重新确认没有训练/评估任务：`ps -eo pid,etime,args | grep -E 'train.py|eval.py|run_stage4f1|stage5a'`、`tmux ls`。本次采集时没有发现。
- [ ] 把 `/home`（尤其 `datasets`、`weights`、`runs`、`projects/BoxPromtIML`）完整备份到 OSS/持久存储；**保留软链**（`tar`/`rsync -a`）。`datasets` 里不要漏掉 `patches/`、`scripts/`、`boxprompt-smoke/`（`fecd` smoke 测试要用）。
- [ ] **不要漏掉 `/hy-tmp/BoxPromtIML/fecd_runs`**：4F1 正式 checkpoint/评估和 stage5a 诊断都在这里；该目录不在 `/home`，旧版清单没有覆盖。
- [ ] 确认备份包含 `fecd/assets` 的 **stage3_evidence 4.15GB**、stage2c cache 643MB、stage4p/4f0/4f1 历史 checkpoint、stage4f1_v；不要只备份代码和 stage2c cache。
- [ ] 用 `migration/checksums.sha256` 校验权重/旧 checkpoint/boxes/bundle/stage3/4 关键产物；用 `migration/checksums-hy-tmp.sha256` 校验 4F1 四组 final.pt/status 和 stage5a 关键文件；至少校验 SAM、TinyViT、旧 checkpoint-20、boxes、4F1 四组 final.pt。
- [ ] 确认 `/root/.hycloud_ossutil_config`、`.ssh`、`.codex/auth.json` 等凭据**不在**备份/传输包里。
- [ ] 在 B 上按 §7 恢复，跑通 `verify_paths.py`、197 个测试、旧 `eval.py`，并与 A 的指标对比后，再考虑释放 A。
- [ ] 在 B 上记录新的 GPU/驱动/torch 版本和实际命令，追加到 B 的笔记；不要静默改变 batch size/数据 split 后还拿旧指标对比。
- [ ] 如果 B 的 `/hy-tmp` 是临时盘，把数据副本放在持久存储，并明确恢复步骤；不要把唯一副本留在 `/hy-tmp`。
- [ ] 不要把 `MIGRATION.md` 里的路径当成 B 的真实路径；先按 §7.0 确认，再替换。
- [ ] **不要为了“清理”删除 `fecd/assets/stage3_evidence`、4F1 run 或 stage5a maps**；它们是不可廉价重建的研究证据（重建分别需要全量证据生成/4 组 20 epoch 训练/CPU 诊断）。

---

## 11. GitHub 私有仓库与 AutoDL 工作流

### 11.1 文件边界

适合提交到 GitHub 的内容：

- 原论文代码：`train.py`、`eval.py`、`boxpromptiml.py`、`utils/`、`Figures/`、`README.md`、`LICENSE`、`BoxPromptIML.yml`。
- 个人代码：`fecd/` 下的 `*.py`、`configs/`、`tests/`、`docs/`、`README.md`；**不包含 `fecd/assets/`**。
- vendored 源码：`TinyViT/`（不含 `__pycache__`、`.bak`）。
- 迁移资料：`MIGRATION.md`、`migration/` 下的文本/脚本/环境文件。
- `.gitignore` 本身。

不要提交、需要单独传输的内容：

- `/home/datasets`（数据）、`/home/weights`（SAM/TinyViT 权重）。
- `/home/runs`、`/home/checkpoints`（旧 checkpoint/日志）。
- `fecd/assets/`（boxes、K1/K8 cache、stage3 evidence、4P/4F0/4F1 checkpoint、逐图诊断等）。
- `/hy-tmp/BoxPromtIML/fecd_runs/`（4F1 正式运行、5A 诊断）。
- `segment-anything/` 权重软链、`migration/BoxPromtIML-git.bundle`（5MB 冗余二进制）。
- 任何 `.env`、Token、SSH key、云厂商配置、`auth.json`。

### 11.2 首次提交与推送（需确认后执行）

**不要**向 `origin`（原作者的公开仓库）推送。保留 `origin` 仅用于查看/同步原论文历史；新私有仓库使用独立 remote 名称，例如 `private`。本次没有写入 `private` remote（新仓库 URL 尚未提供）；确认 URL 后执行：

```bash
cd /home/projects/BoxPromtIML

# 1) 新私有仓库先建好（GitHub 网页操作；必须选 Private）
#    然后把下面占位 URL 换成你的仓库（SSH 示例）：
git remote add private git@github.com:<YOUR_ACCOUNT>/<YOUR_PRIVATE_REPO>.git
git remote -v                 # 应看到 origin 和 private 两个 remote

# 2) 检查待提交内容（建议先 dry-run）
git add -n .
git status --short

# 3) 将生成的 __pycache__ 从索引移除（不删除工作区文件；本次已执行）
git rm --cached utils/__pycache__/*.pyc 2>/dev/null || true

# 4) 暂存全部非忽略内容
git add -A
git diff --cached --stat      # 预计约 280 个文件、2–3MB，不含 assets/data/weights
git diff --cached --name-only --diff-filter=ACMR | grep -E 'fecd/assets|\.pth$|\.pt$|\.npz$|^runs/|^weights/' && echo 'STOP: 大文件被暂存' || true

# 5) 提交（提交信息可按需修改；本次未执行）
git commit -m "Add FECD research code, tests, docs and migration notes"

# 6) 只有你确认后，才推送；不要加 --force
git push -u private main
```

说明：

- 当前 `main` 仍跟踪 `origin/main`；上述 `git push -u private main` 会把 upstream 改成 `private/main`，这是新工作流的预期结果。若想保留 `origin/main` 作为上游，只执行 `git push private main`，不要 `-u`。
- 第一次推送包含原仓库的 3 个提交 + 新增 FECD 提交；这是完整历史，不是把原仓库内容覆盖。
- 如果 GitHub 提示文件超过 100MB，先停止：说明 `.gitignore` 没有覆盖某个大文件，用 `git diff --cached --name-only` 检查，不要用 LFS 盲目提交数据。
- 不要把 `fecd/assets` 加入 Git LFS 作为默认方案；数据/权重/checkpoint 用 OSS/网盘/`scp`/`rsync` 单独传，GitHub 只管理代码。

### 11.3 AutoDL 上克隆与恢复

AutoDL 常见的持久数据盘是 `/root/autodl-tmp`（以实例实际挂载为准）。代码可以克隆到与原路径一致的位置，数据放持久盘：

```bash
# 1) 克隆私有代码（只含代码，不含 assets/data/weights）
mkdir -p /home/projects
git clone git@github.com:<YOUR_ACCOUNT>/<YOUR_PRIVATE_REPO>.git /home/projects/BoxPromtIML
cd /home/projects/BoxPromtIML
git log --oneline -5
git remote -v

# 2) 单独恢复大数据到 /root/autodl-tmp（示例；用你的传输方式）
#    /root/autodl-tmp/datasets
#    /root/autodl-tmp/weights
#    /root/autodl-tmp/runs
#    /root/autodl-tmp/fecd_assets        -> 对应项目内 fecd/assets
#    /root/autodl-tmp/fecd_runs          -> 对应 /hy-tmp/BoxPromtIML/fecd_runs

# 3) 建立代码期望的路径别名
ln -sfn /root/autodl-tmp/datasets /home/datasets
ln -sfn /root/autodl-tmp/weights  /home/weights
ln -sfn /root/autodl-tmp/runs     /home/runs
ln -sfn /root/autodl-tmp/fecd_assets /home/projects/BoxPromtIML/fecd/assets

# 4) /hy-tmp：4F1 训练入口要求真实路径且独立于系统盘。
#    仅把 /hy-tmp 做成指向 /root/autodl-tmp 的软链会被 f1_storage.py 拒绝。
#    推荐 bind mount（需要 root；实例重启后需重新挂载）：
mkdir -p /hy-tmp /root/autodl-tmp/BoxPromtIML/fecd_runs
mount --bind /root/autodl-tmp /hy-tmp
# 或者：直接把 4F1 输出放在 /hy-tmp 的真实数据盘目录，不要用软链。
# 然后确认：
ls -ld /hy-tmp /hy-tmp/BoxPromtIML/fecd_runs
```

如果 AutoDL 实例不允许 bind mount，或者你不想保留 `/hy-tmp` 路径，需要把 `eval.py` 的 `/hy-tmp/runs`、`/hy-tmp/datasets`，`fecd/configs/stage4f1/storage.json`、`fecd/diagnostics/stage5a_io.py` 等硬编码路径改为 AutoDL 实际路径；这是平台适配，不要把它伪装成代码逻辑变化。

恢复后按 §7.5 跑 `verify_paths.py`、197 个测试和 `fecd.tools.audit_stage2c_cache`；旧 `eval.py` 需要 `/hy-tmp` 两套别名都可用。

### 11.4 后续 FECD 开发

```bash
cd /home/projects/BoxPromtIML
git switch -c feat/<short-name>
# 修改 fecd/ 代码、tests/、docs/；不要提交 fecd/assets
git add fecd .gitignore MIGRATION.md
git commit -m "feat: ..."
git push -u private feat/<short-name>
```

- `origin` 保持只读参考；不要 `git push origin`。
- 每次实验的 `fecd/assets`、`/hy-tmp/BoxPromtIML/fecd_runs` 用外部存储备份；GitHub 只记录代码、配置和报告。
- 新实验使用新的 `run_id`/输出目录，不要覆盖 4F1 原始 `FAILED` 证据。

## 附录 A：关键路径速查

```text
项目根                     /home/projects/BoxPromtIML
旧正式 checkpoint 目录     /home/runs/boxprompt_formal_20261016_122447/checkpoints
旧最终 checkpoint          .../checkpoint-20.pth
旧正式 eval 结果           /home/runs/boxprompt_eval_20261019_121727_145312
SAM 权重                   /home/weights/sam_vit_h_4b8939.pth
TinyViT 权重               /home/weights/tiny_vit_5m_22kto1k_distill.pth
原始数据                   /home/datasets/raw
正式 split                 /home/datasets/processed/boxprompt_paper_scale_20261016
旧 preflight split         /home/datasets/processed/boxprompt_preflight_20261016
FECD boxes                 /home/projects/BoxPromtIML/fecd/assets/boxes_train.json
FECD K1 cache              .../fecd/assets/stage2c_cache/k1/d7ccb769414f
FECD K8 cache              .../fecd/assets/stage2c_cache/k8/2bff688546a7
FECD 阶段 3 证据           .../fecd/assets/stage3_evidence/raw
FECD 阶段 3R-A split       .../fecd/assets/stage3ra_split_manifest.json
FECD 4F1 正式运行          /hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a
FECD 4F1 最终权重          .../artifacts/{P0,P1,P2F,P3F}/final.pt
FECD 4F1-V 复验            /home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2
FECD 阶段 5A 诊断          /hy-tmp/BoxPromtIML/fecd_runs/stage5a/diagnosis_20261030
FECD 任务说明              /home/note.md
FECD 研究设计              /home/deep-research-report.md
迁移资料                   /home/projects/BoxPromtIML/migration
```

## 附录 B：B 上最小验收命令

```bash
cd /home/projects/BoxPromtIML
ENV=/usr/local/miniconda3/envs/boxpromptiml

$ENV/bin/python migration/verify_paths.py
$ENV/bin/python migration/verify_paths.py --hash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -m unittest discover -s fecd/tests -p 'test_*.py'
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $ENV/bin/python -m fecd.tests.run_stage2b_tests
$ENV/bin/python -m fecd.tools.audit_stage2c_cache

# 有 GPU 时：
CUDA_VISIBLE_DEVICES=0 $ENV/bin/python eval.py
```

全部通过后，再把 B 平台的新指标与 §4.8 的参考值对比；任何显著偏差先查路径/数据/版本，不要先怀疑模型。

## 附录 C：本次复核记录（2026-11-02 14:45 CST）

- Git：`main` @ `2b5b7c8`；`eval.py` +142/−51、`train.py` 1/1；未提交、未推送；`git bundle verify` 通过；`code-local-changes.patch` 与工作树一致。
- 环境：Python 3.10.22、torch 2.2.1+cu121、torchvision 0.17.1+cu121、CUDA 12.1、驱动 535.230.02、`pip check` 通过；与 2026-10-27 一致；GPU 仍不可见（`nvidia-smi` 无设备，`torch.cuda.is_available()==False`）。
- 数据：`/home/datasets` 9,163,709,381 B；raw 9,093,879,098 B；processed 69,632,888 B；30,787 个链接映射后 0 失败；manifest 5,737 + 230 条 image/mask 0 缺失。
- FECD 资产：`fecd/assets` 6,432,406,786 B；stage2c cache K1/K8 各 5,737；stage3 evidence 5,737 NPZ / 4.15GB；stage4p/4f0/4f1 历史 checkpoint；4F1 四组 final.pt 在 `/hy-tmp`；stage5a 1.19GB。
- 测试：197 tests OK（CPU，21.7s）；`run_stage2b_tests` 9 checks passed。
- 4F1：原始 status FAILED；4F1-V validation PASS(engineering)/NO CLEAR GAIN；四组 F1 0.345859/0.336740/0.346021/0.342369；主对照 CI 均跨零。
- 5A：validation PASS / NO CLEAR GAIN；530 张 new_dev、4 个 checkpoint、2,120 个概率图；未训练。
- 现状：没有 `train.py`/`eval.py`/4F1/stage5a 进程；`tmux` 无会话；只有 jupyter-lab；`/hy-tmp` 仅含 `BoxPromtIML/fecd_runs`。
- 未改动：没有修改核心代码、没有删除文件、没有安装依赖、没有上传/下载数据、没有中断任何任务；本次只更新了 `MIGRATION.md` 和 `migration/` 下的迁移资料。
