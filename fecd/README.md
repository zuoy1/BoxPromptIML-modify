# FECD — Forensic Evidence-Calibrated Distillation

这是 BoxPromptIML 之上新增的个人研究代码，目标是用取证证据和教师可靠性改进弱 Box 监督的伪标签蒸馏。原始论文代码仍保留在仓库根目录；本目录不覆盖原始训练入口。

## 当前状态（2026-11-02）

- 阶段 1–2C：数据/坐标协议、SAM 多提示教师、K=1/K=8 cache、Dataset 与测试已完成。
- 阶段 3：取证证据 + 教师可靠性诊断，工程完成，研究判定 **NO-GO**。
- 阶段 3R / 3R-A / 3R-B0：失败复盘、q_A 开发诊断、参考区域预检，均已 **STOP**。
- 阶段 4P / 4F0：受控学生蒸馏 smoke 与单 epoch 预检，工程 PASS。
- 阶段 4F1：P0/P1/P2F/P3F 四组各 20 epoch 正式训练；原始验收 `FAILED`（float32 loss 断言误报），独立 4F1-V 复验 `PASS (engineering)`，科研结论 **NO CLEAR GAIN**。
- 阶段 5A：530 张 new_dev 的 CPU 失败机制诊断，工程 PASS，科研 **NO CLEAR GAIN**，交付后停止。

不要把这个仓库描述成“FECD 已经有效”；当前结论是没有明确增益。

## 代码地图

```text
fecd/
├── coordinates.py / model_loading.py
├── confidence/          # c_sam 等可靠性启发式
├── evidence/            # Noise/JPEG/FFT 取证证据与归一化
├── forensic_reference/  # 参考区域条件化
├── diagnostics/         # 阶段 3/3R-A/5A 诊断、风险-覆盖、bootstrap
├── training/            # 4P/4F0/4F1 训练、数据、loss、checkpoint、storage
├── evaluation/          # 严格 new_dev 评估、allowlist、指标
├── execution/           # 4F1 执行器、锁、原子状态、关机证据
├── pseudo/              # SAM 多提示、K1/K8 cache
├── datasets/            # FECD Dataset（不读 GT）
├── tools/               # 各阶段入口/验收/导出
├── tests/               # 197 个 CPU 可跑测试
├── configs/             # 冻结配置
├── docs/                # 阶段报告
└── assets/              # 不提交到 GitHub；单独传输
```

`fecd/assets/` 包含 Box 导出、K1/K8 cache、阶段 3 证据、checkpoint、逐图诊断等大文件，已在 `.gitignore` 中排除。克隆本仓库只得到代码、配置、测试和文档；运行实验前必须按根目录 `MIGRATION.md` §6 单独恢复 assets。

## 环境

推荐使用根目录 `migration/install-env.sh` 重建环境：

```bash
bash migration/install-env.sh
# 或按 migration/pip-requirements.txt + 与 B 平台匹配的 torch 安装
```

A 服务器参考环境：Python 3.10.22、torch 2.2.1+cu121、torchvision 0.17.1+cu121、numpy 1.26.4。不要升级 numpy 到 2.x。

## 测试

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s fecd/tests -p 'test_*.py'    # 197 tests
```

## 关键文档

- `docs/stage2b_report.md` / `stage2c_report.md`：cache 与数据协议。
- `docs/stage3_report.md`、`stage3r_*`、`stage3ra_report.md`、`stage3rb0_*`：阶段 3 及失败复盘。
- `docs/stage4p_report.md`、`stage4f0_report.md`：工程预检。
- `docs/stage4f1_training_report_20261030.md`、`stage4f1_revalidation_report.md`：正式 20 epoch 与独立复验。
- `docs/stage5a_failure_diagnosis.md`：5A 机制诊断与下一轮建议。
