# 恒源云数据盘清空后的项目恢复手册

核查日期：2026-11-11。项目：BoxPromptIML / FECD。

项目目录：`/home/projects/BoxPromtIML`。项目 Python：`/usr/local/miniconda3/envs/boxpromptiml/bin/python`。

本文整理本次实际检查结果及后续恢复步骤。**本次仅进行了检查并编写本文，没有修复软链接、恢复数据盘文件、安装依赖或启动训练。** 下文恢复命令均为后续操作指引；未实际运行的命令会明确标注。

## 1. 恢复结论与优先级

**不需要从阶段 1 全部重新开始。** 项目主体代码、数据本体、预训练权重、SAM 缓存、阶段 3 资产、4P/4F0 checkpoint 和最新研究报告仍在系统盘 `/home`。

目前损失集中在数据盘 `/hy-tmp/BoxPromtIML/fecd_runs`：4F1 正式运行、5A 诊断、5C0/5C1 运行产物，以及部分保存在运行目录内的代码和预检文件。数据集和权重的历史路径别名也已消失，导致大量文件看起来无法访问。

按以下顺序处理：

1. 向恒源云确认旧数据目录、原宿主机、快照或备份是否还可恢复。
2. 将目前仍在 `/home` 的数据和代码备份到经确认的持久存储或本地电脑，包含未提交的 5C 代码。
3. 明确不再等待原盘恢复，或已完成原盘保护后，再修复 `/hy-tmp` 路径别名。
4. 校验已有文件、缓存、数据隔离和运行环境，修复缺失的运行依赖。
5. 恢复 GPU 可见性。
6. 根据需要恢复的证据范围，只重跑缺失的实验；重跑前先解决历史路径与哈希绑定。
7. 每次新实验完成后立即备份并在异地复核，最后再停止实例。

**“找回历史文件”和“重新实验”是两条不同路径。** 重新实验得到的是新产物，必须保留新的运行身份与校验记录，不能冒充原始文件或沿用旧 PASS 证据。

## 2. 本次检查确认了什么

### 2.1 磁盘与挂载

| 路径 | 实际情况 | 恢复含义 |
|---|---|---|
| `/` | overlay 系统盘，总配额约 30GiB；后续复核剩余约 6.9GiB | 不足以再放一份全部数据归档；使用量会随会话变化 |
| `/home` | 位于系统盘，项目和数据仍可见 | 当前主要恢复来源；并非已验证的异地备份 |
| `/hy-tmp` | 独立 XFS 挂载，当前配额约 50GiB，目录为空 | 当前挂载正常，不能简单解释为“盘未挂载” |
| `/hy-netdisk` | 当前没有独立挂载，解析到系统盘 overlay | 不能仅凭名字把这里当成持久网盘或异地备份 |
| `/hy-model` | 挂载可见但当前目录为空 | 未发现本项目备份；宿主机共享磁盘占用不代表本实例数据仍在 |

`/hy-tmp` 当前源路径为 `/dev/nvme0n1[/local/I2be5a164b603f01fcb]`。容器没有可直接操作的对应 NVMe 块设备节点，因此不能在容器内据此直接进行原盘文件恢复。

用户反馈为停机三天后数据消失。本次检查能确认“当前数据目录为空”，**不能证明平台删除时间、停机保留规则、宿主机是否发生切换，或底层是否永久不可恢复**；这些需要平台确认。

### 2.2 仍然存在的资产

下表大小为普通文件逻辑大小，不解引用软链接；GB 按十进制近似，不能直接当作系统盘配额占用。

| 资产 | 位置 | 本次检查结果 |
|---|---|---|
| 主项目及 Git | `/home/projects/BoxPromtIML` | 存在；HEAD 为 `5c542e2`，2026-11-02 提交 |
| FECD 阶段报告 | `fecd/docs/` | 阶段 1 至 5C1 的现存报告仍可读 |
| 原始数据 | `/home/datasets/raw` | 23,827 个普通文件，约 9.09GB；文件数包含说明与归档等辅助文件 |
| 数据划分与清单 | `/home/datasets/processed` | 清单、派生文件和软链接仍在；软链接目标前缀失效 |
| SAM/TinyViT 权重 | `/home/weights` | 2 个权重文件，合计约 2.59GB |
| 原始 BoxPromptIML checkpoint | `/home/runs/boxprompt_formal_20261016_122447/checkpoints` | `checkpoint-1.pth` 至 `checkpoint-20.pth` 均在 |
| 旧训练/评估目录 | `/home/runs` | 约 0.60GB，含原始 checkpoint 和旧评估结果 |
| FECD 资产总目录 | `fecd/assets` | 约 6.43GB，主体仍在 |
| 冻结训练 boxes | `fecd/assets/boxes_train.json` | 5,737 条记录 |
| K1 SAM 缓存 | `fecd/assets/stage2c_cache/k1/d7ccb769414f` | 5,737 份 NPZ，配置和 Dataset 索引存在 |
| K8 SAM 缓存 | `fecd/assets/stage2c_cache/k8/2bff688546a7` | 5,737 份 NPZ，配置和 Dataset 索引存在 |
| 阶段 3 证据缓存 | `fecd/assets/stage3_evidence` | 5,737 份 NPZ，目录约 4.15GB，历史缓存审计文件仍在 |
| 派生 split | `fecd/assets/stage3ra_split_manifest.json` | 历史审计记录 student_train/new_dev/new_confirm 为 3,530/530/530 |
| 4P checkpoint | `fecd/assets/stage4p/{P0,P1,P2,P3}/final.pt` | 四组均在，大小与记录相符 |
| 4F0 checkpoint | `fecd/assets/stage4f0/{P0,P1,P2F,P3F}/epoch1.pt` | 四组均在，大小与记录相符 |
| 4F1 早期局部 checkpoint | `fecd/assets/stage4f1/P0/last.pt` | 存在，但不是正式四组 20 epoch 的 final |
| 4F1 独立复验文件 | `fecd/assets/stage4f1_v/revalidation_20261030_v2` | 存在；记录 engineering PASS、research NO CLEAR GAIN |
| 迁移资料 | `migration/`、`MIGRATION.md` | 环境清单、校验和、Git bundle、恢复检查脚本仍在 |

### 2.3 失效软链接与真实文件丢失的区别

原有路径示例：

```text
/home/datasets/processed/.../train/Tp/某样本
  -> /hy-tmp/datasets/processed/.../pool/.../某样本
  -> /hy-tmp/datasets/raw/.../某样本
```

真实文件仍在 `/home/datasets/raw`，但 `/hy-tmp/datasets` 已不存在。因此，原来可用的多级链接全部或部分失效。

**本次逐跳模拟检查：30,787 个数据集软链接，按 `/hy-tmp/datasets` → `/home/datasets` 映射后，找不到文件本体的数量为 0。** 模拟检查没有实际创建链接。

权重也存在类似情况：仓库内 SAM 链接、torch hub 中的 TinyViT 链接仍指向 `/hy-tmp/weights`。这两种权重的本体仍在 `/home/weights`。

### 2.4 Git 未提交内容

本次 `git status --short` 显示 5C0/5C1 多项文件为 `??`，包含：

- `fecd/configs/stage5c0/`、`fecd/configs/stage5c1/`。
- `fecd/docs/stage5c0_report.md`、`fecd/docs/stage5c1_report.md`。
- 区域选择、训练、恢复、诊断和统计相关的 `region_*.py`。
- `train_region_smoke.py`、`preflight_region_formal.py`、`train_region_formal.py`。
- 对应的 5C 测试文件。

**Git 仓库存在，不等于这些文件已进入版本历史或远端备份。** 仅重新 clone 仓库、仅保存已有 Git bundle，不能保证恢复这些现存未提交文件。应完整备份当前工作目录；不要执行 `git clean`、强制 checkout 或重置。

## 3. 当前不可见的文件与额外阻塞

### 3.1 数据盘运行产物

| 阶段 | 历史位置 | 目前缺失的主要内容 |
|---|---|---|
| 4F1 正式训练 | `/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a` | 四组 final checkpoint、训练/评估明细、原执行器状态、统计与审计原始文件 |
| 5A | `/hy-tmp/BoxPromtIML/fecd_runs/stage5a/diagnosis_20261030` | 保存的概率图、逐图诊断、案例及最终校验产物 |
| 5B 依赖 | `/hy-tmp/BoxPromtIML/fecd_runs/stage5b` | 至少包括 5C1 引用的 `results/training_input_hashes.json`；没有足够现存原始材料宣称 5B 全部可审计 |
| 5C0 | `/hy-tmp/BoxPromtIML/fecd_runs/stage5c0` | 恢复探针、单 epoch checkpoint、选择 mask、运行日志、`results/source_hashes.json` 及数据盘内文档/工具 |
| 5C1 | `/hy-tmp/BoxPromtIML/fecd_runs/stage5c1` | R0/R1 final、40 轮 checkpoint、选择记录、严格评估、统计、审计与冻结清单 |

本次在检查过的本地路径中未找到这些正式 final 的替代副本。**未访问云控制台、OSS 对象列表或其他电脑，不能据此断言没有任何外部备份。**

### 3.2 5C1 有部分代码与运行清单也丢了

现存测试 `fecd/tests/test_stage5c1_evaluation_wrapper.py` 会直接加载：

```text
/hy-tmp/BoxPromtIML/fecd_runs/stage5c1/code/evaluate.py
```

该文件目前不存在。因此不能宣称“所有代码都在”，也不能预计完整测试发现流程会直接通过。

现存 `preflight_region_formal.py` 和正式 runner 还依赖：

- `stage5c0/results/source_hashes.json`。
- `stage5b/results/training_input_hashes.json`。
- `stage5c1/results/runtime_frozen.json`。
- 正式预检生成的 `stage5c1/results/preflight.json`、`candidates.npz` 等。

这些属于原先的身份核对与执行门禁。**不能创建空 JSON、删除验证或把新生成的清单冒充历史清单来绕过。** 如果不能从备份找回，需要一次独立、可审查的恢复修复：对现存数据重新核验，建立明确标注日期的新基线，并保留“历史原件不可恢复”的说明。

### 3.3 GPU 当前不可用

本次实际结果：

```text
nvidia-smi: No devices were found
Python: 3.10.22
PyTorch: 2.2.1+cu121
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

依赖导入检查通过：torch、torchvision、cv2、timm、albumentations、sklearn、tensorboardX、thop、segment_anything。

这说明项目环境主体仍在，但不代表具备 CUDA 训练条件。5C1 现存预检还要求已验证的 RTX 4090；换卡时必须重新核对硬件门禁和复现条件，不能简单删除检查。

## 4. 实际执行过的检查与证据边界

以下检查已在本次会话执行：

| 检查 | 实际结果 |
|---|---|
| 磁盘、挂载、目录检查 | `/hy-tmp` 已挂载且为空；`/hy-netdisk` 当前不是独立网盘挂载 |
| Git 状态与历史 | 主仓库存在；5C 多项内容尚未提交 |
| 阶段报告、迁移文档、恢复脚本检查 | 已读，最新研究状态见第 10 节 |
| 路径恢复检查脚本 | 退出码 1，`RESULT: FAIL (24 checks failed)` |
| 数据软链接虚拟映射检查 | 30,787 条，映射后 0 个本体缺失 |
| `/home` 迁移清单 SHA256 | 42 项匹配，0 项缺失，0 项不匹配 |
| 项目环境依赖 import | 上述依赖通过 |
| GPU 检测 | 0 张可用 GPU |

路径检查实际命令：

```bash
cd /home/projects/BoxPromtIML
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python migration/verify_paths.py --fecd-runs /hy-tmp/BoxPromtIML/fecd_runs
```

24 项失败包括多个针对同一根因的断言，例如“别名存在”和“别名指向相同数据”，**不表示 24 组独立数据全部丢失**。主要根因为历史别名消失、数据软链失效，以及 4F1 正式运行缺失。5A 缺失在该工具中主要作为 warning；工具没有完整覆盖后来新增的 5C 依赖。

42 项 SHA256 覆盖迁移清单中列明的 SAM/TinyViT 权重、20 个原始 checkpoint、boxes、部分缓存元数据、4P/4F0 checkpoint、4F1 局部 checkpoint 和独立复验关键文件等。它**不等于**逐一解码了全部原始图像和 11,474 份 SAM 缓存，也不覆盖已丢失的 5C 原始产物。

本次未运行：完整训练、学生 forward、SAM forward、完整 unittest、全量缓存逐份加载审计、严格模型评估、平台快照恢复和外部备份检索。

## 5. 步骤一：向平台确认原始数据能否找回

**状态：未运行，需要用户通过恒源云控制台或客服处理。**

建议提供实例编号、最近一次正常使用时间、发现丢失时间，以及原数据路径 `/hy-tmp/BoxPromtIML/fecd_runs`，询问：

1. 该实例 `/hy-tmp` 的停机保留期限及实际清理记录是什么？
2. 实例是否换过宿主机或临时数据目录？原宿主机上的旧目录是否仍在？
3. 是否有可恢复快照、系统盘镜像、数据盘备份、回收站或平台自动归档？
4. 能否先保护原目录或做副本，再进行恢复，避免覆盖现有 `/home`？
5. 后续应采用哪一种存储产品，才能跨停机/迁移保留实验产物？

可向客服发送：

> 我的 GPU 实例停机后再次启动，`/hy-tmp` 已正常挂载但为空，原目录 `/hy-tmp/BoxPromtIML/fecd_runs` 中的实验 checkpoint、日志和部分代码消失。系统盘 `/home` 尚在。请协助核查临时盘清理记录、是否更换宿主机，以及原目录、快照或备份是否可恢复。请先保护旧数据，恢复到副本或新目录，避免覆盖现有系统盘。

如果平台仍在尝试底层恢复，先不要在 `/hy-tmp` 新建目录、下载数据或启动训练。当前容器的常规读操作不会证明底层数据不可恢复；不要自行对宿主机共享磁盘运行文件系统修复、格式化或扫描恢复工具。

## 6. 步骤二：备份现存数据和未提交代码

**状态：未运行。优先于修复链接和训练。**

### 6.1 备份范围

完整备份至少覆盖：

- `/home/projects/BoxPromtIML`：包括 `.git`、未提交代码、`fecd/assets`、阶段报告、迁移资料和本文。
- `/home/projects/segment-anything-main`：本地安装的 SAM 源码。
- `/home/datasets`：原始图像/GT、processed、划分清单、准备脚本和软链接。
- `/home/weights`：两个预训练权重。
- `/home/runs`、`/home/checkpoints`：旧训练和评估产物。
- `/home/note.md`、`deep-research-report.md`、`conversation_20261026.md`：研究与历史操作说明。

以上都在 `/home`，优先整体归档。当前逻辑体量约 19GB 量级，建议目标留足至少 30GiB，并以备份时的实际 `du` 和可用空间为准。环境重建资料已在 `migration/`；没有必要为本次恢复先重新安装整个环境。

### 6.2 选择真实备份目的地

可用目的地为：经确认的持久网盘、对象存储，或本地电脑/另一台服务器。

注意：

- 当前 `/hy-netdisk` 只是系统盘目录，不能直接把文件放进去就认为已经完成持久化。
- 当前 `/hy-tmp` 是本次丢失发生的位置，不能作为唯一备份。
- 系统盘目前只剩约 6.9GiB，不应在 `/home` 或 `/tmp` 再生成一份完整归档。
- 普通 Git 提交不包含被忽略的大型资产；仅备份 Git 也不够。

先检查目的地挂载与平台保留政策，再设置环境变量 `RECOVERY_BACKUP_DIR` 为实际、已存在的备份目录。本文不提供虚构的 OSS bucket、SSH 地址或网盘路径。

### 6.3 在独立备份挂载上归档 `/home`

以下命令**未运行**。前置条件：已设置 `RECOVERY_BACKUP_DIR`，目的地确实为独立挂载，且其持久保留策略已经确认。脚本拒绝写入系统盘或 `/hy-tmp` 所在文件系统，使用 tar 默认行为保留软链接本身，不展开链接。

```bash
set -euo pipefail
: "${RECOVERY_BACKUP_DIR:?请先设置为已确认的独立持久备份目录}"
export RECOVERY_BACKUP_DIR

python3 - <<'PY'
import os
import shutil
import subprocess
from pathlib import Path

target = Path(os.environ['RECOVERY_BACKUP_DIR']).resolve(strict=True)
if not target.is_dir():
    raise SystemExit('备份目的地不是目录')
if target.stat().st_dev in {Path('/').stat().st_dev, Path('/hy-tmp').stat().st_dev}:
    raise SystemExit('拒绝使用系统盘或当前临时数据盘作为独立备份')
size = int(subprocess.check_output(['du', '-sb', '/home'], text=True).split()[0])
free = shutil.disk_usage(target).free
if free < size + 1024**3:
    raise SystemExit(f'备份空间不足：源逻辑大小={size}，可用={free}')
print(f'检查通过：目的地={target}，源逻辑大小={size}，可用={free}')
PY

archive="$RECOVERY_BACKUP_DIR/boxpromptiml_home_$(date +%Y%m%d_%H%M%S).tar"
test ! -e "$archive"
tar --acls --xattrs -cpf "$archive" -C / home
sha256sum "$archive" > "$archive.sha256"
sha256sum -c "$archive.sha256"
tar -tf "$archive" | sed -n '1,35p'
```

备份期间停止会修改被备份项目的数据处理或训练作业。若 tar 报告文件在读取期间变化，不应直接宣布一致备份成功。应定位变化文件并补做一致副本。

如没有独立备份挂载，使用平台正式文件下载/对象存储工具，或从本地电脑经 SSH 拉取 `/home`。传输完成后在目标端复核大小、目录和 SHA256；具体命令需使用用户自己的连接信息，本次没有实际传输。

## 7. 步骤三：修复历史路径别名

**状态：未运行。只有原盘保护完成、或确认按重建路径处理后再执行。**

### 7.1 最小修复策略

恢复以下别名即可让大量旧链接重新解析，无需移动原始文件或重建整个数据集：

| 兼容路径 | 指向的现存目录 |
|---|---|
| `/hy-tmp/datasets` | `/home/datasets` |
| `/hy-tmp/weights` | `/home/weights` |
| `/hy-tmp/runs` | `/home/runs` |
| `/hy-tmp/projects` | `/home/projects` |

这些链接只恢复访问路径，**不会恢复已丢失的 `/hy-tmp/BoxPromtIML/fecd_runs` 文件**。

`/hy-tmp/runs` 是原始 BoxPromptIML 旧运行目录的别名；`/hy-tmp/BoxPromtIML/fecd_runs` 是 FECD 新运行产物目录，两者不能混淆。

### 7.2 安全创建别名

以下命令**未运行**。脚本先确认数据盘是独立的精确挂载点，验证全部现有路径没有冲突，再只创建缺失的别名；不会覆盖已有文件、目录或指向不同位置的软链接。

```bash
set -euo pipefail
test "$(findmnt -rn -T /hy-tmp -o TARGET)" = /hy-tmp

PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import os
from pathlib import Path

data = Path('/hy-tmp')
if data.stat().st_dev == Path('/').stat().st_dev:
    raise SystemExit('/hy-tmp 不是独立数据文件系统，停止')
pairs = [
    (data / 'datasets', Path('/home/datasets')),
    (data / 'weights', Path('/home/weights')),
    (data / 'runs', Path('/home/runs')),
    (data / 'projects', Path('/home/projects')),
]
pending = []
for alias, source in pairs:
    if not source.is_dir():
        raise SystemExit(f'源目录不存在：{source}')
    if os.path.lexists(alias):
        if not alias.is_symlink() or not alias.exists() or not alias.samefile(source):
            raise SystemExit(f'已有路径冲突，需人工核对：{alias}')
    else:
        pending.append((alias, source))
for alias, source in pending:
    alias.symlink_to(source, target_is_directory=True)
    print(f'CREATED {alias} -> {source}')
for alias, source in pairs:
    if not alias.samefile(source):
        raise SystemExit(f'别名复核失败：{alias}')
print('PASS：四个兼容路径指向现存数据')
PY
```

### 7.3 校验数据链接与权重路径

以下命令**未运行**，应在 7.2 后执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path

links = [p for p in Path('/home/datasets').rglob('*') if p.is_symlink()]
bad = [str(p) for p in links if not p.is_file()]
print('dataset_links:', len(links))
print('unresolved:', len(bad))
print('examples:', bad[:10])
if len(links) != 30787 or bad:
    raise SystemExit('数据链接数量或可读性不符，停止后续训练')
for path in [
    Path('/home/projects/BoxPromtIML/segment-anything/sam_vit_h_4b8939.pth'),
    Path('/root/.cache/torch/hub/checkpoints/tiny_vit_5m_22kto1k_distill.pth'),
]:
    print(path, 'readable:', path.is_file())
    if not path.is_file():
        raise SystemExit(f'权重路径仍不可读：{path}')
print('PASS：数据链接和历史权重路径可解析')
PY
```

如果权重本体存在但 hub 缓存链接仍缺失，先检查实际链接目标；可使用项目已有的 `fecd/model_loading.py` 本地 TinyViT 加载适配器进行模型检查。不要先下载同名权重覆盖本体，也不要修改冻结数据身份来掩盖路径错误。

## 8. 步骤四：校验现存文件、缓存与环境

### 8.1 重新核对 42 项迁移校验和

以下标准命令**本身未运行**；本次已经使用等价的分块 SHA256 核验逻辑确认全部 42 项匹配：

```bash
cd /home
sha256sum -c /home/projects/BoxPromtIML/migration/checksums.sha256
```

预期每项为 `OK`。不匹配时保留当前文件和原清单，先查原因，不要把清单更新为当前值然后宣称历史文件完整。

恢复到原位置的 4F1/5A 原件，可另外核对：

```bash
cd /hy-tmp
sha256sum -c /home/projects/BoxPromtIML/migration/checksums-hy-tmp.sha256
```

第二条命令**未运行**，且仅适用于已经从备份找回的历史原件；对新训练的产物不能要求其冒充历史 SHA256。

### 8.2 复跑路径检查

命令见第 4 节。路径修复后即使仍报 4F1 final 缺失，也符合当前事实；不要为了让检查变绿，将 4P、4F0 或局部 P0 的 checkpoint 复制成正式 final。

完整恢复原始 4F1 后，该检查才有机会全部通过。它要求历史原执行器状态保持 `FAILED`，并与独立 4F1-V 的工程 PASS 并存；二者不应互相覆盖。

### 8.3 全量 SAM 缓存复验

**本次未运行逐份缓存复验。** 项目 `audit_stage2c_cache` 虽以“只读审计”描述输入访问，但实际会写审计 JSON、统计 CSV 和可视化；默认位置会覆盖历史资产。恢复时必须指定全新输出目录。

以下命令**未运行**，前置条件为原盘保护结束、别名可读、已有资产已备份且空间充足：

```bash
set -euo pipefail
cd /home/projects/BoxPromtIML
recovery_audit="/hy-tmp/BoxPromtIML/recovery_checks/cache_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$recovery_audit"

PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python \
  -m fecd.tools.audit_stage2c_cache \
  --audit "$recovery_audit/cache_audit.json" \
  --csv "$recovery_audit/cache_statistics.csv" \
  --visuals "$recovery_audit/visualizations" \
  > "$recovery_audit/audit.log" 2>&1

PYTHONDONTWRITEBYTECODE=1 python3 - "$recovery_audit/cache_audit.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
print('status:', result['status'])
print('counts:', result['counts'])
if result['status'] != 'passed' or result['error_count'] != 0:
    raise SystemExit('缓存审计未通过，停止')
PY
```

预期 K1/K8 各 5,737，unique cache keys 为 11,474，错误为 0。该入口会遍历缓存和学生 RGB 数据，不进行 SAM 推理，也不应重新生成缓存。脚本会保留较多数组，运行前检查内存余量；它不替代阶段 3 evidence 全量内容校验。

阶段 3 证据缓存需要在继续相关分析前，用现有 schema、统计文件和原审计约束重新逐份加载验证。若发现少量损坏，应按冻结配置补算受损项，而不是重建 split 或全量重算覆盖旧数据。

### 8.4 环境报告与轻量回归

以下命令**未运行**：

```bash
cd /home/projects/BoxPromtIML
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_environment
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tests.run_stage2b_tests
```

环境报告中 `boxpromptiml_import.ok` 应为 true；检查报告 JSON，而不只看进程返回码。旧 2B 回归历史记录为 `9 checks passed`，本次不能提前宣布同样结果。

完整测试应在 5C 缺失脚本恢复后执行：

```bash
cd /home/projects/BoxPromtIML
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /usr/local/miniconda3/envs/boxpromptiml/bin/python \
  -m unittest discover -s fecd/tests -p 'test_*.py'
```

**未运行。** 历史 5C1 报告中的 244 项测试通过是历史结果，不是本次结果。恢复缺失的 `stage5c1/code/evaluate.py` 前，完整 discovery 预期会遇到导入问题；不要删除该测试来规避。

## 9. 步骤五：恢复 GPU 和训练前置条件

**状态：未运行 GPU 恢复。** 先在恒源云控制台确认当前实例已经分配 GPU、容器属于正确实例、设备是否正常挂入；必要时让平台检查。不要在没有证据时先重装 PyTorch/CUDA。

恢复后执行以下检查，命令**未运行**：

```bash
nvidia-smi

PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python - <<'PY'
import torch

print('torch:', torch.__version__)
print('build_cuda:', torch.version.cuda)
print('available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit('CUDA 不可用，禁止启动正式训练')
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

通过标准：至少 1 张 GPU 可用，型号与复现实验条件相符。现存 5C1 预检显式要求 4090；如实际分配了其他卡，应先形成新的硬件复验记录和必要的受控修复。

可再做只加载本地权重的学生 forward，以下命令**未运行**：

```bash
cd /home/projects/BoxPromtIML
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python - <<'PY'
import torch
from fecd.model_loading import build_boxpromptiml_with_local_tinyvit

if not torch.cuda.is_available():
    raise SystemExit('需要可用 GPU')
model, info = build_boxpromptiml_with_local_tinyvit(
    '/home/weights/tiny_vit_5m_22kto1k_distill.pth'
)
model = model.to('cuda:0').eval()
with torch.inference_mode():
    output = model(torch.zeros(1, 3, 224, 224, device='cuda:0'))
if tuple(output.shape) != (1, 1, 224, 224) or not torch.isfinite(output).all().item():
    raise SystemExit('学生 forward 输出形状或有限性异常')
print('PASS', tuple(output.shape), info)
PY
```

该检查不会更新参数，不会读取 GT，也不表示完整训练和评估已经通过。

## 10. 哪些阶段需要重做

以下按项目实际执行过的细分阶段判断，不将其机械映射为最早任务书中的七个大阶段。

| 阶段 | 是否必须整阶段重做 | 后续处理 |
|---|---|---|
| 原始 BoxPromptIML 正式训练 | 否 | 20 个 checkpoint 和旧结果仍在；修复路径后可复用 |
| 1：原仓库审计 | 否 | 报告和代码仍在；本文补充本次事故状态 |
| 2A：坐标/数据/环境协议 | 否 | 复验路径、环境与数据角色，不改协议 |
| 2B：缓存实现 | 否 | 实现和测试仍在；运行轻量回归 |
| 2C：全量 K1/K8 缓存 | 否 | 文件数和索引仍在；先全量加载复验，仅补损坏项 |
| 3：取证证据与教师诊断 | 暂不需要 | 主要资产及报告仍在；继续使用前复验完整性；历史结论 NO-GO |
| 3R/3R-A/3R-B0 | 暂不需要 | 现存报告与主要分析资产可复用；保留原 STOP/证据不足结论 |
| 4P：受控 smoke | 否 | 四组 final 在；运行环境改变时只重做必要 smoke |
| 4F0：单 epoch 预检 | 否 | 四组 epoch1 在；完整性复验即可 |
| 4F1：四组各 20 epoch | 恢复完整正式模型与证据时需要 | 优先找原备份；找不到且确需原始证据时重新训练、严格评估和统计 |
| 4F1-V：独立复验 | 不能把旧 PASS 用于新运行 | 原件恢复后按原绑定复核；新训练需要适用于新运行的新验收 |
| 5A：学生失败诊断 | 需要原始概率图/诊断证据时需要 | 依赖完整 4F1；绑定原路径与快照，不能直接对新运行照旧执行 |
| 5B | 需先澄清范围 | 至少有下游依赖文件丢失；从外部备份恢复，或重建有明确来源的新基线 |
| 5C0：区域困难负类预检 | 重新执行 5C1 前需要新的工程预检 | 主体代码在；原探针、单轮、选择记录及冻结清单缺失 |
| 5C1：R0/R1 各 20 epoch | 恢复 checkpoint/逐图证据时需要 | 先恢复脚本及门禁依赖，再完成新预检、训练、严格评估与统计 |

### 10.1 需要保留的历史科研结论

- 阶段 3：NO-GO；3R-A 和 3R-B0 停止。
- 4F1：独立复验工程 PASS，科研 NO CLEAR GAIN。
- 5A：工程 PASS，科研 NO CLEAR GAIN。
- 5C1：工程 PASS，科研 **NEGATIVE SIGNAL**。

5C1 报告记录：R1−R0 的 image-macro F1 为 `+0.007115`，95% CI 为 `[-0.016446, 0.026407]`；small Recall 为 `-0.074119`，large Recall 为 `-0.122101`，Columbia F1 为 `-0.059300`，触发原先冻结的安全负面条件。

这些数值目前来自仍在的报告，**本次无法用已丢失的逐图结果独立复算**。保留报告意味着历史结论没有消失，不代表原始证据已经恢复。

不应因产物丢失就自动进入下一研究阶段、追加 seed、开放锁定数据或修改方法。首先明确是“仅保留结论继续讨论”，还是“恢复模型和可审计原始证据”；前者无需把所有历史实验重跑一遍。

## 11. 路径 A：能从备份找回历史原件

**状态：未运行。此路径优先。**

1. 将平台找回的数据或备份先恢复到独立暂存目录，核对来源、时间、归档目录层级与校验和，不直接覆盖现存 `/home`。
2. 优先找回整个 `BoxPromtIML/fecd_runs`，不要只取几个 `final.pt`。训练日志、选择 mask、逐图评估、GT 访问记录、冻结清单和数据盘内代码都需要。
3. 对 4F1/5A 原件按 `migration/checksums-hy-tmp.sha256` 核验；对 5C1 使用报告记录的 final SHA256 及找回的原始冻结/审计清单核验。
4. 本次 5C1 报告记录的 final SHA256：

   - R0：`b54149c2e27566d13f3308bbecdebb9e89832a4304f98e96af78ef43558cb277`。
   - R1：`1cfe5a79d2501ac58388be1f68c1cb5f8f7b0d3bac7eccb2f1a35ff7443c4454`。

5. 全部确认后，将历史目录恢复到旧工具需要的位置，或为恢复副本建立经过验证的路径映射；保留只读备份。
6. 复核 `stage5c1/code/evaluate.py`、5B/5C0 的预检清单和 `runtime_frozen.json` 是否齐全，再跑完整测试。
7. 重新执行可只读复核的身份、哈希、统计和访问边界检查，结果输出到新目录，保留旧验收原件。

注意，4F1-V 工具 `stage4f1_v2_run.py` 不只是一个通用验收器：它绑定原 run-id、四组历史 checkpoint SHA256，并要求输出目录预先存在且含 `inputs_before.json`。即使找回原始产物，也要先准备正确的独立审计快照；不能仅建一个空目录就运行并声称通过。

## 12. 路径 B：找不到备份，需要重新实验

**状态：未运行。本文没有授权或执行任何新训练。**

### 12.1 先恢复工程闭环

开始新训练前应完成一次单独的恢复修复，交付内容至少包括：

1. 现存 5C 代码、配置、split、缓存和权重的快照及 SHA256。
2. 缺失 `stage5c1/code/evaluate.py` 的找回或受控重建，测试通过且保持 GT 访问边界。
3. 明确记录 5B/5C0 历史清单无法找回；以完整数据核验建立新的恢复基线，不能假装原清单仍在。
4. 恢复 5C1 预检/运行冻结生成流程，检查各输出目录的创建顺序；现存脚本不一定会自动创建所有父目录。
5. 为新实验建立单独输出目录及版本化身份；5C runner 当前输出路径为常量，需要受控调整，不能与历史身份混用。
6. 解决 4F1 旧校验器数值误报及旧 4F1-V 的历史哈希绑定，使新运行有适用的独立验收；保留原规则与旧失败证据。
7. 新恢复代码跑定向测试、静态检查及完整测试，通过后才进入实际实验阶段。

以上是后续需要完成的实际修复，不是本文已经实现的功能。

### 12.2 4F1：确需恢复四组正式模型时

需要 P0、P1、P2F、P3F 各 20 epoch，每组 17,660 更新，使用冻结的 3,530 张 student_train 和既定评估协议。不能拿现存 4P/4F0 checkpoint 当作 20 epoch final，也不能只恢复 P0 就宣称四组完整。

下面是**未运行、带前置条件的训练命令**：仅在 12.1 已完成、明确决定重跑 4F1、GPU/数据/空间/备份门禁通过后执行。它会真实训练并写产物，不属于检查命令。

```bash
set -euo pipefail
cd /home/projects/BoxPromtIML
test "$(findmnt -rn -T /hy-tmp -o TARGET)" = /hy-tmp
mkdir -p /hy-tmp/BoxPromtIML/fecd_runs/stage4f1
recovery_run_id="recovery4f1_$(date +%Y%m%d_%H%M%S)"

PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python \
  -u -m fecd.tools.run_stage4f1_autoshutdown \
  --no-shutdown \
  --run-id "$recovery_run_id" \
  --output-root /hy-tmp/BoxPromtIML/fecd_runs/stage4f1
```

这里使用新的 run-id，关闭自动关机。使用前台命令便于观察状态；如需后台执行，可在确认 `tmux` 可用后采用原工具支持的 `--detach`，并持续查看状态和日志。

现有执行器要求系统盘至少 5GiB、数据盘至少 8GiB 余量。历史四组总耗时约 5.1 小时（RTX 4090），只是旧记录，不能保证本次时长。当前系统盘余量离门槛较近，应在启动前重新检查。

旧执行器可能仍因 float32 loss 分解断言在最后验收报错。**未完成 12.1 的验收修复前，不建议付出数小时成本后再次撞上已知门禁问题。** 不要修改旧历史 `status.json` 来美化结果。

### 12.3 5A：需要重建原始诊断证据时

先恢复或重新完成四组 4F1。现存 `fecd/diagnostics/stage5a_io.py` 固定了历史 `RUN` 和 `OUTPUT`，`run_stage5a_inference.py` 还依赖旧 `inputs_before.json` 和旧逐图评估。

因此，不能对新 4F1 结果直接执行旧 5A 命令。需要将输入绑定到新运行，建立对应的新保护快照、allowlist 和输出目录，并核对严格评估口径后，才重新推理、诊断、可视化和验收。

该阶段不需要重新训练学生，但 CPU 推理可能较慢。GT 仅允许进入独立诊断，不能进入学生训练。新诊断不能覆盖 `stage5a_failure_diagnosis.md` 中的历史记录。

### 12.4 5C0：重新取得工程预检证据

在 12.1 恢复工程基础后，按现存接口完成：

1. R0、R1 各自独立少步更新和磁盘恢复探针。
2. 两组共同初始化、下一步 loss/mask/model/optimizer/memory/RNG 完全一致性检查。
3. R0、R1 各 1 个完整 epoch，每组 883 次更新。
4. 逐图正类相同、框内/框外负类预算一致、R1 排序正确、训练不访问 GT 的核验。
5. 新运行的冻结源码、缓存身份、访问日志及审计文件备份。

现存入口为 `python -m fecd.tools.train_region_smoke --variant R0 --mode probe`，R1 替换 variant；完整单轮使用 `--mode epoch`。**这些入口本次未运行，且在缺失目录、清单生成流程未修复前不应直接调用。**

不需要为了 5C1 的初始化重跑全部 4F1：5C 使用 fresh 初始化，不继承 4F1 final。但需要明确替代缺失历史保护证据的方法，不能跳过访问与身份门禁。

### 12.5 5C1：确需恢复 R0/R1 模型和完整证据时

正式训练前，先跑恢复后的 `preflight_region_formal`，验证所有 3,530 个训练成员、K8 缓存、共同初始化、恢复探针、候选集合与新运行冻结清单。当前工具还要求独立数据盘至少 15GiB 余量及已验证的 4090 环境。

预检通过后，R0/R1 各 20 epoch、每组 17,660 更新；固定 epoch20 final，不按开发集成绩选择 checkpoint。两组全部训练结束后，各在 530 张 new_dev 上严格评估一次，raw sigmoid、固定阈值 0.5、正确恢复 ExternalAttention memory。

随后重新生成来源代理组配对 bootstrap、安全分层、选择频次与覆盖诊断、训练访问审计和独立最终验收。

现存训练接口为 `python -m fecd.tools.train_region_formal --variant R0`，R1 替换 variant；工具有 `--resume`，但它只恢复存在且通过完整性检查的 epoch 边界。**原 checkpoint 已丢失时，不能使用 `--resume` 恢复不存在的训练。**

上述入口本次均未运行。完整启动与评估命令需要在新输出目录和缺失代码恢复后确定，本文不提供一个表面完整、实际会在缺失门禁上失败的串联命令。

历史 5C1 两组训练总耗时约 3.34 小时（不含完整恢复预检、评估、统计与备份），仅作资源估计。保留单 seed 限制，不因重新实验就自动增加 seed 或调整停止条件。

## 13. 恢复后的备份与防止再次丢失

将存储职责分开，并以平台实际保留政策为准：

| 内容 | 日常工作位置 | 必须存在的备份 |
|---|---|---|
| 代码、配置、测试、报告 | 项目目录 | Git 远端及包含未提交文件的目录快照 |
| 原始数据、冻结 split、预训练权重 | 现存 `/home` 或已确认持久盘 | 对象存储/本地或其他独立持久存储 |
| 可再生成的缓存 | 实验工作盘 | 校验清单；耗时高的缓存最好保存副本 |
| 正式 checkpoint、逐图结果、日志、审计与运行目录代码 | 工作盘 | 每个阶段完成后同步到独立存储并校验 |

至少在每个完整 epoch 或可接受损失窗口内保存恢复 checkpoint，并定期同步；每阶段结束后立即同步完整运行目录。同步范围不能只有 final checkpoint，还包括预检、源码快照、配置、评估代码、逐图结果和审计记录。

备份完成的标准是：目标端能够列出文件、校验和匹配，并抽查能恢复读取；仅显示“上传完成”或本地存在同名压缩包不够。校验后再停止实例。

大型缓存和 checkpoint 不应直接塞入普通 Git 历史；是否采用大文件存储或对象存储，按已有平台能力选择。本次未新增依赖、未配置自动同步或自动关机。

## 14. 恢复验收清单

### 14.1 可以继续一般项目工作的条件

- [ ] 已确认平台原盘恢复结果，保存客服结论或快照来源。
- [ ] `/home` 现存数据和未提交代码有一份独立备份，目标端校验通过。
- [ ] `/hy-tmp` 挂载身份明确，路径别名指向正确的现存目录。
- [ ] 30,787 个数据软链接全部可读；SAM/TinyViT 本体与历史加载路径可解析。
- [ ] 42 项迁移校验和通过，未通过项有明确处理记录。
- [ ] K1/K8 缓存全量复验通过，既定 split 和训练/诊断 GT 边界不变。
- [ ] 缺失的评估脚本、预检和冻结清单已恢复，或通过独立恢复修复建立了新基线。
- [ ] 适用测试、静态检查通过；历史通过记录没有被当成本次测试结果。

### 14.2 可以启动正式训练的附加条件

- [ ] 已明确本次要恢复的阶段及证据范围，不自动扩展到新研究阶段。
- [ ] CUDA 可用，设备型号和确定性条件通过复验。
- [ ] 学生本地权重加载与 forward smoke 通过。
- [ ] 新运行目录、运行身份、磁盘余量与备份目的地已确认。
- [ ] 对应阶段的共同初始化、数据身份、恢复探针与训练 GT 隔离门禁通过。
- [ ] 已知旧验收误报和历史哈希绑定已有正确处理，不靠删检查绕过。

### 14.3 可以宣称历史证据已恢复的条件

- [ ] 找回的原件与原 SHA256 一致，或明确标注为新运行、新证据。
- [ ] 需要的 checkpoint、日志、逐图评估、统计、访问记录与代码快照齐全。
- [ ] 独立验收针对本次实际文件执行并通过。
- [ ] 科研结论根据对应原始证据复核，保留负面结果与局限性。
- [ ] 全部恢复产物已经同步到独立存储并在目标端复核。

## 15. 当前交付状态与后续最小行动

本文交付状态为：**检查完成，恢复未执行；正式训练条件 BLOCKED。**

下一步优先联系平台查原始备份，并将现存 `/home` 备份。确认走重建路径后，先执行第 7 节的路径修复，再执行第 8 节的校验；这部分不需要重训任何模型。

只有需要找回正式模型或完整原始实验依据时，才进入第 11 或第 12 节。若只是继续讨论研究方向，保留当前代码、缓存和报告即可，不应为“恢复进度”无差别重复全部失败实验。

## 16. 参考文件

- [项目迁移说明](MIGRATION.md)：截至 2026-11-02 的整理，主要覆盖至 5A；不能替代后来 5C 报告。
- [恢复路径检查脚本](migration/verify_paths.py)。
- [系统盘资产 SHA256 清单](migration/checksums.sha256)。
- [历史数据盘资产 SHA256 清单](migration/checksums-hy-tmp.sha256)。
- [阶段 4F1 独立复验报告](fecd/docs/stage4f1_revalidation_report.md)。
- [阶段 5A 失败机制诊断报告](fecd/docs/stage5a_failure_diagnosis.md)。
- [阶段 5C0 预检报告](fecd/docs/stage5c0_report.md)。
- [阶段 5C1 正式实验报告](fecd/docs/stage5c1_report.md)。
