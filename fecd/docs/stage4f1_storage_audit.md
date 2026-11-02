# FECD 阶段4F1-S：磁盘审计与安全输出路径

日期：2026-10-29。**存储验收 PASS；本次没有启动或继续正式训练，没有运行 new_dev 评估。** 新输出目录已准备，完成后停止等待验收。训练执行仍需验收后重新进行新路径、新入口版本的正式预检，不能用旧代码版本的预检哈希直接放行。

## 1. 先停止上一轮后台工作

接到4F1-S指令时，上轮正式训练仍在后台运行。已先停止训练协调进程组142897（含P0训练进程142906）及等待评估的进程组144226，确认没有遗留训练或自动评估进程。停止不是数值失败，也不将未完成实验改写为正式结果。

原系统盘 `fecd/assets/stage4f1/` 原样保留。P0已完成4个Epoch，`P0/last.pt` 的update=3532、position=`{"epoch":4,"next_batch":0}`；第5轮未完成日志仍在原位，不能当作已保存完整模型状态。P1/P2F/P3F未启动，四组正式实验没有完成，评估流程没有启动。停止记录见 `fecd/assets/stage4f1_storage_audit/interruption.json`。本次没有迁移或删除这些产物，也没有尝试恢复或延续P0训练。

## 2. 实际挂载、容量与权限

已实际执行用户指定的四条命令，原始输出保存在 `fecd/assets/stage4f1_storage_audit/disk_commands.txt`：

```bash
df -hT / /home /hy-tmp
df -i / /hy-tmp
findmnt -T /hy-tmp
lsblk -f
```

| 路径 | 文件系统 | 容量 | 已用 | 可用 | st_dev |
| --- | --- | --- | --- | --- | --- |
| /、/home | overlay | 30 GiB | 约20.83 GiB（df显示21G） | 约9.17 GiB（9.2G） | 67 |
| /hy-tmp | XFS | 50 GiB | 本次快照约0 | 约50 GiB | 66304 |

`findmnt` 确认 `/hy-tmp` 是实际挂载：source=`/dev/nvme0n1[/local/I2be5a164b603f01fcb]`，target=`/hy-tmp`，fstype=`xfs`，options=`rw,relatime,attr2,inode64,logbufs=8,logbsize=32k,prjquota`。它与 `/` 的逻辑文件系统和device ID均不同，不是overlay上的普通目录。该视图是宿主XFS子目录及项目配额挂载；容器内lsblk信息不能据此证明独占一块物理盘。

两处df所见inode总量375,073,792，已用约6,844,114、可用约368,229,678、使用率2%；这是宿主文件系统提供的inode视图，不代表两个各自独占的inode池。没有inode不足。当前用户可写，已通过真实写入、fsync、原子替换和回读探针验证。没有格式化、分区或修改挂载。

## 3. 真实磁盘占用清单

使用 `du -x -B1` 与文件 `st_blocks*512` 统计实际分配占用，同时记录逻辑字节数和硬链接数。目录包含父子重叠，不能把表中各项相加。完整目录和大文件清单在 `fecd/assets/stage4f1_storage_audit/inventory.json`。原始数据及GT仅做文件系统元数据统计，没有打开GT文件内容。

| 目录 | 分配占用约GiB | 分类与处理 |
| --- | ---: | --- |
| /home/datasets | 8.60 | 原始/处理数据，全部保留 |
| /home/datasets/raw | 8.52 | 原始数据，全部保留 |
| /home/projects/BoxPromtIML/fecd/assets | 6.02 | 各阶段受保护实验产物，保留 |
| fecd/assets/stage3_evidence | 3.88 | 历史取证缓存，仍保留NO-GO/STOP状态，不迁移 |
| /home/datasets/raw/NIST16 | 3.77 | 原始数据，不动 |
| /home/datasets/raw/CASIAv2 | 3.32 | 原始数据，不动 |
| /home/weights | 2.41 | 冻结模型权重，不动 |
| /home/datasets/raw/Columbia | 1.15 | 原始数据，不动 |
| fecd/assets/stage2c_cache | 0.62 | K1/K8冻结SAM缓存，路径及身份保持不变 |
| fecd/assets/stage4f0 | 0.57 | 4F0预检、评估和checkpoint，不动 |
| /home/runs | 0.56 | 历史checkpoint/评估，不动 |
| fecd/assets/stage4p | 0.53 | 历史4P checkpoint及smoke，不动 |
| fecd/assets/stage4f1（旧系统盘位置） | 约0.098 | 中断训练及预检记录，保留原位 |

最大的单文件包括：

| 文件 | 逻辑字节数 | 类别 |
| --- | ---: | --- |
| /home/weights/sam_vit_h_4b8939.pth | 2,564,550,879 | 冻结SAM权重 |
| /home/datasets/raw/Columbia/4cam_auth.tar.bz2 | 245,227,548 | 原始归档 |
| /home/datasets/raw/Columbia/4cam_splc.tar.bz2 | 156,194,726 | 原始归档 |
| fecd/assets/stage3ra_per_image.jsonl | 123,321,600 | 历史诊断明细 |
| fecd/assets/stage3_diagnostics/per_image.jsonl | 99,265,927 | 历史诊断明细 |
| /home/datasets/raw/CASIAv1/CASIA 1.0 dataset.zip | 95,120,960 | 原始归档 |
| /home/datasets/raw/NIST16/NC2016_Test0613/reference/splice/NC2016-splice-ref.csv | 76,374,269 | 原始参考元数据，仅stat |
| fecd/assets/stage4f1/P0/last.pt | 70,991,310 | 已中断P0的完整Epoch4状态 |

4P与4F0 checkpoint单文件也均约70,991,000字节。相同大小不等于相同内容，不能据此判定可删除的重复文件。

临时目录 `/tmp` 总占用约50.13 MiB，其中 `/tmp/opencode/verify-repo` 约10.39 MiB；没有确认可安全清理的大型本任务临时文件。`migration/` 约4.84 MiB，是项目迁移备份，不当作垃圾。没有进行全盘内容去重，也没有把原始压缩包视作可删除的解压重复。

## 4. 迁移判断与独立路径记录

**推荐历史迁移清单为空，实际迁移清单为空，历史删除清单为空。** 系统盘尚有约9.17 GiB，且未来新增产物可以直接落盘到50 GiB数据配额，无需改动历史文件。当前没有必须迁移受保护文件的理由；若未来需要，须另列文件、引用和内容校验方案获准后再复制验证。本次没有创建软链接，没有改动原数据或缓存引用。

已创建新目录：

```text
/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/
```

输出位置单独登记在 `fecd/configs/stage4f1/storage.json`；不放进四组训练配置，不改变其config hash或4F0协议。训练、正式预检、严格评估包装、统计、曲线、验收及后续流程入口均支持 `--output-root`。后续流程会把该参数传给各子入口。严格evaluator本身按传入的绝对checkpoint路径读取，无需修改其冻结实现。

路径解析拒绝相对路径；对 `/hy-tmp` 下的目标，若它退化为系统盘目录或经软链接离开数据盘则拒绝写入。本次未创建链接。原子checkpoint的 `.pending.pt` 与 `last.pt` 位于同一目录、同一文件系统。

原9份入口源码已在本次审计的 `code_before/` 内保留副本及修改前哈希，修改前后哈希见 `validation.json`。旧P0、旧预检、旧日志和旧正式配置36个文件均保持哈希不变。由于入口代码版本已变化，旧预检的runtime hash自然不再对应当前入口；没有重写旧预检、更新其通过记录或绕过检查。数据盘新目录在探针清理后为空，不会默默接续旧P0。

## 5. checkpoint与评估路径实测

运行了存储专用检查，**没有运行正式学生训练入口**：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_stage4f1_storage --output-root /hy-tmp/BoxPromtIML/fecd_runs/stage4f1
```

- 在数据盘临时目录创建合成模型的checkpoint，验证 `.pending.pt` 与最终路径同device，真实原子替换成功；完整Adam、模型、memory、Python/NumPy/Torch/CUDA RNG恢复哈希相同，下一步合成更新精确一致。
- 另从本地TinyViT构造未训练的真实BoxPromptIML，保存update=0的完整checkpoint；冻结严格loader从数据盘加载，验证config identity、模型/memory hash及文件SHA256；两次合成RGB推理精确一致，磁盘checkpoint哈希未变。非空Adam恢复由前一项合成模型检查支持；不声称对旧P0完成了实际恢复续训。
- 两项探针只使用自行创建的临时测试文件，完成后清理，未移动或删除历史实验文件。正式学生反向更新=0，GT访问=0。全程启用空RGB/cache白名单审计，数据目录和SAM缓存打开次数均为0。

完整证据见 `storage_probe.json` 与 `storage_probe.log`。所有8个相关CLI的 `--help` 检查通过，见 `entry_help.log`，没有借检查CLI启动训练。

## 6. 磁盘容量预算

依据真实checkpoint和已存在的4个完整Epoch日志测量，未使用内存或GPU显存推算磁盘：

| 项目 | 估计字节数 | 约MiB |
| --- | ---: | ---: |
| 单个完整训练checkpoint | 70,991,310 | 67.70 |
| 四组last及四组final | 567,930,480 | 541.62 |
| 额外一个原子写入临时checkpoint（保守） | 70,991,310 | 67.70 |
| 80个Epoch日志、逐图输出、访问和汇总 | 494,126,080 | 471.24 |
| 四组严格评估（4F0实际参考） | 8,902,638 | 8.49 |
| 哈希清单/元数据预留 | 20,971,520 | 20 |
| 统计、曲线预留 | 33,554,432 | 32 |
| 保守峰值合计 | 1,196,476,460 | 约1.12 GiB |

已测单Epoch全部日志为6,166,484–6,176,576字节，按最大值乘80，包含实际逐图和访问明细。预检临时恢复探针阶段可能同时有4个checkpoint，约271 MiB，但在正式训练开始前清理，未与正式保留checkpoint累加。没有计划保留全部80个大checkpoint。

建议将工作量按2 GiB预算；启动前数据盘至少8 GiB空闲，包含不少于5 GiB额外余量；系统盘也至少保留5 GiB。当前数据盘约50 GiB、系统盘约9.17 GiB均满足。没有释放历史文件来制造余量。四组20Epoch计算耗时约5.05小时是4F0的线性估算，本次没有执行这项训练。

## 7. 冻结完整性、测试和最终状态

下列值重新计算并匹配：

- formal20 proposal：`8b25d14ae1d90c00db59e5794eaff4d038f82829f63abbd1513404923e8d46d2`。
- evaluation protocol：`54baa31ac65ec3df978af09c43b2454fd5515db1c8117b86c6ff2cd7d267416f`。
- student_train成员：`cc98c8c0222ecf20d6ba887fafa954eff07983b6fa360a43c5e0057757e50110`。

17,716个历史保护/4F0交付文件哈希不变；另有36个既存4F1产物和正式配置文件哈希不变。四组配置仍精确等于原冻结配置。没有修改原模型、train.py、eval.py、utils、4P/4F0配置、SAM缓存、学生成员或已有checkpoint；原始工作树已有的train.py/eval.py差异没有修改或撤回。FECD-R NO-GO、q_A STOP、B-R STOP继续有效。

新增路径行为先有失败测试，再实现通过。定向检查通过；完整fecd测试158项通过，12个新增/变更Python文件通过现有AST语法及未解析全局名静态检查，无新增依赖；该检查不等于完整类型检查器。实际验证命令包括：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest fecd.tests.test_stage4f1_storage fecd.tests.test_stage4f1_training fecd.tests.test_stage4f1_statistics
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest discover -s fecd/tests
```

完整证据目录：`fecd/assets/stage4f1_storage_audit/`，包括挂载原始输出、占用清单/空间预算、停止记录、源码修改前副本、红/绿测试、完整测试、数据盘checkpoint探针和最终validation。结果hash=`bf6b7deadd9792b0d58c1ff8dc45f380b61b491fa565e61b4102795277f185de`。

**最终判断：4F1-S存储条件PASS。** 当前没有运行训练或自动评估任务；本次正式训练更新=0、new_dev/new_confirm/teacher_diag/正式测试GT读取=0。新输出路径可用，但没有据此自动放行4F1：等待验收后，须对新路径及已记录的入口版本重新预检，再按明确的后续训练指令执行。未自动恢复P0，未开启其他组或多seed。
