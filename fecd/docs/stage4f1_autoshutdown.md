# 阶段4F1：无人值守执行、完整验收与成功关机

日期：2026-10-29。**当前最终状态：BLOCKED / SHUTDOWN_NOT_VERIFIED。** 代码、测试、完整模拟dry-run及新路径技术预检已完成；未启动四组正式20Epoch，未执行真实关机，未宣称已停止GPU计费。缺失条件是官方平台停止实例的效果及数据持久性/长期关机保留规则，不能以OS命令存在或容器内root权限代替。

## 1. 云环境与关机能力检查

当前是Docker托管容器：PID1=`docker-init`，`systemd-detect-virt`返回`docker`，cgroup路径属于`/docker/`；不是已证实具有宿主控制权限的完整VM。当前用户为root，`sudo -n -l`允许命令，但effective capability中没有CAP_SYS_BOOT，缺少`/run/systemd/system`和systemd总线。不能把`systemctl poweroff`等同于停止云实例、释放GPU或停止平台计费。

存在`/usr/local/bin/shutdown`，实际为静态链接ELF而非普通系统shell脚本，SHA256=`30b9ea6f23173078ccfe57d9b15ffe8caf5ba1442477f32c2c170ea0e97771d7`。它可能是平台包装程序，但来源及语义未获官方证据确认。**没有执行该程序，连`--help`都未调用；没有调用shutdown、poweroff或任何停止/销毁API。** 只做了文件属性、哈希、capability、进程和挂载检查。未猜测API端点，未读取或记录API密钥。

本机历史记录曾提及恒源云，但不能据此确认当前官方服务条款。未找到足以核实以下条件的官方文档或控制面证据：

- 容器内候选命令是否真正请求平台停止实例，而非只结束容器。
- GPU资源是否释放、GPU算力计费何时停止、其他存储计费是否继续。
- `/hy-tmp`在正常停止/再次开机后是否保留。
- 长期关机、余额不足等情形是否自动释放实例，及数据清理期限。

因此`fecd/configs/stage4f1_autoshutdown/shutdown_evidence.json`保留明确的未核验状态。当前实现只为现有候选包装程序预留受审查的停止边界，没有未验证的API实现，也没有OS关机兜底。以后必须先提供并核对官方停止、计费、持久性说明URL、保留规则、平台身份、当前环境指纹和包装程序哈希，才能使用已核验状态；不能仅把status手改成VERIFIED。当前证据不满足入口检查。

非破坏性环境原始证据：`fecd/assets/stage4f1_autoshutdown/capability_audit/capabilities.json`。

## 2. 输出、状态与冻结保护

统一基础目录仍为：`/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/`。新执行器每次创建独立run_id子目录；大型checkpoint、逐图结果、训练日志和统计均写数据盘：

```text
stage4f1/<run_id>/
  artifacts/                 正式预检及四组新实验目录
  logs/                      各阶段同步子进程日志
  status.json                当前状态、阶段、已完成Epoch和最近checkpoint
  events.jsonl               阶段/错误/关机意图与响应事件
  runtime_and_history.json   运行版本与受保护引用哈希
  completion_report.json     全部工程验收后的结果及配对统计
  completion_report.md       主指标、主对照CI及风险解释
  output_manifest.json       产物SHA256和大小
  logs_manifest.json         已关闭阶段日志的SHA256和大小
  reports_manifest.json      报告及运行版本清单SHA256
```

仅真实训练完整通过全部验收后才可创建`SUCCESS.json`。纯模拟永不创建该标记，不能成为正式结果。工程成功允许阴性效果，不要求P3F优于P2F。

项目中另存小型镜像索引：`fecd/assets/stage4f1_autoshutdown/index/<run_id>.json`。重启后先读该索引，再查看对应数据盘status/events/logs。进度中的epoch为已完整提交的轮数，正在训练的轮为下一轮；最近checkpoint是完整提交状态，不冒充未保存的内存权重。镜像索引也可在数据盘掉挂载、主状态写入失败时保留FAILED信息。

状态/事件使用flush、fsync及同目录原子替换，终止关机前执行os.sync。这里证明的是当前文件系统的写入一致性，**不能证明平台长期保留数据**，所以持久性仍是实际启用关机前的阻断条件。

原系统盘P0完成Epoch4的`last.pt`与第5轮部分日志原样保留；不恢复为新实验初始化。4P、4F0、4F1-S冻结报告/配置/交付清单及旧P0按原哈希核对。没有修改历史产物或冻结入口，没有移动/删除历史文件。新增模块都在fecd内。

三个协议值保持不变：formal20=`8b25d14ae1d90c00db59e5794eaff4d038f82829f63abbd1513404923e8d46d2`，evaluation=`54baa31ac65ec3df978af09c43b2454fd5515db1c8117b86c6ff2cd7d267416f`，student_train=`cc98c8c0222ecf20d6ba887fafa954eff07983b6fa360a43c5e0057757e50110`。四组config hash逐项等于原配置，没有引入输出路径或关机超参数。

## 3. 执行顺序与成功门槛

入口：`fecd/tools/run_stage4f1_autoshutdown.py`。同一基础输出目录使用flock单实例锁，重复run_id拒绝覆盖。默认shutdown_on_failure=False。正式入口首先检查独立数据挂载、系统盘至少5GiB和数据盘至少8GiB空闲，以及关机证据；证据不通过即落盘BLOCKED，不启动训练子进程。

证据通过后由同一个同步控制流程依次运行真实存在的CLI：

1. `fecd.tools.preflight_formal_student`，在该run的新artifacts目录核验当前版本与全部冻结身份。
2. `fecd.tools.train_formal_student --variant P0`、P1、P2F、P3F，均不传resume，重新共同初始化。
3. `fecd.tools.evaluate_formal_students`，只在四组完整20Epoch后评估final。
4. `fecd.tools.summarize_formal_students`及`fecd.tools.plot_formal_curves`，计算冻结2000次配对来源组bootstrap及全部分域/面积结果。
5. 完整fecd unittest及`fecd.tools.check_stage4f1`。
6. 保存完整报告、产物/日志/报告hash和完成标记，关闭文件句柄，sync，再经过单独的关机边界发出已核验停止请求。

每个子进程均等待真实完成并检查返回码；另检查所需文件和内容：每组20个完整Epoch汇总、每组17660更新、3530成员、配置一致性、可加载final checkpoint及完整模型/memory身份、530个dev成员、分域F1/IoU及面积、三个配对差值/CI、全部曲线和最终GT/哈希审计。没有只根据exit=0判定成功；缺失任何一项都不写成功标记。

四组正式约束不变：共同seed20261029、batch4、20Epoch、883更新/轮、FP32、Adam 1e-4、constant LR、clip0.5、原模型和本地TinyViT、原平衡Masked BCE、冻结几何增强和P2F/P3F固定原始像素集合。P2E不加入，无中间dev选优。新结果仅为单seed开发比较，new_confirm、teacher_diag及正式测试GT保持锁定。

数据盘路径在创建运行、状态写入及子进程监控中检查；掉挂载时拒绝向系统盘同名位置写状态并终止子进程，项目镜像保留错误。子进程以独立进程组执行，控制器异常会终止并等待该组，避免遗留后续训练或评估。

## 4. 关机状态和失败策略

- `SUCCESS_SHUTDOWN_REQUESTED`：所有工程结果完成、持久化后进入平台停止请求阶段。事件中区分REQUEST_PENDING与实际命令返回，`billing_stopped_verified`始终为false，不能解读为平台确认停止计费。
- `FAILED`：任一训练/评估/校验/关机操作失败。保存阶段、异常类型/信息、日志引用及最近完整checkpoint，停止后续阶段；关机失败单独保存shutdown_status=FAILED及返回信息，不回退到OS关机。
- `BLOCKED`：平台停止或持久性尚未核验，或本次只是明确标注的dry-run。模拟通过另外记录dry_run_status=PASS及simulated_status，不伪造正式SUCCESS。

提供显式`--shutdown-on-failure`，默认不启用。只有平台停止能力已核验，失败信息和已落盘checkpoint同步后才可请求失败关机。CUDA/进程崩溃后无法凭空补存尚未提交的内存模型，恢复点仍是最近完整Epoch checkpoint，部分日志保留。关机请求失败不被该开关重复重试。没有无条件shell EXIT trap。

真实包装程序stdout/stderr仅记录hash，避免凭据进入报告；当前候选命令不带密钥参数。以后若平台需要API凭据，必须另行审查支持方式，不得写入代码、JSON、日志、checkpoint或Git。

## 5. 实际完成的测试和dry-run

新增13项行为测试先失败再实现通过，覆盖顺序编排、训练/评估异常、checkpoint缺失/损坏、只调用模拟关机、真实关机边界在dry-run中不执行、失败关机策略、关机失败不重复请求、重复run和锁、掉挂载、旧文件保留、零数据目录读取及掉挂载时系统盘错误镜像。完整fecd测试171项PASS；11个新增Python文件通过现有AST语法及未解析全局名静态检查。该检查不等于完整类型检查器，没有安装新依赖。

实际完整模拟run_id=`dryrun_autoshutdown_20261029a`，另实际用tmux执行脱离终端模拟run_id=`dryrun_tmux_autoshutdown_20261029a`。二者均完整执行模拟阶段并得到dry_run_status=PASS、shutdown_status=DRY_RUN_MOCK，无真实训练或真实GT读取，没有正式SUCCESS标记。

真实入口门槛测试run_id=`blocked_autoshutdown_20261029a`，返回BLOCKED、SHUTDOWN_NOT_VERIFIED、completed_stages为空，未启动任何正式实验子进程。真实关机包装程序调用次数0。

实际命令（以上run_id已存在，不可覆盖重用）：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest fecd.tests.test_stage4f1_autoshutdown
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest discover -s fecd/tests
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.run_stage4f1_autoshutdown --dry-run --run-id dryrun_autoshutdown_20261029a
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.run_stage4f1_autoshutdown --dry-run --detach --run-id dryrun_tmux_autoshutdown_20261029a
```

`--detach`使用已存在的tmux，把同一个受控执行器放在独立会话中；不创建无条件关机脚本。启动响应仅为LAUNCH_REQUESTED，须查看状态索引确认实际运行结果。真正关机证据未通过时，即使传detach也只记录BLOCKED，不启动后台训练。

## 6. 新版本数据盘技术预检的实际结果

使用最新冻结训练入口、数据盘全新路径实际运行GT-free预检：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.preflight_formal_student --output-root /hy-tmp/BoxPromtIML/fecd_runs/stage4f1/technical_preflight_autoshutdown_20261029a/artifacts
```

**技术预检PASS。** 全3530成员的RGB/两套缓存身份、共同20轮顺序规则、P2F/P3F全量选择hash与逐图正负floor预算均通过；跨Epoch固定性由未改动的选择函数及既有20轮行为测试保障。四组共同完整初始化hash均为`1086c22f6f2a7d0756d4ad9c7de33f52c5d4e6fffeee5e064a77690bb148d516`。真实CUDA探针验证Adam、全部RNG、model/memory恢复及下一更新精确一致。

探针合计12次物理更新，探针checkpoint已在专用临时目录完成检查后清理；这不是正式20Epoch预算，也不作为正式初始化。正式训练更新=0，GT访问=0。正式预检的保护清单、缓存身份和运行代码通过哈希核对；另119个执行版本/历史引用文件在技术预检前后不变。

预检result_hash=`627eda784a7bc656859c450dfd7bef6c1b3387cd1a208eceada940a2e813c0e1`。结果索引为`fecd/assets/stage4f1_autoshutdown/technical_preflight.json`；原始日志和预检产物在上述数据盘目录。技术预检目录仅为本次检查证据，不会悄悄被执行器当作正式run复用；实际执行器仍会在自己的新run目录重新预检。

## 7. 当前交付与解除阻断条件

新增：两个工具入口、fecd/execution下的编排/状态/子进程/产物门槛/模拟/报告/关机模块、13项测试、未核验关机证据JSON、本说明及独立状态索引。未修改冻结训练参数、原模型或历史产物。

当前总状态位于`fecd/assets/stage4f1_autoshutdown/status.json`。本次停在配置阶段，**没有留下等待自动启动正式训练的后台进程**。只有取得当前平台官方停止实例、停止GPU算力计费、数据盘保留及长期停机清理规则的可核实证据，审查当前包装程序后，才具备启用真实无人值守任务的条件。没有声称平台已关机或已经停止计费。
