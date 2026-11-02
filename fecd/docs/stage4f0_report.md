# BoxPromptIML 阶段 4F0：协议冻结、严格评估与全数据预检

日期：2026-10-29。**最终工程判断 PASS。具备申请正式受控训练的工程条件，但正式20 Epoch未获执行授权，本次到此停止。**

原 FECD-R NO-GO、q_A STOP、B-R STOP及阶段4P配置/8更新smoke原样保留。本轮实际完成P0/P1/P2F/P3F各1个完整Epoch和各530张new_dev严格评估。未训练P2E，未运行正式20 Epoch，未打开new_confirm、teacher_diag或正式测试GT，没有新增取证或可靠性公式。单Epoch分数偏低且有前景过预测风险；不能据此判定教师K8或可靠性筛选的正式效果，未挑组、未调参。

## 1. 数据、冻结与可复现性

student_train严格使用冻结3530成员。split hash=022da143a26833f1613db722ae3be152117948330621e028f02b824f2d1680d0，canonical成员hash=cc98c8c0222ecf20d6ba887fafa954eff07983b6fa360a43c5e0057757e50110。四组共同初始化hash=1086c22f6f2a7d0756d4ad9c7de33f52c5d4e6fffeee5e064a77690bb148d516；共同Epoch0顺序hash=3ae9a1283720d6cab5ae9a8ef3a0b62cf16aaf95b8a64f9d588bb2d8e96d8521。

本地TinyViT SHA256=e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d；适配器strict加载296 tensors。每组从相同随机学生层及本地预训练TinyViT重新构造，未从历史5737张B0或4P最终checkpoint继续训练。

在读取本轮真实GT、训练四组之前，已固化配置、停止规则和gates.json：评估协议hash=54baa31ac65ec3df978af09c43b2454fd5515db1c8117b86c6ff2cd7d267416f，正式提案hash=8b25d14ae1d90c00db59e5794eaff4d038f82829f63abbd1513404923e8d46d2。最终核验16个gate代码文件hash不变、17610个历史文件/权重hash不变。原train.py/eval.py的工作树差异在本轮开始时已存在，本轮没有修改或撤回它们。

| 数据集 | 每组实际训练图 | 训练来源代理组 | new_dev图 | new_dev来源代理组 |
| --- | --- | --- | --- | --- |
| CASIAv2 | 3278 | 130 | 410 | 202 |
| NIST16 | 172 | 23 | 80 | 11 |
| Columbia | 52 | 12 | 26 | 6 |
| Coverage | 28 | 28 | 14 | 14 |

四组逐步sample_key、几何增强和更新数相同，最后batch=2、不drop_last。每组3530张均完成前向与反向；883逻辑更新，另有一次第5步恢复重放，所以884物理更新。重放样本不另计为训练成员或额外逻辑监督预算。来源隔离继续沿用历史代理及3,020张CASIAv2大连通分量约束；不能证明所有真实来源身份独立，NIST历史来源风险不因本轮训练消失。new_dev是已使用过的开发集合，绝非独立确认集。

## 2. 4P前景比例的精确定义及损失核对

4P记录的P1学生67.96%来自每个batch `mean(sigmoid(logits)>=0.5)`，跨8个等大batch取均值；它是32张训练图在不断变化的模型下的阈值前景占比，不是sigmoid概率均值，也不是最终checkpoint固定模型的预测面积。6.55%是同批K8 hard伪标签的前景像素比例。4P未保存当时完整logits或概率均值，不能事后把本轮或最终checkpoint推理值当成该历史量。

四个量在新日志中独立存储：teacher_hard_positive_fraction、probability_mean、student_positive_fraction，以及selected_positive/negative_full_fraction。选中正负类还分别保存精确candidate/selected像素计数，不能用被选比例代替教师/学生前景面积。

复用原4P损失，无权重变化。对图像i，L_i=0.5*sum(S*y*BCE)/max(sum(S*y),1)+0.5*sum(S*(1-y)*BCE)/max(sum(S*(1-y)),1)，再对batch图像平均。空类分子为0、该0.5不转移给另一类；全空选择有可微零损失；忽略像素梯度严格为0。原损失行为测试覆盖精确梯度、空类及等图像权重。

正负类均存在时，每个正像素与负像素的系数比为n_neg/n_pos。在6.55%正类的示意条件下约14.27倍；逐图比例各异，不能把总体比例当每图实际权重。若一张图只能输出一个常数概率，且两类均存在，此平衡目标最优为p=0.5，不以其前景prevalence为常数最优值。因此该损失可能促进稀少前景、使阈值面积偏大；随机初始化、仅8更新、特征和标签结构也可能参与。上述数学性质不是已经证明67.96%由某单一机制引起。未做改变权重的因果实验。

| 组 | 教师hard面积 | 学生概率均值 | 学生>=0.5面积 | 选正/全图 | 选负/全图 |
| --- | --- | --- | --- | --- | --- |
| P0 | 8.9134% | 40.9220% | 29.7590% | 8.9134% | 91.0866% |
| P1 | 8.8376% | 40.9334% | 29.8265% | 8.8376% | 91.1624% |
| P2F | 8.8376% | 40.7657% | 29.4076% | 4.4183% | 45.5807% |
| P3F | 8.8376% | 41.3073% | 31.4362% | 4.4183% | 45.5807% |

上表是本轮全部3530张、更新前且模型参数持续变化的图像均值；与4P前32张不同，不能只比较比例就归因。P0教师空正类0张；P1/P2F/P3F各1张，选后空正类同为1张；四组空负类均0张。那1张仅有负类的有效总类系数为0.5。P2F/P3F类预算与空类情况逐图完全一致。

## 3. 跨Epoch监督预算

| 组 | 标签/策略 | Epoch依赖 | 20Epoch原始唯一像素覆盖 |
| --- | --- | --- | --- |
| P0 | K1 hard，全监督 | 不变 | 100% |
| P1 | K8 hard，全监督 | 不变 | 100% |
| P2F | K8正负分别floor(n/2)，固定随机 | 选择epoch固定0 | 正49.994279%；负49.999445% |
| P3F | 同预算c_sam降序，随机打破并列 | 选择epoch固定0 | 与P2F相同 |
| P2E | K8正负动态随机；本轮未训练 | 真实epoch | 独立均匀抽样假设下期望见下文 |

优先级复用经验证4P SHA256(seed:epoch:sample_key:class)固定排列；P2F/P3F固定priority epoch=0，P3F按(-c_sam,随机优先级)排序。原始224网格先选择，再与RGB/hard标签同步执行实际epoch几何变换。全平分置信度在256个sample_key中检查没有栅格顺序优先；6种变换均完成逆变换对齐测试。

全3530张K8预检耗时113.00秒，候选正15,653,295、负161,467,985；两组均保留正7,825,752、负80,733,097。固定选择全量检查Epoch0/1/19，所有20Epoch不变另由不依赖epoch的实现及20轮逻辑单测保证；没有进行20Epoch训练。

P2E每类n中每轮k=floor(n/2)，若各轮为独立均匀优先级，20轮期望唯一比例1-(1-k/n)^20。按全量候选加权，正99.9999044%、负99.9999046%；这不是对SHA256确定性抽样作数学独立性证明。仅每域sample_key前8张、共32张实际构造20轮mask，均有20个不同集合，union覆盖100%；不可将子样本结果冒充全量实测。空候选覆盖为N/A，n=1时floor=0始终不选。原4P的P2/P3未覆盖，P2E与P2F保持不同编号。

## 4. 全数据训练结果与恢复

| 组 | 首步loss | 末步loss | 步均loss | 步均正贡献 | 步均负贡献 | 热步中位ms | 全程秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.691038 | 0.398374 | 0.556604 | 0.263156 | 0.293448 | 137.47 | 213.36 |
| P1 | 0.690831 | 0.352453 | 0.555877 | 0.262741 | 0.293136 | 139.88 | 222.99 |
| P2F | 0.689142 | 0.395695 | 0.555773 | 0.263903 | 0.291870 | 139.11 | 225.99 |
| P3F | 0.692186 | 0.345896 | 0.534845 | 0.253768 | 0.281077 | 139.30 | 246.86 |

四组峰值allocated=650,857,472 bytes（620.71 MiB），reserved=715,128,832 bytes（682 MiB），是PyTorch分配器峰值而非全部驱动显存。热步计时含H2D、前后向、诊断及优化器，不含外部数据加载；总耗时包含数据读取、保存和恢复重放。loss不要求单调；更低训练loss也不证明更好分割。

每一步检查logits、梯度、参数有限，clip_norm=0.5后范数<=0.50001，loss等于正负贡献之和；全部通过。优化器Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，float32，constant LR，四组完全相同。

各组update4.pt从磁盘恢复模型、完整注意力memory、Adam及Python/NumPy/Torch/CUDA RNG；重做第5步，与不中断分支loss及模型参数/buffer/memory digest精确一致。resume.json中的full_state_exact字段在本实现仅由loss及模型/memory哈希支持，不能扩大为已逐tensor比较Adam及所有RNG状态。旧checkpoint契约单测另覆盖恢复行为。末尾epoch1.pt可被独立evaluator完整载入并校验state+memory hash；本轮没有执行第2Epoch，也不将预检checkpoint用作正式初始化。

## 5. 严格评估与ExternalAttention

固定batch=1，PIL RGB BILINEAR到224并ImageNet标准化，无增强；保持冻结224坐标预处理。先载入完整学生state与非state_dict memory并验hash，之后eval，TinyViT据当前attention biases重建评估缓存。原始logits直接sigmoid，固定0.5，无逐图min-max。GT只在专用evaluation/data.py中按冻结metadata SHA及exact new_dev成员验证dataset/id/RGB SHA/native尺寸后读取，native二值0/255经NEAREST224并>127。空GT且空预测F1/IoU=1，单侧空=0。

原ExternalAttention.memory_bank不是普通state_dict内容，且eval forward会更新它；实际memory混合仅训练态开启。严格evaluator使用独立新模型，每图前恢复checkpoint完整起始memory，训练模型不存在共享对象，修改不会传播回训练。第一版禁止batch>1。

真实原模型memory gate使用4P P1 checkpoint和前2张student_train RGB：同图重复、正逆序、独立冷重建logits均torch.equal，最大差0；快照及磁盘checkpoint未变。另对本轮P0 Epoch1最终checkpoint重复真实检查，全部通过。原eval不reset的同图重复在该模型也相等，符合eval不混合memory代码语义；不能宣称原eval在已测样本上出现了实际顺序差异。严格协议仍采用逐图reset，明确排除未来状态传播。仅这两张RGB执行过完整正逆/冷构造检查，未把它夸大为530张全排列实测；全部530张均使用相同强制reset路径。

原eval逐图min-max语义与本严格raw sigmoid协议不同，不能把旧成绩直接并列作性能归因。

## 6. new_dev完整单次评估闭环（仅开发工程结果）

| 组 | image-macro F1 | image-macro IoU | pooled micro F1 | pooled micro IoU | 域等权F1 | 域等权IoU | 评估秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.186627 | 0.121262 | 0.245636 | 0.140014 | 0.284857 | 0.195586 | 29.03 |
| P1 | 0.193589 | 0.127398 | 0.251704 | 0.143971 | 0.298708 | 0.210257 | 28.73 |
| P2F | 0.188188 | 0.122896 | 0.245487 | 0.139918 | 0.294549 | 0.204916 | 28.70 |
| P3F | 0.187117 | 0.121600 | 0.250764 | 0.143356 | 0.304197 | 0.210647 | 28.62 |

image-macro先每图求指标再平均；micro合并像素混淆计数，域等权先求各域image-macro再四域平均。没有将像素视为独立重复实验，没有计算虚构显著性或用开发CI替代独立确认。

| 域 | 组 | 图 | 来源代理组 | macro F1 | macro IoU | micro F1 | micro IoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CASIAv2 | P0 | 410 | 202 | 0.161794 | 0.101427 | 0.197182 | 0.109374 |
| CASIAv2 | P1 | 410 | 202 | 0.165418 | 0.104122 | 0.204698 | 0.114019 |
| CASIAv2 | P2F | 410 | 202 | 0.161075 | 0.100865 | 0.196357 | 0.108867 |
| CASIAv2 | P3F | 410 | 202 | 0.157636 | 0.098045 | 0.198316 | 0.110073 |
| NIST16 | P0 | 80 | 11 | 0.207970 | 0.139541 | 0.353548 | 0.214733 |
| NIST16 | P1 | 80 | 11 | 0.222294 | 0.151163 | 0.332604 | 0.199475 |
| NIST16 | P2F | 80 | 11 | 0.213072 | 0.144042 | 0.341499 | 0.205908 |
| NIST16 | P3F | 80 | 11 | 0.205184 | 0.137831 | 0.323318 | 0.192832 |
| Columbia | P0 | 26 | 6 | 0.430310 | 0.328432 | 0.444889 | 0.286082 |
| Columbia | P1 | 26 | 6 | 0.474790 | 0.378119 | 0.485166 | 0.320276 |
| Columbia | P2F | 26 | 6 | 0.449707 | 0.350859 | 0.464280 | 0.302320 |
| Columbia | P3F | 26 | 6 | 0.514261 | 0.394085 | 0.528994 | 0.359614 |
| Coverage | P0 | 14 | 14 | 0.339353 | 0.212942 | 0.356266 | 0.216742 |
| Coverage | P1 | 14 | 14 | 0.332329 | 0.207625 | 0.353184 | 0.214465 |
| Coverage | P2F | 14 | 14 | 0.354343 | 0.223897 | 0.368620 | 0.225956 |
| Coverage | P3F | 14 | 14 | 0.339710 | 0.212627 | 0.357003 | 0.217288 |

这些数值完整保留，包括P3F的image-macro F1低于P2F这一单Epoch观察；不据此选择或剔除任何主组。域等权和image-macro可能给出不同表象，既不代表独立复现，也不能宣称已证实排序有益或无益。所有主对照仍冻结为P1−P0、P2F−P1、P3F−P2F。

## 7. 预测面积、数值分布与风险

| 组 | 预测面积 | 概率均值 | 全零 | >=99%面积 | <=1%面积 | 均值饱和 | 预测正类均值p | 预测负类均值1-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 26.2493% | 0.370105 | 0 | 0 | 0 | 0 | 0.709295 | 0.744957 |
| P1 | 26.3725% | 0.371339 | 0 | 0 | 0 | 0 | 0.703400 | 0.739290 |
| P2F | 27.3856% | 0.379819 | 0 | 0 | 1 | 0 | 0.713135 | 0.738755 |
| P3F | 28.5564% | 0.380127 | 0 | 0 | 0 | 0 | 0.731835 | 0.755277 |

new_dev GT平均前景7.2513%；四组预测26.2493%–28.5564%，明显面积偏大且F1仅0.1866–0.1936，是应披露的低质量/过预测风险。它们不是全零或全前景崩溃。P2F有1张面积<=1%，但没有<=0.1%异常低面积样本；四组>=95%异常高面积均0。按概率图像均值<=0.01或>=0.99定义的饱和均0，不等于没有局部接近0/1像素。没有通过调阈值掩盖该风险。

| 组 | 全局logit min | 全局logit max | 图均logit q01/q50/q99 | 图均概率q01/q50/q99 |
| --- | --- | --- | --- | --- |
| P0 | -5.09969 | 10.06444 | -2.47430/-0.82589/3.02895 | 0.08979/0.31611/0.91468 |
| P1 | -5.79335 | 11.08037 | -2.38038/-0.80835/2.85486 | 0.09844/0.31927/0.90798 |
| P2F | -4.86985 | 10.79828 | -2.33940/-0.78028/3.13695 | 0.09920/0.32517/0.91990 |
| P3F | -5.35464 | 12.41519 | -2.47787/-0.81854/3.61554 | 0.08770/0.31876/0.93718 |

分位点是逐图求分位后图像等权平均，不是全像素pooled分位。完整7档分位、预测正负置信度、空GT数和全套分域异常计数保存在output_diagnostics.json和原始逐图JSONL；该派生汇总只读既有日志，未重新读取GT。

| 域 | 组 | 训练教师面积 | 训练过程学生面积 | 训练概率均值 | dev固定模型面积 | dev概率均值 | dev全零 | dev>=99% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CASIAv2 | P0 | 8.958% | 29.968% | 0.4096 | 25.897% | 0.3642 | 0 | 0 |
| CASIAv2 | P1 | 8.887% | 30.015% | 0.4095 | 25.761% | 0.3647 | 0 | 0 |
| CASIAv2 | P2F | 8.887% | 29.622% | 0.4081 | 26.742% | 0.3725 | 0 | 0 |
| CASIAv2 | P3F | 8.887% | 31.655% | 0.4136 | 27.401% | 0.3704 | 0 | 0 |
| NIST16 | P0 | 5.606% | 23.889% | 0.3930 | 26.075% | 0.3831 | 0 | 0 |
| NIST16 | P1 | 5.597% | 24.367% | 0.3959 | 27.577% | 0.3882 | 0 | 0 |
| NIST16 | P2F | 5.597% | 23.650% | 0.3910 | 29.318% | 0.4014 | 0 | 0 |
| NIST16 | P3F | 5.597% | 25.505% | 0.3933 | 30.943% | 0.4049 | 0 | 0 |
| Columbia | P0 | 16.364% | 31.694% | 0.4187 | 27.081% | 0.3855 | 0 | 0 |
| Columbia | P1 | 15.720% | 31.882% | 0.4208 | 27.164% | 0.3878 | 0 | 0 |
| Columbia | P2F | 15.720% | 31.303% | 0.4179 | 26.431% | 0.3927 | 0 | 0 |
| Columbia | P3F | 15.720% | 32.514% | 0.4174 | 32.240% | 0.4091 | 0 | 0 |
| Coverage | P0 | 10.123% | 37.808% | 0.4462 | 36.028% | 0.4416 | 0 | 0 |
| Coverage | P1 | 10.155% | 37.453% | 0.4468 | 35.941% | 0.4377 | 0 | 0 |
| Coverage | P2F | 10.155% | 36.133% | 0.4406 | 36.951% | 0.4457 | 0 | 0 |
| Coverage | P3F | 10.155% | 40.241% | 0.4606 | 41.921% | 0.4705 | 0 | 0 |

训练过程和dev固定模型是不同成员/参数状态，不能直接相减作泛化或损失偏置因果估计。未用本轮成绩更改类权重。缺少正式收敛曲线、多seed及独立确认，本阶段不证明强分割能力，也不声称改进教师或筛选一定改善学生。

## 8. 数据隔离、保护与验收

训练入口及Dataset完全不导入evaluator，不接受GT路径或tensor。四组DataReadAudit各记录7060个唯一允许文件（3530 RGB + 对应K1或K8缓存）；实际打开集合与冻结白名单精确相等，denied为空。此为CPython文件open事件与受审代码路径检查，不冒充操作系统级沙箱或对任意恶意代码的形式证明。

独立evaluator每组恰530次GT读取，总2120次，逐项日志均为冻结new_dev成员及配对prepared_mask。new_confirm GT访问0，teacher_diag/正式测试GT访问0；没有复制GT到训练缓存/索引。GT路径仅保存在明确evaluation目录的专用allowlist中。eval前后每个磁盘checkpoint SHA相同；每组保存配置、checkpoint config/model hash、成员canonical hash、GT访问日志hash、逐图结果hash及summary result_hash。

验收器check_stage4f0.py只读取日志、metadata及文件哈希，不打开GT图像；验证全部监督floor、逐图选择hash、跨组顺序增强、梯度、loss、恢复及统计重聚合。17610个历史文件哈希未变，16个训练前冻结代码哈希未变。完整fecd 147项单测PASS；17个新增Python源文件通过stdlib AST语法和符号表未解析全局名检查，issues=0，配置JSON与冻结字典一致；未安装/新增第三方linter或完整类型检查器。

## 9. 实际新增文件与运行证据

- training：f0_selection.py、f0_data.py、f0_protocol.py、f0_diagnostics.py、f0_engine.py。
- evaluation：data.py、metrics.py、memory.py、memory_audit.py、strict.py。
- tools：audit_fixed_selection.py、train_epoch_preflight.py、evaluate_student_strict.py、check_stage4f0.py。
- tests：test_stage4f0_selection.py、test_stage4f0_diagnostics.py、test_stage4f0_evaluation.py。
- configs/stage4f0：P0/P1/P2F/P3F/P2E.json、evaluation_protocol.json、formal20_proposal.json。
- docs：本报告、stage4f0_formal_protocol.md。
- assets/stage4f0：selection_audit、两个真实memory gate、训练前gates、protected_before、static_and_tests、output_diagnostics、最终validation；四组config/steps/train_outputs/access/result/training_summary/resume/update4.pt/epoch1.pt；evaluation四组config/allowlist/access/per_image/summary；logs下15份原始运行日志。

以下命令在/home/projects/BoxPromtIML执行，Python=/usr/local/miniconda3/envs/boxpromptiml/bin/python，日志已保存在assets/stage4f0/logs。模型注册重复提示来自原TinyViT导入，未改动原文件规避提示。

```bash
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest fecd.tests.test_stage4f0_selection fecd.tests.test_stage4f0_diagnostics fecd.tests.test_stage4f0_evaluation
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest discover -s fecd/tests
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.audit_fixed_selection
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_epoch_preflight --variant P0
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_epoch_preflight --variant P1
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_epoch_preflight --variant P2F
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_epoch_preflight --variant P3F
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.evaluate_student_strict --checkpoint fecd/assets/stage4f0/P0/epoch1.pt --output-dir fecd/assets/stage4f0/evaluation/P0
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.evaluate_student_strict --checkpoint fecd/assets/stage4f0/P1/epoch1.pt --output-dir fecd/assets/stage4f0/evaluation/P1
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.evaluate_student_strict --checkpoint fecd/assets/stage4f0/P2F/epoch1.pt --output-dir fecd/assets/stage4f0/evaluation/P2F
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.evaluate_student_strict --checkpoint fecd/assets/stage4f0/P3F/epoch1.pt --output-dir fecd/assets/stage4f0/evaluation/P3F
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_stage4f0
```

两个memory gate通过Python直接调用fecd.evaluation.memory_audit.audit_memory：分别输入fecd/assets/stage4p/P1/final.pt和本轮P0/epoch1.pt，输出memory_gate.json及epoch1_memory_gate.json。训练/评估/最终验收入口拒绝覆盖已有目录或结果；不能直接重跑上述命令覆盖冻结产物。GT-free选择审计实际调用时的参数见脚本与日志，默认规则即本轮冻结规则。

## 10. 正式预算及阶段行动

后续正式提案详见stage4f0_formal_protocol.md及formal20_proposal.json：一个共同seed、batch4、20Epoch、每组17660逻辑更新、四组总70640；同Adam/初始化/数据序/增强/损失，固定Epoch20 final。P2E单独登记、未自动纳入；多seed须所有主组等同另行授权。基于本次端到端每组213–247秒/Epoch作粗略线性预算，四主组约5.1小时训练，建议预留约6小时及评估/IO余量；不是已执行耗时，也不保证长期性能和吞吐不变。

预先停止线：技术异常立即BLOCKED；new_dev至少50%图全零或面积>=0.99且image-macro F1<=0.05，或至少90%图概率均值<=0.01/ >=0.99，停止正式申请。四组均未触发。这不将低F1或面积偏大解释为已解决；完整开发数值和风险保留，后续不得根据某组的一次成绩改各组协议。

**结论：4F0工程PASS，formal_request_ready=true仅指具备提交冻结受控实验申请的技术条件。没有学生优势结论，没有独立确认结果。等待用户验收和下一阶段授权；不自动进入4F1，不访问锁定GT，不把本轮checkpoint用于正式初始化。**

最终只读验收result_hash=0d2a4be47a48325f2ff4072e98e68fed2135233858f5b8f1977ea0d95bac4f53。
