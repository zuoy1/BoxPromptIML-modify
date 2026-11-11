# BoxPromptIML 阶段5C0：区域分层困难负类学习机制预检

日期：2026-11-05。大型产物根目录：`/hy-tmp/BoxPromtIML/fecd_runs/stage5c0`。

## 工程结果与研究边界

**工程PASS：R0/R1满足逐图、逐区域公平性，在线困难选择、单Epoch训练和完整checkpoint恢复闭环通过。**
具备提交5C1受控正式对照申请的工程条件；不代表R1有效或有新颖性，不构成下一阶段授权。
历史4F1的NO CLEAR GAIN、FECD-R/q_A/B-R停止结论保持不变。
训练进程未读取任何GT；本阶段未解码或使用new_dev GT，未读取new_confirm、teacher_diag或正式IND/OOD GT，没有SAM推理。历史完整性检查只对5A已保存产物作字节级SHA256，其中可能包含GT参考副本的封装文件；没有将其数组载入训练或评估。

## 冻结实验与实际实现

R0/R1只改变hard-negative候选区域内的排序依据。teacher为K8 v2，float32(mu)>=0.5；历史正类不被Box区域覆盖。
原始224网格的正类用seed=20261029、sample_key、pos与固定epoch0随机顺序取floor(n_pos/2)。
hard负类分为Box并集内外；每区各取floor(n_region/2)，0或1个候选时选0，不补齐、不重分配。
R0负类按seed、epoch、sample_key、region的SHA256派生NumPy RNG随机选；R1按当前forward的FP32 sigmoid概率降序，同值复用该随机优先级。
R1先detach，按实际增强操作逆变换回canonical网格，选择后同步正向变换mask；不做第二次forward，选择不参与反传。
固定六离散几何与原4F1数据顺序；原逐图0.5所选正类均值+0.5所选负类均值BCE不变，空类为0且不重分配。框内外负类合在同一负类均值中，不另加区域损失或区域权重。
Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0；constant LR、FP32、clip_norm=0.5、Batch4、drop_last=False。

## 真实训练、初始化与恢复

|组|图像|逻辑更新|平均训练Loss|平均训练前向预测面积|耗时秒|裁剪后最大梯度|全零/近全前景|
|---|---:|---:|---:|---:|---:|---:|---:|
|R0|3530|883|0.555868|29.3957%|267.05|0.500000|0/0|
|R1|3530|883|0.637632|17.7670%|311.77|0.500000|13/0|

两组仅各训练1个完整Epoch（epoch索引0），末batch均为2。四域计数均为CASIAv2=3278、NIST16=172、Columbia=52、Coverage=28。
共同模型+memory初始hash：`1086c22f6f2a7d0756d4ad9c7de33f52c5d4e6fffeee5e064a77690bb148d516`。原始网络不改，TinyViT从已验证本地预训练适配器严格载入，其余新seed初始化；未读历史final。
共同顺序hash：`dad1acb6f8fdbd7dc3f3aa382b7ff45bc811e69187a8b92e55436060a32581d0`；几何hash：`e648c800b40c2a717b6713c44515eebf41062173d2d81239aeda179a95b51edb`。
先做独立单batch更新，再保存update1，比较未中断下一步与重新构造模型并从磁盘恢复的下一步。两组loss、canonical mask、诊断、完整model/Adam/memory/Python/NumPy/Torch/CUDA RNG哈希和训练位置均精确一致。
checkpoint包含完整模型、optimizer、未注册为普通state_dict的ExternalAttention memory、全部RNG、config和训练位置；单Epoch终态为update883、epoch1、next_batch0。
训练前审查发现并修复诊断概率精度不一致，并补真实六增强接线测试；旧探针保存在probe_before_diagnostic_fix，最终探针在probe。每组前后各3次物理探针更新，状态hash证明仅诊断精度修正没有改变训练轨迹。
因此每组物理更新为883+6=889，总计1778；其中正式单Epoch轨迹为883×2=1766次。所有探针模型均丢弃，完整Epoch从共同seed重新初始化。
**表中Loss和预测面积来自参数持续变化的训练前向，不是held-out评估、收敛结论或效果比较。没有用Smoke Loss选优、改预算、阈值或权重。**
**实际安全信号：R1有13张训练前向输出全背景（CASIAv2 12、NIST16 1），13张K8教师均有正类；R0为0张。** 这是风险提示，不是独立GT漏检率，也不是最终checkpoint退化的评估。相关身份和概率统计见results/zero_output_cases.json。不能将预测面积下降当成改善，5C1必须严查微小目标Recall/FN和域退化。没有出现数值异常、空监督或恢复失败，因此本信号不被伪装成工程故障，也不被解释为安全性已证明。

## 监督公平性与困难负类诊断

独立审计复核7060张packed canonical mask、3530成员的区域标签与选择预算，正类集合在两组完全相同；逐图每区域负类均符合floor。R0使用独立SHA256/RNG重建，区域候选与正类标签保持原缓存。
R1每区selected概率下界不低于unselected概率上界，允许实际FP32 sigmoid饱和并列；并列规则与ignored梯度由契约测试和精确恢复补证。未保存完整逐像素学生概率，因此产物不能独立重建每一处并列的全排序，不能夸大审计能力。

|组/区域|负类候选|选中|选中mu均值|选中var均值|选中c_sam均值|选中/未选学生概率|选中/未选BCE|
|---|---:|---:|---:|---:|---:|---:|---:|
|R0/in|12858918|6428547|0.058529|0.008430|0.870690|0.503189/0.503105|0.854698/0.854566|
|R0/out|148609067|74304063|0.000934|0.000131|0.998158|0.384475/0.384485|0.558740/0.558782|
|R1/in|12858918|6428547|0.063684|0.008263|0.863383|0.538515/0.399631|0.860901/0.523715|
|R1/out|148609067|74304063|0.000934|0.000124|0.998195|0.461732/0.360982|0.649686/0.451510|

以上是区域内像素描述性加权均值，相关像素不作为独立统计样本。逐图n/均值/min/q10/q50/q90/max分布保存在epoch/<R>/images.jsonl；四域、极小Box≤1%、大Box>25%及多Box≥2分层保存在diagnostic_summary.csv/json。
原始mu、var、c_sam只做诊断，不参与R0/R1选择或损失加权。负类mu接近0.5、方差较高或低c_sam可能提示伪负类风险，但无训练GT，不能确认它们是真背景或SAM漏检。

|组/区域|选中mu≥0.4（占该区所选负类）|选中var≥0.1|同输出快照下一Epoch选择Jaccard图均|空/单候选区域图数|
|---|---:|---:|---:|---:|
|R0/in|225499 (3.5078%)|166962 (2.5972%)|0.333173|1/1|
|R0/out|10470 (0.0141%)|24868 (0.0335%)|0.333353|15/0|
|R1/in|247133 (3.8443%)|155940 (2.4257%)|0.999949|1/1|
|R1/out|9861 (0.0133%)|22115 (0.0298%)|0.999993|15/0|

这两个数值区间仅作预定义描述性分箱，不是GT条件规则，也未用于训练决策。
**跨Epoch限制：只训练一个Epoch，每张图仅参与一次真实更新，无法观察实际跨多Epoch的重复压制。** 已保存的是同一当前学生输出快照下仅将epoch0改为epoch1的反事实选择变化；R1非并列高分区域可维持选择，不能把这当作已发生的多轮强化或因果机制。5C1草案要求逐Epoch记录真实集合变化。
危险监督区域需要在5C1通过独立开发评估的微小目标Recall/FN等安全指标观察；本阶段不能用训练GT消除这些风险或据此事后设规则。

## 文献与下一阶段申请

四篇文献均完成全文核对，见docs/literature.md和literature/sources.json（正式PDF、URL、HTTP状态、SHA256）：OHEM（Shrivastava等，CVPR2016）、BoxInst（Tian等，CVPR2021）、DiscoBox（Lan等，ICCV2021）、WeakMed（Jun Wei、Hui Huang，CVPR2026，Rethinking Box Supervision）。
OHEM是当前难例损失排序；BoxInst是投影及颜色关系约束；DiscoBox是结构化教师与跨图关系；WeakMed是M2B投影/反投影与尺度一致性。R1仅为常规在线难负类挖掘加区域分层的强基线，本阶段未加入篡改特有证据，不能宣称创新。
5C1正式申请草案见docs/stage5c1_proposal.md：同seed新初始化R0/R1各20Epoch、固定Epoch20 final、严格new_dev评估；主指标image-macro Pixel F1，微小/大目标Recall、四域/域等权F1、FP/预测面积、组配对bootstrap及面积偏差作安全指标。
草案预先给出继续/停止门槛；未经用户验收与另行授权不执行。本次没有据开发集成绩调采样比例、阈值或损失；new_dev评估为0。单seed不得声称跨seed稳定。

## 实际命令、测试与历史保护

工作目录`/home/projects/BoxPromtIML`，Python `/usr/local/miniconda3/envs/boxpromptiml/bin/python`，真实设备RTX4090。命令和日志索引见commands.json。
训练命令：`PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_region_smoke --variant R0 --mode probe`；R1同命令替换variant；完整单Epoch以`--mode epoch`执行。
新增17项契约测试通过；项目完整unittest共214项通过；stdlib trace记录定向测试行覆盖，选择模块67/67可执行行，GPU runner由真实探针/完整Epoch覆盖，不能宣称其单元或分支覆盖100%。audit回调在CPython跟踪抑制期间执行，trace缺失不等于未通过边界测试。
静态AST/编译/symtable未定义全局检查见results/static_analysis.json；没有安装新依赖。训练前review两项问题已修复，见docs/runtime_review.md。
历史只读快照复核3522文件，变化0；范围覆盖原tracked文件、历史4F1/5A、完整5B。训练3530缓存与5B及本轮双组哈希核对。只宣称已快照范围，不声称全盘逐文件验证。
独立产物审计状态：`PASS`，详见docs/artifact_verification.md。最终计划验收见docs/verification.md。

## 七项验收回答

1. 公平性：满足。逐图正类相同，框内/框外负类数量相同，原损失归一化、初始化、数据顺序、增强与883更新相同。
2. 困难选择：符合定义。选当前forward的实际sigmoid概率最高负类，确定性随机tie，不向选择传播梯度。
3. 危险监督：存在潜在伪负类风险，分布与特殊Box层已披露；未用训练GT确认/排除。仅一Epoch不能断言跨轮重复压制已发生或安全。
4. 单Epoch：两组完整运行正常，无非有限值或异常空监督；训练前向全零/近全前景计数见表。
5. 恢复：两组从磁盘恢复下一步精确一致，含optimizer、memory、全部RNG及位置。
6. 5C1：具备提交正式受控训练申请的工程条件；效果与安全性仍未证明，须用户另行批准。
7. 阶段结果：PASS。到此停止，不运行20Epoch、多seed或开放确认集。
