# BoxPromptIML 阶段5C1正式困难负类学习受控实验报告

日期：2026-11-06（北京时间）。产物：`/hy-tmp/BoxPromtIML/fecd_runs/stage5c1/`。

**工程 PASS；科研 NEGATIVE SIGNAL。** 两组各20 Epoch、17660次更新完成；Epoch20 final各严格评估530张new_dev一次。R1未达到预注册增益要求，且微小/大目标Recall与Columbia F1触发安全负面门槛。原样保留失败证据，到此停止。

R1仅为区域分层OHEM强基线。历史4F1 NO CLEAR GAIN、FECD-R/q_A/B-R停止结论保持不变；本结果不是FECD创新成果。

## 冻结设计与真实接口

共同seed=20261029，student_train3530张，K8 v2 hard阈值0.5。固定随机floor(n_pos/2)正类；hard负类按弱Box并集inside/outside各取floor(n_region/2)。不覆盖框外K8正类，0/1候选严格floor不补齐。R0按seed/epoch/key/region随机重采样；R1在当前forward后以detach FP32 sigmoid概率取区域最高半数，用冻结随机优先级处理并列。

原224 canonical网格上逆几何选择，再与图像同步正向几何；六种冻结离散增强、每轮数据顺序一致。相同逐图0.5正类均值+0.5负类均值BCE，inside/outside合并负类均值；空类为0不重分配。原网络不变，无新增分支/损失/特征/SAM推理。

Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、constant LR、FP32、clip norm=0.5、batch4、drop_last=False；每轮883更新，末批2。共同fresh TinyViT本地预训练及其他层seed初始化，不继承5C0探针/单轮或历史4F1 final。

共同初始model+memory SHA256：`1086c22f6f2a7d0756d4ad9c7de33f52c5d4e6fffeee5e064a77690bb148d516`。TinyViT SHA256：`e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d`。

申请草案实际路径为`stage5c0/docs/stage5c1_proposal.md`，项目fecd/docs没有同名副本；已读取实际冻结草案，路径差异记录在results/protocol_interface_notes.json，未改变研究方法。原草案“未授权”历史文字不改，本次用户授权覆盖执行状态。

## 训练工程与恢复

RTX4090，/hy-tmp为独立xfs数据盘。按R0后R1顺序完整执行，无训练中dev评估、无调参/早停/重启；自动关机始终关闭。每组70600图次，两组总35320逻辑更新。正式配置独立恢复探针每组3物理更新，共6次额外物理更新；探针丢弃，正式fresh重建，总物理更新35326。

每轮完整保存模型、Adam、ExternalAttention memory、Python/NumPy/Torch/CUDA RNG及训练位置。预检两组磁盘恢复下一步loss/诊断/mask/完整状态精确一致；实际正式训练未中断，未发生恢复。独立CPU验收40个checkpoint完整状态hash、位置和Adam步数。

正式前复审修复PASS提交顺序、恢复位置核对、参数实际更新证明和组级BLOCKED拒绝；runtime冻结v1/v2及差异保留。所有检查通过后才启动正式训练，不更改选择/优化方法。

预先冻结工程塌缩门禁为连续两轮≥90%训练图次全背景或近全前景；非有限值、GT/身份/预算/位置/恢复异常立即BLOCKED。本次未触发。

|组|Epoch|更新|耗时秒|Epoch20训练Loss|Epoch20训练前向面积|全背景图次|近全前景图次|
|---|---|---|---|---|---|---|---|
|R0|20|17660|5557.368244|0.178463|0.127621|0|0|
|R1|20|17660|6464.212242|0.249368|0.114942|31|0|

R0逐轮全背景均0；R1逐轮计数为13,1,2,2,6,4,0,2,0,0,1,0,0,0,0,0,0,0,0,0，共31图次。两组近全前景均0。训练不读GT，这些不是GT漏检率；训练Loss和面积不用于判断有效性。逐轮梯度、GPU峰值、Loss贡献、时间、全背景身份见results/training_summary.json及CSV与各epoch日志。

## 逐图公平性、困难排序与选择风险

40轮逐图/逐区域floor预算、固定正类mask、候选集合、同epoch顺序与增强均通过。每轮每组选择正类7825752、inside负类6428547、outside负类74304063，总88558362像素；20轮分别156515040、128570940、1486081260，总1771167240像素监督事件。两组相同，非唯一像素数。

R0全部70600个完整canonical选择mask独立随机重建逐位一致。R1全部70600图的140860个非空区域满足selected学生概率min≥unselected max，340个零选择区域正确；5696个边界并列允许实际FP32 sigmoid相同。梯度detach/ignored梯度为零及确定性tie由冻结契约和恢复测试补证。

**审计限制：没有保存完整逐像素学生概率，不能独立重建每个R1并列的完整排序；边界检查不能宣称完整排序重放。**

实际相邻Epoch Jaccard与累计唯一覆盖由保存的canonical mask独立重算，不使用5C0反事实next_epoch字段。下表频次分母为固定hard负类候选，区域内像素相关，非独立统计样本。

|组|区域|候选像素|20轮累计覆盖|每轮均选像素|每轮均选占比|
|---|---|---|---|---|---|
|R0|inside|12858918|0.999999|8|0.000001|
|R0|outside|148609067|0.999999|120|0.000001|
|R1|inside|12858918|0.923590|1335771|0.103879|
|R1|outside|148609067|0.992726|3367757|0.022662|

R1存在持续选择同一批像素，框内更明显。R0/R1累计覆盖不同是必须披露的混杂，不能把最终差异全部归因为单纯难度排序。没有训练GT，不能认定这些像素是真背景或SAM漏检，也未据此设计GT规则。

已选/未选的原K8 mu、var、c_sam、学生概率和BCE，每图分布及域内、极小/多Box信息保留在images.jsonl，逐轮域汇总见training_summary。mu>=0.4与var>=0.1沿用5C0描述性分箱，不参与决策。分域Loss/梯度为域图次加权的混合batch exposure，非独立域损失；分位数均值是逐图分位数平均，非像素池化分位数。

## 严格new_dev评估

双方全部训练结束后，每组独立模型各完整评估530张一次：raw logits→sigmoid、threshold0.5、batch1、每图恢复checkpoint起始ExternalAttention memory，无逐图min-max。协议hash `54baa31ac65ec3df978af09c43b2454fd5515db1c8117b86c6ff2cd7d267416f`。

独立memory重复/逆序/cold-model门禁全部通过，仅用两张student_train RGB，GT=0。真正GT只由冻结独立evaluator读取，两组各530条访问；metadata阶段零像素读取。访问allowlist外立即拒绝；new_confirm、teacher_diag、IND/OOD GT实际访问均0，checkpoint评估前后hash一致。

评估入口初审发现lazy import临时目录探测被严格写审计拒绝，正式评估前通过零数据预加载修复并做真实新进程RED/GREEN；保持读写边界，v1证据及v2冻结保留，独立复审APPROVED。

|指标（比例0–1）|R0|R1|R1−R0|
|---|---|---|---|
|f1|0.349719|0.356834|0.007115|
|iou|0.272901|0.283619|0.010718|
|precision|0.296551|0.322918|0.026366|
|recall|0.651001|0.590990|-0.060011|
|micro_f1|0.442607|0.418671|-0.023937|
|micro_iou|0.284198|0.264759|-0.019439|
|prediction_area|0.164294|0.141618|-0.022676|
|gt_area|0.072513|0.072513|0.000000|
|area_bias|0.091782|0.069106|-0.022676|
|area_mae|0.105824|0.095132|-0.010692|
|tp|1393652|1192048|-201604.000000|
|fp|2975467|2574048|-401419.000000|
|fn|534694|736298|201604.000000|
|all_background_images|0|0|0.000000|
|severe_overprediction_images|78|69|-9.000000|

Precision/Recall为image-macro；micro F1/IoU由pooled混淆计数计算。严重过预测沿用4F1：预测面积≥20%且≥GT面积3倍；描述案例不调参。

## 来源代理组配对Bootstrap

固定2000次、seed20261029。以(dataset,source_group_id)联合分组，每域有放回抽同数来源组并保留组内全图；两组、各指标和small/large层共用抽样。层在抽样后筛，未定义draw不记有效；域等权要求点估计有该层的域均有效。95% percentile区间，像素不作独立样本。

主指标为new_dev image-macro Pixel F1，唯一主比较R1−R0。整体530图、233来源代理组。来源代理并非已证明真实独立来源；沿用CASIA巨型连通分量和NIST本地重建协议限制。

|指标|权重|差值|95% CI|有效次数|
|---|---|---|---|---|
|f1|image_equal|0.007115|[-0.016446, 0.026407]|2000|
|f1|group_equal|0.006431|[-0.014947, 0.024837]|2000|
|f1|dataset_equal|-0.021844|[-0.062249, 0.013435]|2000|
|iou|image_equal|0.010718|[-0.010585, 0.028405]|2000|
|iou|group_equal|0.010519|[-0.006939, 0.026894]|2000|
|iou|dataset_equal|-0.019428|[-0.057658, 0.013651]|2000|
|recall|image_equal|-0.060011|[-0.101957, -0.019771]|2000|
|recall|group_equal|-0.061868|[-0.102079, -0.025053]|2000|
|recall|dataset_equal|-0.099764|[-0.162443, -0.046891]|2000|
|area_mae|image_equal|-0.010692|[-0.025855, 0.005708]|2000|
|area_mae|group_equal|-0.006226|[-0.021685, 0.010383]|2000|
|area_mae|dataset_equal|-0.012989|[-0.034720, 0.010081]|2000|

## 四域与Small/Large安全分析

|域|图|组|R0 F1|R1 F1|F1差|95% CI|R0 IoU|R1 IoU|
|---|---|---|---|---|---|---|---|---|
|CASIAv2|410|202|0.326536|0.345805|0.019269|[-0.000260, 0.036700]|0.246905|0.270470|
|Columbia|26|6|0.872496|0.813196|-0.059300|[-0.100333, -0.017078]|0.826180|0.759696|
|Coverage|14|14|0.458006|0.439819|-0.018187|[-0.136042, 0.073687]|0.327983|0.319941|
|NIST16|80|11|0.279677|0.250519|-0.029158|[-0.152781, 0.043397]|0.216678|0.189929|

|层|图|有效组|R0 Recall|R1 Recall|差|95% CI|有效次数|
|---|---|---|---|---|---|---|---|
|small|126|82|0.563617|0.489497|-0.074119|[-0.215345, 0.048518]|2000|
|large|40|26|0.754999|0.632898|-0.122101|[-0.185498, -0.058701]|2000|

small=GT≤1%；large=GT>25%，均按冻结224 GT网格面积。逐域有效来源组和有效重抽样次数：

|层|域|图|组|有效重抽样|
|---|---|---|---|---|
|small|CASIAv2|98|76|2000|
|small|Columbia|0|0|0|
|small|Coverage|0|0|0|
|small|NIST16|28|6|2000|
|large|CASIAv2|22|19|2000|
|large|Columbia|14|6|2000|
|large|Coverage|0|0|0|
|large|NIST16|4|1|1311|

R1 FP减少401419，但FN增加201604；总体Recall下降，pooled micro F1也下降。主image-macro F1虽略升，CI跨0，且安全分层明显退化，不能因为FP或面积偏差下降就宣布有效。

## 冻结科研判据

|正向条件|是否满足|
|---|---|
|main_f1_gain|False|
|main_ci_positive|False|
|small_recall_point|False|
|large_recall_point|False|
|small_recall_ci|False|
|large_recall_ci|False|
|every_domain_f1|False|
|dataset_equal_f1|False|
|area_mae_not_increased|True|

|安全负面条件|是否触发|
|---|---|
|small_recall|True|
|large_recall|True|
|domain_f1|True|
|fp_fn_tradeoff_no_gain|False|

**最终科研判断：NEGATIVE SIGNAL。** small Recall差−0.074119、large差−0.122101均小于−0.05；Columbia F1差−0.059300小于−0.05。主F1差+0.007115不足+0.01、CI下界−0.016446≤0，域等权差−0.021844。面积MAE改善不足以抵消这些门槛。

仅一个训练seed，不得声称跨seed稳定。new_dev已用于历史开发分析，本次固定协议也不是独立确认。对区域误激活机制、因果解释和论文创新均无正向证据；不自动修改方法或追加实验。

## 验收与交付索引

训练完整性/逐图区域公平性：results/training_verification.json；困难排序：results/selection_order_verification.json；训练逐轮汇总：results/training_summary.json、CSV；20轮实际频次：results/selection_frequency.json。

历史保护：results/preservation_verification.json，3643历史文件、3530 K8缓存、42运行文件hash变化0，仅宣称预先快照范围。checkpoint两组final SHA256分别为 `b54149c2e27566d13f3308bbecdebb9e89832a4304f98e96af78ef43558cb277`、`1cfe5a79d2501ac58388be1f68c1cb5f8f7b0d3bac7eccb2f1a35ff7443c4454`。

正式checkpoint：R0/final.pt、R1/final.pt；全部40轮checkpoint保留在各epochXX/checkpoint.pt。训练日志logs/R0_train.log、R1_train.log及每轮steps/images/coverage/access；两组strict结果evaluation/R0、R1；paired统计analysis/paired_bootstrap.json。

运行命令及返回码见results/execution.json；Python为/usr/local/miniconda3/envs/boxpromptiml/bin/python。测试、静态与独立最终验收文档见logs及docs。未安装新依赖；AST/compile/符号未定义检查通过，环境无Ruff/Mypy，不能宣称独立第三方lint/type通过。

工程PASS和科研负面结论分开。全部原始退化证据保留；不运行第二seed，不开放锁集，不重新选Epoch，不自动关机，由用户在云平台确认产物后停止实例。

### 独立统计复核补充

`docs/statistics_verification.md` 与 `results/statistics_verification.json` 对 1,060 条已保存逐图记录和既有统计结果作了 CPU-only 独立复核：992 项数值核对通过，最大误差 5.55e-17；Bootstrap、三种权重、三层分层及冻结科研裁决全部一致。small 层为126图/82组，large 层为40图/26组；NIST16 large 仅4图/1组，有效重抽样1311/2000，已按预注册协议披露。

### 选择顺序独立复核

`results/selection_order_verification.json` 由 CPU-only 审计生成：R0 70,600 个完整 canonical mask 逐位重建一致；R1 70,600 图、140,860 个非空区域满足所选概率不低于未选概率，340 个零选择区域正确，5,696 个边界并列按规则允许。训练未读取 GT；由于未保存逐像素概率，不能重放所有 R1 并列的完整排序。

### 域等权绝对性能

|指标|R0|R1|R1−R0|95%配对CI|
|---|---|---|---|---|
|四域等权F1|0.484179|0.462335|-0.021844|[-0.062249, 0.013435]|

### 测试证据

项目完整unittest 244项通过；数据盘产物工具测试17项通过。选择、覆盖、恢复、统计、GT边界和失败拒绝均有可运行契约。独立验收器的合成CPU checkpoint与排序测试见logs/independent_verifier_tests_final.log；静态检查见results/static_analysis_seal.json。

### 预注册门槛原值

只有以下条件全部满足才为有条件POSITIVE SIGNAL：主F1差≥+0.01、主95%CI下界>0；small与large Recall点差各≥−0.02、对应CI下界各≥−0.05；每域F1点差≥−0.03；域等权F1差≥0；平均绝对面积误差不增加。本次仅最后一项通过。

安全负面门槛为：small或large Recall点差<−0.05；任一域F1点差<−0.05；或FP减少、FN增加且主F1没有改善。本次前两类触发，第三类因主F1点差为正未触发。判据在结果前冻结，未修改。
