# BoxPromptIML 阶段 4P：受控学生蒸馏基线与训练闭环

日期：2026-10-29。**最终工程判断 PASS；没有学生效果结论。**

原 FECD-R NO-GO、q_A STOP、B-R STOP 及负控解释复核均原样保留。本次暂停新增取证特征和可靠性公式，只实现 P0–P3 的同协议 hard-label BCE 基线并运行最小 smoke。没有运行正式 20 epoch、多 seed 性能比较、GT 验证、new_confirm 或正式 IND/OOD 测试。

## 1. 实际范围与结果

四组均完成单 batch 前向/反向、有限梯度检查、裁剪、Adam 更新，以及 8 个逻辑更新的训练闭环；每组 batch=4，实际消费相同 32 张图，不满一 epoch。Dataset 固定长度为 student_train 3,530，而非把 32 张另建为训练划分。

按固定种子得到的前 32 张恰好全部属于 CASIAv2，仅 5 个来源代理组，其中 28 张来自 3,020 张大连通分量，未因结果改数据顺序或补挑好案例。这是 smoke 的明显覆盖限制：不能声称本次在真实 NIST16/Columbia/Coverage 输入上跑过学生反向，也不代表全量 epoch 的性能或稳定性已验证。跨域旧缓存验证和新合成数据契约测试不能冒充本轮真实跨域训练结果。

每组在第 4 步保存 checkpoint，实际从磁盘重建模型/优化器并恢复，验证第 5 步与未中断分支 loss、模型和注意力内存哈希完全相等。最终第 8 步 checkpoint 已保存。127 项 fecd 单测、9 项旧回归检查通过；新增 13 个 Python 文件静态检查 issues=0；17,551 个受保护旧文件/权重哈希未变。

## 2. 四个受控实验

P 编号独立于历史 B0–B5。所有实验使用同一个损失实现，不含原 structure_loss 或新 Box loss。

| 实验 | 教师hard标签 | 选择 | 唯一目的 |
| --- | --- | --- | --- |
| P0 | 冻结K1 mu≥0.5 | 全像素 | 新的K1 BCE基线 |
| P1 | 冻结K8 mu≥0.5 | 全像素 | 相对P0隔离教师标签变化 |
| P2 | 冻结K8 mu≥0.5 | 每图正负各floor(n/2)，随机 | 相对P1检查减少监督像素 |
| P3 | 冻结K8 mu≥0.5 | 每图正负各floor(n/2)，c_sam降序 | 相对P2检查同预算可靠性排序 |

P1–P3 的候选正负集合始终由同一 K8 hard 标签定义。没有 q_A、Noise、JPEG、FFT、B-R、soft target、Consistency 或 Confidence Head。c_sam 仅复用冻结公式；未按 smoke loss 调阈值或公式。P0/P1 可因教师标签不同有不同正负候选数，P2/P3 每图每类保留数严格相同。

## 3. 固定数据和预处理协议

派生 split hash：022da143a26833f1613db722ae3be152117948330621e028f02b824f2d1680d0。

student_train 3,530 个 sample_key 的 canonical membership hash：cc98c8c0222ecf20d6ba887fafa954eff07983b6fa360a43c5e0057757e50110。四组实际构造的身份序列相同，按 sample_key 排序后再使用 torch.randperm(generator seed=20261029) 生成训练顺序；共同 order hash 为 3ae9a1283720d6cab5ae9a8ef3a0b62cf16aaf95b8a64f9d588bb2d8e96d8521。

数据组成固定为 CASIA 3,278、NIST 172、Columbia 52、Coverage 28。构造时核验原派生划分/父哈希及身份隔离，不改变旧 train_fit/teacher_diag/new_dev/new_confirm。SAM K1/K8 v2 只读；缓存目录分别为 d7ccb769414f 和 2bff688546a7，完整 config hash 在 data.py 中固定。

ControlledDataset 包装已经验证的 FECDDataset，关闭其随机增强；读取时补充 RGB bytes SHA256/SHA1、缓存 sample_key/dataset/id/K/config/Box hash/native 尺寸核验。Dataset 只接受由冻结 student_train 索引导出的 RGB/cache/Box 字段，没有 GT 入口或 prepared mask 路径。

复用既有图像坐标定义：PIL RGB BILINEAR 显式 resize 到 224×224，ImageNet mean/std。这里方形映射是既有学生/SAM 统计协议，未改变坐标或 Box 导出。hard 标签直接取缓存 224 均值阈值；选择 mask 先在同一缓存网格计算，再与图像、标签共同做离散几何变换，没有插值 mask。

增强操作为 identity/hflip/vflip/rot90:1/2/3，由 SHA256(seed:epoch:sample_key:geometry) 前8字节决定，四组同图同 epoch 完全相同，和选择模式无关。无光度增强、裁剪、MixUp 或重采样。训练数据未使用 new_dev；本轮连专用评估 GT 都未打开。

## 4. 损失与监督预算定义

对每张图 b，P_b/N_b 为已经选入的正/负像素，L_b 为：

L_b = 0.5 × mean_{i in P_b} BCEWithLogits(z_i,1)
    + 0.5 × mean_{i in N_b} BCEWithLogits(z_i,0)

再对 batch 内图像等权平均。某类为空则该项为0、不重分配其0.5权重；全选集为空为可微0；忽略像素的分类梯度为0。分母只含该图该类实际选入数，不用全图像素数稀释。数值计算在 FP32，以原始 logits 进入 BCE，不先 sigmoid 再 BCE。

每图每类的像素系数是 0.5/(batch_size×该类已选数)。因此正负类像素数量虽相差很大，存在两类时各类总系数相同；这不意味着两类数值 loss 或梯度贡献必须相等。空类的有效总系数下降已显式定义。本批32图四组正负类均有至少一个已选像素，所以每步有效正负类系数都为0.5；空类/单像素/全空由合成行为测试实际覆盖，不能声称真实smoke遇到了这些边界。

P2/P3 对 n=0/1 的类别都选0，不用 min-one 规则。随机种子为 SHA256(seed:epoch:sample_key:pos或neg) 前8字节 big-endian；NumPy default_rng 给出类内排列。P2 取排列前 k，P3 用 c_sam 降序并使用同一排列优先级打破 ties，取相同 k=floor(n/2)。这也保证全同分时 P3 与 P2 一致，避免 ties 引入固定空间顺序偏差。epoch 进入种子，未来不同epoch可重新抽样，但本轮仅epoch0。

这是新的统一 BCE 蒸馏协议，不是原论文 structure_loss 的严格复现；不能将它与完整5,737图、不同损失的历史 B0 成绩直接相减并归因于新监督。

## 5. 初始化、优化器与恢复

原始 boxpromptiml.py 与 TinyViT 不编辑，使用既有 fecd/model_loading.py 本地适配器。TinyViT权重 /home/weights/tiny_vit_5m_22kto1k_distill.pth，SHA256=e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d。实际 strict load 296 tensor，missing/unexpected keys均空，未联网下载，未加载历史 B0。其余学生层按相同 seed初始化。

四组初始完整模型哈希均为1086c22f6f2a7d0756d4ad9c7de33f52c5d4e6fffeee5e064a77690bb148d516。每组独立进程、同seed20261029、batch4、Adam(lr=1e-4, betas=.9/.999, eps=1e-8,weight_decay=0)、constant LR、FP32、clip_grad_norm_=0.5、8逻辑更新、workers0。无梯度累积、AMP、scheduler变化或挑选最佳loss checkpoint。

CUDA deterministic algorithms开启，cudnn deterministic=true、benchmark=false、TF32关闭，CUBLAS_WORKSPACE_CONFIG=:4096:8。硬件RTX4090，PyTorch2.2.1+cu121，CUDA12.1；跨硬件/库版本的bitwise复现不作保证。

原 ExternalAttention.memory_bank 是会随forward改变的普通属性，不属于state_dict。checkpoint除state_dict/BN buffers/optimizer外，显式保存该内存、Python/NumPy/Torch/CUDA RNG、epoch/next_batch、完整config/hash和model+memory哈希。恢复先重建原网络，再还原全部状态，避免重建消耗RNG造成漂移。

第4步后先运行一个未中断参考第5步，再丢弃该分支，磁盘恢复第4步重做第5步；比较loss精确相等与完整hash相等。每组物理执行9次update，其中1次只作被丢弃的恢复探针，最终轨迹仍为相同8个逻辑更新。恢复探针使用同一已加载batch，没有额外训练样本。总elapsed含探针、重建和哈希开销，不能当作纯训练吞吐。

当前入口只允许固定8步smoke，拒绝改配置或覆盖已有输出；不是已批准的通用20epoch或任意checkpoint CLI。底层restore函数及真实续训已验证；未来完整epoch循环/独立恢复命令须在另行批准后按正式预算完善。

## 6. 真实运行结果

所有loss均来自对应实验实际步骤；不同batch内容不同，首末变化既不是同一个batch上的收敛曲线，也不能作方法排名。没有GT精度、Pixel F1或IoU结果。

| 实验 | 逻辑步/实际图 | 首步loss | 末步loss | 8步均值loss | 平均正贡献 | 平均负贡献 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 8 / 32 | 0.691038 | 0.680002 | 0.676531 | 0.286454 | 0.390077 |
| P1 | 8 / 32 | 0.690831 | 0.680111 | 0.676267 | 0.286180 | 0.390087 |
| P2 | 8 / 32 | 0.689142 | 0.678744 | 0.675938 | 0.285930 | 0.390008 |
| P3 | 8 / 32 | 0.692186 | 0.690778 | 0.675403 | 0.284740 | 0.390663 |

| 实验 | 累计已选正像素 | 累计已选负像素 | 平均预测正比例 | 平均预测负比例 | 裁剪前范数范围 | 裁剪后最大 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 105223 | 1500409 | 0.680100 | 0.319900 | 1.990342/4.313044 | 0.500000 |
| P1 | 105179 | 1500453 | 0.679640 | 0.320360 | 2.020929/4.276022 | 0.500000 |
| P2 | 52581 | 750218 | 0.679727 | 0.320273 | 2.037140/4.244085 | 0.500000 |
| P3 | 52581 | 750218 | 0.685978 | 0.314022 | 2.124167/4.569154 | 0.500000 |

P2/P3累计预算完全相同：52,581正、750,218负；更强的验证是逐batch、逐图、逐正负候选数/已选数全部相同，已由check_stage4p断言。四组数据索引、sample_keys及operation逐项相同，P1–P3候选类别逐项相同。每步pos_mean/neg_mean、有效类别数、类贡献、候选数与已选数均保存在steps.jsonl。

上述结果只证明loss/梯度可计算、预算执行正确；P3 loss略低/略高均不能解释为错误监督更少或模型更好。本轮未回答K8是否提升学生、c_sam是否优于随机，三个核心问题仍须正式同协议训练与独立new_dev评估。

| 实验 | CUDA峰值allocated MiB | 峰值reserved MiB | 首步秒 | 后7步中位秒 | 含恢复探针总秒 | 恢复第5步 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 620.706055 | 682.000000 | 1.388747 | 0.135882 | 4.204552 | loss及model+memory hash精确相等 |
| P1 | 620.706055 | 682.000000 | 1.188815 | 0.130371 | 4.002496 | loss及model+memory hash精确相等 |
| P2 | 620.706055 | 682.000000 | 1.152936 | 0.130211 | 4.028013 | loss及model+memory hash精确相等 |
| P3 | 620.706055 | 682.000000 | 1.118840 | 0.133123 | 4.269108 | loss及model+memory hash精确相等 |

step_seconds含batch已在CPU后的H2D/forward/backward/梯度检查/裁剪/Adam/同步，不含Dataset解码和checkpoint IO；allocated/reserved是PyTorch计数，不是整卡所有进程总占用。所有步logits、loss、已有梯度与更新参数有限，梯度非零，裁剪后范数≤0.5+1e-5，未发生NaN/Inf或对齐错误。TinyViT注册名与timm现有名字重合的UserWarning原样保留在run.log，是已存在导入行为，未改模型规避日志。

## 7. 测试、隔离和保护

针对性22项新测试通过，完整fecd suite 127 tests passed，旧阶段harness 9 checks passed。新增测试先红后绿：loss/selection、真实PNG+v2 NPZ的Dataset契约、checkpoint状态恢复；另测试访问边界拒绝非student图像、未批准cache及写入。覆盖空正/空负/全空、单像素、忽略梯度、类权重与逐图归一化、ties可复现、增强图像/标签/mask同步、坏hash/尺寸/非法角色/字段、checkpoint错配置及拒绝覆盖。

DataReadAudit在实际训练期间监控/home/datasets与stage2c_cache的Python open事件，仅允许student_train RGB及当前教师cache路径，拒绝其他图像/缓存和写模式。各组实际128次open、64个唯一输入文件（32 RGB+32cache，因加强身份校验与复用FECDDataset每图/cache重复读），denied=0。角色/字段限制与读路径日志联合支撑本轮GT读取0；这不是全操作系统文件监控，不能将计数器单独当作任意进程无GT访问证明。

new_dev GT本轮未打开；没有学生验证指标。new_confirm、teacher_diag及正式测试未打开；学生Dataset没有GT字段或读取函数。没有新evidence模块介入。

保护快照核对17,551个旧文件，包括冻结SAM/evidence产物、阶段3/3R/3RA/3RB报告与结果、原模型/训练/评估/utils源码和TinyViT权重，changed=0。既有train.py/eval.py及utils pyc工作树脏改动未触碰、不归入本次变更；没有覆盖原checkpoint。新增13个Python文件用现有AST syntax及symbol-table unresolved-global静态检查，issues=0；环境限定检查不冒充第三方完整lint/typechecker。完整证据在assets/stage4p/validation.json；运行前旧文件哈希快照归档为assets/stage4p/protected_before.json。

## 8. 真实文件与执行命令

新增实现：
- training/data.py：student_train限定Dataset。
- training/selection.py、losses.py：固定选择与统一masked BCE。
- training/protocol.py、checkpoint.py、engine.py、access.py：固定配置、完整状态恢复、更新步与数据读边界。
- tools/train_controlled_baseline.py、check_stage4p.py：独立smoke入口及跨组验收。
- tests/test_stage4p_objective.py、test_stage4p_data.py、test_stage4p_checkpoint.py、test_stage4p_access.py。
- configs/stage4p/P0.json 至 P3.json。
- docs/stage4p_report.md（本报告）。

训练产物位于assets/stage4p/P0至P3：config.json、steps.jsonl、run.log、update4.pt、final.pt、resume_check.json、access.json、result.json。独立目录且拒绝覆盖。顶层validation.json/log、target_tests.log、all_tests.log、regression.log及先红日志保存检查证据。checkpoint合计约542MiB量级（含4组两份完整状态），未自动清理或混入历史checkpoint目录。

实际命令（cwd=/home/projects/BoxPromtIML）：

~~~bash
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_controlled_baseline --variant P0
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_controlled_baseline --variant P1
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_controlled_baseline --variant P2
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.train_controlled_baseline --variant P3
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest fecd.tests.test_stage4p_access fecd.tests.test_stage4p_checkpoint fecd.tests.test_stage4p_data fecd.tests.test_stage4p_objective
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest discover -s fecd/tests -p 'test_*.py'
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tests.run_stage2b_tests
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_stage4p
~~~

| 实验 | 冻结smoke config hash |
| --- | --- |
| P0 | 54a82a6c83d20cb003c11a119e4178bec71da82d30a8030b5a470c84a5a727a8 |
| P1 | 41af2fd19c29f6746a86e65a2a9655b12514e36e6e7c4136e47ef1ea7aef4627 |
| P2 | dd8f0bf0f75aa702b6e1269339bb97c88df8f5eae76cf60b6c8e1c14394bf386 |
| P3 | 8dce770af639565a7005f96607b4bf5a91311369cac95892a5c0834efcd22a38 |

## 9. 正式实验预算与预备设计（未运行，须单独批准）

建议下一阶段先明确批准如下共同协议，再实现正式有界epoch循环与new_dev专用evaluator；本轮不通过编辑smoke配置启动它。以下是预备设计，不是已运行/已经独立确认的结果。

1. **数据与对照**：P0–P3全部使用同一student_train3530与同样224输入，不用历史5737 B0初始化；教师hard、选择定义、类权重不变。任何恢复全量训练或增加soft target都另立独立对照，不能同时改筛选和loss。三项计划对比为P1−P0、P2−P1、P3−P2，最后一项才隔离可靠性排序。

2. **预算**：建议20epoch、batch4、drop_last=false，末batch2，ceil(3530/4)=883步/epoch、17,660更新/组/seed、70,600图像呈现/组/seed。先申请一个共同seed的四组，若预算允许预先登记三个seed20261029/30/31，四组共12run、211,920逻辑更新、847,200次图像呈现；不能看一个seed成绩再选择性补好方法。smoke的原始预训练初始化方案与Adam1e-4/无weight_decay/constant LR/FP32/.5clip保持，所有组同epoch采样与augmentation；若需改batch/lr，先共同修订并重验，不对某组单独降预算。

3. **模型选择**：主结果固定epoch20 final，不按new_dev逐组取最佳epoch或最低loss。可预定epoch5/10/15/20输出曲线但不改变主checkpoint。无性能早停；非有限loss/梯度、身份失败、OOM为工程失败并停该受控组批次查因，不能悄悄减batch或只删失败seed。正式统计前锁定该方案并记录版本；本轮未使用任何验证数据调参。

4. **专用验证**：仅evaluator持有冻结new_dev530的角色allowlist并读取GT；训练入口不传入GT。用原始sigmoid概率，不逐图min-max，固定阈值≥0.5，GT native尺寸校验、NEAREST到224、>127。输出TP/FP/FN、Pixel F1=2TP/(2TP+FP+FN)、IoU=TP/(TP+FP+FN)、逐图/pooled micro、等图像macro及各域结果，显式标N/A和空GT约定。建议空GT且空预测F1/IoU=1，空GT有预测=0；该约定在未来任何GT访问前写入evaluator测试与协议。原 ExternalAttention 在eval forward仍更新memory_bank，未来evaluator应从checkpoint建立独立模型副本，不把验证后的memory带回训练实例。没有本轮学生验证结果。

5. **统计与解释**：正式比较配对seed、相同图像/来源组；同一图多seed的依赖不得当成独立样本。逐域完整报告，必要时以来源代理组配对bootstrap及组等权敏感性，并单独显示seed变异。不将开发集成绩称new_confirm独立确认。teacher_diag不参与新模型选择，new_confirm与正式IND/OOD仍锁定；任何确认申请均另行授权。

6. **运行容量**：本轮热更新约0.13秒/batch（不含数据和验证），按此仅优化计算估计单组单seed17,660步约0.64–0.67小时，12run约7.7–8.0小时；真实全量解码、哈希、写盘、验证及域分布会增加，不能把估算声明为实测20epoch耗时。建议至少按半天级GPU窗口安排并先记录完整epoch实际吞吐，再在不改变组间预算的前提排队。当前八份smoke checkpoint约542MiB；正式建议只保留可恢复last与final加必要审计，12run双份完整状态约1.6GiB量级，评估日志额外预算；未创建或运行正式checkpoint。

## 10. 最终边界

**PASS只表示受控学生蒸馏训练闭环可运行。** K8相对K1、c_sam相对随机、类别贡献平衡后是否仍有收益，这三个研究问题仍没有正式学生评估答案。smoke loss不能回答它们。

不恢复FECD-R/q_A/B-R研究、不新增模块、不运行20epoch对比、不打开new_confirm或正式测试。本阶段完成后停止，等待验收；正式训练必须单独批准。
