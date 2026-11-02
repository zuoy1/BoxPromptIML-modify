# FECD 阶段 3R-B0：参考区域条件化机制预检

日期：2026-10-29。仅 GT-free 机制可行性，不是伪标签错误检测实验。

## 1. 判断

**整体行动：STOP，不进入有标签 B-R 开发诊断。** 工程核验 PASS，参考区域在大多数图像可计算；但当前固定机制未通过预冻结负控门槛，尚未建立超出普通参考内部变异的可识别基础。本结论不能写成“已证明真实 TP/FP 无法区分”：本轮没有读取 GT，参考负控也未经 GT 确认为真实背景。

64×64 主尺度：60/62 张图有匹配参考，1,716/2,490 个审计锚点可匹配（68.92%）；其中纯 SAM 正类候选 245/486（50.41%）可匹配。在 60 张可配对图中，目标差异不超过同图普通参考代理 q95 的等图像平均比例为 75.6389%，超过冻结的 50% 停止线。32×32 为 76.9864%，不能用较高可用率替代主尺度失败。

未读取 new_dev、new_confirm、旧 teacher_diag 或正式测试 GT；没有 TP/FP、AUROC、Risk–Coverage、新 c_sam 融合、SAM 推理或学生训练。原阶段 3、3R、3R-A 的结论、公式、划分与产物保持不变。

## 2. 冻结样本和版本

从冻结 stage3ra_split_manifest.json 的 new_dev 中，按每域 sample_key 升序取前 min(16,n) 张。实际 CASIAv2 16、Columbia 16、Coverage 14、NIST16 16，共 62 图、46 个来源代理组（依次 16/6/14/10）。没有根据 GT、旧错误指标或取证响应选图。本轮不拟合归一化/参考库，故未加载 student_train 图像；未打开确认集 RGB。

协议 fecd-reference-feasibility-v1，在首次样本加载前写入：
- protocol hash：ced4d483b9a3cb15a40269825dceb37ce599b2bc10cbe33ac8e36e75ffc2ecaa
- 派生 split hash：022da143a26833f1613db722ae3be152117948330621e028f02b824f2d1680d0
- 62 条完整选定记录 canonical hash：9f5e64fca07c0b9b469828c0e7a8c981b3f283ad083ee418882b0088d7974287

精确名单、身份、来源组与路径见 assets/stage3rb0/protocol.json。每域最多 16 图是可行性小样本，不保证代表整个域；Columbia/NIST 来源组少。哈希是文件定义的 canonical 内容哈希，不混同文件字节哈希。

## 3. 实际参考算法

### 3.1 工作尺度与 patch

原图 RGB 解码为 float [0,1]，以统一比例缩至最长边≤512，INTER_AREA，不放大；两轴 round 带来的舍入差显式保存在 scale_xy。保持长宽比，不先拉为方形。64×64 为主尺度，32×32 仅是固定敏感性检查。

stride=size/2，补最后一个贴边起点，所有 patch 完整位于图内；无 padding，小图尺寸不足则无候选。SAM/Box 保持冻结 224×224 来源定义，在工作像素中心以 (i+0.5)*224/work_length−0.5 作 nearest 采样。Box union 先在 224 上以 11×11 方核向外膨胀 5 像素，再映射工作网格。xyxy 边界为半开区间；native/work/224 边界按各轴精确比例变换，不暗中改变 native 长宽比。

候选分类：
- pos：patch 内 K8 mu≥0.5 比例≥0.75。
- neg：预测正类比例≤0.25。
- box_inside：Box union 比例≥0.75。
- box_edge：0.05<Box union 比例<0.95。

分类可重叠。每图每类按 grid-order 索引等距 linspace 取最多 16 个锚点，再取四类并集；同一类别可能额外包含其他类别选入的锚点。普通参考代理控制也从合格参考池等距取最多 16 个锚点。匹配时使用完整参考池。此固定抽样用于控制 CPU 成本，未按残差挑锚点；报告匹配率分母为实际审计锚点，不能冒充所有重叠 patch 的总体适用率。

### 3.2 参考条件与内容匹配

只在同图、同尺度内选择参考，需同时满足：
- patch 全部处于膨胀 Box 外。
- K8 正类比例≤0.05。
- 至少 90% 像素为稳定负类：mu<0.5、原 c_sam≥0.90、var≤0.01。
- 候选/参考不重叠，保留参考之间也不重叠。

内容描述为灰度均值 brightness、Gaussian sigma=1 平滑后 3×3 Sobel（scale=1/8）的梯度模均值 gradient，以及 gray−Gaussian(gray,sigma=2) 的标准差 texture。固定匹配门槛：

abs(brightness_C−brightness_R)≤0.10；

abs(log((gradient_C+0.005)/(gradient_R+0.005)))≤log(1.5)；

texture 同样使用 epsilon=0.005、log(1.5)。

低幅区域受固定地板影响，不能将其误读为严格原值比率 1.5。按三个差异各除对应门槛后的平方和排序，再以 patch id 打破 ties。没有最近空间距离优先，也不以残差差异排序。贪心选最多 5 个互不重叠参考，至少 3 个才适用；不足明确 N/A，不补零、不放宽门槛。贪心不是最大独立集合优化，可能限制可用率，本轮不按结果改算法。

Box 是历史 GT 离线派生的弱监督输入，Box 外选择引入背景假设；本轮没有新增读取 GT。高一致性 SAM 背景也可能是 FN，同图可能有多处篡改，因此参考不能被称为确认真实背景。来源代理也不提供每个区域真实成像管线的证明。

## 4. 唯一主区域统计

仅一种噪声残差：e=gray−median3(gray)，signed、未 clip、在完整工作图上滤波。median 图像边界 replicate；Gaussian/Sobel 用 REFLECT_101。每 patch 只从去除四周 2 像素的中心计算残差统计，避免 patch 独立滤波产生假边界。

保存 raw MAD=median(abs(e−median(e)))，不乘 1.4826；q_j 为 .05,.10,...,.95 共 19 个 signed 分位数。

D_raw(C,R)=mean_j(abs(q_j(C)−q_j(R)))

D(C,R)=D_raw(C,R)/(0.005+(MAD_C+MAD_R)/2)

**唯一主统计为对 3–5 个参考的 D 取中位数。** 它是中心分位数距离的尺度化近似，不能称完整经验 Wasserstein-1。辅助描述量为 raw 分位距离和 abs(log((MAD_C+.001)/(MAD_R+.001)))，不另选最佳分支；记录删一参考后中位数范围检验稳定性。

常值图可合法得到零；参考不足为 None/N/A，不能混为零差异。分数不代表正确概率，没有规定“差异越大，SAM 正类越正确”，也不将异常负类自动翻正。最长边缩放、设备、去噪、内容纹理与历史压缩都可能影响该统计；对 TIFF 可计算不等于证明它无压缩历史。

## 5. 小样本实际结果

### 5.1 主尺度可用率

| 域 | 图/组 | 有匹配图 | 审计候选 | 有效候选 | 匹配率 | 正类匹配 | 负类匹配 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 62/46 | 60/62 | 2490 | 1716 | 68.92% | 245/486 | 1205/1628 |
| CASIAv2 | 16/16 | 15/16 | 495 | 349 | 70.51% | 14/48 | 279/371 |
| Columbia | 16/6 | 16/16 | 843 | 448 | 53.14% | 87/258 | 294/459 |
| Coverage | 14/14 | 14/14 | 610 | 513 | 84.10% | 127/140 | 287/357 |
| NIST16 | 16/10 | 15/16 | 542 | 406 | 74.91% | 17/40 | 345/441 |

64 尺度生成 9,347 个完整网格 patch，其中 9,347 个符合至少一类候选，实际抽取并集 2,490 个审计锚点。60 张有匹配的图都至少有一个 pos 或 Box 目标可匹配，但不等于各图都有纯正类 patch。CASIA/Columbia/NIST 正类可用率仅 29.17%/33.72%/42.50%，不能被大量易匹配负类掩盖。

| 最终参考数（含不适用） | 候选锚点数 |
| --- | --- |
| 0 | 352 |
| 1 | 187 |
| 2 | 235 |
| 3 | 242 |
| 4 | 210 |
| 5 | 1264 |

| 无法匹配主因（顺序归因，互斥） | 数量 |
| --- | --- |
| joint_content_lt3 | 439 |
| mutually_nonoverlap_selection_lt3 | 237 |
| reference_pool_lt3 | 98 |

| 池/统计 | n | min | median | q95 | max |
| --- | --- | --- | --- | --- | --- |
| 每图Box外patch数 | 62 | 0.000000 | 101.000000 | 146.900000 | 156.000000 |
| 每图稳定参考池 | 62 | 0.000000 | 101.000000 | 146.900000 | 156.000000 |
| 主D | 1716 | 0.000000 | 0.170360 | 0.836884 | 2.132148 |
| raw距离 | 1716 | 0.000000 | 0.001267 | 0.009204 | 0.022574 |
| log MAD比 | 1716 | 0.000000 | 0.168395 | 0.883596 | 2.444653 |
| 删一参考范围宽度 | 1716 | 0.000000 | 0.042617 | 0.306143 | 1.362902 |

有效统计均有限，无 NaN/Inf。不适用保留原因而非补零；即使池较大，也可能因内容门槛或参考相互重叠而不足 3 个。n 是描述性锚点计数，候选 patch 存在重叠，未当作独立统计样本。

| 内容差（每候选对参考的均值） | 筛选前中位数 | 筛选后中位数 | 筛选后q95 | D与筛选后差的Spearman |
| --- | --- | --- | --- | --- |
| brightness | 0.138416 | 0.029706 | 0.063286 | 0.160868 |
| gradient | 0.420917 | 0.103736 | 0.233708 | 0.161517 |
| texture | 0.513776 | 0.109686 | 0.246992 | 0.195499 |

筛选前质量来自最终有效锚点的所有非重叠参考候选，筛选后为保留参考。门槛收窄了内容差异；但摘要未对缺少共同支持的锚点给出反事实匹配质量。梯度/纹理差与 D 的相关均未达到冻结 |rho|≥0.7 且≥30 对门槛，不足以证明已排除内容干扰。

### 5.2 普通区域代理负控与停止线

从同样合格参考池取控制锚点，与其他不重叠参考作完全相同匹配。这些是自然区域代理，未经 GT 确认；控制与目标也可能共享参考。每图先求控制 D 的 q95，再求该图 pos 或 Box 目标 D≤q95 的比例，最后等图像平均。仅具 neg 类别且不属于 Box 目标的候选不纳入目标比较，避免负控与自身重复造成必然相似。

首次样本加载前冻结的保守停止规则：至少 10 张可配对图且上述宏比例≥50%，则未建立超出普通参考内部变异的机制基础。该规则不是“95% 显著性”、真实篡改判别检验或最优阈值，也没有利用 TP/FP 定阈值。

| 域 | 配对图 | 目标D≤控制q95的图像宏比例 | 控制D中位数 | 控制D最大值 |
| --- | --- | --- | --- | --- |
| ALL | 60 | 75.64% | 0.110787 | 2.028159 |
| CASIAv2 | 15 | 83.58% | 0.197303 | 1.207789 |
| Columbia | 16 | 67.08% | 0.057759 | 1.281087 |
| Coverage | 14 | 92.24% | 0.206292 | 2.028159 |
| NIST16 | 15 | 61.34% | 0.041419 | 0.892014 |

四域均超过该停止门槛，未排除不利域。它不证明未来有标签错误检测绝无信息，只说明当前无 GT 检查尚无清晰的特殊成像锚点，不支持直接转成可靠性分数。

补充固定 seed=20261029 的程序化未篡改内容压力检查：constant、stationary noise、连续变化纹理加同一噪声、连续边缘加同一噪声。这些不是自然照片；全图模拟稳定负类，只检验数值与普通内容响应，未依据结果改主规则。

| 程序化控制 | 尺度 | 适用/锚点 | D中位数 | D最大值 |
| --- | --- | --- | --- | --- |
| constant | 64 | 16/16 | 0.000000 | 0.000000 |
| constant | 32 | 16/16 | 0.000000 | 0.000000 |
| stationary_noise | 64 | 16/16 | 0.022473 | 0.046269 |
| stationary_noise | 32 | 16/16 | 0.033615 | 0.075865 |
| continuous_varying_texture_same_noise | 64 | 16/16 | 0.000001 | 0.049740 |
| continuous_varying_texture_same_noise | 32 | 16/16 | 0.054367 | 0.115674 |
| continuous_edges_same_noise | 64 | 16/16 | 0.000001 | 0.023135 |
| continuous_edges_same_noise | 32 | 16/16 | 0.045113 | 0.135337 |

程序化控制没有极大响应，不能凭此消除真实照片的代理负控混淆。32 下部分普通纹理响应比 64 更高，提示尺度/量化敏感性；它不是篡改定位能力的证明。

## 6. 适用范围与独立性风险

| 尺度 | 全网格patch | 审计锚点 | 有效锚点 | 可用率 | 有匹配图 | 目标/控制q95宏比例 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 9347 | 2490 | 1716 | 68.92% | 60/62 | 75.64% |
| 32 | 40147 | 3180 | 2762 | 86.86% | 60/62 | 76.99% |

| 原始格式（有目标匹配） | 图数 | 每图目标D中位数的中位数 | q95 |
| --- | --- | --- | --- |
| JPEG | 19 | 0.217798 | 0.696732 |
| TIFF | 41 | 0.210137 | 0.627395 |

| 每图D中位数与变量的Spearman | rho |
| --- | --- |
| native_longest | -0.158427 |
| high_texture | 0.508086 |
| mean_candidate_box_fraction | -0.114865 |
| num_boxes | 0.195034 |
| 缓存Box面积与主64匹配率 | -0.638713 |

JPEG/TIFF 图像统计中位数比约 1.036，未触发≥4、每层至少 10 个有效图像的格式警戒线。但它们与数据集、内容、分辨率混杂，不能推断格式无影响。原始 JPEG 量化表和可用元信息字段已保存；没有估计未知相机或历史压缩质量。

分辨率仅按连续相关性检查，未执行独立的分辨率分层比值门槛；协议 format_resolution_flag 实际只执行了 format 比值部分，不能称完整排除了分辨率依赖。尤其大幅缩小会损失 native 噪声及块统计语义，低整体相关不排除非线性或域内偏差。

每图 D 与候选平均纹理约 rho=0.508，既不能称主要由纹理主导，也不能称提取了独立成像关系。Box 面积与可用率 rho≈−0.639，是相关而非因果结论。所有主表同时给图像数和组数，不把像素/重叠 patch 当作独立推断单元；本轮不作有标签效应检验或像素级显著性宣称。

### 固定样本内极值案例

| 案例 | 域/样本 | Box数/面积 | 64候选匹配 | 64目标匹配 |
| --- | --- | --- | --- | --- |
| smallest_nonzero_cached_box | NIST16/NC2016_5702 | 3/0.30% | 15/20 | 4/4 |
| largest_cached_box | CASIAv2/Tp_D_NRN_L_N_ani10109_ani10108_10226 | 1/100.00% | 0/45 | 0/45 |
| largest_box_count | Columbia/canonxt_kodakdcs330_sub_18 | 3/20.07% | 28/42 | 18/30 |
| highest_mean_grid_texture | Coverage/055 | 1/20.46% | 25/32 | 16/20 |

该批 Box<1% 有 7 图、>50% 有 5 图、多 Box 有 8 图。极小 Box 可能没有 64×64 纯正类 patch，只能由 Box 边缘/混合候选代表。Box 覆盖全图的 CASIA 案例两尺度都没有参考，明确不适用。最高纹理 Coverage055 的目标全部位于同图控制 q95 内，控制 q95≈1.6843；普通参考间也可有强差异。最小 Box 案例的目标/控制关系在 64/32 改变，不能用单案例宣称机制稳定。完整身份、Box/SAM 面积、实际匹配和控制结果见 geometry_audit.json。

选择过度依赖 SAM 稳定负类是实质风险：它不构成独立真实背景锚点。亮度/梯度/纹理均值相近也不保证材质、景深、照明、去噪相近，同图不同区域仍可有普通内容残差变化。来源元数据不足以证明真实采集管线一致；本轮没有补造来源身份。

## 7. 空间、身份、GT 禁读核验

输入核对索引 sample_key/sample_id/dataset/image_path、split=train、RGB SHA256/SHA1、K8 v2 config、Box hash、native 尺寸和缓存目录。冻结派生 manifest 的来源代理隔离先行核验，没有从 source manifest 的 mask 路径加载数据。

ReadGuard 在 RGB/cache 加载解码期间通过 Python audit hook，仅允许打开当前 RGB 与 K8 文件，拒绝其他文件及写模式；PIL/ZIP/编码器事先初始化，未通过放宽白名单解决延迟导入。guard 外读取的派生 manifest、协议和结果是明确的代码路径。因此这是受控入口的文件访问防线与记录，不是全操作系统所有进程的 GT 访问证明。

完整预检无 denied input opens。独立几何复核仅重读同 62 张 RGB/K8：49,494 个 patch 边界/坐标、7,590 个保存锚点完整描述子复现、30,864 对候选/参考非重叠与门槛、57,706 对参考/参考非重叠，全通过。N/A 和参考数量一致。所有分区 GT 读取 0，new_confirm GT 关闭。

## 8. 实际变更、命令、日志和异常

新增源文件：
- forensic_reference/patches.py、matching.py、statistics.py。
- forensic_reference/protocol.py、inputs.py、summary.py。
- tools/audit_reference_feasibility.py、check_reference_controls.py。
- tests/test_reference_patches.py、test_reference_matching.py、test_reference_feasibility.py、test_reference_summary.py。
- docs/stage3rb0_report.md。

新增产物独立位于 assets/stage3rb0：protocol.json、per_image.jsonl、input_access.jsonl、summary.json、run.json、procedural_controls.json、geometry_audit.json、implementation.json 及日志。geometry_audit_script.py 是真实复核脚本归档，已有结果时拒绝覆盖。未修改原代码、checkpoint 或任何前阶段冻结结果。原有 train.py/eval.py/utils pyc 脏改动未触碰、不归入本次变更。

实际执行，cwd=/home/projects/BoxPromtIML：

~~~bash
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest fecd.tests.test_reference_feasibility fecd.tests.test_reference_patches fecd.tests.test_reference_matching
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.audit_reference_feasibility
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_reference_controls
PYTHONPATH=/home/projects/BoxPromtIML /usr/local/miniconda3/envs/boxpromptiml/bin/python /tmp/audit_rb0_geometry.py
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m unittest discover -s fecd/tests -p 'test_*.py'
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tests.run_stage2b_tests
~~~

完整 62/62 运行含汇总约 75.04 秒。完整单测 105 项、旧回归 harness 9 项通过。静态分析覆盖新增 12 个 Python 文件、issues=0，17,507 个原有文件哈希未变；最终成员/访问日志核验详见 validation.json；AST/symbol-table 是现有环境的限定检查，不冒充第三方类型检查器。

两次真实启动失败保留：首先 whitelist 拦截 NumPy 延迟导入 zipfile，随后拦截标准库 cp437；均在首图缓存加载期间失败，逐图产物 0 行，不是身份/坐标失败。新增实际 PNG/NPZ 读取集成测试，预初始化库后测试通过，再启动同一固定协议。失败目录为 assets/stage3rb0_failed_start 和 stage3rb0_failed_start2；traceback 为主目录 first_run_failure.log、second_run_failure.log。没有隐藏失败、放宽 GT 防线或将失败尝试宣称完成。

汇总后补齐相关性 gate 的≥30对保护，本批有效 1,716 个，数值及决策不变。只重新汇总既有 JSON，未重新提特征；一次 Python dict 比较因 JSON 将整数键变字符串而失败，改为 JSON 规范化比较后确认一致，未修改实验数据。implementation.json 为运行后捕获源码哈希，不冒充全源码运行前锁定。geometry_audit.log 明确为事后执行记录，不伪造未保存的原始 stdout；其 /tmp 脚本首次缺 PYTHONPATH 导入失败，修正后通过。

报告初次写入命令因工具字符串解析失败未执行；恢复后继续用已有真实产物生成报告，没有因连接故障重跑实验。

## 9. 是否进入下一步

工程可计算性、身份及空间测试 PASS；B0 整体行动 STOP，不能标记 READY FOR B-R DEVELOPMENT DIAGNOSTIC。当前固定机制不值得申请开放 new_dev GT 做新的错误排序。该结论仅限本候选与保守预检，不否定所有潜在参考机制，也不证明 GT 标签下无任何信息。

若未来有来源可靠的 authentic 参考或处理历史，应由用户另行批准和预注册；本轮不新增数据、搜索匹配公式、翻转统计方向，也不自动开放任何 GT。原阶段 3、3R、3R-A 结果保持原样。完成后停止，等待验收；不进入学生训练。
