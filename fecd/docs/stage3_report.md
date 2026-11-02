# FECD 阶段 3：取证证据与教师可靠性诊断报告

日期：2026-10-29。工程关卡 3A、3B、3C 已实际运行并完成；研究判定：**NO-GO**。本报告不建议进入阶段 4。未训练学生、未实现 FECD Loss、Confidence Head 或 Consistency，未运行 B1–B5。

固定候选 Rule-FECD 未证明相对 SAM-only 的增量价值，而且在预先冻结的核心指标上显著退化。高一致性正类的逐图错误检测 AUROC 从 0.81025 降为 0.55488，差值 **−0.25537，95% CI [−0.28369, −0.22813]**。相同 50% 正类 coverage 下，逐图平均错误率从 2.611% 增至 3.361%，差值 **+0.749 个百分点，95% CI [+0.572, +0.937] 个百分点**。

结论仅针对本次冻结的描述子、归一化和规则，不否定所有可能的取证方法。没有根据锁定诊断 GT 调整特征、阈值、质量因子、融合方式或公式；成功和负面结果均保留。

## 1. 数据协议与 source-group 隔离

先阅读了阶段 1、2A、2B、2C 历史审计，以及已有 Dataset、cache schema、坐标和 Box 导出实现。内部划分详见 [stage3_data_protocol.md](stage3_data_protocol.md)，原始 5,737 张训练成员不变。seed=20261029，目标 teacher_diag 比例 20%。

| 数据集 | 原 train | train_fit 图 | fit 组 | teacher_diag 图 | diag 组 |
| --- | --- | --- | --- | --- | --- |
| CASIAv2 | 5123 | 4098 | 525 | 1025 | 461 |
| NIST16 | 414 | 332 | 43 | 82 | 9 |
| Columbia | 130 | 104 | 24 | 26 | 6 |
| Coverage | 70 | 56 | 56 | 14 | 14 |
| 合计 | 5737 | 4590 | 648 | 1147 | 490 |

划分版本 `fecd-stage3-split-v1`，内容哈希 `7f18b459cded7cc2a8577167c5e2958cfac715691217a32bf457c8ae611cd66e`；records 哈希 `30cd447039aa20f90e7e87a5b1b4e589316d262a9f395a0ee5a9e9a1c46fbcb5`。独立重建 `stage3_split_reproduction.json` 与冻结 manifest 完全一致。1,138 个代理组跨内部集合数为 **0**。

CASIAv2 原 `group` 是逐文件 ID，不能当作原始来源。现将文件名中的两个 donor token 建立传递连通分量，最大分量 3,020 张整组留在 train_fit；诊断集因此只代表其余较小分量。Columbia 跨 camera-pair 合并相同 `_sub_NN` 场景编号。二者合计 **5,253 张只有文件名/场景代理，真实来源身份并未唯一核实**；“无法解析 token”的记录为 0，并不意味着来源身份确定。NIST16 使用已有 base/source group，Coverage 使用来源图编号。这是元数据能力范围内的保守隔离，不声称获得完整 provenance。

NIST16 的历史跨 train/test 风险继续保留：`world/NC2016_0359.jpg` 的历史 train `NC2016_5290`、`NC2016_7882` 与历史 test `NC2016_0796`、`NC2016_9257` 同源。未重写历史划分，未读取正式 IND/OOD 测试图像或 GT 做选择。

后续如使用 train_fit，B0–B5 的严格对照都必须统一为 4,590 张及相同训练预算。完整 5,737 张训练的旧 B0 只能作为历史参考。本次不自动恢复全量训练；任何后续特征/规则修订也不能再把已经用于诊断的 teacher_diag 冒充新的独立确认集。

## 2. 诊断前冻结的定义与判定

机器协议为 `fecd/diagnostics/protocol.py` 和 `fecd/assets/stage3_analysis_protocol.json`，哈希 `90c1ec0a68625ca692e5ae4461ffd5ca2b3664e48746cdb6610241577574f4ce`。在首次诊断 GT 访问前已写入。

教师预测 `pred=(mu>=0.5)`，错误 `pred!=gt`。专用 `audit_pseudo.py` 按 `(dataset,id,image_sha256)` 连接原 train manifest，核对 native 尺寸和 prepared GT 的 `{0,255}` 二值性，PIL NEAREST 映射到 224×224 后 `>127` 为前景。GT 只在该入口的 `load_diagnostic_gt` 打开；模块及学生 Dataset 不读取 GT。所有 GT 入口先经过 3A 完整性门禁。

高一致性主要子集固定为 `c_sam>=0.90 AND var<=0.01`；另报告低 variance 子集、正负候选集和 Box 条件。Box union 向内腐蚀 3 像素为内部，剩余 union 为边缘，向外膨胀 5 像素减去 union 为周边环，其他为远外部。该“边缘”是 Box 边缘，不是 GT 篡改轮廓。

所有 score 高表示可靠，错误检测使用 `1-score`；AUROC 和非插值 AP 越高越好。单类 AUROC 标 N/A，无错误 AP 标 N/A，全错误 AP=1 并保留单类原因。对相同候选集合按可靠性降序保留 10%、25%、50%、75%、90%，边界同分采用相同期望比例选取，避免按数组位置打破并列。少于 10 个期望选中像素时 risk/precision 为 N/A。AURC 是该并列组内期望累计风险函数对 coverage 的解析积分，不是离散右端点求和。

主要指标为逐图 high_pos AUROC 的 S4−S0；次要指标为逐图 pos 的 50% coverage risk 差。2,000 次整个 source group 有放回配对 bootstrap，seed=20261029，报告图像加权宏平均差和百分位 95% CI。GO 要求前者 CI 下界>0、后者 CI 上界<0，且至少 20 张诊断图、50 个正类错误的数据集不退化。两项点估计均有利但不确定性/域反转时才允许 CONDITIONAL GO；其他为 NO-GO。少量数据集与错误数不足必须披露。

## 3. 四种取证证据的真实定义

输入是解码 native RGB uint8；浮点输入限于有限 `[0,1]`，乘 255 四舍五入后处理，BGR 必须显式声明并转换。不应用 EXIF 自动旋转，沿用 SAM 原有解码坐标。

过滤工作网格最长边至多 512，保持长宽比、仅缩小，尺寸四舍五入，`INTER_AREA`。JPEG 重压缩发生在 native 图像，残差才下采样到工作网格。最后所有空间图明确以 `INTER_LINEAR` 映射到原协议的 224×224 方形 SAM 网格；这一步沿用原学生直接 resize 的坐标拉伸定义，不是静默增加 crop、padding 或新的长宽比变换。输出四个 float32 `[224,224]` 非负有限异常图。

普通滤波边界 `BORDER_REFLECT_101`。局部残差共同采用 9×9 均值池化，`L=log(pooled+1/255)`，异常为 `abs(L−GaussianBlur(L,sigma=16))`；Gaussian 的核尺寸采用 OpenCV `(0,0)` 按 sigma 自动确定，版本已记录。此局部背景对比属于预定义描述子，不是逐图 min-max。

| 描述子 | 实际参数与定义 | 解释边界 |
|---|---|---|
| HP | BT.601 灰度，`abs(gray−Gaussian(gray,sigma=1))` 后局部 log contrast | 高频/物体边界会响应；只作诊断与单独消融，不进入默认 median |
| Noise | 灰度减 3×3 median 的绝对残差，clip 到 0.1；除以 `1+8*Gaussian(gradient,sigma=1)`；gradient 来自 sigma=1 平滑图的 Sobel3、scale=1/8；随后局部 log contrast | 中值和截断/梯度补偿抑制部分结构干扰，未证明能够可靠分开自然纹理与真实噪声不一致 |
| JPEG | Pillow native RGB JPEG Q90、4:4:4/subsampling=0、optimize=False、progressive=False；RGB 绝对残差通道均值/255，缩小后局部 log contrast | 不是绝对残差篡改概率；非 JPEG 图只表示反事实重编码失真，不能解释成原 JPEG 取证痕迹 |
| FFT | 32×32、stride=8，窗口前去均值，二维可分离对称 Hann；rFFT 功率恢复横向正负频率权重；径向频率>=0.25 cycles/work-pixel 的功率/总功率，分母+1e-12；与 tile-grid sigma=4 Gaussian 背景的绝对差 | 总功率包括加 Hann 后重新出现的 DC；配置中 AC 指预去均值，不是显式删除 DC bin。尺度受工作网格和 native 分辨率影响 |

FFT 使用 symmetric padding，首 tile 支持相对名义 anchor 的 `[-16,+15]` 采样，stride=8；按 anchor 网格 remap（双线性，越界 replicate）回工作像素，再映射 224。偶数窗口支持的几何中心与名义 anchor 相差半像素，这属于本版本固定滤波相位定义，不改变图像/Box 坐标。小尺寸与 singleton 输入有测试。

运行环境：Python 3.10.22、NumPy 1.26.4、OpenCV 4.9.0、Pillow 12.3.0、JPEG codec 6.2。配置哈希同时绑定 extractor 源文件 SHA 和这些运行版本。本次没有新装库。

## 4. GT-free 归一化与证据缓存

| 特征 | 全局 q01 / low | 全局 median | 全局 q99 / high | 采样像素数 |
| --- | --- | --- | --- | --- |
| hp | 0.008191037 | 0.212487802 | 1.205368757 | 4700160 |
| noise | 0.007355639 | 0.155318618 | 0.830352545 | 4700160 |
| jpeg | 0.003554274 | 0.072191458 | 0.417580903 | 4700160 |
| fft | 0.002914940 | 0.047379978 | 0.257255403 | 4700160 |

仅 train_fit 4,590 张参与拟合：每图从 224×224 展平网格取包含端点的 1,024 个确定性 linspace 索引，每图等权；每特征 4,700,160 个观测，NumPy linear 分位数。统一应用 `clip((x−low)/(high−low),0,1)`，退化 high<=low 映为 0；缺失或非有限值直接报错，不填补。没有逐张图 min-max，也没有根据 teacher_diag GT 修改统计。

统计版本 `fecd-evidence-normalization-v1`，hash `9a92977d313d68b4c23ad317b1bcb487978bf8cdee5e81729154ccc871a22d5c`；拟合成员 hash `5098a6a03db9504c0edfad33c7c3217a1aa312fea252f81b4de38cf3b59b6b9a`。固定融合 `r=median(noise_norm,jpeg_norm,fft_norm)`，HP 排除。

原始 evidence cache：`fecd/assets/stage3_evidence/raw/05e1736aaeca/`，5,737 条，**4,152,523,947 bytes（约 3.87 GiB）**。原始格式 TIFF 3,259、JPEG 2,478。NPZ 保存四个 float32 图、身份/源 hash、SAM/Box/config 身份、native 尺寸、原格式、原 JPEG 量化表范围与表 hash；不从表反推精确原 Q。重压缩配置完整记录，metadata 与 maps 均有 checksum。没有保存 GT 或覆盖 SAM v2。

先运行 12 张前缀 pilot，再运行四数据集各 3 张的 GT-free pilot；临时 pilot scales 只用于预览，没有用于诊断。随后生成全量并 `--validate-only` 全遍历：**5,737/5,737 成功，失败 0，复用 5,737，写入 0**，sample-key 集与 K8 精确对齐，图像 SHA-256/SHA1、K8 schema/source/Box 身份和尺寸均核验。源特征配置 hash `05e1736aaeca77ed4a61be59d36b9b335e3f22df6c297f26a9f257fc66f00a7c`。

train_fit 上归一化图的逐图均值如下；完整分布和每图 p10/p50/p90 见统计分布文件与 bias JSONL。

| train_fit 数据集 | Noise | JPEG | FFT | 融合 r |
| --- | --- | --- | --- | --- |
| CASIAv2 | 0.25335 | 0.23710 | 0.22542 | 0.22336 |
| NIST16 | 0.16381 | 0.10740 | 0.26914 | 0.13783 |
| Columbia | 0.17557 | 0.16605 | 0.26439 | 0.16180 |
| Coverage | 0.26583 | 0.31064 | 0.23444 | 0.24623 |

## 5. SAM-only 与固定 Rule-FECD

`u=4*agreement*(1−agreement)`；`c_sam=(1−clip(4*var,0,1))*(1−u)*sqrt(2*abs(mu−0.5))`；`c_pos=c_sam*(0.5+0.5*r)`；`c_neg=c_sam`。这是启发式稳定性分数，绝不解释为校准正确概率。

所有实际诊断图上的 score 均有限且在 `[0,1]`，单测验证 variance 和 threshold uncertainty 增加时不反向升高、前背景对称和正负分支约束。S1/S2/S3 分别以 Noise/JPEG/FFT 替代 r，S4 为固定 median，S_HP 单独报告 HP。S5 仅用 evidence：正类可靠性 r（error score=1−r），负类可靠性 1−r（error score=r），用于检查异常对 FP/FN 的方向假设，不属于当前负类规则。

原始教师预测始终不变；任何 evidence 都没有新增预测正类。S4 只降低正类分数，真正要验证的是同样 coverage 下排序是否更好。

## 6. K1/K8 伪标签与错误分布

| 数据集 | K1 micro F1/IoU | K8 micro F1/IoU | K1 macro F1/IoU | K8 macro F1/IoU |
| --- | --- | --- | --- | --- |
| ALL | 0.87039 / 0.77052 | 0.87063 / 0.77090 | 0.88179 / 0.81377 | 0.88117 / 0.81326 |
| CASIAv2 | 0.86682 / 0.76494 | 0.86902 / 0.76837 | 0.88618 / 0.81845 | 0.88635 / 0.81859 |
| Columbia | 0.82592 / 0.70346 | 0.80210 / 0.66959 | 0.80035 / 0.70672 | 0.76831 / 0.67892 |
| Coverage | 0.94335 / 0.89278 | 0.94177 / 0.88994 | 0.94063 / 0.89199 | 0.94120 / 0.89223 |
| NIST16 | 0.90601 / 0.82817 | 0.90768 / 0.83096 | 0.84264 / 0.77583 | 0.84191 / 0.77576 |

诊断最终唯一成员为 1,147 张、57,551,872 个网格像素；两张 pilot 没有在汇总中重复计数。K8 FP=170,352，FN=855,220；相对全像素比例分别为 0.296% 和 1.486%。预测正类 3,621,240 中 FP prevalence=4.704%；预测负类 53,930,632 中 FN prevalence=1.586%。不能用全图低错误率代表目标区域可靠。

K8 TP/FP/TN/FN 的像素加权平均 c_sam 分别为 0.82981 / 0.42034 / 0.99388 / 0.61862。总体上 SAM 稳定性有排序价值，但仍保留大量高稳定错误。

| 子集 | 候选像素 | 错误像素 | 条件错误率 | 含错误图像 |
| --- | --- | --- | --- | --- |
| high_neg | 52869507 | 348194 | 0.659% | 1090 |
| high_pos | 2532018 | 47039 | 1.858% | 530 |
| lowvar_neg | 53221102 | 561117 | 1.054% | 1129 |
| lowvar_pos | 2863662 | 67075 | 2.342% | 764 |

高一致性错误合计 395,233 像素（47,039 FP、348,194 FN）；其出现图像数不能把正负两类图像数直接相加，因为可重叠。逐图记录含各区域 FP/FN/count。

## 7. S0–S5：逐图 macro 与 pooled micro

macro 每图等权，单指标 N/A 排除并保存有效数；micro 汇总同一批像素，采用冻结的 4,096 等宽 reliability bins、bin 内视为并列，只作描述。两者不是相同估计量。下表 AP 同步给出 error prevalence；不同 prevalence 的 AP 不作横向优劣结论。

### 全部 SAM 预测正类

错误 prevalence：micro 4.704%，逐图 macro 5.374%。

| 方法 | macro AUROC | macro AP | macro AURC | micro AUROC | micro AP | micro AURC |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.87900 | 0.31046 | 0.02868 | 0.80998 | 0.19181 | 0.01400 |
| S1 | 0.79847 | 0.28244 | 0.03703 | 0.77564 | 0.18909 | 0.01928 |
| S2 | 0.77367 | 0.28327 | 0.03614 | 0.75181 | 0.18416 | 0.02207 |
| S3 | 0.81345 | 0.28857 | 0.03626 | 0.76716 | 0.18720 | 0.02082 |
| S4 | 0.80118 | 0.28676 | 0.03636 | 0.76687 | 0.18712 | 0.02032 |
| S5 | 0.48972 | 0.09582 | 0.05421 | 0.50049 | 0.04588 | 0.04523 |
| S_HP | 0.79538 | 0.28561 | 0.03677 | 0.78170 | 0.19087 | 0.01828 |

### 预定义高一致性正类

错误 prevalence：micro 1.858%，逐图 macro 3.109%。

| 方法 | macro AUROC | macro AP | macro AURC | micro AUROC | micro AP | micro AURC |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.81025 | 0.22124 | 0.02151 | 0.72422 | 0.04227 | 0.00840 |
| S1 | 0.55760 | 0.11433 | 0.03091 | 0.58471 | 0.03242 | 0.01487 |
| S2 | 0.50828 | 0.11666 | 0.02959 | 0.50437 | 0.01921 | 0.01777 |
| S3 | 0.59151 | 0.13668 | 0.03018 | 0.53824 | 0.02213 | 0.01721 |
| S4 | 0.55488 | 0.12309 | 0.03069 | 0.54056 | 0.02163 | 0.01644 |
| S5 | 0.49130 | 0.09405 | 0.03274 | 0.49580 | 0.01749 | 0.01824 |
| S_HP | 0.54763 | 0.12550 | 0.03109 | 0.59913 | 0.03190 | 0.01391 |

### 全部 SAM 预测负类

错误 prevalence：micro 1.586%，逐图 macro 1.885%。

| 方法 | macro AUROC | macro AP | macro AURC | micro AUROC | micro AP | micro AURC |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.95716 | 0.56288 | 0.00498 | 0.94324 | 0.45849 | 0.00177 |
| S1 | 0.95716 | 0.56288 | 0.00498 | 0.94324 | 0.45849 | 0.00177 |
| S2 | 0.95716 | 0.56288 | 0.00498 | 0.94324 | 0.45849 | 0.00177 |
| S3 | 0.95716 | 0.56288 | 0.00498 | 0.94324 | 0.45849 | 0.00177 |
| S4 | 0.95716 | 0.56288 | 0.00498 | 0.94324 | 0.45849 | 0.00177 |
| S5 | 0.65255 | 0.03400 | 0.01588 | 0.55464 | 0.01868 | 0.01284 |
| S_HP | 0.95716 | 0.56288 | 0.00498 | 0.94324 | 0.45849 | 0.00177 |

pos AUROC 有效 830 图，其余 317 图无 FP；high_pos 有效 526 图，其余为空候选 37、无错误 580、全错误 4；neg 有效 1,139 图，其余 8 图无 FN。以上全部 N/A 原因在 per-image JSONL 中保留，没有补零。高一致性 AP 含 4 个全错误 AP=1 的图，因此其有效样本不等于 AUROC。

S5 的正类 micro AUROC 0.50049、high_pos 0.49580，接近随机排序；本版独立异常并不能稳定识别 FP。所有 S1–S4 的负类逐图指标、直方图和可信选择均与 S0 完全一致，没有改善稳定漏检。

## 8. 相同 coverage 的 Risk–Coverage 与可信精度

### 预测正类：相同候选集合与相同 coverage

| coverage | S0 macro risk | S4 macro risk | S0 micro risk | S4 micro risk |
| --- | --- | --- | --- | --- |
| 0.1 | 2.084% | 3.400% | 0.520% | 1.539% |
| 0.25 | 2.243% | 3.236% | 0.520% | 1.747% |
| 0.5 | 2.611% | 3.361% | 0.987% | 1.822% |
| 0.75 | 3.275% | 3.717% | 2.087% | 2.098% |
| 0.9 | 4.182% | 4.325% | 2.722% | 2.710% |

### 高一致性正类：相同候选集合与相同 coverage

| coverage | S0 macro risk | S4 macro risk | S0 micro risk | S4 micro risk |
| --- | --- | --- | --- | --- |
| 0.1 | 1.480% | 2.907% | 0.520% | 1.457% |
| 0.25 | 1.536% | 2.583% | 0.520% | 1.650% |
| 0.5 | 1.763% | 2.592% | 0.678% | 1.638% |
| 0.75 | 2.084% | 2.681% | 1.036% | 1.727% |
| 0.9 | 2.377% | 2.741% | 1.479% | 1.790% |

S4 在所有五档逐图 macro coverage 上都更差。pooled micro 的 90% 正类 coverage 存在很小点估计改善，但它不能覆盖其他 coverage 的退化，更不能替代预注册的 50% 逐图主要比较。Trusted Positive Precision 在同覆盖率下等于 `1−risk`；完整所有方法 precision/coverage、各图有效数见 summary.json。

负类 S0/S4 的 macro risk 在 10/25/50/75/90% coverage 分别为 0.217/0.253/0.318/0.622/1.084%，逐项相等，相应 trusted-negative precision 分别为 99.783/99.747/99.682/99.378/98.916%。

| 固定 score>=0.90 | 正类 precision micro | 正类 conditional coverage | 正类 full-image coverage | 负类 precision micro | 负类 conditional coverage |
| --- | --- | --- | --- | --- | --- |
| S0 | 98.146% | 70.213% | 4.418% | 99.341% | 98.049% |
| S1 | 98.640% | 2.016% | 0.127% | 99.341% | 98.049% |
| S2 | 98.425% | 3.840% | 0.242% | 99.341% | 98.049% |
| S3 | 97.869% | 1.891% | 0.119% | 99.341% | 98.049% |
| S4 | 98.695% | 0.946% | 0.060% | 99.341% | 98.049% |
| S5 | 97.233% | 0.744% | 0.047% | 98.810% | 29.803% |
| S_HP | 98.786% | 1.763% | 0.111% | 99.341% | 98.049% |

固定阈值下 S4 看似 precision 更高，但只保留 34,252 个正类像素（原候选的 0.946%），S0 保留 2,542,564（70.213%）；这是强烈的 coverage 变化，不能宣称错误识别提升。S4/S0 负类均保留 52,878,399 像素，占全图 91.880%，precision=99.341%，完全不变。

## 9. 配对 bootstrap 与跨数据集一致性

| 数据集 | high_pos ΔAUROC S4−S0 [95% CI] | 有效图/组 | pos Δrisk@50 S4−S0 [95% CI] | 有效图/组 |
| --- | --- | --- | --- | --- |
| ALL | -0.25537 [-0.28369, -0.22813] | 526/329 | 0.00749 [0.00572, 0.00937] | 1144/490 |
| CASIAv2 | -0.24880 [-0.27681, -0.21983] | 474/307 | 0.00818 [0.00628, 0.01017] | 1023/461 |
| Columbia | -0.48261 [-0.71533, -0.39676] | 11/5 | 0.00107 [0.00059, 0.00152] | 25/6 |
| Coverage | -0.54415 [-0.66926, -0.40798] | 12/12 | 0.01847 [0.00832, 0.02975] | 14/14 |
| NIST16 | -0.15703 [-0.30055, -0.05560] | 29/5 | -0.00099 [-0.00926, 0.00448] | 82/9 |

表中 risk 差以 0–1 单位表示，乘 100 为百分点。NIST16 的小幅 50% coverage 改善 CI 跨零，不能称作可靠正面结果。Columbia 仅 6 个诊断组，高一致性有效组仅 5；Coverage 只有 14 图，低于 20 图跨域门槛，仍全部披露。ALL 的图像权重主要来自 CASIAv2（1,025/1,147），不能解释为四数据集等权结论。

| 方法−S0 | high_pos ΔAUROC [95% CI] | pos ΔAURC [95% CI] | pos Δrisk@50 [95% CI] |
| --- | --- | --- | --- |
| S1 | -0.25265 [-0.27820, -0.22787] | 0.00835 [0.00660, 0.01023] | 0.00834 [0.00673, 0.01016] |
| S2 | -0.30197 [-0.33127, -0.27409] | 0.00745 [0.00598, 0.00910] | 0.00814 [0.00646, 0.01000] |
| S3 | -0.21874 [-0.24406, -0.19415] | 0.00757 [0.00564, 0.00972] | 0.00725 [0.00540, 0.00924] |
| S4 | -0.25537 [-0.28369, -0.22813] | 0.00767 [0.00582, 0.00965] | 0.00749 [0.00572, 0.00937] |
| S5 | -0.31895 [-0.34834, -0.29070] | 0.02553 [0.02176, 0.02977] | 0.02658 [0.02261, 0.03113] |
| S_HP | -0.26262 [-0.28755, -0.23669] | 0.00809 [0.00613, 0.01027] | 0.00788 [0.00619, 0.00969] |

这些探索性组件差值没有用于选择最优模块，未作多重检验校正。主要结论依赖预先冻结的 S4 两个端点。bootstrap 的抽样单位是记录的 source-group 代理，不能修复代理本身不准确或遗漏的来源关系。

## 10. Box 与篡改面积条件

| 区域 | 候选数 | 错误数 | 条件错误率 | S0 macro AUROC | S4 macro AUROC |
| --- | --- | --- | --- | --- | --- |
| box_inside | 5524317 | 892345 | 16.153% | 0.84037 | 0.50794 |
| box_edge | 867461 | 112071 | 12.919% | 0.83042 | 0.76556 |
| box_ring | 1211164 | 15330 | 1.266% | 0.92392 | 0.95189 |
| box_far | 49948930 | 5826 | 0.012% | 0.86057 | 0.86108 |
| pos_box_inside | 3420359 | 123162 | 3.601% | 0.86764 | 0.77427 |
| pos_box_edge | 183586 | 29959 | 16.319% | 0.79845 | 0.75600 |
| pos_box_ring | 15258 | 15194 | 99.581% | 0.62583 | 0.63899 |

Box-ring 的预测正类几乎全为错误，与粗框支持范围有关；远外部海量简单背景会显著稀释全图错误率。不能只展示背景主导的全图结果。

| GT 面积档 | pos 候选数 | FP prevalence | S0 micro AUROC | S4 micro AUROC | S0 micro risk50 | S4 micro risk50 |
| --- | --- | --- | --- | --- | --- | --- |
| 1-10% | 1028937 | 7.311% | 0.88173 | 0.80755 | 0.567% | 2.479% |
| 10-30% | 1033718 | 3.229% | 0.85662 | 0.79058 | 0.251% | 1.092% |
| <1% | 87168 | 6.048% | 0.82361 | 0.77623 | 1.425% | 2.323% |
| >=30% | 1471417 | 3.838% | 0.81763 | 0.79953 | 1.453% | 1.421% |

面积分层只是描述性诊断，不用于选择归一化或阈值。每数据集各面积档、所有方法的完整指标保存在 summary.json 的 area_strata。

## 11. 证据方向、纹理与压缩偏差

| 教师类别 | Noise mean | JPEG mean | FFT mean | 融合 r mean |
| --- | --- | --- | --- | --- |
| TP | 0.22510 | 0.25621 | 0.23545 | 0.21514 |
| FP | 0.20061 | 0.26128 | 0.24027 | 0.20868 |
| TN | 0.23778 | 0.22310 | 0.21345 | 0.20682 |
| FN | 0.24685 | 0.29187 | 0.23143 | 0.23538 |

FP 的融合 r 均值仅略低于 TP，JPEG/FFT 反而略高；均值差本身不是有效排序证据。S5 的接近随机 AUROC 与 S4 的实际退化说明，不能把低异常直接解释成“更多错误”的普适可靠性规律。

| 全 train 域 | r 逐图均值 | r/纹理 Spearman rho | 梯度高四分位 r | 梯度低四分位 r |
| --- | --- | --- | --- | --- |
| CASIAv2 | 0.22163 | 0.29181 | 0.23903 | 0.22741 |
| NIST16 | 0.13493 | 0.79015 | 0.17057 | 0.12194 |
| Columbia | 0.16390 | 0.37548 | 0.22105 | 0.13203 |
| Coverage | 0.24854 | 0.40912 | 0.28592 | 0.23894 |

纹理由 224 RGB Sobel 强度定义，边缘/平滑区域分别为每图梯度 >=q75 / <=q25，仅用于偏差描述，不改变特征。NIST16 rho=0.79015，表明明显的纹理关联。数据集、native 分辨率、压缩和设备因素混合，现有信息不足以分离设备因果影响，不能声称已排除采集设备偏差。保存的 Spearman p 值仅为描述性输出，未作为独立像素或来源组的显著性证据。

| 数据集（各固定 8 图） | Q75 JPEG norm MAE vs Q90 | Q95 MAE | Q75 融合 r mean Δ | Q95 mean Δ |
| --- | --- | --- | --- | --- |
| CASIAv2 | 0.14849 | 0.12333 | 0.03009 | -0.02360 |
| Columbia | 0.08109 | 0.06308 | 0.02220 | -0.01878 |
| Coverage | 0.15190 | 0.13357 | 0.02783 | -0.03372 |
| NIST16 | 0.05014 | 0.03813 | 0.01337 | -0.00971 |

Q75/Q95 仅在预先固定的各数据集 hash 前 8 张 teacher_diag 图上检查，共 32 图、64 个质量变体；使用原 Q90 的统计量，不重拟合。它们没有参与 S0–S5 方法选择，也没有生成替代训练 cache。JPEG 在自然纹理和非 JPEG 源上响应、对质量因子敏感；这支持偏差风险，不能据此证明某一具体误差完全由压缩引起。

## 12. micro 数值近似的真实敏感性

| 固定 32 图 pooled ALL | AUROC 最大绝对偏差 | AP 最大绝对偏差 | AURC 最大绝对偏差 |
| --- | --- | --- | --- |
| pos | 0.0001774 | 0.0005716 | 0.0000261 |
| high_pos | 0.0006544 | 0.0000947 | 0.0000335 |
| neg | 0.0107905 | 0.0060266 | 0.0010494 |

所有 dataset/region/method 单元中，最大 AUROC 偏差为 0.0263436（NIST16 neg S0），最大 AP 偏差为 0.0125274（CASIAv2 neg S0），最大 AURC 偏差为 0.0023420（Columbia neg S0）。因此不能把背景分数接近 1 时的直方图微指标说成精确值，也不能过度解释 pooled negative 曲线的平坦区。这里仅报告固定子集的实测偏差，不把它当作全量误差上界。主要逐图指标、配对 CI、固定阈值计数均为精确计算，不受该近似影响；负类 S0/S4 相等也不受影响。

完整结果为 `micro_sensitivity.json`，含 exact 与 histogram 的全部指标和固定样本 key。

## 13. 重点案例与人工检查

| 案例 | 样本 ID | GT | pos risk@50 S0→S4 | 图 |
| --- | --- | --- | --- | --- |
| historical_animal | Tp_D_CND_M_N_ani00018_sec00096_00138 | 无：train_fit | N/A | [PNG](../assets/stage3_diagnostics/cases/historical_animal.png) |
| historical_zero | Tp_S_CNN_M_N_sec00091_sec00091_10355 | 无：train_fit | N/A | [PNG](../assets/stage3_diagnostics/cases/historical_zero.png) |
| historical_high_coverage | Tp_D_NNN_L_N_ani00010_nat00099_10217 | 无：train_fit | N/A | [PNG](../assets/stage3_diagnostics/cases/historical_high_coverage.png) |
| historical_max_variance | NC2016_2255 | 无：train_fit | N/A | [PNG](../assets/stage3_diagnostics/cases/historical_max_variance.png) |
| historical_k1_k8_difference | NC2016_3660 | 无：train_fit | N/A | [PNG](../assets/stage3_diagnostics/cases/historical_k1_k8_difference.png) |
| high_confidence_fp | Tp_S_CNN_M_N_arc00067_arc00067_10250 | 有 | 65.548% → 70.275% | [PNG](../assets/stage3_diagnostics/cases/high_confidence_fp.png) |
| high_confidence_fn | Tp_S_CNN_M_N_art00048_art00048_10259 | 有 | 2.199% → 8.031% | [PNG](../assets/stage3_diagnostics/cases/high_confidence_fn.png) |
| natural_texture_alarm | Tp_D_NNN_M_N_txt00060_txt00061_10186 | 有 | 0.000% → 0.000% | [PNG](../assets/stage3_diagnostics/cases/natural_texture_alarm.png) |
| jpeg_natural_alarm | Tp_S_CNN_M_N_txt10113_txt10113_10865 | 有 | 0.000% → 0.000% | [PNG](../assets/stage3_diagnostics/cases/jpeg_natural_alarm.png) |
| evidence_success | Tp_S_NNN_S_N_cha20016_cha20016_02015 | 有 | 18.750% → 7.375% | [PNG](../assets/stage3_diagnostics/cases/evidence_success.png) |
| evidence_failure | Tp_S_NNN_S_N_ani00007_ani00007_00127 | 有 | 0.249% → 32.031% | [PNG](../assets/stage3_diagnostics/cases/evidence_failure.png) |

11 张均已实际生成并人工预览。前五张恰在 train_fit，严格没有读取其 GT；其图含 RGB+Box、K1、K8 mu/var、c_sam、四证据、r、c_pos 和匹配 50% coverage 的选择差异，不能据它们判定准确率。其余六张另有 GT、FP、FN，只保存在明确的诊断目录。

河边动物图的 SAM 正类主要在上方水面，证据在岸线/纹理响应；因无 GT，本报告不凭视觉将水面标签判错。最高覆盖率图为海面/海鸟，coverage=94.976%，同样不作 GT 正误结论。`NC2016_2255` 与 `NC2016_3660` 的高 variance 令 c_sam 在大区域被压低；它们的历史极值身份保持不变。

高置信 FP 案例为建筑/天空图 `arc00067...10250`，有 10,344 个高一致性 FP；高置信 FN 案例为装饰图 `art00048...10259`，有 13,926 个稳定 FN。个别成功例 `cha20016...02015` 在相同 50% coverage 的风险从 18.750% 降为 7.375%；失败例 `ani00007...00127` 从 0.249% 升为 32.031%。这些是结果后按公开客观排序挑选的解释案例，不能代替全体统计。

自然纹理例为织物图，TN 区域平均 r=0.43751；JPEG 自然响应例为 JPEG 编码的水滴/纹理图，TN 区域 JPEG=0.51226。后者不能单独证明压缩因果。特征“异常误报”不等于规则新增 SAM FP，因为规则不改变 pred；其风险在于相对排序偏向高纹理错误。

“全零 K8”严格指 `mu>=0.5` 的硬 mask 全零：实际 mu∈[0.020493,0.436035]，var mean=0.023849，c_sam mean=0.657453，绝非 mu/var 全零 tensor。焦点样本无 GT 数值见 `focus_gtfree_statistics.json`。

| 历史焦点 | 前景覆盖率 | var mean | c_sam mean | GT |
| --- | --- | --- | --- | --- |
| Tp_D_CND_M_N_ani00018_sec00096_00138 | 35.782% | 0.003807 | 0.957507 | 未读 |
| Tp_S_CNN_M_N_sec00091_sec00091_10355 | 0.000% | 0.023849 | 0.657453 | 未读 |
| Tp_D_NNN_L_N_ani00010_nat00099_10217 | 94.976% | 0.023133 | 0.806636 | 未读 |
| NC2016_2255 | 44.715% | 0.192938 | 0.012959 | 未读 |
| NC2016_3660 | 17.853% | 0.171510 | 0.019879 | 未读 |

## 14. 实际文件、产物和运行命令

新增实现均在 fecd/：

- `evidence/{extractors,normalize,cache}.py`；`confidence/teacher_scores.py`。
- `diagnostics/{protocol,metrics,risk_coverage,bootstrap,visualizations,sample_audit,cases,artifact_validation}.py`。
- `tools/{prepare_stage3_split,generate_evidence_cache,fit_evidence_stats,audit_pseudo,stage3_bias_checks,summarize_stage3,check_stage3}.py`。
- `tests/{test_stage3_split,test_evidence,test_evidence_cache,test_teacher_scores,test_diagnostics,test_sample_audit,test_stage3_bias,test_summary,test_stage3_cases,test_stage3_checks}.py`。
- `docs/stage3_data_protocol.md`、本文；冻结 split/protocol/stats、evidence NPZ、门禁日志与专用 diagnostics 产物。

核心产物：`stage3_diagnostics/per_image.jsonl`（逐图完整指标和候选错误数）、`histograms.npz`/`hist_index.json`（2,156 scopes）、`summary.json`（全部数据集/方法/面积档/CI）、`gt_access_log.json`、`auxiliary_gt_access_log.json`、`bias_checks.json`/`per_image_bias.jsonl`、`cases_selection.json`、`micro_sensitivity.json`。15 张风险曲线在 `stage3_diagnostics/risk_curves/`；诊断 GT 图只在 `stage3_diagnostics/cases/`。

工作目录 `/home/projects/BoxPromtIML`；以下命令均实际运行，Python 全路径为 `/usr/local/miniconda3/envs/boxpromptiml/bin/python`（下文用 `$PY` 表示，不涉及修改环境）：

```bash
PY=/usr/local/miniconda3/envs/boxpromptiml/bin/python
$PY -m fecd.tools.prepare_stage3_split
$PY -m fecd.tools.prepare_stage3_split --output fecd/assets/stage3_split_reproduction.json
$PY -m fecd.tools.generate_evidence_cache --subset train_fit --limit 12 --workers 2
$PY -m fecd.tools.generate_evidence_cache --subset all --workers 4
$PY -m fecd.tools.generate_evidence_cache --subset all --workers 4 --validate-only
$PY -m fecd.tools.fit_evidence_stats
$PY -m fecd.tools.check_stage3 --require-3a --output fecd/assets/stage3a_gate.json
$PY -m fecd.tools.audit_pseudo --limit 2 --output-dir fecd/assets/stage3_diagnostics/pilot
$PY -m fecd.tools.audit_pseudo
$PY -m fecd.tools.stage3_bias_checks
$PY -m fecd.tools.summarize_stage3
$PY -m fecd.tools.audit_pseudo --cases-only
$PY -m unittest discover -s fecd/tests -p 'test_*.py'
$PY -m fecd.tests.run_stage2b_tests
$PY -m compileall -q fecd
$PY -m fecd.tools.check_stage3 --require-3a --protected --output fecd/assets/stage3_final_checks.json
```

多数据集 pilot 通过 `generate_record` 对每数据集 train_fit 前 3 张执行，绘图使用临时 pilot stats；保存 `stage3_evidence/pilot_report.json` 与 `pilot/`。焦点无 GT 统计通过只读 `load_cache`/`sam_confidence` 导出。没有 GPU 教师重推或学生训练。

真实日志为 `fecd/assets/stage3_{evidence_pilot,evidence_generate,evidence_validate,fit_stats,audit_pilot,audit,bias_checks,summary,cases,all_tests,regression,final_checks}.log`。全量教师诊断实测耗时 399.26 秒（不含后续汇总/案例）；其他阶段未记录可靠壁钟总时间，因此不编造耗时。

## 15. 子关卡验收与最终判断

3A：四数据集 12 图 pilot 通过；5,737 证据缓存全量验证通过；4,590 图 GT-free 拟合通过；split 重建一致，来源组代理 crossing=0。

3B：完整 1,147 图诊断成功，无跳过/失败；所有方法 region 候选 n/error 相同；GT access 限于 teacher_diag。S0/S4 负类逐图字典及 histogram 完全相等。公式和阈值没有改动。

3C：2,000 次 source-group paired bootstrap、全部数据集/面积/Box 分层、15 张 Risk–Coverage 图、11 张重点案例及 32 图 exact/micro 和 Q75/Q95 检查全部实际生成。汇总前校验 run status、protocol、membership、scope 完整性及每 scope 候选/错误计数一致，防止未完成或旧新产物混合。

最终 unittest **60 tests passed**；旧阶段 harness **9 checks passed**；compileall 通过。静态分析对 29 个阶段 3 Python 文件执行 AST 语法、符号表未解析全局名和 GT 字段边界检查，issues=0。环境没有 ruff/pyflakes/pylint，未将该限定检查冒充完整第三方 lint/type checker。测试先红后绿证据来自拆分、描述子、可信度、缓存、指标和汇总完整性测试；一次测试命令误写 `test_stage3_summary` 导致模块找不到，已纠正为真实 `test_summary` 并完成最终全套，无未解决失败。

原始保护快照校验 **11,500 文件，变化 0**：原始 train.py、eval.py、boxpromptiml.py、utils、checkpoint、Box、冻结 K1/K8 缓存及原 train manifest 保持任务开始时的字节内容。工作区原有修改没有还原或覆盖。

**最终：NO-GO；不建议进入阶段 4。** 核心高一致性 FP 识别显著变差，同覆盖率正类监督错误显著增加；跨数据集不一致，唯一 NIST16 risk50 轻微改善缺乏统计确定性。纹理/域/压缩关联加重解释风险，而当前负类公式不能解决稳定 FN。

限制包括：来源身份代理与 CASIAv2 大分量留 fit 的选择性；Columbia/NIST16 有效来源组较少；4096-bin micro 近似；工作网格对分辨率/设备因素的混淆；固定特征没有证明真实取证定位能力；Box 已由训练 GT 离线导出，因此本实验只评估固定 GT-derived Box 提示下的教师可靠性；历史训练划分的 NIST16 跨测试来源风险。局部成功案例不改变总体判断。

阶段 3 到此结束。不得继续叠加 Confidence Head、Consistency 或 FECD Loss 掩盖本结果。若用户未来授权新的研究假设，应另立版本和独立确认协议；本次不修改任何后续训练协议，等待验收。
