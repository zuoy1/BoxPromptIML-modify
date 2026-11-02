# 阶段 4F0 冻结的正式训练预算提案

本文件是后续申请方案，未执行、未授权阶段 4F1。本阶段单 Epoch checkpoint 不用于正式初始化，不据此选择实验组或调整超参数。

## 固定主对照

P0=K1 hard全像素；P1=K8 hard全像素；P2F=K8 hard逐图正负类固定随机各floor(n/2)；P3F=相同K8候选集合按冻结c_sam降序各floor(n/2)。P2F/P3F使用同一固定随机优先级处理选择及并列，不依赖Epoch；在原始224网格选择后同步几何增强。P2E是每Epoch重采样随机策略，单独登记，未在4F0训练，也不自动纳入正式主对照。

## 预算、初始化和优化

一个共同seed=20261029，student_train全部3530图，batch=4、drop_last=False，每Epoch883更新，20Epoch每组17660更新，四主组总70640逻辑更新；不含另行授权的恢复校验开销。每个Epoch均读完全部四域。固定Epoch20 final，不挑各组不同的best-dev Epoch。多seed须另外对所有主组等同授权。

统一原BoxPromptIML，使用已核验的本地TinyViT适配器及相同seed初始化其他层，不加载历史5737张B0或本轮预检checkpoint。Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，constant学习率，float32，clip norm=0.5。损失是4P同一个逐图正负平衡masked BCE，各已选类均值乘0.5；空类贡献0、不重分配。六种离散几何增强由seed:epoch:sample_key:geometry确定；排序sample_key后torch.Generator(seed+epoch)产生共同randperm。四组初始化、数据序、增强、更新步数完全一致，不因某组表现调整。

## 评估与解释

采用evaluation_protocol.json：独立模型、batch=1、每图恢复checkpoint memory；PIL RGB BILINEAR224及ImageNet标准化；raw logits sigmoid、固定阈值0.5，无逐图min-max；配对GT NEAREST224后>127。主指标new_dev image-macro Pixel F1，次指标IoU、各域、dataset-equal和pooled micro。空GT且空预测F1/IoU=1，只有一者为空=0。

主差值P1−P0、P2F−P1、P3F−P2F。new_dev仅开发比较，不是独立确认；new_confirm、teacher_diag和正式IND/OOD保持锁定。未来soft label必须另设对照，不能同时改筛选与损失。本统一BCE不是论文structure_loss严格复现，历史B0不能作同条件性能归因。

## 预先固定的停止条件

评估顺序依赖、GT进入训练、身份/监督预算/选择策略不一致、NaN/Inf或恢复不一致立即BLOCKED。new_dev至少50%图预测全零或面积>=0.99，且image-macro F1<=0.05，停止正式申请；或至少90%图概率均值<=0.01或>=0.99，停止申请。其他异常面积和分域风险完整报告，不事后修改门槛。禁止根据单Epoch分数挑组、改阈值、改权重或重写负面结果。

机器协议：fecd/configs/stage4f0/formal20_proposal.json，canonical hash=8b25d14ae1d90c00db59e5794eaff4d038f82829f63abbd1513404923e8d46d2。严格评估hash=54baa31ac65ec3df978af09c43b2454fd5515db1c8117b86c6ff2cd7d267416f。阶段4F0结束后等待用户验收，不自动开始20Epoch。
