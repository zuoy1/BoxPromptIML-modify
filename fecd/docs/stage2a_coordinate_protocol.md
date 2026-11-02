# FECD 阶段 2A：Box、SAM 与 cache 坐标协议

状态：设计已确定；当前只完成 smoke 级测试，不生成正式 cache。

## 唯一坐标约定

1. 原图以解码后的 RGB `uint8[H,W,3]` 表示，native `(H,W)` 是源坐标系。样本身份由数据 manifest 的 `(dataset, split, id, image_sha256)` 确认，不从文件名 stem 推断。
2. 离线 Box 导出是唯一允许读取训练 GT 的阶段。轮廓和 box 坐标在 native resolution 计算，统一使用 SAM 的 `XYXY` 像素坐标约定；记录图像尺寸、导出算法/版本和 box 内容哈希。该导出器只能读取 train manifest 指定的 GT，不接受 test/validation 路径。
3. SAM 对 native RGB 调用 `SamPredictor.set_image()`；SAM 自己将图像和 box 变换到 encoder 坐标。Prompt 仍以 native `XYXY` 传给 predictor。返回的 logits 是原图空间 `[H,W]` 的 raw logits，不能当作概率。
4. 先对 logits 逐像素做 sigmoid 得到连续 probability，再以 bilinear、`align_corners=False` resize 到学生 `224×224`。GT 导出的 box mask 用 nearest-neighbor resize 到 `224×224`；box 边界来自 native box 同比例缩放，保存时按像素中心/半开区间规则实现并以单元测试固定。
5. 学生 RGB 由 native 图像按既定 resize 规则得到 224 输入。cache 中的 `mu/var/agreement/evidence/box masks` 都在同一 224 网格；后续几何变换必须对学生图像和全部空间图应用同一个离散变换（首轮仅 HFlip、VFlip、Rot90）。首轮禁用随机 crop、在线 copy-move、inpainting，以及任何不和 cache 同步的 resize/几何增强。光度增强只能改变 RGB，不改变空间图。
6. 学生 Dataset 只接收 train-only 图像清单和 cache，不接收 GT 路径、GT tensor 或能间接加载 GT 的通用 audit manifest。GT box 导出、GT 诊断、测试评估各自使用独立入口；测试 GT 不能进入训练 sampler、batch、confidence 计算或 loss。

## Cache 键

不以 stem 作为 key。确定性 `sample_key`：

```text
SHA256(dataset || NUL || split || NUL || sample_id || NUL || image_sha256)
```

伪标签 cache key：

```text
SHA256(sample_key || NUL || native_box_sha256 || NUL || sam_config_sha256 || NUL || cache_schema_version)
```

`native_box_sha256` 基于规范化、固定 dtype/字节序的原始坐标和 native 图像尺寸；`sam_config_sha256` 包含模型/checkpoint SHA、prompt variants、box top-k、`multimask_output`、`return_logits`、resize 规则与阈值设置。每个 cache 同时记录以上字段用于校验。不同数据集同名样本、相同 stem 的文件和不同 box/config 不能互相覆盖。

## 当前验证边界

已验证一个 256×256 smoke RGB 可交给 SAM，Box 在 native XYXY 下输出 `(1,256,256)` logits；sigmoid 后数值在 `[0,1]`。新增的最小坐标单测实际通过：native probability resize 到 `[224,224]`、非正方形源图的 XYXY 等比缩放和 box rasterize、Rot90 对 RGB/teacher/evidence 的空间同步。完整 cache schema、prompt 多框变换及几何增强闭环仍留在阶段 2B 的小样本测试中；该闭环通过前不启动全量缓存。
