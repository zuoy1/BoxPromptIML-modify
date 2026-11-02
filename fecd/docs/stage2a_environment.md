# FECD 阶段 2A：环境与模型 Smoke Test

审计日期：2026-10-23。所有命令使用现存 Conda 环境，不修改系统 Python/CUDA 栈。

## 环境选择与依赖

复用了 `/usr/local/miniconda3/envs/boxpromptiml`（Python 3.10.22）。本次没有安装、升级或卸载任何包。环境依赖版本：

| 组件 | 实际版本 |
|---|---|
| PyTorch | 2.2.1+cu121 |
| torchvision | 0.17.1+cu121 |
| CUDA runtime | 12.1 |
| NumPy | 1.26.4 |
| timm | 0.9.12 |
| albumentations | 1.3.1 |
| tensorboardX | 2.6.5 |
| scikit-learn | 1.7.2 |
| thop | 0.1.1 (`0.1.1-2209072238` distribution build) |
| segment-anything | 1.0 |
| OpenCV (`cv2`) | 4.9.0 |
| SciPy | 1.15.3 |
| Pillow | 12.3.0 |
| tqdm | 4.70.1 |

环境内 `PyYAML` 为 6.0.3。原仓库 `train.py`/`eval.py` 直接 import 的关键包均可导入。没有发现缺失的本阶段运行依赖。系统 Python 3.11 环境仍缺少模型依赖，不应用于 FECD。

使用方式：

```bash
cd /home/projects/BoxPromtIML
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_environment
```

也可先加载 Conda shell hook 后激活该 prefix。复现时不要运行不带约束的 `pip install -U`，不要替换 torch/torchvision/NumPy；优先复用这个已验证环境。SAM Python package 已安装，权重从 `/home/weights` 显式读取，无需再次下载。

## TinyViT 权重路径与网络行为

原始 `BoxPromptIML.__init__` 调用 `tiny_vit_5m_224(pretrained=True)`。仓库 `TinyViT/models/tiny_vit.py` 将 `pretrained=True` 配置为 TinyViT model-zoo GitHub URL：若本地 timm cache 不含该权重，默认路径可能触发下载。不能用此默认构造路径保证离线运行。

新增 `fecd/model_loading.py` 临时替换 `boxpromptiml` 模块导入的 TinyViT factory：用 `pretrained=False` 构造相同 backbone，再从 `/home/weights/tiny_vit_5m_22kto1k_distill.pth` 严格加载 state dict，所有 296 个 tensor key 均匹配，无 missing/unexpected keys；结束后恢复原 factory。该适配器不更改 BoxPromptIML 结构或原始源文件，也不触发模型 URL 加载。本次没有通过网络获取权重。

## Smoke test 结果

实际命令：

```bash
/usr/local/miniconda3/envs/boxpromptiml/bin/python -m fecd.tools.check_environment
/usr/local/miniconda3/envs/boxpromptiml/bin/python - <<'PY'
from fecd.tests.test_model_smoke import student_smoke
print(student_smoke())
PY
/usr/local/miniconda3/envs/boxpromptiml/bin/python - <<'PY'
from fecd.tests.test_model_smoke import sam_smoke
print(sam_smoke())
PY
```

结果：

- `from boxpromptiml import BoxPromptIML`：成功。
- 学生：RTX 4090 / `cuda:0`，输入 `[1,3,224,224]`，输出 `[1,1,224,224]`，有限值；TinyViT strict load 296/296，网络下载 `False`。一次实测 logits 范围约 `[-0.2419, 0.3406]`。
- SAM：ViT-H checkpoint `/home/weights/sam_vit_h_4b8939.pth` 实际加载成功，smoke RGB `[256,256,3]`，Box `[38.4,38.4,217.6,217.6]`；`multimask_output=False, return_logits=True` 输出 `float32 [1,256,256]`，logits 范围约 `[-18.97,10.10]`，全为有限值；sigmoid probability 范围约 `[5.76e-9,0.999959]`。两者已明确区分。
- 不覆盖任何已有学生 checkpoint 或 SAM/TinyViT 权重。

前向 smoke 测试不代表训练性能、cache 正确性或正式数据评估结果。
