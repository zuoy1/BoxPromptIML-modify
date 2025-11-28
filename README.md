<div align="center">
<h1> [AAAI'26] From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations </h1>
</div>

## 📢 News
* **[2025-11]** Our paper is accepted by **AAAI 2026**! 🎉🎉🎉
* **[2025-11]** The code are being organized and will be released shortly. Please star this repo for updates!

## ✨ Highlights
* 🚀 Novel Weakly-Supervised IML Paradigm: We propose BoxPromptIML, a novel weakly supervised framework that requires only coarse bounding box annotations (7 seconds per image) instead of expensive pixel-level masks (23 minutes per image), achieving over 98% reduction in annotation cost while maintaining competitive localization performance.


* 🧠 Memory-Guided Active Perception: Inspired by human subconscious memory mechanisms, we design a Memory-Guided Gated Fusion Module (MGFM) that employs a dual-guidance strategy, actively contextualizing real-time observational cues with recalled prototypical patterns from a learnable memory bank, significantly enhancing localization accuracy and robustness.


* ⚡ Lightweight & Efficient Design: Our framework achieves exceptional efficiency with only 5.5M parameters and 1.4G FLOPs, making it the most lightweight model among compared methods while enabling fast convergence (competitive performance in just 20 epochs) and practical deployment.

* 🎯 Superior Performance & Generalization: The proposed method outperforms or rivals fully-supervised models on in-distribution datasets (F1: 0.619) and demonstrates strong out-of-distribution generalization (F1: 0.285), while maintaining high robustness against social media compression (Facebook, Weibo, WeChat, WhatsApp) and complex quality degradations.

##  Introduction
Official repository for the AAAI2026 paper “*From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations*” [[paper]](https://arxiv.org/abs/2511.20359) 

<div align="center">
    <img width="600" alt="image" src="Figures/figure1.jpg?raw=true">
</div>

Comparison of annotation cost and supervision quality in different IML paradigms. Pixel-level masks take 23 minutes per image, image-level labels require 4 seconds. Our IML framework from coarse to fine adopts rough boxes which only takes 7 seconds to annotate, while retaining spatial clues.

## 📻 Overview

<div align="center">
    <img width="1000" alt="image" src="Figures/method.jpg?raw=true">
</div>


Overview of the proposed framework for fine-grained manipulation localization using coarse prompts. A frozen SAM generates pseudo-masks from coarse annotations (e.g., bounding boxes) as soft supervision. The student model learns to replicate these masks via knowledge distillation. To improve localization, we design a Memory-Guided Gated Fusion Module (MGFM) that fuses multi-scale features with guidance from both real-time and memory-recalled signals. This enables refined mask prediction using only coarse supervision.

## 🎮 Quick start

### 1. Clone Project
```bash
git clone https://github.com/vpsg-research/BoxPromtIML
```


### 2. Install Environment

To set up the experimental environment, please follow the specific requirements for each baseline model. Taking **BoxPromptIML** as an example, you can create and install the environment using the provided script:

```bash
conda env create -f BoxPromptIML.yml

conda activate boxpromptiml

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download the pre-trained SAM weights(sam_vit_h_4b8939.pth)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 2. Prepare Datasets
| Dataset     | Nums        |  #CM          | #SP          | #IP          |  #Train          |  #Test          | 
| :----:      |    :----:   |         :----:|:----:        |    :----:    |         :----:   |         :----:  |
| CASIAv2   | 5123        | 3295          |1828          |    0         |        5123      |        0        |
| CASIAv1   | 920         | 459           |461           |    0         |        0         |        920      |
| Coverage    | 100         | 100           |0             |    0         |        70        |        30       |
| Columbia    | 180         | 0             |180           |    0         |         130      |        50       |
| NIST16      | 564         | 68            |288           |    208       |        414       |        150      |
|CocoGlide| 512 | - |  - | - | 0  |  512  |
|In-the-Wild      |201  | 0 | 201|-  | 0  |  201  |
|Korus   | 220 | -|-|-|0|220|
|DSO|100|-|-|-|0|100|
|IMD2020|2010|-|-|-|0|2010|


- CASIAv2 [Download](https://github.com/SunnyHaze/IML-Dataset-Corrections)
- CASIAv1 [Download](https://github.com/SunnyHaze/IML-Dataset-Corrections)
- Columbia  [Download](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- Coverage  [Download](https://github.com/wenbihan/coverage?tab=readme-ov-file)
- NIST16    [Download](https://mfc.nist.gov/users/sign_in)
- CocoGlide [Download](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip)
- In-the-Wild [Download](https://minyoungg.github.io/selfconsistency/)
- Korus [Download](https://pkorus.pl/downloads/dataset-realistic-tampering)
- DSO [Download](http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip)
- IMD2020 [Download](https://staff.utia.cas.cz/novozada/db/)


### 3. Train

To train our BoxPromptIML model, first ensure that the pretrained [TinyVit](https://github.com/wkcn/tinyvit/) weights are ready, and then simply run the corresponding training script.
```bash
# Make sure your directory paths are set correctly!
python train.py
```
### 4. Test
```bash
python eval.py
```


## ✨ Visualization results 

<div align="center">
    <img width="1000" alt="image" src="Figures/vision.jpg?raw=true">
</div>

<div align="center">
Visualization results of different methods.
</div>

## Citation
If you find our code useful, please consider citing us and give us a star!
```bibtex
@article{guo2025passive,
  title={From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations},
  author={Guo, Zhiqing and Xi, Dongdong and Li, Songlin and Yang, Gaobo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 👍Credit
Our implementation is partly built upon [segment-anything](https://github.com/facebookresearch/segment-anything) and [TinyViT](https://github.com/wkcn/tinyvit), while the comparative experiments were conducted in the environment provided by [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo). We sincerely thank the authors for their open-source contributions.
