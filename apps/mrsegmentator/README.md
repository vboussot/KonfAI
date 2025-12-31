[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/mrsegmentator-konfai.svg?color=blue)](https://pypi.org/project/mrsegmentator-konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/KonfAI_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/KonfAI_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# MRSegmentator-KonfAI 

**Fast and lightweight CLI for whole-body MRI segmentation using MRSegmentator models within the KonfAI framework.**

---

## 🧩 Overview

**MRSegmentator-KonfAI** is a lightweight **command-line interface (CLI)** for running  
**[MRSegmentator](https://github.com/hhaentze/MRSegmentator)** models through the  
[KonfAI](https://github.com/vboussot/KonfAI) deep learning framework.

It provides **fast and efficient inference** for whole-body MRI segmentation, including on low-resource hardware.  
Pretrained models are automatically downloaded from  
[Hugging Face Hub](https://huggingface.co/VBoussot/MRSegmentator-KonfAI).

---

## 🚀 Installation

From PyPI:
```bash
python -m pip install mrsegmentator-konfai
```

From source:
```bash
git clone https://github.com/vboussot/KonfAI.git
python -m pip install -e apps/mrsegmentator
```

---

## ⚙️ Usage

Run inference on an MRI scan:
```bash
mrsegmentator-konfai -i path/to/input.nii.gz -o ./Output/
```

### Optional arguments

| Flag | Description | Default |
|------|--------------|----------|
| `-i`, `--input` | Path to the input MRI volume | *required* |
| `-o`, `--output` | Path to save the segmentation | `./Output/` |
| `--gt` | Path to reference segmentation (ground truth), if available (enables evaluation workflows) | *unset* |
| `--mask` | Path to region-of-interest mask used for evaluation and uncertainty analysis | *unset* |
| `-f`, `--folds` | Number of model folds to ensemble (1–5) | `2` |
| `-uncertainty` | Save uncertainty maps | `False` |
| `--gpu` | GPU list (e.g. `0` or `0,1`) | CPU if unset |
| `--cpu` | Number of CPU cores (if no GPU) | `1` |
| `-q`, `--quiet` | Suppress console output | `False` |

### Example

```bash
mrsegmentator-konfai -i path/to/input.nii.gz -o ./Output/ --gt path/to/reference.nii.gz --mask path/to/mask.nii.gz --gpu 0 -f 3 -uncertainty
```

---

## 🧠 Features

- ⚡ **Fast inference** powered by [KonfAI](https://github.com/vboussot/KonfAI)
- 🤗 **Automatic model download** from Hugging Face
- 🧩 **Multi-model ensembling**
- 🧠 **Supports evaluation workflows with reference data, and uncertainty estimation without reference**
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK


---

## 📖 Reference

If you use **MRSegmentator-KonfAI** in your work, please cite the original MRSegmentator work in addition to this CLI tool.

- Häntze, H. *et al.* (2025).  
  **Segmenting Whole-Body MRI and CT for Multiorgan Anatomic Structure Delineation.**  
  *Radiology: Artificial Intelligence*, 7(6). https://doi.org/10.1148/ryai.240777

- Boussot, V., & Dillenseger, J.-L. (2025).  
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**.  
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## 🔗 Links

- 🧠 **Original MRSegmentator:** [github.com/hhaentze/MRSegmentator](https://github.com/hhaentze/MRSegmentator)  
- 🤗 **Model Hub:** [huggingface.co/VBoussot/MRSegmentator-KonfAI](https://huggingface.co/VBoussot/MRSegmentator-KonfAI)  
- 📦 **PyPI Package:** [pypi.org/project/mrsegmentator-konfai](https://pypi.org/project/mrsegmentator-konfai)
