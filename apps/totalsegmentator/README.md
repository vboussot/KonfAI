[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/mrsegmentator-konfai.svg?color=blue)](https://pypi.org/project/mrsegmentator-konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/KonfAI_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/KonfAI_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# TotalSegmentator-KonfAI

**Fast and lightweight CLI for whole-body CT or MRI segmentation using TotalSegmentator models within the KonfAI framework.**

---

## 🧩 Overview

**TotalSegmentator-KonfAI** is a lightweight **command-line interface (CLI)** for running  
**[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)** models for **multi-organ medical image segmentation**,  
through the [KonfAI](https://github.com/vboussot/KonfAI) deep learning framework.

It provides **fast and efficient inference** for segmentation tasks, including on low-resource hardware.  
Pretrained models are automatically downloaded from  
[Hugging Face Hub](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI).

---

## 🚀 Installation

From PyPI:
```bash
python -m pip install totalsegmentator-konfai
```

From source:
```bash
git clone https://github.com/vboussot/KonfAI.git
python -m pip install -e apps/totalsegmentator
```

---

## ⚙️ Usage

Perform segmentation on an input volume:

```bash
totalsegmentator-konfai total -i path/to/image.nii.gz -o ./Output/
```

### Optional arguments

| Flag | Description | Default |
|------|--------------|----------|
| `TASK` | Input modality / model name on Hugging Face | `total`, `total_mr`, `total_3mm`, `total_mr_3mm`|
| `-i`, `--input` | Path to the input medical image | *required* |
| `-o`, `--output` | Path to save the segmentation | `./Output/` |
| `--gt` | Path to reference segmentation (ground truth), if available (enables evaluation workflows) | *unset* |
| `--mask` | Path to region-of-interest mask used for evaluation and uncertainty analysis | *unset* |
| `--gpu` | GPU list (e.g. `0` or `0,1`) | CPU if unset |
| `--cpu` | Number of CPU cores (if no GPU) | `1` |
| `-q`, `--quiet` | Suppress console output | `False` |

### Example

```bash
totalsegmentator-konfai total -i path/to/input.nii.gz -o ./Output/ --gt path/to/reference.nii.gz --mask path/to/mask.nii.gz --gpu 0 -uncertainty

```

---

## 🧠 Features

- ⚡ **Fast inference** powered by [KonfAI](https://github.com/vboussot/KonfAI)
- 🤗 **Automatic model download** from Hugging Face
- 🧠 **Supports evaluation workflows with reference data**
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 📖 Reference

If you use **TotalSegmentator-KonfAI** in your work, please cite the original TotalSegmentator work in addition to this CLI tool.

- Wasserthal, J. *et al.* (2023).  
  **TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images.**  
  *Radiology: Artificial Intelligence*, 5(5). https://doi.org/10.1148/ryai.230024

- Akinci D’Antonoli, T. *et al.* (2025).  
  **TotalSegmentator MRI: Robust Sequence-independent Segmentation of Multiple Anatomic Structures in MRI.**  
  *Radiology*, 314(2). https://doi.org/10.1148/radiol.241613

- Boussot, V., & Dillenseger, J.-L. (2025).  
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**.  
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## 🔗 Links

- 🧠 **Original TotalSegmentator:** [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)  
- 🤗 **Model Hub:** [huggingface.co/VBoussot/TotalSegmentator-KonfAI](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI)  
- 📦 **PyPI Package:** [pypi.org/project/totalsegmentator-konfai](https://pypi.org/project/totalsegmentator-konfai)

---
