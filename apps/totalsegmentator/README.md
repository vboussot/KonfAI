[![PyPI version](https://img.shields.io/pypi/v/totalsegmentator-konfai.svg?color=blue)](https://pypi.org/project/totalsegmentator-konfai/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![KonfAI](https://img.shields.io/badge/framework-KonfAI-orange.svg)](https://github.com/vboussot/KonfAI)

# TotalSegmentator-KonfAI

**Fast and lightweight TotalSegmentator inference using the KonfAI framework**

---

## 🧩 Overview

**TotalSegmentator-KonfAI** is a lightweight command-line interface (CLI) for running [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) models a **multi-organ medical image segmentation**, through the [KonfAI](https://github.com/vboussot/KonfAI) deep learning framework.

It provides **fast** inference for segmentation tasks, even on low-resource hardware.  
Models are automatically downloaded from [Hugging Face Hub](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI).

---

## 🚀 Installation

From PyPI:
```bash
pip install totalsegmentator-konfai
```

From source:
```bash
git clone https://github.com/vboussot/TotalSegmentator-KonfAI.git
cd TotalSegmentator-KonfAI
pip install .
```

---

## ⚙️ Usage

Perform segmentation on an input volume:

```bash
totalsegmentator-konfai -i path/to/image.nii.gz -o path/to/seg.nii.gz
```

### Optional arguments

| Flag | Description | Default |
|------|--------------|----------|
| `-i`, `--input` | Path to the input medical image | *required* |
| `-o`, `--output` | Path to save the segmentation | `Seg.nii.gz` |
| `-ta`, `--task` | Choose model type: `total` or `total_mr` | `total` |
| `-f`, `--fast` | Use faster low-resolution model (≈3 mm) | `False` |
| `-g`, `--gpu` | GPU list (e.g. `0` or `0,1`) | *CPU if unset* |
| `--cpu` | Number of CPU cores (used if no GPU) | `1` |
| `-q`, `--quiet` | Suppress console output | `False` |

### Example

```bash
totalsegmentator-konfai -i patient01.nii.gz -o seg_patient01.nii.gz --gpu 0 --task total --fast
```

---

## 🧠 Features

- ⚡ **Fast inference** using [KonfAI](https://github.com/vboussot/KonfAI)
- 🤗 **Automatic model download** from Hugging Face  
- 🧩 **Multi-model support**
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 📖 Reference

This package is based on the original **[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)** by  
**Wasserthal et al.**, a deep learning framework for whole-body CT and MR segmentation.

For scientific use, please cite the original TotalSegmentator work in addition to this CLI tool.

---

## 🧾 License

Released under the **Apache 2.0 License**.  
The original TotalSegmentator and KonfAI licenses remain with their respective authors.

---

## 🔗 Links

- 🧠 **Original TotalSegmentator:** [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)  
- ⚙️ **KonfAI Framework:** [github.com/vboussot/KonfAI](https://github.com/vboussot/KonfAI)  
- 🤗 **Model Hub:** [huggingface.co/VBoussot/TotalSegmentator-KonfAI](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI)  
- 📦 **PyPI Package:** [pypi.org/project/totalsegmentator-konfai](https://pypi.org/project/totalsegmentator-konfai)

---
