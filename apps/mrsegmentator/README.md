
[![PyPI version](https://img.shields.io/pypi/v/mrsegmentator-konfai.svg?color=blue)](https://pypi.org/project/mrsegmentator-konfai/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![KonfAI](https://img.shields.io/badge/framework-KonfAI-orange.svg)](https://github.com/vboussot/KonfAI)

# MRSegmentator-KonfAI 
**Fast and lightweight MRI segmentation built on the MRSegmentator model using the KonfAI framework**

---

## 🧩 Overview

**MRSegmentator-KonfAI** is a lightweight command-line interface (CLI) for running **[MRSegmentator](https://github.com/hhaentze/MRSegmentator)**  
models through the [KonfAI](https://github.com/vboussot/KonfAI) deep learning framework.

It provides **fast** inference for MRI segmentation tasks, even on low-resource hardware.  
Models are automatically downloaded from [Hugging Face Hub](https://huggingface.co/VBoussot/MRSegmentator-KonfAI).

---

## 🚀 Installation

From PyPI:
```bash
pip install mrsegmentator-konfai
```

From source:
```bash
git clone https://github.com/vboussot/MRSegmentator-KonfAI.git
cd MRSegmentator-KonfAI
pip install .
```

---

## ⚙️ Usage

Run inference on an MRI scan:
```bash
mrsegmentator-konfai -i path/to/input.nii.gz -o path/to/output_seg.nii.gz
```

### Optional arguments

| Flag | Description | Default |
|------|--------------|----------|
| `-i`, `--input` | Path to the input MRI volume | *required* |
| `-o`, `--output` | Path to save the segmentation | `Seg.nii.gz` |
| `-f`, `--folds` | Number of model folds to ensemble (1–5) | `2` |
| `-g`, `--gpu` | List of GPUs to use (e.g. `0`, `0,1`) | CPU if unset |
| `--cpu` | Number of CPU cores to use (if no GPU) | `1` |
| `-quiet` | Suppress console output | *off* |

Example:
```bash
mrsegmentator-konfai -i subject01.nii.gz -o seg_subject01.nii.gz --gpu 0 -f 3
```

---

## 🧠 Features

- ⚡ **Fast inference** using [KonfAI](https://github.com/vboussot/KonfAI)  
- 🤗 **Automatic model download** from Hugging Face  
- 🧩 **Support for ensemble models (multi-fold prediction)**  
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 📖 Reference

This package is based on the original **[MRSegmentator](https://github.com/hhaentze/MRSegmentator)** by  
**Henrik Haentze et al.**, a deep-learning-based method for full body MRI segmentation.

For scientific use, please cite the original MRSegmentator work in addition to this CLI tool.

---

## 🧾 License

This repository is released under the **Apache 2.0 License**.  
Original MRSegmentator license and copyright remain with their respective authors.

---

## 🔗 Links

- 🧠 **Original MRSegmentator:** [github.com/hhaentze/MRSegmentator](https://github.com/hhaentze/MRSegmentator)  
- ⚙️ **KonfAI Framework:** [github.com/vboussot/KonfAI](https://github.com/vboussot/KonfAI)  
- 🤗 **Model Hub:** [huggingface.co/VBoussot/MRSegmentator-KonfAI](https://huggingface.co/VBoussot/MRSegmentator-KonfAI)  
- 📦 **PyPI Package:** [pypi.org/project/mrsegmentator-konfai](https://pypi.org/project/mrsegmentator-konfai)
