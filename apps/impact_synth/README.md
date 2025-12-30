[![PyPI version](https://img.shields.io/pypi/v/impact_synth_konfai.svg?color=blue)](https://pypi.org/project/impact_synth_konfai/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![KonfAI](https://img.shields.io/badge/framework-KonfAI-orange.svg)](https://github.com/vboussot/KonfAI)

# IMPACT-Synth-KonfAI

**Fast and lightweight CLI for synthetic CT generation using IMPACT-Synth models within the KonfAI framework.**

---

## 🧩 Overview

**IMPACT-Synth-KonfAI** is the **command-line interface (CLI)** for performing **inference** and **uncertainty estimation** with the *IMPACT-Synth* models.  
It provides a streamlined way to generate **synthetic CT (sCT) images** from MR or CBCT scans, leveraging the [KonfAI](https://github.com/vboussot/KonfAI) framework for efficient inference, test-time augmentation (TTA), model ensembling, and uncertainty quantification.  

The underlying **IMPACT-Synth** models are a family of **supervised convolutional neural networks (CNNs)** dedicated to **sCT generation**. 
They build upon the research presented in **“Why Registration Quality Matters: Enhancing sCT Synthesis with IMPACT-Based Registration” (Boussot et al., 2025)**.  
These models are trained on **carefully aligned MR–CT pairs**, where alignment is optimized through the **IMPACT-Reg loss** to minimize spatial bias. Their training further integrates the **IMPACT-Synth loss**, a **perceptual loss derived from semantic representations of segmentation networks**. Together, **precise spatial alignment** and **semantic perceptual supervision** reinforce **anatomical fidelity** and **realistic tissue contrast** in the synthesized CT images.  

The official **IMPACT-Synth models** are available on [Hugging Face](https://huggingface.co/VBoussot/ImpactSynth) and can be executed directly through this CLI.

---

## 🚀 Installation

From PyPI:
```bash
pip install Impact-Synth-KonfAI
```

From source:
```bash
git clone https://github.com/vboussot/Impact-Synth-KonfAI.git
cd Impact-Synth-KonfAI
pip install .
```
---

## ⚙️ Usage

Perform image-to-sCT synthesis:

```bash
impact-synth-konfai -i input.nii.gz -o sCT.nii.gz
```

### Optional arguments

| Flag | Description | Default |
|------|--------------|----------|
| `-i`, `--input` | Path to the input file | *required* |
| `-o`, `--output` | Path to save the synthetic CT | `sCT.mha` |
| `-m`, `--model` | Model name on Hugging Face | `MR` |
| `--tta` | Number of test-time augmentations (TTA) | `2` |
| `--ensemble` | Number of models to ensemble | `5` |
| `--mc_dropout` | Monte Carlo dropout samples for uncertainty | `1` |
| `--uncertainty` | Save uncertainty maps | `False` |
| `-g`, `--gpu` | GPU list (e.g. `0` or `0,1`) | CPU if unset |
| `--cpu` | Number of CPU cores (if no GPU) | `1` |
| `-q`, `--quiet` | Suppress console output | `False` |

### Example

```bash
impact-synth-konfai -i patient01.nii.gz -o sCT_patient01.nii.gz --gpu 0 --tta 2 --ensemble 5 --uncertainty
```

---

## 🧠 Features

- ⚡ **Fast inference** powered by [KonfAI](https://github.com/vboussot/KonfAI)
- 🤗 **Automatic model download** from Hugging Face
- 🧩 **Multi-model ensembling** and **test-time augmentation (TTA)**
- 🧠 **Handles uncertainty estimation** 
- 🧾 **Multi-format compatibility:** supports all major medical image formats handled by ITK

---

## 📚 References

If you use **IMPACT-Synth-KonfAI** in your work, please cite:

- Boussot, V., Hémon, C., Nunes, J.-C., & Dillenseger, J.-L. (2025).  
  **Why Registration Quality Matters: Enhancing sCT Synthesis with IMPACT-Based Registration.**  
  *arXiv preprint* [arXiv:2510.21358](https://arxiv.org/abs/2510.21358)

- Boussot, V., & Dillenseger, J.-L. (2025).  
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**.  
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## 🔗 Links

- 🧠 **KonfAI Framework:** [github.com/vboussot/KonfAI](https://github.com/vboussot/KonfAI)  
- 🤗 **Model Hub:** [huggingface.co/VBoussot/IMPACT-Synth](https://huggingface.co/VBoussot/ImpactSynth)  
- 📦 **PyPI Package:** [pypi.org/project/impact_synth_konfai](https://pypi.org/project/impact_synth_konfai)  

---
