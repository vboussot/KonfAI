[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vboussot/KonfAI/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/konfai)](https://pypi.org/project/konfai/)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# 🧠 KonfAI
<img src="https://raw.githubusercontent.com/vboussot/KonfAI/main/logo.png" alt="KonfAI Logo" width="250" align="right"/>

**KonfAI** is a flexible and extensible deep learning framework built on PyTorch, designed for **fully YAML-driven configuration**.  
It provides a clean separation between configuration and implementation, allowing users to orchestrate entire workflows, from data loading to evaluation, with no hardcoded logic.

KonfAI natively supports **multi-model training**, **patch-based learning**, **test-time augmentation**, and **loss scheduling**, making it ideal for medical imaging research and large-scale experimentation.

**KonfAI** has been used in several top-performing challenge projects:  
[🔗 SynthRAD2025 – Task 1](https://github.com/vboussot/Synthrad2025_Task_1) •  
[🔗 SynthRAD2025 – Task 2](https://github.com/vboussot/Synthrad2025_Task_2) •  
[🔗 CURVAS PDACVI 2025](https://github.com/vboussot/CurvasPDACVI) •  
[🔗 TrackRAD 2025](https://github.com/vboussot/TrackRAD2025) •  
[🔗 Panther](https://github.com/vboussot/Panther) •  
[🔗 CURVAS](https://github.com/vboussot/CURVAS) •  

For more details on the design principles and scientific background, refer to the paper:  
> [**KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**](https://www.arxiv.org/abs/2508.09823)


---

## 🔧 Key Features

- 🔀 Full training/prediction/evaluation orchestration via YAML configuration file
- 🧩 Modular plugin-like structure (transforms, augmentations, models, losses, schedulers)
- 🔄 Dynamic criterion scheduling per head / target
- 🧠 Multi-branch / multi-output model support
- 🖥️ Cluster-ready
- 📈 TensorBoard and custom logging support

---

## 🚀 Installation

### From PyPI

Install KonfAI from PyPI:

```
pip install konfai
```

This will install the command-line tools:

```
konfai --help
konfai-cluster --help
```

---

### From GitHub

Clone the repository and install:

```
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
pip install -e .
```

---

## 🧪 Usage

```bash
konfai TRAIN --gpu 0
konfai PREDICTION --gpu 0
konfai EVALUATION
```

---

## 🧩 TODO & Perspectives

### 📘 Documentation

The official KonfAI documentation is in progress and will be released soon.

### 🤖 KonfAI-MCP Server

We are actively developing KonfAI-MCP, an extension of the framework enabling language-driven deep learning experimentation.
Through the Model Context Protocol (MCP), KonfAI will serve as the deterministic and transparent execution layer for agentic LLMs, allowing large language models to specify, launch, and refine deep learning experiments directly through natural language.

Imagine instructing an AI to:
“Train a model for lung tumor segmentation from this dataset and optimize the Dice score.”
KonfAI-MCP aims to turn such instructions into reproducible, verifiable experiments.

This represents the next stage of AI-assisted scientific research, where language becomes a medium of empirical discovery.
