[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vboussot/KonfAI/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/konfai)](https://pypi.org/project/konfai/)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# 🧠 KonfAI
<img src="https://raw.githubusercontent.com/vboussot/KonfAI/main/logo.png" alt="KonfAI Logo" width="250" align="right"/>

**KonfAI** is a modular and highly configurable deep learning framework built on PyTorch, driven entirely by YAML configuration files.

It is designed to support complex medical imaging workflows, flexible model architectures, customizable training loops, and advanced loss scheduling, without hardcoding anything.

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

🧩 TODO & Perspectives

📘 Documentation

The official KonfAI documentation is in progress and will be released soon.

🤖 KonfAI-MCP Server

We are actively developing KonfAI-MCP, an extension of the framework enabling language-driven deep learning experimentation.
Through the Model Context Protocol (MCP), KonfAI will serve as the deterministic and transparent execution layer for agentic LLMs, allowing large language models to specify, launch, and refine deep learning experiments directly through natural language.

Imagine instructing an AI to:
“Train a model for lung tumor segmentation from this dataset and optimize the Dice score.”
KonfAI-MCP aims to turn such instructions into reproducible, verifiable experiments.

This represents the next stage of AI-assisted scientific research, where language becomes a medium of empirical discovery.
