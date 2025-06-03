
# 🧠 KonfAI
<img src="https://raw.githubusercontent.com/vboussot/KonfAI/main/logo.png" alt="KonfAI Logo" width="200" align="right"/>

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

### From PyPI (recommended)

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
