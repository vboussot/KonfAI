# 🧠 KonfAI

**KonfAI** is a modular and highly configurable deep learning framework built on PyTorch, driven entirely by YAML configuration files.

It is designed to support complex medical imaging workflows, flexible model architectures, customizable training loops, and advanced loss scheduling — without hardcoding anything.

---

## 🔧 Key Features

- 🔀 Full training/prediction/evaluation orchestration via YAML configuration file
- 🧩 Modular plugin-like structure (datasets, models, losses, schedulers)
- 🔄 Dynamic criterion scheduling per head / target
- 🧠 Multi-branch / multi-output model support
- 🖥️ Cluster-ready
- 📈 TensorBoard and custom logging support

---

## 🚀 Installation

```bash
pip install -e .[full]
```

---

## 🧪 Usage

```bash
konfai train --config path/to/config.yaml
```
