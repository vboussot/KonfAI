[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vboussot/KonfAI/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/konfai)](https://pypi.org/project/konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/KonfAI_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/KonfAI_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

# 🧠 KonfAI
<img src="https://raw.githubusercontent.com/vboussot/KonfAI/main/logo.png" alt="KonfAI Logo" width="250" align="right"/>

**KonfAI** is a flexible and extensible deep learning framework built on PyTorch, designed for **fully YAML-driven configuration**.  
It provides a clean separation between configuration and implementation, allowing users to orchestrate entire workflows, from data loading to evaluation, with no hardcoded logic.

KonfAI natively supports **multi-model training**, **patch-based learning**, **test-time augmentation**, and **loss scheduling**, making it ideal for medical imaging research and large-scale experimentation.

In practice, KonfAI is meant to help you:

- design and iterate on low-level deep learning workflows through YAML
- keep experiments reproducible and easy to inspect
- turn mature workflows into reusable **KonfAI Apps** for inference, evaluation, fine-tuning, or remote execution

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

- 🔀 Full training/prediction/evaluation orchestration via YAML configuration files
- 🧩 Modular plugin-like structure (transforms, augmentations, models, losses, schedulers)
- 🔄 Dynamic criterion scheduling per head / target
- 🧠 Multi-branch / multi-output model support
- 🧱 Patch-based learning and reconstruction
- 📦 Packaged workflows through KonfAI Apps
- 🖥️ Cluster-ready
- 📈 TensorBoard and custom logging support

---

## 🚀 Installation

### From PyPI

Install KonfAI from PyPI:

```bash
pip install konfai
```

This installs the core command-line tools:

```bash
konfai --help
konfai-apps --help
```

Optional extras:

```bash
pip install "konfai[server]"
pip install "konfai[cluster]"
```

Use these extra commands when the matching optional dependencies are installed:

```bash
konfai-apps-server --help
konfai-cluster --help
```

---

### From GitHub

Clone the repository and install:

```bash
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
pip install -e .
```

This is the best option if you want to edit YAML examples, develop custom models, or work on KonfAI itself.

---

## ✨ Start Here

If you are discovering KonfAI, the easiest path is:

1. install the package
2. start from one of the examples in [`examples/`](https://github.com/vboussot/KonfAI/tree/main/examples)
3. run the workflow with `konfai`
4. package the stabilized workflow as a **KonfAI App** if you want a simpler reusable interface

Recommended entry points:

- [`examples/Synthesis`](https://github.com/vboussot/KonfAI/tree/main/examples/Synthesis) for image synthesis workflows
- [`examples/Segmentation`](https://github.com/vboussot/KonfAI/tree/main/examples/Segmentation) for a clean segmentation baseline
- [`apps/`](https://github.com/vboussot/KonfAI/tree/main/apps) for packaged workflows ready to use

---

## 🐳 Docker

KonfAI can be packaged as a GPU-ready Docker image that exposes the existing
CLI entrypoints: `konfai`, `konfai-apps`, `konfai-apps-server`, and
`konfai-cluster`.

Build the image:

```bash
docker build -f docker/Dockerfile -t konfai .
```

This build installs KonfAI from PyPI, and preinstalls the CUDA-enabled PyTorch wheel. To target another KonfAI or PyTorch version, override the corresponding build arguments.

The official Docker image is published on Docker Hub as `vboussot/konfai`.

Pull the published image directly:

```bash
docker pull vboussot/konfai
```

If you built the image locally, replace `vboussot/konfai` with your local image tag, for example `konfai`.

Run the main CLI from your current working directory:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai TRAIN --gpu 0 -c examples/Synthesis/Config.yml
```

Run KonfAI Apps:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai konfai-apps infer my_app -i input.mha -o ./Output
```

Run the apps server image with the `server` extra installed by default:

```bash
docker run --rm -it -p 8000:8000 \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -e KONFAI_API_TOKEN=my-token \
  vboussot/konfai konfai-apps-server --host 0.0.0.0 --port 8000 --apps tests/assets/apps.json
```

Notes:

- The container defaults to `konfai --help` when no command is provided.
- If the first argument is not a known KonfAI executable, it is forwarded to
  `konfai`.
- To override the KonfAI version from PyPI, rebuild with
  `docker build -f docker/Dockerfile --build-arg KONFAI_PYPI_VERSION=1.5.3 -t konfai .`
- For custom optional dependencies, rebuild with
  `docker build -f docker/Dockerfile --build-arg KONFAI_EXTRAS=server,cluster -t konfai .`
- To override the PyTorch wheel source, rebuild with
  `docker build -f docker/Dockerfile --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t konfai .`
- To force a CPU-only image, rebuild with
  `docker build -f docker/Dockerfile --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t konfai .`

---

## 🧪 Usage

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models checkpoint.pt
konfai EVALUATION -y --config Evaluation.yml
```

For complete, ready-to-adapt workflows, see:

- [`examples/Synthesis/README.md`](https://github.com/vboussot/KonfAI/tree/main/examples/Synthesis)
- [`examples/Segmentation/README.md`](https://github.com/vboussot/KonfAI/tree/main/examples/Segmentation)

---

## 📦 KonfAI Apps

A **KonfAI App** is a self-contained workflow package built with KonfAI.  
It defines how a model is executed, how outputs are generated, and how optional evaluation or uncertainty workflows are performed.

Several ready-to-use KonfAI Apps are available directly in the repository under the [`apps/`](https://github.com/vboussot/KonfAI/tree/main/apps) directory.

They can be executed **identically** from:

| Interface | Command |
|----------|---------|
| 🖥️ CLI | `konfai-apps infer / eval / uncertainty / pipeline <app>` |
| 🧠 3D Slicer | Via **SlicerKonfAI** GUI https://github.com/vboussot/SlicerKonfAI |
| 🐍 Python API | Via `konfai.app.KonfAIApp` |
| 🌐 Remote Server (client/server mode) |

---

### 📂 Structure of a KonfAI App

```text
my_konfai_app/
├── app.json                # Metadata for UI + behaviors
├── Prediction.yml          # Inference workflow (required)
├── Evaluation.yml          # Evaluation workflow (optional)
├── Uncertainty.yml         # Uncertainty workflow (optional)
└── checkpoint.pt           # Trained model (single or ensemble)
```

Example `app.json`:

```json
{
    "display_name": "Lung Lobe Segmentation",
    "short_description": "Segmentation of lung lobes on CBCT scans.",
    "description": "This App synthesizes CT-like contrast from CBCT then segments lung lobes.",
    "tta": 4,
    "mc": 0
}
```

---

### 🚀 Using a KonfAI App (CLI)

Inference:
```bash
konfai-apps infer my_app -i input.mha -o ./Predictions --tta 4
```

Evaluation:
```bash
konfai-apps eval my_app -i input/ --gt labels/
```

Uncertainty:
```bash
konfai-apps uncertainty my_app -i input.mha
```

Pipeline (inference → evaluation → uncertainty):
```bash
konfai-apps pipeline my_app -i input.mha --gt gt.mha -uncertainty
```

Fine-tuning:
```bash
konfai-apps fine-tune my_app name -d ./Dataset --epochs 20
```

The very same commands can be executed on a remote KonfAI Apps server by adding
`--host`, `--port`, and `--token`. This allows heavy workloads to run on shared GPU
machines while keeping a lightweight local client.

More detailed documentation and usage examples for each app are available in the corresponding subdirectories of the `apps/` folder.

---

## 🧩 TODO & Perspectives

### 📘 Documentation

The official KonfAI documentation is in progress and will be released soon.

In the meantime, the most useful documentation is currently:

- the examples in [`examples/`](https://github.com/vboussot/KonfAI/tree/main/examples)
- the packaged workflows in [`apps/`](https://github.com/vboussot/KonfAI/tree/main/apps)
- the Docker guide in [`docker/README.md`](https://github.com/vboussot/KonfAI/tree/main/docker/README.md)

### 🤖 KonfAI-MCP Server

We are actively developing KonfAI-MCP, an extension of the framework enabling language-driven deep learning experimentation.
Through the Model Context Protocol (MCP), KonfAI will serve as the deterministic and transparent execution layer for agentic LLMs, allowing large language models to specify, launch, and refine deep learning experiments directly through natural language.

Imagine instructing an AI to:
“Train a model for lung tumor segmentation from this dataset and optimize the Dice score.”
KonfAI-MCP aims to turn such instructions into reproducible, verifiable experiments.

This represents the next stage of AI-assisted scientific research, where language becomes a medium of empirical discovery.
