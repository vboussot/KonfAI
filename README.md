[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vboussot/KonfAI/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/konfai)](https://pypi.org/project/konfai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/vboussot/KonfAI/actions/workflows/KonfAI_ci.yml/badge.svg)](https://github.com/vboussot/KonfAI/actions/workflows/KonfAI_ci.yml)
[![Documentation Status](https://readthedocs.org/projects/konfai/badge/?version=latest)](https://konfai.readthedocs.io/en/latest/?badge=latest)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)


# 🧠 KonfAI
<img src="https://raw.githubusercontent.com/vboussot/KonfAI/main/logo.png" alt="KonfAI Logo" width="250" align="right"/>

**KonfAI** is a modular deep learning framework for medical imaging built
around **YAML-driven workflows**.

It lets you define complete pipelines, from data loading to prediction and
evaluation, through configuration instead of orchestration scripts.

- reproducible workflows with explicit configs and outputs
- modular components for datasets, transforms, models, losses, and metrics
- designed for research, experimentation, and agent-driven workflows

KonfAI has been used in several top-performing challenge projects:
[🔗 SynthRAD2025 – Task 1](https://github.com/vboussot/Synthrad2025_Task_1) •
[🔗 SynthRAD2025 – Task 2](https://github.com/vboussot/Synthrad2025_Task_2) •
[🔗 CURVAS PDACVI 2025](https://github.com/vboussot/CurvasPDACVI) •
[🔗 TrackRAD 2025](https://github.com/vboussot/TrackRAD2025) •
[🔗 Panther](https://github.com/vboussot/Panther) •
[🔗 CURVAS](https://github.com/vboussot/CURVAS)

Paper:
> [**KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**](https://www.arxiv.org/abs/2508.09823)

---

## 🚀 Why KonfAI?

Most frameworks focus on models.
**KonfAI focuses on pipelines.**

- 🧩 Compose full workflows from modular components
- 🔁 Iterate without rewriting Python scripts
- 📦 Turn experiments into reusable **KonfAI Apps**
- 🤖 Use KonfAI as a backend for LLM-driven experimentation through **KonfAI-MCP**

---

## ⚡ Quickstart

Install and run your first workflow:

```bash
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
python -m pip install -e .
cd examples/Segmentation

python -m pip install -U "huggingface_hub[cli]"
hf download VBoussot/konfai-demo \
  --repo-type dataset \
  --include "Segmentation/**" \
  --local-dir Dataset
mv Dataset/Segmentation/* Dataset/
rmdir Dataset/Segmentation
rm -rf Dataset/.cache
```

This downloads a small public demo subset and prepares the layout expected by
the example:

```text
Dataset/
├── 1PC006/
│   ├── CT.mha
│   └── SEG.mha
└── ...
```

- `CT` is the input image
- `SEG` is the segmentation label map

Then launch the first training run:

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

Then:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/SEG_BASELINE/<checkpoint>.pt
konfai EVALUATION -y --config Evaluation.yml
```

If you do not have a GPU available, replace ``--gpu 0`` with ``--cpu 1``.

Edit these files first:

- `Config.yml` → training
- `Prediction.yml` → exported outputs
- `Evaluation.yml` → metrics on saved predictions

Notebook entry points:

- `examples/Segmentation/Segmentation_demo.ipynb`
- `examples/Synthesis/Synthesis_demo.ipynb`

For editable installs and optional extras such as `server` or `cluster`, see:

- [`docs/source/getting-started/installation.md`](docs/source/getting-started/installation.md)

---

## 🤖 Agent-Ready by Design

KonfAI is designed to serve as a **deterministic backend for LLM-driven experimentation**.

Through **KonfAI-MCP Server**, agents can:

- inspect datasets
- generate or refine configurations
- launch experiments
- analyze results and iterate

All executions remain:
- reproducible
- structured
- grounded in YAML workflows

👉 KonfAI bridges the gap between **LLM reasoning** and **real experimental execution**.

---

## 📦 KonfAI Apps

A **KonfAI App** is a self-contained workflow package built with KonfAI.

It can expose:

- inference
- evaluation
- uncertainty estimation
- full pipelines
- fine-tuning

Apps live in [`apps/`](https://github.com/vboussot/KonfAI/tree/main/apps) and
can be used through:

| Interface | Entry point |
| --- | --- |
| 🖥️ CLI | `konfai-apps` |
| 🐍 Python API | `konfai.app.KonfAIApp` |
| 🌐 Remote server | `konfai-apps-server` + `konfai-apps --host ...` |
| 🧠 3D Slicer | [SlicerKonfAI](https://github.com/vboussot/SlicerKonfAI) |

Use Apps when a workflow is already stable and you want a cleaner user-facing
interface than the low-level YAML CLI.

---

## 📚 Documentation

The README is only the entry point. The full documentation is available here:

- https://konfai.readthedocs.io/en/latest/

---

## 🐳 Docker

KonfAI ships a Docker setup for CLI-oriented workflows.

- Dockerfile: [`docker/Dockerfile`](docker/Dockerfile)
- guide: [`docs/source/usage/docker.md`](docs/source/usage/docker.md)
- image: `vboussot/konfai`

Example:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai TRAIN --gpu 0 -c examples/Synthesis/Config.yml
```

---

## 🤝 Contributing

Contributions are welcome.

Typical ways to help:

- improve examples and notebooks
- clarify documentation
- add tests for real user paths
- extend models, transforms, or apps

Local setup:

```bash
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
python -m pip install -e .
python -m pip install pytest pre-commit
```

Useful commands:

```bash
pytest -q
pre-commit run --all-files
make -C docs html
```

Contributor guide:

- [`docs/source/contributing.md`](docs/source/contributing.md)
