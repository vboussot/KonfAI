# Installation

KonfAI targets **Python 3.10+** and depends on PyTorch, SimpleITK, TensorBoard,
and a set of medical-imaging utilities.

## Install from PyPI

```bash
python -m pip install konfai
```

This installs the core CLI entrypoints:

- `konfai`
- `konfai-apps`

Optional extras exposed by the package metadata:

```bash
python -m pip install "konfai[server]"
python -m pip install "konfai[cluster]"
python -m pip install "konfai[vtk]"
python -m pip install "konfai[lpips]"
```

These extras enable optional entrypoints or integrations:

- `konfai-apps-server` with `server`
- `konfai-cluster` with `cluster`
- VTK-dependent features with `vtk`
- LPIPS-based metrics with `lpips`

## Install from source

Use an editable install when you want to:

- work on the framework itself
- modify example YAML files
- create local Python modules next to your configs
- build the documentation

```bash
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
python -m pip install -e .
```

## PyTorch and GPU notes

KonfAI declares `torch` as a dependency, but the correct GPU-enabled PyTorch
wheel still depends on your platform, drivers, and CUDA setup. In practice:

- if your default PyTorch install already matches your machine, `pip install konfai` is enough
- if you need a specific CUDA or CPU-only wheel, install PyTorch first, then install KonfAI
- for containerized usage, see {doc}`../usage/docker`

## Verify the installation

Check that the package imports correctly:

```bash
python -c "import konfai; print(konfai.__version__)"
```

Check that the main CLIs are available:

```bash
konfai --help
konfai-apps --help
```

If you installed the `server` or `cluster` extras:

```bash
konfai-apps-server --help
konfai-cluster --help
```

## Common installation issues

### `ModuleNotFoundError` after installation

This usually means the package was installed into a different Python
environment than the one you are currently using. Re-run the install with the
same interpreter you will use to launch KonfAI:

```bash
python -m pip install -e .
```

### GPU is available in Python but not in KonfAI

KonfAI relies on PyTorch device discovery and `CUDA_VISIBLE_DEVICES`. Check both:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
echo "$CUDA_VISIBLE_DEVICES"
```

### `konfai-apps-server` is missing

Install the `server` extra:

```bash
python -m pip install "konfai[server]"
```

### `konfai-cluster` is missing

Install the `cluster` extra:

```bash
python -m pip install "konfai[cluster]"
```

## See also

- {doc}`../quickstart`
- {doc}`../usage/docker`
- {doc}`../reference/cli`
