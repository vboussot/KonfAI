# Docker usage

KonfAI ships Docker assets in the repository under `docker/`.

The Docker image is designed for **CLI-oriented workflows** executed from a
mounted workspace.

## Official image

The repository documentation states that the official image is published as:

```bash
docker pull vboussot/konfai
```

## Build locally

```bash
docker build -f docker/Dockerfile -t konfai .
```

Useful build arguments documented in `docker/README.md`:

- `KONFAI_PYPI_VERSION`
- `KONFAI_EXTRAS`
- `TORCH_INDEX_URL`

## Run the low-level CLI

Training:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai TRAIN --gpu 0 -c examples/Synthesis/Config.yml
```

Prediction:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai PREDICTION --models checkpoint.pt --gpu 0 -c examples/Synthesis/Prediction.yml
```

Evaluation:

```bash
docker run --rm -it \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai EVALUATION -c examples/Synthesis/Evaluation.yml
```

## Run the app layer

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai konfai-apps infer my_app -i input.mha -o ./Output
```

## Notes

- The container defaults to `konfai --help`.
- GPU visibility still depends on the host runtime and NVIDIA container support.
- For host troubleshooting, see `docker/README.md`.

See also
--------

- :doc:`../getting-started/installation`
- :doc:`../reference/cli`
