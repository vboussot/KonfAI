# Examples

KonfAI ships two low-level, YAML-driven examples under `examples/`. They are
the best starting point when you want to understand the framework before
building a reusable KonfAI App.

Both examples are backed by public demo data on Hugging Face:

- `VBoussot/konfai-demo/Synthesis`
- `VBoussot/konfai-demo/Segmentation`

Each example also includes a notebook intended for a fresh environment,
including Google Colab.

```{toctree}
:maxdepth: 1

synthesis
segmentation
```

## Choosing an example

Start with **Segmentation** when you want the smallest conservative baseline:

- one input group (`CT`)
- one label-map target (`SEG`)
- built-in `UNet`
- training with `CrossEntropyLoss`
- final evaluation with Dice

Start with **Synthesis** when you want to understand more of KonfAI's
configuration model:

- custom local Python modules loaded through `classpath`
- paired image-to-image training
- masked evaluation
- shared prediction and evaluation configs
- a GAN variant with nested patching scopes

## Working from the repository

All example commands in this documentation assume you are running from the
example directory itself, for example:

```bash
cd examples/Segmentation
```

or:

```bash
cd examples/Synthesis
```

That matters because the shipped YAML files refer to local modules and dataset
paths relative to the current working directory.

## See also

- {doc}`../quickstart`
- {doc}`../concepts/configuration`
- {doc}`../usage/custom-models`
