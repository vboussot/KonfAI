# Model graph and output naming

KonfAI models are not treated as opaque single-output blocks. A model is a
**named module graph**, and KonfAI lets you attach losses, metrics, and exported
datasets to specific named outputs.

## Networks

The core abstractions live in `konfai.network.network`:

- `Network`
- `ModelLoader`
- `OptimizerLoader`
- `TargetCriterionsLoader`
- `Measure`

The selected model class is configured under `Model.classpath`, then further
configured under a section named after that class.

Example:

```yaml
Model:
  classpath: segmentation.UNet.UNet
  UNet:
    dim: 2
    nb_class: 41
```

## Addressing outputs

Losses and metrics are attached through `outputs_criterions`. Keys in this
mapping correspond to named modules or outputs in the model graph.

Example from the segmentation baseline:

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:
    targets_criterions:
      SEG:
        criterions_loader:
          torch:nn:CrossEntropyLoss:
            is_loss: true
```

If an output key does not match any module path, KonfAI raises a configuration
error at runtime.

## Targets and metrics

For each output group you can define one or more target groups, then one or more
criteria for each target:

```yaml
outputs_criterions:
  Head:Tanh:
    targets_criterions:
      CT:
        criterions_loader:
          MAE:
            is_loss: true
```

This structure lets you express:

- multiple heads
- multiple targets per head
- multiple losses or metrics per target
- independent scheduler weights per criterion

## Dataset patching vs model patching

KonfAI supports patching at two different levels:

- **dataset patching** with `Dataset.Patch`
- **model patching** with `Model.<Class>.Patch`

Dataset patching controls what reaches the model. Model patching controls how a
network internally re-processes those tensors.

The `examples/Synthesis` GAN variant is the clearest example:

- `Dataset.Patch` provides a 3D chunk to the whole GAN
- `Model.Gan.UNetpp5.Patch` reprocesses the chunk slice-wise inside the generator

## `;accu;` outputs

The `;accu;` marker appears in some advanced workflows, especially when model
patching is enabled. Its semantics are **inferred from the shipped examples and
the network patch/accumulation logic**.

In practice it refers to patch-wise outputs **before final re-assembly**.

This matters in the synthesis GAN example:

- `Generator_A_to_B:;accu;Head:Tanh` is used for patch-wise reconstruction loss
- `Discriminator_pB:Head:Conv` is used after the generator output has been re-assembled

## Prediction outputs

Inference uses a separate `outputs_dataset` mapping to decide what should be
written to disk.

Example:

```yaml
outputs_dataset:
  Head:Tanh:
    OutputDataset:
      name_class: OutSameAsGroupDataset
      group: sCT
      reduction: Mean
```

This lets you control:

- which model output is exported
- how multiple predictions are reduced
- what final transforms are applied before writing files

## See also

- {doc}`configuration`
- {doc}`datasets`
- {doc}`../config_guide/prediction`
- {doc}`../usage/custom-models`
