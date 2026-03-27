# Using custom models and transforms

One of KonfAI's strongest workflows is the ability to keep project-specific
Python code next to the YAML files that use it.

## When to create a local module

Create a local module when you need:

- a custom model architecture
- a custom transform
- a project-specific output helper

This is the pattern used in `examples/Synthesis`.

## Local model example

`examples/Synthesis/Model.py` defines local classes such as:

- `UNetpp5`
- `Discriminator`
- `Gan`

The YAML then imports them explicitly:

```yaml
Model:
  classpath: Model:UNetpp5
```

or:

```yaml
Model:
  classpath: Model:Gan
```

## Local transform example

`examples/Synthesis/UnNormalize.py` is used from YAML like this:

```yaml
final_transforms:
  UnNormalize:UnNormalize: {}
```

## Practical rules

- Put the Python file next to the YAML when possible.
- Use explicit `module:ClassName` classpaths for local modules.
- Keep constructor argument names in snake_case so they map cleanly from YAML.
- Start from a working example and change one layer at a time.

## Debugging tips

- If KonfAI fails to import the classpath, verify the file name, class name, and working directory.
- If the YAML is accepted but a loss/output key fails later, verify the named module paths inside the model.
- If prediction and training should share the same predictor, keep the exported class name stable across checkpoints.

See also
--------

- :doc:`../concepts/configuration`
- :doc:`../concepts/model-graph`
- :doc:`../examples/synthesis`
