# Configuration model

KonfAI is fundamentally a **configuration-driven object builder**.

The YAML file does not just pass values into a fixed script. It determines
which Python classes are instantiated and how they are connected.

## Root workflow objects

The root key of the YAML selects the high-level workflow object:

- `Trainer` for training
- `Predictor` for inference
- `Evaluator` for metrics

These names map directly to the public classes in:

- `konfai.trainer.Trainer`
- `konfai.predictor.Predictor`
- `konfai.evaluator.Evaluator`

## How YAML becomes Python objects

This behavior is implemented by `konfai.utils.config.Config`, `config()`, and
`apply_config()`.

In practice, the mapping is straightforward:

1. a class or function is annotated with `@config("...")`
2. `apply_config()` inspects the constructor signature
3. YAML fields are matched against constructor parameter names
4. nested objects are recursively instantiated from nested mappings

This is why KonfAI configuration keys should generally:

- use **snake_case**
- match the actual Python constructor arguments
- stay close to the shipped examples when you introduce a custom class

## `classpath`

Many configurable components are selected dynamically through a `classpath`
string. The exact resolution logic is implemented by
`konfai.utils.utils.get_module()`.

Typical examples:

```yaml
Model:
  classpath: segmentation.UNet.UNet
```

```yaml
Model:
  classpath: Model:UNetpp5
```

The two main styles are:

- `package.module.ClassName`-style references resolved relative to a KonfAI namespace
- `module:ClassName` references for explicit imports, often used for local files next to the YAML

Use the second form when you add custom files inside an example or project
directory. It is usually the least ambiguous option.

## `default|...` values

The `default|...` prefix is an important KonfAI convention. Its behavior is
inferred directly from `konfai.utils.config.Config._get_input_default()`.

It is used to express a fallback value that can still be overridden by config or
interactive generation. Examples from the codebase include:

- `train_name: str = "default|TRAIN_01"`
- `classpath: str = "default|segmentation.UNet.UNet"`
- default dictionary keys such as `default|Labels`

In practice, you can read it as:

- **use the value after the pipe if nothing else is provided**

## Configuration is recursive

Because nested constructors are instantiated recursively, the shape of the YAML
mirrors the shape of the Python object graph. For example, a training config can
nest:

- `Trainer`
- `Model`
- a chosen model class
- `optimizer`
- `schedulers`
- `outputs_criterions`

This is why KonfAI examples are such a good source of truth: they show real
constructor trees that the framework accepts.

## Practical mapping rules

When a config does not behave as expected, check these rules first:

- the YAML root must match the workflow you are launching
- nested section names must match constructor parameters or `@config(...)` keys
- local `classpath` modules must be importable from the current working directory
- the YAML shape should mirror the Python object graph, not just the names you
  want conceptually

## When to use local Python modules

Use a local module when you need:

- a custom model architecture
- a custom transform
- a project-specific helper that is not part of the built-in package

The `examples/Synthesis` workflow is the clearest repository example:

- `Model.py` defines local model classes
- `UnNormalize.py` defines a local transform
- the YAML references them with `Model:...` and `UnNormalize:...`

## See also

- {doc}`datasets`
- {doc}`model-graph`
- {doc}`../config_guide/index`
- {doc}`../usage/custom-models`
