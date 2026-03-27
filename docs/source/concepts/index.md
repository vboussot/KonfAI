# Core concepts

KonfAI is easiest to understand when you keep four ideas in mind:

1. **YAML builds Python objects** rather than acting as a loose parameter blob.
2. **Datasets are organized by groups** such as `CT`, `MR`, `SEG`, or `MASK`.
3. **Model outputs are addressable by module path**, which is how losses,
   metrics, and exported predictions are attached.
4. **The same low-level workflow can later be packaged as a KonfAI App**.

```{toctree}
:maxdepth: 1

configuration
datasets
model-graph
execution-flow
apps
```
