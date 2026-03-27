# Using KonfAI Apps

KonfAI Apps are packaged workflows that expose a stable user interface on top of
KonfAI's low-level prediction, evaluation, uncertainty, and fine-tuning logic.

## The `konfai-apps` CLI

The app CLI currently exposes these subcommands:

- `infer`
- `eval`
- `uncertainty`
- `pipeline`
- `fine-tune`

The main command pattern is:

.. code-block:: bash

   konfai-apps <command> <app> [options]

## App identifiers

Apps can be local or remote repository identifiers. The repository examples and
tests show Hugging Face style identifiers such as:

- ``VBoussot/ImpactSynth:MR``
- ``VBoussot/ImpactSynth:CBCT``
- ``VBoussot/TotalSegmentator-KonfAI:total``

## Common app workflows

Inference:

.. code-block:: bash

   konfai-apps infer VBoussot/ImpactSynth:CBCT \
     -i input.mha -o ./Output --gpu 0

Evaluation:

.. code-block:: bash

   konfai-apps eval VBoussot/ImpactSynth:CBCT \
     -i prediction.mha --gt ct.mha --mask mask.mha --gpu 0

Pipeline:

.. code-block:: bash

   konfai-apps pipeline VBoussot/ImpactSynth:CBCT \
     -i input.mha --gt ct.mha --mask mask.mha -o ./Output -uncertainty

## Grouped inputs

The CLI accepts grouped inputs by repeating `--inputs` / `-i`. This matches the
grouping behavior documented in `konfai.main.add_common_konfai_apps()`.

Use this when an app expects multiple input groups or multiple files per group.

## Fine-tuning

Fine-tuning is available through:

.. code-block:: bash

   konfai-apps fine-tune <app> <name> -d ./Dataset --epochs 10 --gpu 0

Under the hood, the app installs training assets, links the dataset, then calls
the low-level training flow in resume mode.

## Local vs remote

If you add `--host`, the same command switches from local execution to
client/server mode automatically.

See also
--------

- :doc:`remote-server`
- :doc:`../concepts/apps`
- :doc:`../reference/cli`
