Evaluation workflows
====================

This guide covers the low-level ``konfai EVALUATION`` workflow.

Evaluation in KonfAI compares saved dataset groups, not in-memory model outputs.
That is why a typical workflow is:

1. train
2. predict
3. evaluate the written predictions

Minimal command
---------------

.. code-block:: bash

   konfai EVALUATION -y --config Evaluation.yml

What evaluation writes
----------------------

Evaluation writes to:

- ``Evaluations/<train_name>/Metric_TRAIN.json``
- optionally ``Evaluations/<train_name>/Metric_VALIDATION.json``

The output directory is controlled by:

- ``Evaluator.train_name`` in the YAML
- ``--evaluations-dir`` on the CLI

What the JSON contains
----------------------

The evaluator writes JSON with two sections:

- ``case`` for per-case values
- ``aggregates`` for summary statistics such as mean, std, percentiles, min, max, and count

This structure is implemented by `konfai.evaluator.Statistics`.

Pairing targets and predictions
-------------------------------

Evaluation relies on ``dataset_filenames`` and ``groups_src`` to align:

- the predicted output group
- the reference target group
- any optional mask or auxiliary group

For example, the synthesis evaluation example combines:

- ``./Dataset:a:mha``
- ``./Predictions/TRAIN_01/Dataset:i:mha``

The ``i`` flag keeps only cases present in both sources.

Validation reports
------------------

``Evaluator.Dataset.validation`` can optionally point to a case list. When it is
set, KonfAI writes a separate validation metrics JSON in addition to
``Metric_TRAIN.json``.

Common evaluation mistakes
--------------------------

- prediction and target datasets do not share the same case names
- output group names in ``metrics`` do not exist in the loaded dataset
- the evaluation file still points to an old prediction folder
- label definitions in the metric do not match the dataset encoding

See also
--------

- :doc:`../config_guide/evaluation`
- :doc:`prediction`
- :doc:`../reference/app-server-api`
