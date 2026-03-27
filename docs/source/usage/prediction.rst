Prediction workflows
====================

This guide covers the low-level ``konfai PREDICTION`` workflow.

Prediction in KonfAI is driven by two inputs:

- a ``Prediction.yml`` file that defines the inference dataset and exported outputs
- one or more checkpoints passed with ``--models``

Minimal command
---------------

.. code-block:: bash

   konfai PREDICTION -y --gpu 0 --config Prediction.yml \
     --models Checkpoints/SEG_BASELINE/<checkpoint>.pt

You can also pass multiple checkpoints:

.. code-block:: bash

   konfai PREDICTION -y --gpu 0 --config Prediction.yml \
     --models ckpt_a.pt ckpt_b.pt ckpt_c.pt

When multiple checkpoints are provided, the predictor combines them using the
``combine`` strategy from the YAML, usually ``Mean`` or ``Median``.

What prediction writes
----------------------

Prediction writes to:

- ``Predictions/<train_name>/``

The exact substructure depends on ``outputs_dataset``. KonfAI also copies the
active ``Prediction.yml`` into the prediction directory for reproducibility.

The role of ``outputs_dataset``
-------------------------------

``outputs_dataset`` is the key prediction-specific section. It tells KonfAI:

- which model output should be exported
- what output group name to write
- what transforms should run before writing files
- how to combine predictions across TTA or ensembles

This is why prediction configs can be shared between different checkpoints as
long as the exported output name stays consistent.

Patch-based inference
---------------------

Use ``Dataset.Patch`` in ``Prediction.yml`` when:

- the full input does not fit in memory
- you want slice-wise or sliding-window inference
- you need the same spatial strategy as training

If reassembly is needed after model-level patching, configure it through
``outputs_dataset`` and model patch settings.

Troubleshooting prediction configs
----------------------------------

- If KonfAI says an output group does not exist, the key in
  ``outputs_dataset`` does not match a real model output path.
- If predictions are written into the wrong folder, check ``train_name``.
- If geometry or intensity range is wrong, review the final transforms in
  ``outputs_dataset``.

See also
--------

- :doc:`../config_guide/prediction`
- :doc:`../concepts/model-graph`
- :doc:`training`
- :doc:`evaluation`
