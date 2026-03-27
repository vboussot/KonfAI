Training workflows
==================

This guide covers the low-level ``konfai TRAIN`` and ``konfai RESUME``
workflows.

Use this mode when you want full control over:

- the dataset structure
- preprocessing and augmentation
- the model graph
- losses and metrics
- checkpointing and validation

Start from a shipped example
----------------------------

The repository currently provides two strong starting points:

- ``examples/Segmentation/Config.yml`` for a simple multiclass segmentation baseline
- ``examples/Synthesis/Config.yml`` for a richer image-synthesis workflow

For most new users, the segmentation example is the easiest first template.

Minimal command
---------------

From the example directory:

.. code-block:: bash

   konfai TRAIN -y --gpu 0 --config Config.yml

CPU-only:

.. code-block:: bash

   konfai TRAIN -y --cpu 1 --config Config.yml

What training writes
--------------------

Training writes into two top-level directories:

- ``Checkpoints/<train_name>/`` for model checkpoints
- ``Statistics/<train_name>/`` for TensorBoard logs, copied configs, and train/validation case lists

The output folder name comes from ``Trainer.train_name`` in the YAML.

TensorBoard
-----------

Enable TensorBoard from the CLI:

.. code-block:: bash

   konfai TRAIN -y --gpu 0 --config Config.yml -tb

KonfAI allocates a free local port automatically when TensorBoard is enabled.

Resume training
---------------

Resume from an existing checkpoint with ``RESUME``:

.. code-block:: bash

   konfai RESUME -y --config Config.yml \
     --model Checkpoints/SEG_BASELINE/<checkpoint>.pt

You can also change the output directories:

.. code-block:: bash

   konfai TRAIN -y --config Config.yml \
     --checkpoints-dir ./Checkpoints \
     --statistics-dir ./Statistics

Training checklist
------------------

Before launching a new run, verify:

- ``dataset_filenames`` points to the right data
- every group named in ``groups_src`` exists on disk
- ``train_name`` is unique unless you intend to overwrite
- output names used in ``outputs_criterions`` match real model modules
- ``validation`` is appropriate for your dataset size

Advanced training patterns
--------------------------

KonfAI supports several advanced training patterns visible in the codebase and
examples:

- dataset-level patch extraction through ``Dataset.Patch``
- model-level patching through ``Model.<Class>.Patch``
- multiple criteria per output and per target
- EMA through ``ema_decay``
- selective logging with ``data_log``
- multi-process execution through the distributed runner

For a concrete advanced example, see the GAN variant in
``examples/Synthesis/Config_GAN.yml``.

See also
--------

- :doc:`../config_guide/training`
- :doc:`../concepts/model-graph`
- :doc:`prediction`
- :doc:`evaluation`
