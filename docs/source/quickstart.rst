Quickstart
==========

This quickstart shows the **smallest useful end-to-end KonfAI workflow**:
install the package, prepare a demo dataset, train a baseline model, run
prediction, then evaluate the saved outputs.

The example used here is the shipped segmentation baseline in
``examples/Segmentation`` because it is the simplest starting point in the
repository.

Prerequisites
-------------

- Python 3.10 or newer
- a working KonfAI installation
- a terminal in the repository root

Install KonfAI
--------------

From PyPI:

.. code-block:: bash

   python -m pip install konfai

From source:

.. code-block:: bash

   git clone https://github.com/vboussot/KonfAI.git
   cd KonfAI
   python -m pip install -e .

Verify the install:

.. code-block:: bash

   konfai --help

What to keep in mind before you start:

- run the commands from the directory that contains the YAML files
- `Config.yml` is the training workflow
- `Prediction.yml` writes model outputs to disk
- `Evaluation.yml` compares those saved outputs against references

Download the demo dataset
-------------------------

Run these commands from the repository root. The example expects you to work
from the example directory itself so that local YAML references and Python
modules resolve correctly.

.. code-block:: bash

   cd examples/Segmentation

.. code-block:: bash

   python -m pip install -U "huggingface_hub[cli]"
   hf download VBoussot/konfai-demo \
     --repo-type dataset \
     --include "Segmentation/**" \
     --local-dir Dataset
   mv Dataset/Segmentation/* Dataset/
   rmdir Dataset/Segmentation
   rm -rf Dataset/.cache

After the download, the example expects this layout:

.. code-block:: text

   examples/Segmentation/
   ├── Dataset/
   │   ├── 1PC006/
   │   │   ├── CT.mha
   │   │   └── SEG.mha
   │   └── ...
   ├── Config.yml
   ├── Prediction.yml
   └── Evaluation.yml

Train a baseline
----------------

At this stage, KonfAI reads ``Config.yml`` and builds a ``Trainer`` object from
it.

.. code-block:: bash

   konfai TRAIN -y --gpu 0 --config Config.yml

If you do not have a GPU available, use ``--cpu 1`` instead of ``--gpu 0``.

Training creates, at minimum:

- ``Checkpoints/SEG_BASELINE/``
- ``Statistics/SEG_BASELINE/``

Run prediction
--------------

Use one checkpoint from ``Checkpoints/SEG_BASELINE``. ``Prediction.yml`` defines
which outputs are written and under which group names.

.. code-block:: bash

   konfai PREDICTION -y --gpu 0 --config Prediction.yml \
     --models Checkpoints/SEG_BASELINE/<checkpoint>.pt

Prediction writes:

- ``Predictions/SEG_BASELINE/``

Run evaluation
--------------

``Evaluation.yml`` does not run the model again. It compares saved prediction
groups against reference groups on disk.

.. code-block:: bash

   konfai EVALUATION -y --config Evaluation.yml

Evaluation writes:

- ``Evaluations/SEG_BASELINE/Metric_TRAIN.json``

What to inspect
---------------

- The copied YAML files inside ``Statistics/``, ``Predictions/``, and
  ``Evaluations/`` for reproducibility
- The prediction dataset written under ``Predictions/SEG_BASELINE/Dataset/``
- The aggregated metrics in ``Metric_TRAIN.json``

Common first issues
-------------------

- **``--gpu`` rejects your device id**

  ``konfai`` validates GPU ids against ``CUDA_VISIBLE_DEVICES``. Use ``--cpu``
  if no GPU is available, or check the visible devices with a small PyTorch
  snippet.

- **The command asks whether it should overwrite an existing run**

  Add ``-y`` to skip the interactive confirmation.

- **Dataset groups do not match the YAML**

  KonfAI expects the group names used in ``groups_src`` to exist on disk. In
  this example that means ``CT.mha`` and ``SEG.mha`` for every case directory.

- **The workflow runs, but evaluation cannot find predictions**

  Check that ``Prediction.yml`` and ``Evaluation.yml`` use the same
  ``train_name`` and that evaluation points to the correct prediction dataset.

- **A metric or output group name is rejected**

  Output names in ``outputs_criterions`` and ``outputs_dataset`` must match real
  model module paths. Start from the shipped examples before introducing custom
  names.

Next steps
----------

- :doc:`getting-started/installation`
- :doc:`concepts/configuration`
- :doc:`config_guide/index`
- :doc:`usage/training`
- :doc:`examples/index`
- ``examples/Segmentation/Segmentation_demo.ipynb`` if you prefer a notebook walkthrough
