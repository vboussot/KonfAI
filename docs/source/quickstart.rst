Quickstart
==========

This quickstart shows the **smallest useful end-to-end KonfAI workflow**:
install the package, download the public demo dataset, train a baseline model,
run prediction, then evaluate the saved outputs.

The example used here is the shipped segmentation baseline in
``examples/Segmentation`` because it is the most conservative starting point in
the repository.

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

Download the demo dataset
-------------------------

Run these commands from the repository root:

.. code-block:: bash

   cd examples/Segmentation

.. code-block:: python

   from pathlib import Path
   import shutil
   from huggingface_hub import snapshot_download

   dataset_dir = Path("Dataset")
   snapshot_download(
       repo_id="VBoussot/konfai-demo",
       repo_type="dataset",
       local_dir=str(dataset_dir),
       allow_patterns=["Segmentation/**"],
   )

   nested = dataset_dir / "Segmentation"
   if nested.exists():
       for item in nested.iterdir():
           target = dataset_dir / item.name
           if target.exists():
               if target.is_dir():
                   shutil.rmtree(target)
               else:
                   target.unlink()
           shutil.move(str(item), str(target))
       shutil.rmtree(nested)

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

.. code-block:: bash

   konfai TRAIN -y --gpu 0 --config Config.yml

CPU-only alternative:

.. code-block:: bash

   konfai TRAIN -y --cpu 1 --config Config.yml

Training creates, at minimum:

- ``Checkpoints/SEG_BASELINE/``
- ``Statistics/SEG_BASELINE/``

Run prediction
--------------

Use one checkpoint from ``Checkpoints/SEG_BASELINE``:

.. code-block:: bash

   konfai PREDICTION -y --gpu 0 --config Prediction.yml \
     --models Checkpoints/SEG_BASELINE/<checkpoint>.pt

Prediction writes:

- ``Predictions/SEG_BASELINE/``

Run evaluation
--------------

.. code-block:: bash

   konfai EVALUATION -y --config Evaluation.yml

Evaluation writes:

- ``Evaluations/SEG_BASELINE/Metric_TRAIN.json``

What to inspect
---------------

- The copied YAML files inside ``Statistics/`` and ``Evaluations/`` for reproducibility
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
