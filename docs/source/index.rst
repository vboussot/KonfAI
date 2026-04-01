KonfAI
======

KonfAI is a **YAML-driven deep learning framework for medical imaging** built on
top of PyTorch.

At the low level, KonfAI is organized around three workflows:

- ``TRAIN`` for fitting models from configuration files
- ``PREDICTION`` for exporting model outputs to structured datasets
- ``EVALUATION`` for computing metrics on saved predictions

On top of these workflows, KonfAI also provides **KonfAI Apps**: packaged
workflows that can run locally or through a remote server.

The framework is especially useful when you want to:

- keep experiments reproducible and easy to inspect
- describe models, datasets, transforms, metrics, and schedulers in YAML
- iterate on medical imaging workflows without rewriting orchestration code
- package mature workflows into reusable apps

If you are new to the project, the fastest path is:

1. read :doc:`quickstart`
2. start from one of the shipped examples
3. come back to :doc:`concepts/index` when you want to adapt the YAML

If you want one concrete recommendation: start with
``examples/Segmentation`` and the ``konfai TRAIN`` command before looking at
KonfAI Apps.

Quick links
-----------

- :doc:`getting-started/installation`
- :doc:`quickstart`
- :doc:`concepts/index`
- :doc:`reference/cli`
- :doc:`examples/index`

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting-started/installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Core concepts

   concepts/index
   config_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Guides

   usage/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Project

   examples/index
   troubleshooting
   contributing
   architecture
