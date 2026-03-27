KonfAI
======

KonfAI is a **YAML-driven deep learning framework for medical imaging** built on
top of PyTorch. It is organized around three low-level workflows:

- ``TRAIN`` for fitting models from configuration files
- ``PREDICTION`` for exporting model outputs to structured datasets
- ``EVALUATION`` for computing metrics on saved predictions

On top of these workflows, KonfAI also provides **KonfAI Apps**: packaged
inference and evaluation bundles that can run locally or through a remote
server.

The framework is especially useful when you want to:

- keep experiments reproducible and easy to inspect
- describe models, datasets, transforms, metrics, and schedulers in YAML
- iterate on medical imaging workflows without rewriting orchestration code
- package mature workflows into reusable apps

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
