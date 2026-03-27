Workflows API
=============

The low-level KonfAI workflows are exposed through three entrypoint functions
and three root classes:

- training
- prediction
- evaluation

These are the main Python APIs behind the ``konfai`` CLI.

Training
--------

.. currentmodule:: konfai.trainer

.. autofunction:: train
   :no-index:

.. autoclass:: Trainer
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: EarlyStopping
   :members:
   :show-inheritance:
   :no-index:

Prediction
----------

.. currentmodule:: konfai.predictor

.. autofunction:: predict
   :no-index:

.. autoclass:: Predictor
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: OutputDataset
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: OutputDatasetLoader
   :members:
   :show-inheritance:
   :no-index:

Evaluation
----------

.. currentmodule:: konfai.evaluator

.. autofunction:: evaluate
   :no-index:

.. autoclass:: Evaluator
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Statistics
   :members:
   :show-inheritance:
   :no-index:

See also
--------

- :doc:`configuration`
- :doc:`data`
- :doc:`models`
