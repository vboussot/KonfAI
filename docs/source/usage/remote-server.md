# Remote server and client mode

KonfAI Apps can run remotely through the FastAPI server exposed by
`konfai-apps-server`.

## Start the server

The server requires a JSON file listing the available apps:

.. code-block:: bash

   konfai-apps-server --host 0.0.0.0 --port 8000 --apps konfai-apps/tests/assets/apps.json

Bearer-token authentication is enabled by default:

.. code-block:: bash

   export KONFAI_API_TOKEN="my-secret-token"
   konfai-apps-server --apps konfai-apps/tests/assets/apps.json

Important server options from `konfai_apps.cli.main_apps_server()`:

- `--host`
- `--port`
- `--auth off|bearer`
- `--token-env`
- `--token`
- `--apps`
- `--download`
- `--check`

## Run a remote job

Any `konfai-apps` command becomes remote as soon as `--host` is provided:

.. code-block:: bash

   konfai-apps infer VBoussot/ImpactSynth:CBCT \
     -i input.mha -o ./Output \
     --host my.server.org --port 8000 --token "$KONFAI_API_TOKEN"

The client then:

1. uploads inputs
2. submits a job
3. streams logs over SSE
4. downloads the result archive

## Scheduling model

From `konfai_apps.app_server`, jobs are:

- queued
- optionally assigned GPUs
- executed in isolated temporary workspaces
- cleaned up after a grace period

The server also exposes:

- health and device endpoints
- app metadata endpoints
- job status, log, result, and kill endpoints

See also
--------

- :doc:`apps`
- :doc:`../reference/app-server-api`
