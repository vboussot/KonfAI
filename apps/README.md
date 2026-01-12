# 📦 KonfAI Apps

A **KonfAI App** is a self-contained workflow package built on top of
KonfAI.\
It defines *how a model is executed*, *how data is prepared*, and *how
optional evaluation or uncertainty workflows are performed*.

KonfAI Apps provide a **uniform execution interface** across:

-   🖥️ Command Line Interface (CLI)
-   🧠 3D Slicer via **SlicerKonfAI**
-   🐍 Python API
-   🌐 Remote Server (client/server mode)

Several ready-to-use apps are available in this directory.

------------------------------------------------------------------------

## ✨ Key Principles

-   **Same interface everywhere**\
    The very same command works locally or remotely.

-   **Self-contained**\
    An app bundles its configuration, checkpoints, and metadata.

-   **Reproducible**\
    Workflows are fully described via YAML and versioned artifacts.

-   **Composable**\
    Apps can be chained (inference → evaluation → uncertainty).

------------------------------------------------------------------------

## 📂 Structure of a KonfAI App

    my_konfai_app/
    ├── app.json                # Metadata for UI and defaults
    ├── Prediction.yml          # Inference workflow (required)
    ├── Evaluation.yml          # Evaluation workflow (optional)
    ├── Uncertainty.yml         # Uncertainty workflow (optional)
    └── checkpoint.pt           # Trained model (single or ensemble)

Example `app.json`:

``` json
{
  "display_name": "Lung Lobe Segmentation",
  "short_description": "Segmentation of lung lobes on CBCT scans.",
  "description": "This App synthesizes CT-like contrast from CBCT then segments lung lobes.",
  "tta": 4,
  "mc": 0
}
```

This file is used by UIs (e.g. SlicerKonfAI) and can define default
behaviors.

------------------------------------------------------------------------

## 🚀 Using a KonfAI App (CLI)

### Inference

``` bash
konfai-apps infer my_app -i input.mha -o ./Predictions --tta 4
```

### Evaluation

``` bash
konfai-apps eval my_app -i input/ --gt labels/
```

### Uncertainty

``` bash
konfai-apps uncertainty my_app -i input.mha
```

### Pipeline (inference → evaluation → uncertainty)

``` bash
konfai-apps pipeline my_app -i input.mha --gt gt.mha -uncertainty
```

### Fine-tuning

``` bash
konfai-apps fine-tune my_app name -d ./Dataset --epochs 20
```

All commands support:

-   grouped inputs (`-i` multiple times)
-   GPU or CPU execution (`--gpu` / `--cpu`)
-   quiet mode (`--quiet`)
-   custom config files (`Prediction.yml`, `Evaluation.yml`, ...)

------------------------------------------------------------------------

## 🌐 Remote Execution (Client / Server Mode)

KonfAI Apps can be executed on a **remote server** without modifying your
local workflow or your Apps.  
The same commands (`infer`, `eval`, `pipeline`, `fine-tune`, …) work
locally and remotely.

```bash
konfai-apps infer my_app \
    -i input.mha \
    -o ./Predictions \
    --tta 4 \
    --host my.server.com \
    --port 8000 \
    --token YOUR_TOKEN
```

When `--host` is provided, KonfAI switches transparently to *client/server
mode*:

1. Inputs and configuration are uploaded to the server.
2. The job is queued and scheduled (CPU/GPU aware).
3. Logs are streamed live to the client (SSE).
4. Results are packaged and downloaded automatically.

From the user’s perspective, the command behaves exactly like a local run.

This enables:

- Running heavy workloads on dedicated GPU servers  
- Sharing Apps across a lab, hospital, or team  
- Centralizing compute while keeping lightweight clients  
- Integrating KonfAI into GUIs (e.g. 3D Slicer) or web services  
- Reproducible, auditable execution on controlled infrastructure  

---

### 🖥️ Starting a Server

```bash
konfai-apps-server --host 0.0.0.0 --port 8000
```

With authentication (recommended):

```bash
export KONFAI_API_TOKEN="my_secret_token"
konfai-apps-server
```

When authentication is enabled:

- All endpoints require a Bearer token.
- Unauthorized clients receive HTTP 401/403 errors.
- Tokens can be rotated without changing Apps or clients.

This mode is designed for:

- Shared GPU servers
- Clinical or research infrastructures
- Multi-user environments
- Long-running experiments

---

## 🔌 HTTP API (Remote Server)

When running `konfai-apps-server`, a REST API is exposed.  
All endpoints are protected by a Bearer token when authentication is enabled.

Base URL:

```
http://<host>:<port>
```

### Health & System

| Method | Endpoint              | Description                          |
|--------|-----------------------|--------------------------------------|
| GET    | `/health`             | Server health check                  |
| GET    | `/available_devices`  | List available GPU devices           |
| GET    | `/ram`                | System RAM usage                     |
| GET    | `/vram?devices=0&...` | GPU memory usage for given devices   |

These endpoints are typically used by GUIs or clients to inspect the
available hardware and server status before submitting jobs.

---

### Job Submission

Each job submission endpoint creates a new **asynchronous job** and
returns:

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status_url": "/jobs/<id>",
  "logs_url": "/jobs/<id>/logs",
  "result_url": "/jobs/<id>/result"
}
```

| Method | Endpoint                                | Description                    |
|--------|------------------------------------------|--------------------------------|
| POST   | `/apps/{app_name}/infer`                 | Inference                      |
| POST   | `/apps/{app_name}/evaluate`              | Evaluation                     |
| POST   | `/apps/{app_name}/uncertainty`           | Uncertainty estimation         |
| POST   | `/apps/{app_name}/pipeline`              | Full pipeline                  |
| POST   | `/apps/{app_name}/fine_tune`             | Fine-tuning / training         |

All endpoints accept `multipart/form-data` with:

- uploaded files (`inputs`, `gt`, `mask`, `dataset`, …),
- form fields (e.g. `ensemble`, `tta`, `mc`, `gpu`, `cpu`, `quiet`, …).

This makes the API directly usable from:

- Python clients,
- web front-ends,
- 3D Slicer extensions,

---

### Job Management

| Method | Endpoint                     | Description                                  |
|--------|------------------------------|----------------------------------------------|
| GET    | `/jobs/{job_id}`             | Poll job status                              |
| GET    | `/jobs/{job_id}/logs`        | Stream logs (Server-Sent Events)             |
| GET    | `/jobs/{job_id}/result`      | Download result archive (ZIP)                |
| POST   | `/jobs/{job_id}/kill`        | Terminate a running job                      |

Job status values:

- `queued`
- `waiting`
- `running`
- `done`
- `error`
- `killed`

This API layer is intentionally simple and stable.  
It is designed as a thin orchestration surface over KonfAI Apps, enabling
third-party clients to interact with KonfAI without depending on its
internal Python APIs.


------------------------------------------------------------------------

## 🧠 Python API

Apps can be executed programmatically:

``` python
from pathlib import Path
from konfai.app import KonfAIApp

app = KonfAIApp("my_app")
app.infer([[Path("input.mha")]], output=Path("./Predictions"))
```

Or remotely:

``` python
from pathlib import Path
from konfai.app import KonfAIAppClient
from konfai import RemoteServer

client = KonfAIAppClient("my_app", RemoteServer("host", 8000, "TOKEN"))
client.infer([[Path("input.mha")]], output=Path("./Predictions"))
```

------------------------------------------------------------------------

## 🧩 Creating Your Own App

1.  Create a directory under `apps/`

2. Add the following elements:

    - `Prediction.yml`  
    Defines the inference workflow (required).

    - Optional:
    - `Evaluation.yml` – evaluation workflow  
    - `Uncertainty.yml` – uncertainty workflow

    - Model checkpoint(s)  
        (`.pt`, directory of weights, or any format supported by KonfAI)

    - `app.json` metadata  
        Used by UIs and clients to expose defaults and documentation.

    - `requirements.txt` (optional)  
        Lists Python dependencies required by this App (custom models, transforms,
        third-party libraries, etc.).  
        When executed remotely, the server can install these dependencies in an
        isolated environment before running the App.

    - Custom Python code (optional):  
        - `Custom Transform`  
        - `Custom Augmentation`  
        - `Custom Metrics`  
        - `Custom Model`
        - Any custom KonfAI component needed by your YAML workflows  

    These files can live directly inside the app directory and will be
    automatically available at runtime.

3.  Test locally with:

    ``` bash
    konfai-apps infer my_app -i sample.mha
    ```

Your App is now portable across:

-   CLI
-   Python
-   SlicerKonfAI
-   Remote servers

------------------------------------------------------------------------

KonfAI Apps turn complex medical AI pipelines into **portable,
reproducible, and shareable building blocks**.
