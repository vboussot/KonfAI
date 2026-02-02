# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import signal
import subprocess  # nosec B404
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import konfai
from konfai.utils.utils import AppRepositoryError, get_app_repository_info

MAX_ACTIVE_JOBS = 32

_APPS_CONFIG: dict = {}
if "KONFAI_APPS_CONFIG" in os.environ:
    _APPS_CONFIG = json.loads(os.environ["KONFAI_APPS_CONFIG"])

_APPS: dict[str, list] = _APPS_CONFIG.get("apps", [])

app = FastAPI()

security = HTTPBearer(auto_error=False)


def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    expected = os.environ.get("KONFAI_API_TOKEN")
    if not expected:
        return

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    if credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid token")


protected = APIRouter(dependencies=[Depends(require_token)])

GPU_SEM: dict[int, asyncio.Semaphore] = {}


@app.on_event("startup")
async def init_gpu_semaphores():
    devices_index, _ = konfai.get_available_devices()
    for i in devices_index:
        GPU_SEM[int(i)] = asyncio.Semaphore(1)


MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
MAX_TOTAL_BYTES = 6 * 1024 * 1024 * 1024  # 6GB


def save_uploads(
    files: list[UploadFile], dst: Path, max_file_bytes: int = MAX_FILE_BYTES, max_total_bytes: int = MAX_TOTAL_BYTES
) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    total = 0

    for f in files:
        if not f.filename:
            continue

        p = dst / Path(f.filename).name
        written = 0

        with p.open("wb") as w:
            while True:
                chunk = f.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                total += len(chunk)

                if written > max_file_bytes:
                    raise HTTPException(413, f"File too large: {f.filename}")
                if total > max_total_bytes:
                    raise HTTPException(413, "Total upload too large")

                w.write(chunk)

        out.append(p.resolve())

    return out


@dataclass
class Job:
    """
    Represents a single KonfAI job executed on the server.

    A Job encapsulates:
    - its unique identifier (`job_id`)
    - its execution workspace (`run_dir`, `input_dir`, `output_dir`)
    - its lifecycle state (`status`, `error`, `created_at`, `finished_at`)
    - the underlying subprocess (`proc`)
    - GPU scheduling information (`requested_gpus`, `assigned_gpus`)
    - a bounded log queue used for SSE streaming (`log_q`)

    Lifecycle:
        queued   -> waiting (GPU acquisition)
        waiting  -> running
        running  -> done | error | killed

    The job owns a temporary directory which is deleted after completion,
    with a grace period allowing clients to download results.
    """

    job_id: str
    app_name: str
    run_dir: Path
    input_dir: Path
    output_dir: Path
    zip_path: Path
    log_q: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=10_000))
    status: str = "queued"  # queued|running|done|error
    error: str | None = None
    proc: subprocess.Popen | None = None

    requested_gpus: list[int] | None = None  # None => auto
    assigned_gpus: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


JOBS: dict[str, Job] = {}


async def acquire_gpus(job: Job, requested: list[int]) -> list[int]:
    """
    Acquire GPU resources for a job.

    Two modes are supported:

    - Auto mode (requested == []):
        The job waits until *any* GPU becomes available.
        The first free GPU semaphore is acquired.

    - Explicit mode (requested = [0, 2, ...]):
        The job waits until *all* specified GPUs are free.
        Semaphores are acquired in sorted order to avoid deadlocks.

    While waiting, the job status is set to "waiting".

    Parameters
    ----------
    job : Job
        The job requesting GPU resources.
    requested : list[int]
        List of requested GPU indices. An empty list means "auto".

    Returns
    -------
    list[int]
        The list of GPU indices that were successfully acquired.

    Raises
    ------
    HTTPException
        If a requested GPU id does not exist.
    """
    if len(requested) == 0:
        job.status = "waiting"
        while True:
            for gid, sem in GPU_SEM.items():
                if sem.locked() is False and sem._value > 0:
                    await sem.acquire()
                    return [gid]
            await asyncio.sleep(0.1)

    gpus = sorted({int(x) for x in requested})
    job.status = "waiting"
    for gid in gpus:
        if gid not in GPU_SEM:
            raise HTTPException(400, f"Unknown GPU id: {gid}")
    for gid in gpus:
        await GPU_SEM[gid].acquire()
    return gpus


def release_gpus(gpus: list[int]) -> None:
    for gid in gpus:
        sem = GPU_SEM.get(gid)
        if sem:
            try:
                sem.release()
            except ValueError:
                pass


MAX_SSE_GLOBAL = 200  # ajuste
SSE_TTL_S = 600  # 10 minutes

_active_sse_global = 0
_active_sse_jobs: set[str] = set()
_active_sse_lock = asyncio.Lock()


async def sse_log_stream(job: Job):
    """
    Server-Sent Events (SSE) generator for streaming job logs.

    This stream is protected by:
    - a global limit on the number of concurrent streams
    - a per-job limit (only one active stream per job_id)
    - a hard TTL (SSE_TTL_S) after which the stream is closed

    The stream yields:
        data: <log line>\n\n

    Termination conditions:
    - "__DONE__" log marker
    - "__ERROR__" marker
    - TTL expiration
    - client disconnect

    Admission control and cleanup are guaranteed via a lock and a
    try/finally block, ensuring counters remain consistent even on
    client abort.

    Parameters
    ----------
    job : Job
        The job whose logs are streamed.
    """
    global _active_sse_global

    # ---- admission control ----
    async with _active_sse_lock:
        if _active_sse_global >= MAX_SSE_GLOBAL:
            raise HTTPException(429, "Too many log streams")
        if job.job_id in _active_sse_jobs:
            raise HTTPException(429, "Log stream already open for this job")
        _active_sse_global += 1
        _active_sse_jobs.add(job.job_id)

    start = time.time()
    try:
        yield f"data: [KonfAI-Apps] Remote job {job.job_id} log stream connected\n\n"

        while True:
            # ---- TTL ----
            if time.time() - start > SSE_TTL_S:
                yield "data: __DONE__\n\n"
                break

            line = await job.log_q.get()
            yield f"data: {line.strip()}\n\n"

            if line == "__DONE__" or line.startswith("__ERROR__"):
                break

    except asyncio.CancelledError:
        # client disconnected
        pass

    finally:
        # ---- release admission ----
        async with _active_sse_lock:
            _active_sse_jobs.discard(job.job_id)
            _active_sse_global -= 1


@protected.get("/health")
def health():
    return {"status": "ok"}


@protected.get("/available_devices")
def get_available_devices():
    devices_index, devices_name = konfai.get_available_devices()
    return {"devices_index": devices_index, "devices_name": devices_name}


@protected.get("/ram")
def get_ram():
    used_gb, total_gb = konfai.get_ram()
    return {"used_gb": used_gb, "total_gb": total_gb}


@protected.get("/vram")
def get_vram(devices: list[int] = Query(...)):
    used_gb, total_gb = konfai.get_vram(devices)
    return {"used_gb": used_gb, "total_gb": total_gb}


@protected.get("/repo_apps_list")
def get_apps():
    return {"apps": _APPS}


@protected.get("/repo_apps/{app_id:path}")
def get_app_info(app_id: str):

    try:
        app = get_app_repository_info(app_id, False)
    except AppRepositoryError:
        return {
            "app": app_id,
            "available": False,
        }

    result = {
        "app": app_id,
        "available": True,
        "display_name": app.get_display_name(),
        "description": app.get_description(),
        "short_description": app.get_short_description(),
        "checkpoints_name": app.get_checkpoints_name(),
        "checkpoints_name_available": app.get_checkpoints_name_available(),
        "maximum_tta": app.get_maximum_tta(),
        "mc_dropout": app.get_mc_dropout(),
        "has_capabilities": app.has_capabilities(),
    }
    terminology = app.get_terminology()
    if terminology is not None:
        result["terminology"] = {str(k): asdict(v) for k, v in terminology.items()}

    result["inputs"] = {k: asdict(v) for k, v in app.get_inputs().items()}
    result["outputs"] = {k: asdict(v) for k, v in app.get_outputs().items()}

    result_tmp: dict[str, dict[str, dict]] = {}
    for key, entries in app.get_evaluations_inputs().items():
        by_file = result_tmp.setdefault(key.display_name, {})
        by_file[key.evaluation_file] = {name: asdict(entry) for name, entry in entries.items()}
    result["inputs_evaluations"] = result_tmp
    return result


@protected.get("/repo_apps_config/{app_id:path}")
def download_app_repository_configs(app_id: str, background_tasks: BackgroundTasks):
    """
    Download the configuration files of an app as a ZIP archive.
    """
    try:
        app = get_app_repository_info(app_id, False)
    except AppRepositoryError:
        raise HTTPException(404, f"Unknown app '{app_id}'")

    files = app.download_config_file()

    tmp_dir = Path(tempfile.mkdtemp(prefix="konfai_app_"))
    zip_root = tmp_dir / f"{app_id}_configs"
    zip_root.mkdir(parents=True, exist_ok=True)

    for path in files:
        shutil.copy2(path, zip_root / path.name)

    zip_path = tmp_dir / f"{app_id}_configs.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", zip_root)

    # Planifie le nettoyage après l'envoi
    background_tasks.add_task(shutil.rmtree, tmp_dir, ignore_errors=True)

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"{app_id}_configs.zip",
    )


def q_put_drop_oldest(q: asyncio.Queue[str], item: str) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueFull:
            return
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass


def _run_job_sync(
    job: Job,
    cmd: list[str],
):
    """
    Execute a job synchronously in a background thread.

    This function:
    - spawns the subprocess in its own process group
    - captures stdout/stderr line-by-line
    - pushes log lines into the job log queue
    - updates job status and error fields
    - packages the output directory into a zip file on success

    It is always executed inside `asyncio.to_thread(...)` to avoid
    blocking the event loop.

    Parameters
    ----------
    job : Job
        The job being executed.
    cmd : list[str]
        Fully resolved command line to execute.
    """
    job.status = "running"
    q_put_drop_oldest(job.log_q, f"[KonfAI-Apps] Starting job in: {job.run_dir}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(job.run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )  # nosec B603
        job.proc = proc
        if proc.stdout:
            for line in proc.stdout:
                q_put_drop_oldest(job.log_q, line.rstrip("\n"))

        rc = proc.wait()
        if rc != 0:
            job.status = "error"
            job.error = f"Subprocess failed (exit code {rc})"
            q_put_drop_oldest(job.log_q, f"__ERROR__ {job.error}")
            q_put_drop_oldest(job.log_q, "__DONE__")
            return

        zip_base = job.run_dir / "result"
        zip_file = shutil.make_archive(str(zip_base), "zip", root_dir=job.output_dir)
        job.zip_path = Path(zip_file)

        job.status = "done"
        q_put_drop_oldest(job.log_q, f"Result zip created: {job.zip_path}")
        q_put_drop_oldest(job.log_q, "__DONE__")

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        q_put_drop_oldest(job.log_q, f"__ERROR__ {job.error}")
        q_put_drop_oldest(job.log_q, "__DONE__")


async def start_job(job: Job, cmd: list[str], requested_gpus: list[int] | None):
    """
    Orchestrate the full lifecycle of a job:

    1. Acquire GPU resources (auto or explicit mode)
    2. Inject the assigned GPUs into the command line
    3. Run the job in a worker thread
    4. Release GPU resources (always, even on failure)
    5. Mark completion time
    6. Wait for a grace period
    7. Delete the temporary workspace
    8. Remove the job from the registry

    This function guarantees that:
    - GPUs are never leaked
    - temporary directories are cleaned
    - the job registry remains consistent

    Parameters
    ----------
    job : Job
        The job instance.
    cmd : list[str]
        Base command line (without GPU arguments).
    requested_gpus : list[int] | None
        Requested GPU ids, or None for CPU execution.
    """
    gpus = []
    try:
        cmd2 = list(cmd)
        if requested_gpus is not None:
            gpus = await acquire_gpus(job, requested_gpus)
            job.assigned_gpus = gpus

            q_put_drop_oldest(job.log_q, f"[KonfAI-Apps] Assigned GPUs: {gpus}")

            if gpus:
                cmd2 += ["--gpu"] + [str(i) for i in gpus]

        await asyncio.to_thread(_run_job_sync, job, cmd2)

    finally:
        if gpus:
            release_gpus(gpus)

        job.finished_at = time.time()

        await asyncio.sleep(120)

        shutil.rmtree(job.run_dir, ignore_errors=True)

        JOBS.pop(job.job_id, None)


def submit_job():
    """
    Decorator factory for job-submitting endpoints.

    The wrapper produced by this decorator:

    - Enforces a global limit on active jobs
    - Creates a unique temporary workspace
    - Registers a Job object in the global registry
    - Saves uploaded files with size quotas
    - Builds the final command line
    - Spawns the asynchronous job execution task
    - Returns job metadata (id, URLs)

    This provides a uniform execution model for all apps
    (infer, evaluate, pipeline, fine_tune, etc.).
    """

    def deco(fn: Callable[..., Awaitable[list[str]]]):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            active = sum(1 for j in JOBS.values() if j.status in {"queued", "waiting", "running"})
            if active >= MAX_ACTIVE_JOBS:
                raise HTTPException(429, "Server busy: too many active jobs")

            app_name = kwargs.get("app_name")

            job_id = uuid.uuid4().hex[:12]
            run_dir = Path(tempfile.mkdtemp(prefix=f"konfai_job_{job_id}_")).resolve()
            input_dir = run_dir / "Input"
            output_dir = run_dir / "Output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            job = Job(
                job_id=job_id,
                app_name=str(app_name),
                run_dir=run_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                zip_path=run_dir / "result.zip",
            )
            JOBS[job_id] = job

            inputs_upload_files = kwargs.get("inputs")
            inputs = None
            if inputs_upload_files:
                inputs = save_uploads(inputs_upload_files, job.input_dir)

            gt_upload_files = kwargs.get("gt")
            gt = None
            if gt_upload_files:
                gt = save_uploads(gt_upload_files, job.input_dir)

            mask_upload_files = kwargs.get("mask")
            mask = None
            if mask_upload_files:
                mask = save_uploads(mask_upload_files, job.input_dir)

            gpu = kwargs.get("gpu")
            cpu = kwargs.get("cpu")
            quiet = kwargs.get("quiet")

            cmd = await fn(*args, **kwargs)
            cmd += ["--output", str(job.output_dir)]
            if gpu is None:
                gpu_list = None
                cmd += ["--cpu", str(cpu)]
            else:
                gpu_list = [int(x.strip()) for x in gpu.split(",") if x.strip()]

            if inputs:
                cmd += ["--inputs"] + [str(p) for p in inputs]

            if gt:
                cmd += ["--gt"] + [str(p) for p in gt]

            if mask:
                cmd += ["--mask"] + [str(p) for p in mask]

            if quiet:
                cmd += ["--quiet"]

            asyncio.create_task(start_job(job, cmd, gpu_list))

            return {
                "job_id": job_id,
                "status_url": f"/jobs/{job_id}",
                "logs_url": f"/jobs/{job_id}/logs",
                "result_url": f"/jobs/{job_id}/result",
            }

        return wrapper

    return deco


@protected.post("/apps/{app_name:path}/infer")
@submit_job()
async def infer(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    ensemble: Annotated[int, Form()] = 0,
    ensemble_models: Annotated[str, Form()] = "",  # CSV
    tta: Annotated[int, Form()] = 0,
    mc: Annotated[int, Form()] = 0,
    uncertainty: Annotated[bool, Form()] = False,
    prediction_file: Annotated[str, Form()] = "Prediction.yml",
    gpu: Annotated[str | None, Form()] = None,  # CSV "0,1"
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit an inference job.

    This endpoint runs `konfai-apps infer` on the server. The job is executed
    asynchronously and managed by the scheduler.

    Features:
    - Supports ensembling (by count or by explicit model list)
    - Test-time augmentation (TTA)
    - Monte-Carlo sampling (MC dropout)
    - GPU auto-allocation or explicit GPU selection
    - Isolated workspace and result packaging

    Parameters
    ----------
    app_name : str
        Name/path of the KonfAI application.
    inputs : list[UploadFile]
        Input images or volumes.
    ensemble : int
        Number of models to sample when no explicit list is provided.
    ensemble_models : str
        Comma-separated list of model identifiers.
    tta : int
        Number of test-time augmentations.
    mc : int
        Number of Monte-Carlo samples.
    prediction_file : str
        Prediction configuration file.
    gpu : str | None
        Comma-separated GPU ids, or None for auto mode.
    cpu : int
        CPU workers if GPU is not used.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata (job_id and URLs).
    """
    ensemble_models_list = [x.strip() for x in ensemble_models.split(",") if x.strip()]
    cmd = [
        "konfai-apps",
        "infer",
        app_name,
        "--tta",
        str(tta),
        "--mc",
        str(mc),
        "--prediction_file",
        prediction_file,
    ]
    if uncertainty:
        cmd += ["-uncertainty"]
    if len(ensemble_models_list) > 0:
        cmd += ["--ensemble_models"] + ensemble_models_list
    else:
        cmd += ["--ensemble", str(ensemble)]
    return cmd


@protected.post("/apps/{app_name:path}/evaluate")
@submit_job()
async def evaluate(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    gt: Annotated[list[UploadFile], File(...)],
    mask: Annotated[list[UploadFile], File(...)] = [],
    evaluation_file: Annotated[str, Form()] = "Evaluation.yml",
    gpu: Annotated[str | None, Form()] = None,
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit an evaluation job.

    Runs `konfai-apps eval` to compute metrics between predictions and
    ground-truth data.

    Parameters
    ----------
    app_name : str
        Application name.
    inputs : list[UploadFile]
        Predicted outputs or inputs to evaluate.
    gt : list[UploadFile]
        Ground-truth data.
    mask : list[UploadFile]
        Optional masks.
    evaluation_file : str
        Evaluation configuration file.
    gpu : str | None
        GPU selection or auto mode.
    cpu : int
        CPU workers.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata.
    """
    cmd = [
        "konfai-apps",
        "eval",
        app_name,
        "--evaluation_file",
        evaluation_file,
    ]
    return cmd


@protected.post("/apps/{app_name:path}/uncertainty")
@submit_job()
async def uncertainty(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    uncertainty_file: Annotated[str, Form()] = "Uncertainty.yml",
    gpu: Annotated[str | None, Form()] = "",
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit an uncertainty estimation job.

    Executes `konfai-apps uncertainty` to compute uncertainty maps or
    statistics from model outputs.

    Parameters
    ----------
    app_name : str
        Application name.
    inputs : list[UploadFile]
        Input data.
    uncertainty_file : str
        Uncertainty configuration file.
    gpu : str | None
        GPU selection or auto mode.
    cpu : int
        CPU workers.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata.
    """
    cmd = [
        "konfai-apps",
        "uncertainty",
        app_name,
        "--uncertainty_file",
        uncertainty_file,
    ]
    return cmd


@protected.post("/apps/{app_name:path}/pipeline")
@submit_job()
async def pipeline(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    gt: Annotated[list[UploadFile], File(...)],
    ensemble: Annotated[int, Form()] = 0,
    ensemble_models: Annotated[str, Form()] = "",
    tta: Annotated[int, Form()] = 0,
    mc: Annotated[int, Form()] = 0,
    prediction_file: Annotated[str, Form()] = "Prediction.yml",
    mask: Annotated[list[UploadFile] | None, File(...)] = None,
    evaluation_file: Annotated[str, Form()] = "Evaluation.yml",
    uncertainty: Annotated[bool, Form()] = True,
    uncertainty_file: Annotated[str, Form()] = "Uncertainty.yml",
    gpu: Annotated[str | None, Form()] = "",
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit a full end-to-end pipeline job.

    This endpoint chains:
        inference → evaluation → (optional) uncertainty estimation

    It provides a single entry point for complete experiments with
    ensembling, TTA, MC sampling, and metric computation.

    Parameters
    ----------
    app_name : str
        Application name.
    inputs : list[UploadFile]
        Input data.
    gt : list[UploadFile]
        Ground-truth data.
    ensemble : int
        Ensemble size.
    ensemble_models : str
        Explicit ensemble model list.
    tta : int
        Test-time augmentation count.
    mc : int
        Monte-Carlo samples.
    prediction_file : str
        Prediction configuration.
    mask : list[UploadFile] | None
        Optional masks.
    evaluation_file : str
        Evaluation configuration.
    uncertainty : bool
        Enable uncertainty stage.
    uncertainty_file : str
        Uncertainty configuration.
    gpu : str | None
        GPU selection or auto mode.
    cpu : int
        CPU workers.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata.
    """
    ensemble_models_list = [x.strip() for x in ensemble_models.split(",") if x.strip()]
    cmd = [
        "konfai-apps",
        "pipeline",
        app_name,
        "--ensemble",
        str(ensemble),
        "--tta",
        str(tta),
        "--mc",
        str(mc),
        "--prediction_file",
        prediction_file,
        "--evaluation_file",
        evaluation_file,
        "--uncertainty_file",
        uncertainty_file,
    ]
    if uncertainty:
        cmd += ["-uncertainty"]
    if ensemble_models_list:
        cmd += ["--ensemble_models"] + ensemble_models_list
    return cmd


@protected.post("/apps/{app_name:path}/fine_tune")
@submit_job()
async def fine_tune(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    name: Annotated[str, Form()] = "Finetune",
    epochs: Annotated[int, Form()] = 10,
    it_validation: Annotated[int, Form()] = 1000,
    config_file: Annotated[str, Form()] = "Config.yml",
    gpu: Annotated[str | None, Form()] = "",
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit a fine-tuning (training) job.

    Executes `konfai-apps fine-tune` to adapt a model on the provided dataset.
    The job is fully managed by the server (workspace, GPUs, logs, cleanup).

    Parameters
    ----------
    app_name : str
        Application name.
    inputs : list[UploadFile]
        Training dataset.
    name : str
        Run name.
    epochs : int
        Number of training epochs.
    it_validation : int
        Validation interval.
    config_file : str
        Training configuration file.
    gpu : str | None
        GPU selection or auto mode.
    cpu : int
        CPU workers.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata.
    """
    cmd = [
        "konfai-apps",
        "fine-tune",
        app_name,
        name,
        "--config",
        config_file,
        "--epochs",
        str(epochs),
        "--it_validation",
        str(it_validation),
    ]
    return cmd


@protected.get("/jobs/{job_id}")
def job_status(job_id: str):
    """
    Query the current status of a job.

    This endpoint provides lightweight polling access to the job state
    without streaming logs.

    Parameters
    ----------
    job_id : str
        Identifier of the job.

    Returns
    -------
    dict
        {
            "job_id": <str>,
            "status": <queued|waiting|running|done|error|killed>,
            "error": <optional error message>
        }

    Raises
    ------
    HTTPException
        If the job_id is unknown.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")
    return {"job_id": job.job_id, "status": job.status, "error": job.error}


@protected.get("/jobs/{job_id}/logs")
async def job_logs(job_id: str):
    """
    Stream the logs of a job using Server-Sent Events (SSE).

    The stream is protected by:
    - a global limit on concurrent streams
    - a per-job limit (only one active stream per job)
    - a hard time-to-live (TTL)

    Log lines are emitted as SSE events of the form:

        data: <log line>

    The stream terminates when:
    - the job finishes (`__DONE__`)
    - an error is reported (`__ERROR__`)
    - the TTL expires
    - the client disconnects

    Parameters
    ----------
    job_id : str
        Identifier of the job.

    Returns
    -------
    StreamingResponse
        SSE stream of log lines.

    Raises
    ------
    HTTPException
        If the job does not exist or stream limits are exceeded.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")
    return StreamingResponse(sse_log_stream(job), media_type="text/event-stream")


@protected.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    """
    Retrieve the result archive of a completed job.

    Behavior:
    - If the job is still running or waiting:
        returns HTTP 202 (not ready)
    - If the job failed:
        returns HTTP 500 with error information
    - If the job succeeded:
        returns the result zip archive

    Parameters
    ----------
    job_id : str
        Identifier of the job.

    Returns
    -------
    FileResponse | JSONResponse
        - ZIP archive when the job is done
        - JSON status otherwise

    Raises
    ------
    HTTPException
        If the job_id is unknown.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")

    if job.status == "error":
        return JSONResponse(status_code=500, content={"job_id": job.job_id, "status": job.status, "error": job.error})

    if job.status != "done" or not job.zip_path.exists():
        # Not ready yet
        return JSONResponse(status_code=202, content={"job_id": job.job_id, "status": job.status})

    return FileResponse(str(job.zip_path), media_type="application/zip", filename="result.zip")


@protected.post("/jobs/{job_id}/kill")
def kill_job(job_id: str):
    """
    Terminate a running job.

    The job subprocess is started in its own process group.
    This endpoint:

    1. Sends SIGTERM to the entire process group
    2. Waits briefly for graceful shutdown
    3. Sends SIGKILL if the process is still alive
    4. Marks the job as killed
    5. Emits termination markers in the log stream

    This guarantees that:
    - All child processes are terminated
    - GPU resources are eventually released
    - Clients observing logs are notified

    Parameters
    ----------
    job_id : str
        Identifier of the job to terminate.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id")

    proc = job.proc
    if proc is None or proc.poll() is not None:
        return {"job_id": job.job_id, "status": job.status, "message": "Job not running"}

    try:
        # SIGTERM au groupe
        os.killpg(proc.pid, signal.SIGTERM)

        # petite attente + SIGKILL si besoin
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)

        job.status = "killed"
        job.error = "Killed by user"
        q_put_drop_oldest(job.log_q, "__ERROR__ Killed by user")
        q_put_drop_oldest(job.log_q, "__DONE__")

        return {"job_id": job.job_id, "status": "killed", "message": "Kill requested"}

    except Exception as e:
        raise HTTPException(500, f"Failed to kill job: {e}")


app.include_router(protected)
