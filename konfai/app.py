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

import inspect
import os
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import requests
import SimpleITK as sitk  # noqa: N813

from konfai import RemoteServer, check_server, cuda_visible_devices
from konfai.utils.dataset import Dataset
from konfai.utils.utils import (
    SUPPORTED_EXTENSIONS,
    KonfAIAppClientError,
    LocalAppRepository,
    MinimalLog,
    State,
    get_app_repository_info,
)


class CancelProcess(RuntimeError):
    """
    Exception used to convert SIGINT/SIGTERM signals into a regular Python error.

    This is primarily used to ensure that `finally` blocks are executed even when
    the process receives an interrupt/termination signal (Ctrl+C, system kill).

    Notes
    -----
    - This exception is intentionally raised from a signal handler.
    - It should typically be caught at a high level to perform cleanup.
    """

    pass


@contextmanager
def ensure_finally_on_signals():
    """
    Context manager that guarantees `finally` blocks run on SIGINT/SIGTERM.

    Inside this context:
    - SIGINT and SIGTERM handlers are temporarily replaced.
    - Receiving one of these signals raises `CancelProcess`, which unwinds the
      stack normally and therefore triggers `finally` clauses.

    On exit:
    - original signal handlers are restored.

    Typical usage
    -------------
    with ensure_finally_on_signals():
        try:
            ...
        finally:
            cleanup()

    Caveats
    -------
    - Signal handlers are process-global: using this concurrently in multiple
      threads is not recommended.
    - Only SIGINT/SIGTERM are handled here.
    """
    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)

    def _handler(signum, frame):
        # On déclenche une exception -> remonte -> finally exécuté
        raise CancelProcess(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        # On restaure les handlers d'origine
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def run_distributed_app(
    func: Callable[..., None],
) -> Callable[..., None]:
    """
    Decorator that runs a KonfAI app entrypoint inside an isolated temporary workspace.

    This wrapper:
    - Creates (or reuses) a temporary working directory (`tmp_dir`)
    - Changes the current working directory to that temporary directory
    - Adds that directory to `sys.path` (so local imports work)
    - Executes the wrapped function inside a minimal logging context (`MinimalLog`)
    - Restores the user's original working directory
    - Deletes the temporary directory if it was created automatically

    The decorated function may declare a `tmp_dir` argument. If provided, that
    directory is used and NOT automatically deleted (unless it lives under the
    system temp directory and the code chooses to clean it).

    Parameters
    ----------
    func : Callable[..., None]
        Function implementing a local app action (infer/evaluate/etc.).

    Returns
    -------
    Callable[..., None]
        Wrapped function with identical signature and behavior, executed in
        an isolated workspace.
    """

    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        params = sig.parameters
        kwargs_fun = {k: v for k, v in kwargs.items() if k in params}

        bound = sig.bind_partial(*args, **kwargs_fun)
        bound.apply_defaults()

        tmp_dir = bound.arguments.get("tmp_dir")
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="konfai_app_"))
        tmp_dir = tmp_dir.resolve()
        user_dir = os.getcwd()
        try:
            os.makedirs(tmp_dir, exist_ok=True)
            os.chdir(str(tmp_dir))
            sys.path.insert(0, os.getcwd())
            with MinimalLog():
                func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n[KonfAI-Apps] Manual interruption (Ctrl+C)")
            exit(0)
        finally:
            if Path(os.getcwd()).resolve() != Path(user_dir).resolve():
                tmp_dir = Path(os.getcwd()).resolve()
                if str(tmp_dir) in sys.path:
                    sys.path.remove(str(tmp_dir))
                os.chdir(user_dir)
                if tmp_dir.parent == Path(tempfile.gettempdir()):
                    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return wrapper


class AbstractKonfAIApp:

    def __init__(self) -> None:
        super().__init__()


class Cancelprocess(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class KonfAIAppClient(AbstractKonfAIApp):
    """
    Client-side helper to submit jobs to a remote KonfAI app server.

    This class implements:
    - job submission to endpoints like `/apps/{app}/{action}`
    - streaming logs via SSE (`/jobs/{job_id}/logs`)
    - result retrieval (`/jobs/{job_id}/result`)
    - remote job termination (`/jobs/{job_id}/kill`)

    It is intended to mirror the server's execution model:
    submit → stream logs → download results → (optional) kill on interruption.
    """

    def __init__(self, app: str, remote_server: RemoteServer) -> None:
        """
        Create a client bound to a given application and remote server.

        Parameters
        ----------
        app : str
            Application identifier/path on the server.
        remote_server : RemoteServer
            Server connection parameters (base URL, auth headers).

        Raises
        ------
        KonfAIAppClientError
            If the server cannot be reached or does not respond as expected.
        """
        self.app = app
        self.remote_server = remote_server
        ok, msg = check_server(remote_server)
        if not ok:
            raise KonfAIAppClientError(
                f"{msg}."
                "Unable to connect to the KonfAI app server.\n\n"
                "Please verify the host and port, or select another remote server."
            )

    def stream_logs(self, job_id: str, connect_timeout: int = 3, read_timeout: int = 300):
        """
        Stream server-side job logs using Server-Sent Events (SSE).

        This method connects to:
            GET /jobs/{job_id}/logs

        It prints each received SSE "data:" message to stdout. The stream ends when
        one of the terminal markers is received:
        - "__DONE__"
        - "__ERROR__ ..."

        Parameters
        ----------
        job_id : str
            Remote job identifier returned by the server.
        connect_timeout : int
            Max seconds to wait for the initial connection.
        read_timeout : int
            Max seconds to wait for new bytes before considering the stream stalled.

        Raises
        ------
        RuntimeError
            For auth errors, forbidden access, stream stalls, or other request failures.
        """
        url = f"{self.remote_server.get_url()}/jobs/{job_id}/logs"
        try:
            with requests.get(
                url,
                headers=self.remote_server.get_headers(),
                stream=True,
                timeout=(connect_timeout, read_timeout),
            ) as r:

                if r.status_code == 401:
                    raise RuntimeError("Unauthorized: invalid or missing token")
                if r.status_code == 403:
                    raise RuntimeError("Forbidden")

                r.raise_for_status()

                for line in r.iter_lines(decode_unicode=True):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        msg = line[6:]
                        if msg == "__DONE__" or msg.startswith("__ERROR__"):
                            return
                        else:
                            print(msg, flush=True)

        except requests.exceptions.ReadTimeout:
            raise RuntimeError(f"Log stream stalled (no data received for {read_timeout}s)")
        except requests.exceptions.ConnectTimeout:
            raise RuntimeError("Connection timeout")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to stream logs from {url}: {e}") from e

    def kill_job(self, job_id: str, timeout_s: float = 5.0) -> None:
        """
        Request termination of a remote job.

        Sends:
            POST /jobs/{job_id}/kill

        If successful, prints a confirmation message.

        Parameters
        ----------
        job_id : str
            Remote job identifier.
        timeout_s : float
            Timeout (seconds) for the kill request.

        Raises
        ------
        TimeoutError
            If the request times out.
        RuntimeError
            For auth errors or other HTTP failures.
        """
        url = f"{self.remote_server.get_url()}/jobs/{job_id}/kill"
        try:
            r = requests.post(
                url,
                headers=self.remote_server.get_headers(),
                timeout=timeout_s,
            )

            if r.status_code == 401:
                raise RuntimeError("Unauthorized: invalid or missing token")

            r.raise_for_status()
            print(f"[KonfAI-Apps] Remote job {job_id} successfully killed.")
        except requests.exceptions.ConnectTimeout as e:
            raise TimeoutError("Connection timeout while sending kill request") from e

        except requests.exceptions.ReadTimeout as e:
            raise TimeoutError(f"Kill request stalled (no response for {timeout_s:.0f}s)") from e

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to kill job {job_id}: {e}") from e

    def download_result(
        self, job_id: str, out_dir: Path, connect_timeout: int = 3, read_timeout: int = 300, max_wait_s: int = 600
    ):
        """
        Download and unpack the result archive for a remote job.

        Polls:
            GET /jobs/{job_id}/result

        Server behavior expected:
        - HTTP 202: result not ready → keep polling
        - HTTP 200: returns a zip archive → download then unpack

        The downloaded archive is saved as:
            <out_dir>/result.zip

        Then extracted into `out_dir`.

        Parameters
        ----------
        job_id : str
            Remote job identifier.
        out_dir : Path
            Destination directory where the result is extracted.
        connect_timeout : int
            Connection timeout for each poll attempt.
        read_timeout : int
            Read timeout for each download attempt.
        max_wait_s : int
            Maximum total time to wait for the result to become available.

        Returns
        -------
        bool
            True if the result was successfully downloaded and extracted.

        Raises
        ------
        TimeoutError
            If the result does not become ready within `max_wait_s`.
        RuntimeError
            For request failures, auth issues, or download errors.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / "result.zip"

        poll_interval = 0.5
        deadline = time.monotonic() + max_wait_s

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Result not ready after {max_wait_s:.0f}s")

            try:
                with requests.get(
                    f"{self.remote_server.get_url()}/jobs/{job_id}/result",
                    headers=self.remote_server.get_headers(),
                    stream=True,
                    timeout=(connect_timeout, read_timeout),
                ) as r:

                    if r.status_code == 401:
                        raise RuntimeError("Unauthorized: invalid or missing token")

                    if r.status_code == 202:
                        time.sleep(poll_interval)
                        continue

                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
            except requests.exceptions.ConnectTimeout as e:
                raise TimeoutError("Connection timeout while downloading result") from e
            except requests.exceptions.ReadTimeout:
                raise TimeoutError(f"Download stalled (no data for {read_timeout:.0f}s)")
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download result for job {job_id}: {e}") from e

            break
        shutil.unpack_archive(zip_path, out_dir)
        print(f"[KonfAI-Apps] Result written to: {out_dir}")
        return True

    @staticmethod
    def run_remote_job(func: Callable[..., None]) -> Callable[..., None]:
        """
        Decorator for KonfAIAppClient methods that submit work to the remote server.

        The wrapped method is treated as an "action" endpoint. For example, wrapping
        `infer()` will call:
            POST /apps/{self.app}/infer

        Behavior:
        1. Introspects the wrapped function signature to filter kwargs.
        2. Builds a multipart request:
        - file fields for inputs/gt/mask (nested list[list[Path]])
        - scalar fields for other parameters
        3. Submits the job and retrieves a `job_id`.
        4. Streams logs until completion markers are received.
        5. Downloads and extracts results into the requested output directory.
        6. On SIGINT/SIGTERM, triggers cleanup and (if unfinished) kills the remote job.
        7. Always closes local file handles.

        Signal handling:
        - Uses `ensure_finally_on_signals()` so that SIGINT/SIGTERM raises `CancelProcess`,
        ensuring that the `finally` clause runs and remote kill is executed.

        Notes
        -----
        - The decorated methods are "declarative": they do not implement logic
        themselves and typically contain only `pass`.
        - Output directory is taken from the wrapped method's `output` argument.

        Returns
        -------
        Callable[..., None]
            Wrapped method that performs remote submission + monitoring + download.
        """
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> None:
            job_id: str | None = None

            params = sig.parameters
            kwargs_fun = {k: v for k, v in kwargs.items() if k in params}

            bound = sig.bind_partial(self, *args, **kwargs_fun)
            bound.apply_defaults()
            bound.arguments.pop("self", None)

            output = bound.arguments.pop("output", None)
            files = []
            data = {}
            finished = False
            with ensure_finally_on_signals():
                try:
                    data_arguments = ["inputs", "gt", "mask"]
                    for k, v in bound.arguments.items():
                        if k in data_arguments:
                            if v is not None:
                                for group in v:
                                    for p in group:
                                        files.append((k, open(p, "rb")))
                        else:
                            data[k] = v

                    if "ensemble_models" in data:
                        if len(data["ensemble_models"]) > 0:
                            del data["ensemble"]
                            data["ensemble_models"] = ",".join(data["ensemble_models"])
                        else:
                            del data["ensemble_models"]
                    connect_timeout = 3
                    read_timeout: int = 300

                    with requests.post(
                        f"{self.remote_server.get_url()}/apps/{self.app}/{func.__name__}",
                        files=files,
                        data=data,
                        headers=self.remote_server.get_headers(),
                        timeout=(connect_timeout, read_timeout),
                    ) as r:
                        if r.status_code == 401:
                            raise RuntimeError("Unauthorized: invalid or missing token")
                        r.raise_for_status()
                        resp = r.json()
                        job_id = resp["job_id"]

                    self.stream_logs(job_id)
                    self.download_result(job_id, output)
                    finished = True
                except (CancelProcess, KeyboardInterrupt):
                    print("\n[KonfAI-Apps] Interrupted (SIGINT/SIGTERM)")
                except requests.RequestException as e:
                    raise RuntimeError(f"Failed to submit job to remote KonfAI server: {e}") from e
                finally:
                    for _, fh in files:
                        try:
                            fh.close()
                        except (OSError, ValueError):
                            pass

                    if job_id is not None and not finished:
                        self.kill_job(job_id)

        return wrapper

    @run_remote_job
    def infer(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/").resolve(),
        ensemble: int = 0,
        ensemble_models: list[str] = [],
        tta: int = 0,
        mc: int = 0,
        prediction_file: str = "Prediction.yml",
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        pass

    @run_remote_job
    def evaluate(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]],
        output: Path = Path("./Output/"),
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        pass

    @run_remote_job
    def uncertainty(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/"),
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        pass

    @run_remote_job
    def pipeline(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]] | None,
        output: Path = Path("./Output/"),
        ensemble: int = 0,
        ensemble_models: list[str] = [],
        tta: int = 0,
        mc: int = 0,
        prediction_file: str = "Prediction.yml",
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        uncertainty: bool = True,
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        pass

    @run_remote_job
    def fine_tune(
        self,
        dataset: Path,
        name: str = "Finetune",
        output: Path = Path("./Output/"),
        epochs: int = 10,
        it_validation: int = 1000,
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        config_file: str = "Config.yml",
        tmp_dir: Path | None = None,
    ) -> None:
        pass


class KonfAIApp(AbstractKonfAIApp):
    """
    Local runner for KonfAI applications.

    This class executes inference/evaluation/uncertainty/fine-tuning locally by:
    - building a dataset folder structure expected by KonfAI
    - installing the appropriate model/config assets (HF or local directory)
    - invoking KonfAI predictor/evaluator/trainer functions
    - collecting outputs into a user-defined output folder

    The public methods (infer/evaluate/uncertainty/fine_tune) are wrapped by
    `run_distributed_app`, which runs each operation in an isolated temporary
    workspace.
    """

    def __init__(self, app: str) -> None:
        """
        Create a local KonfAI app runner from either a HuggingFace model spec
        or a local directory.

        Parameters
        ----------
        app : str
            Either:
            - "repo_id:revision" (HuggingFace style), or
            - a local path identifying a model directory.

        Notes
        -----
        Sets `self.app_repository` to either:
        - AppRepositoryOnHF
        - AppRepository
        """
        self.app_repository: LocalAppRepository
        app_repository_info = get_app_repository_info(app)
        if not isinstance(app_repository_info, LocalAppRepository):
            raise TypeError(
                f"KonfAI apps can only be executed from a local application repository. "
                f"App '{app}' resolves to a {type(app_repository_info).__name__}, which is not local."
            )
        self.app_repository = app_repository_info

    @staticmethod
    def _match_supported(file: Path) -> bool:
        """
        Check whether the file has an extension supported by KonfAI datasets.

        Parameters
        ----------
        file : Path
            Candidate file.

        Returns
        -------
        bool
            True if the file matches one of `SUPPORTED_EXTENSIONS`.
        """
        lower = file.name.lower()
        return any(lower.endswith("." + ext) for ext in SUPPORTED_EXTENSIONS)

    @staticmethod
    def _list_supported_files(paths: list[Path]) -> list[Path]:
        """
        Expand a list of input paths into a flat list of supported files.

        Each element in `paths` may be:
        - a file: must match supported extensions
        - a directory: recursively scanned for supported files

        Parameters
        ----------
        paths : list[Path]
            Files and/or directories provided by the user.

        Returns
        -------
        list[Path]
            All discovered supported files.

        Raises
        ------
        FileNotFoundError
            If a path does not exist, or contains no supported files.
        """
        files = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: '{path}'")

            if path.is_file():
                if KonfAIApp._match_supported(path):
                    files.append(path)
                else:
                    raise FileNotFoundError(f"No supported file found: '{path.name}' is not a supported format.")
            else:

                for f in path.rglob("*"):
                    if f.is_file() and KonfAIApp._match_supported(f):
                        files.append(f)
                if not files:
                    raise FileNotFoundError(f"No supported files found in directory: '{path}'.")
        return files

    @staticmethod
    def symlink(src: Path, dst: Path) -> None:
        """
        Create a symlink from `dst` pointing to `src`, with safe replacement.

        If `dst` already exists:
        - directories are removed (unless they are symlinks)
        - files are unlinked

        On platforms or filesystems that do not support symlinks (e.g. some Windows
        environments), this falls back to copying:
        - directories via copytree
        - files via copy2

        Parameters
        ----------
        src : Path
            Source file or directory.
        dst : Path
            Destination symlink path.
        """
        if dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
        except OSError:
            # fallback Windows / FS without symlink
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def _write_inputs_to_dataset(self, inputs: list[list[Path]]) -> None:
        """
        Build the on-disk Dataset/ structure for inference/evaluation inputs.

        Expected structure:
            ./Dataset/P{idx}/Volume_{i}{suffix}

        Where:
        - i is the input-group index (e.g., channel/modalities)
        - idx is the patient/case index

        Parameters
        ----------
        inputs : list[list[Path]]
            Nested list of paths. Each inner list is scanned for supported files.
        """
        dataset_path = Path("./Dataset/")
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        for i, input_path in enumerate(inputs):
            for idx, file in enumerate(KonfAIApp._list_supported_files(input_path)):
                suffix = "".join(file.suffixes)
                KonfAIApp.symlink(file, dataset_path / f"P{idx:03d}" / f"Volume_{i}{suffix}")

    def _write_inference_stack_to_dataset(self, inputs: list[list[Path]]) -> None:
        """
        Build the Dataset/ structure for uncertainty estimation.

        This method enforces that each input file is a multi-component volume
        (e.g., an inference stack) by reading metadata with SimpleITK.

        Raises
        ------
        FileNotFoundError
            If a provided input is not multi-channel (single-component).
        """
        dataset_path = Path("./Dataset/")
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        for i, input_path in enumerate(inputs):
            for idx, file in enumerate(KonfAIApp._list_supported_files(input_path)):
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(file))
                reader.ReadImageInformation()
                n_channels = reader.GetNumberOfComponents()
                if n_channels > 1:
                    suffix = "".join(file.suffixes)
                    KonfAIApp.symlink(file, dataset_path / f"P{idx:03d}" / f"Volume_{i}{suffix}")
                else:
                    raise FileNotFoundError(
                        "Invalid input volume for inference: a multi-channel volume stack is required, "
                        "but a single-channel volume was provided."
                    )

    def _write_gt_to_dataset(self, gt: list[list[Path]]) -> None:
        """
        Write ground-truth volumes into the Dataset/ structure.

        Expected structure:
            ./Dataset/P{idx}/Reference_{i}{suffix}

        Parameters
        ----------
        gt : list[list[Path]]
            Ground truth file paths grouped similarly to inputs.
        """
        for i, gt_path in enumerate(gt):
            for idx, file in enumerate(KonfAIApp._list_supported_files(gt_path)):
                suffix = "".join(file.suffixes)
                KonfAIApp.symlink(file, Path(f"./Dataset/P{idx:03d}/Reference_{i}{suffix}"))

    def _write_mask_or_default(self, mask: list[list[Path]] | None) -> None:
        """
        Write mask volumes into the Dataset/ structure or generate default masks.

        If `mask` is None:
        - creates a mask of ones for each case using the shape/metadata of Volume_0

        If `mask` is provided:
        - symlinks mask files as:
            ./Dataset/P{idx}/Mask_{i}{suffix}

        Parameters
        ----------
        mask : list[list[Path]] | None
            Optional mask paths.
        """
        if mask is None:
            dataset = Dataset("Dataset", "mha")
            names = dataset.get_names("Volume_0")
            for name in names:
                data, attr = dataset.read_data("Volume_0", name)
                dataset.write("Mask_0", name, np.ones_like(data), attr)
        else:
            for i, mask_path in enumerate(mask):
                for idx, file in enumerate(KonfAIApp._list_supported_files(mask_path)):
                    suffix = "".join(file.suffixes)
                    KonfAIApp.symlink(file, Path(f"./Dataset/P{idx:03d}/Mask_{i}{suffix}"))

    @run_distributed_app
    def infer(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/").resolve(),
        ensemble: int = 0,
        ensemble_models: list[str] = [],
        tta: int = 0,
        mc: int = 0,
        prediction_file: str = "Prediction.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Run inference locally for the given inputs.

        Steps:
        1. Build Dataset/ from `inputs`
        2. Install inference assets (models/config) via `self.app_repository.install_inference`
        3. Call `konfai.predictor.predict(...)`
        4. Copy generated predictions into `output` if they exist

        Notes
        -----
        - Executes inside an isolated temporary workspace (via run_distributed_app).
        - GPU defaults to `cuda_visible_devices()`.
        """
        self._write_inputs_to_dataset(inputs)
        models_path = self.app_repository.install_inference(tta, ensemble, ensemble_models, mc, prediction_file)
        from konfai.predictor import predict

        predict(models_path, True, gpu, cpu, quiet, False, Path(prediction_file).resolve())
        if Path("./Predictions").absolute().exists():
            shutil.copytree(Path("./Predictions").absolute(), output, dirs_exist_ok=True)

    @run_distributed_app
    def evaluate(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]],
        output: Path = Path("./Output/"),
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Run evaluation locally against ground-truth.

        Steps:
        1. Build Dataset/ from inputs and gt
        2. Ensure masks exist (provided or generated)
        3. Install evaluation assets via `self.app_repository.install_evaluation`
        4. Call `konfai.evaluator.evaluate(...)`
        5. Copy evaluation outputs into `output`

        Notes
        -----
        - Runs inside an isolated workspace (run_distributed_app).
        - GPU defaults to `cuda_visible_devices()`.
        """
        self._write_inputs_to_dataset(inputs)
        self._write_gt_to_dataset(gt)
        self._write_mask_or_default(mask)
        self.app_repository.install_evaluation(evaluation_file)
        from konfai.evaluator import evaluate

        evaluate(True, gpu, cpu, quiet, False, Path(evaluation_file).resolve())
        if Path("./Evaluations").exists():
            shutil.copytree("./Evaluations", output, dirs_exist_ok=True)

    @run_distributed_app
    def uncertainty(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/"),
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        self._write_inference_stack_to_dataset(inputs)
        self.app_repository.install_uncertainty(uncertainty_file)
        """
        Run uncertainty estimation locally.

        Steps:
        1. Validate that inputs are multi-component inference stacks
        2. Install uncertainty assets via `self.app_repository.install_uncertainty`
        3. Call evaluator with an explicit output directory (./Uncertainties)
        4. Copy uncertainty results into `output`

        Notes
        -----
        - Runs inside an isolated workspace (run_distributed_app).
        - GPU defaults to `cuda_visible_devices()`.
        """
        from konfai.evaluator import evaluate

        evaluate(True, gpu, cpu, quiet, False, Path(uncertainty_file).resolve(), Path("./Uncertainties/"))
        if Path("./Uncertainties").exists():
            shutil.copytree("./Uncertainties", output, dirs_exist_ok=True)

    def pipeline(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]] | None,
        output: Path = Path("./Output/"),
        ensemble: int = 0,
        ensemble_models: list[str] = [],
        tta: int = 0,
        mc: int = 0,
        prediction_file: str = "Prediction.yml",
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        uncertainty: bool = True,
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Run a full pipeline locally: inference → evaluation → uncertainty.

        This is a convenience method that orchestrates multiple stages and organizes
        outputs into subfolders:
            <output>/Predictions
            <output>/Evaluations
            <output>/Uncertainties

        Behavior:
        - always runs inference
        - runs evaluation only if `gt` is provided
        - runs uncertainty only if `uncertainty=True`
        """
        self.infer(
            inputs,
            output / "Predictions",
            ensemble,
            ensemble_models,
            tta,
            mc,
            prediction_file,
            gpu,
            cpu,
            quiet,
            tmp_dir,
        )
        outputs = []
        inference_stacks = []
        for f in (output / "Predictions").rglob("*"):
            if f.is_file() and KonfAIApp._match_supported(f):
                if f.name == "InferenceStack.mha":
                    inference_stacks.append(f)
                else:
                    outputs.append(f)
        if gt is not None:
            self.evaluate([outputs], gt, output / "Evaluations", mask, evaluation_file, gpu, cpu, quiet, tmp_dir)
        if uncertainty:
            self.uncertainty([inference_stacks], output / "Uncertainties", uncertainty_file, gpu, cpu, quiet, tmp_dir)

    @run_distributed_app
    def fine_tune(
        self,
        dataset: Path,
        name: str = "Finetune",
        output: Path = Path("./Output/"),
        epochs: int = 10,
        it_validation: int = 1000,
        gpu: list[int] = cuda_visible_devices(),
        cpu: int | None = None,
        quiet: bool = False,
        config_file: str = "Config.yml",
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Run fine-tuning (training) locally.

        Steps:
        1. Install training assets/config via `self.app_repository.install_fine_tune`
        2. Link the user dataset into ./Dataset
        3. Call `konfai.trainer.train(...)` in resume mode

        Notes
        -----
        - Runs inside an isolated workspace (run_distributed_app).
        - GPU defaults to `cuda_visible_devices()`.
        """
        models_path = self.app_repository.install_fine_tune(config_file, name, epochs, it_validation)
        KonfAIApp.symlink(dataset, Path("./Dataset").absolute())
        from konfai.trainer import train

        train(State.RESUME, True, models_path[0], gpu, cpu, quiet, False, config_file)
