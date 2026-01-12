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

import datetime
import os
from importlib import metadata
from pathlib import Path

import psutil
import pynvml
import requests
from torch.cuda import get_device_name

try:
    __version__ = metadata.version("konfai")
except metadata.PackageNotFoundError:
    __version__ = "unknown"


def checkpoints_directory() -> Path:
    return Path(_get_env("KONFAI_CHECKPOINTS_DIRECTORY"))


def predictions_directory() -> Path:
    return Path(_get_env("KONFAI_PREDICTIONS_DIRECTORY"))


def evaluations_directory() -> Path:
    return Path(_get_env("KONFAI_EVALUATIONS_DIRECTORY"))


def statistics_directory() -> Path:
    return Path(_get_env("KONFAI_STATISTICS_DIRECTORY"))


def config_file() -> Path:
    return Path(_get_env("KONFAI_config_file"))


def konfai_state() -> str:
    return _get_env("KONFAI_STATE")


def konfai_root() -> str:
    return _get_env("KONFAI_ROOT")


class RemoteServer:

    def __init__(self, host: str, port: int, token: str | None) -> None:
        self.host = host
        self.port = port
        self.token = token
        self.timeout = 10

    def __str__(self) -> str:
        return f"{self.host}|{self.port}"

    def get_headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def get_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def cuda_visible_devices() -> list[int]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu != ""]
    else:
        import torch

        devices = []
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
    return devices


def get_available_devices(
    remote_server: RemoteServer | None = None, timeout_s: float = 2.0
) -> tuple[list[int], list[str]]:
    if remote_server is not None:
        r = requests.get(
            f"{remote_server.get_url()}/available_devices", headers=remote_server.get_headers(), timeout=timeout_s
        )
        r.raise_for_status()
        data = r.json()
        return data["devices_index"], data["devices_name"]
    else:
        devices_index = cuda_visible_devices()
        return devices_index, [get_device_name(device_index) for device_index in devices_index]


def get_ram(remote_server: RemoteServer | None = None, timeout_s: float = 2.0) -> tuple[float, float]:
    if remote_server is not None:
        r = requests.get(
            f"{remote_server.get_url()}/ram",
            headers=remote_server.get_headers(),
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        return data["used_gb"], data["total_gb"]
    else:
        ram = psutil.virtual_memory()
        used_gb = (ram.total - ram.available) / (1024**3)
        total_gb = ram.total / (1024**3)
        return used_gb, total_gb


def get_vram(
    devices: list[int], remote_server: RemoteServer | None = None, timeout_s: float = 2.0
) -> tuple[float, float]:
    if remote_server is not None:
        r = requests.get(
            f"{remote_server.get_url()}/vram",
            params=[("devices", device_index) for device_index in devices],
            headers=remote_server.get_headers(),
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        return data["used_gb"], data["total_gb"]
    else:
        used_gb = 0.0
        total_gb = 0.0

        pynvml.nvmlInit()
        for device_index in devices:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device_index))
            used_gb += info.used / (1024**3)
            total_gb += info.total / (1024**3)

        return used_gb, total_gb


def current_date() -> str:
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def _get_env(var: str) -> str:
    value = os.environ.get(var)
    if value is None:
        raise RuntimeError(f"Environment variable '{var}' is not set.")
    return value


_KONFAI_DEPS: dict[str, str] = {
    "torch": "torch",
    "tqdm": "tqdm",
    "numpy": "numpy",
    "ruamel.yaml": "ruamel.yaml",
    "psutil": "psutil",
    "tensorboard": "tensorboard",
    "SimpleITK": "SimpleITK",
    "lxml": "lxml",  # often used as lxml.etree
    "h5py": "h5py",
    "nvidia-ml-py": "pynvml",  # IMPORTANT: pip != import
    "requests": "requests",
    "huggingface_hub": "huggingface_hub",
}


def _try_import(import_name: str) -> str | None:
    try:
        __import__(import_name)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def check_server(remote_server: RemoteServer, timeout_s: float = 2.0) -> tuple[bool, str]:
    try:
        r = requests.get(
            f"{remote_server.get_url()}/health",
            headers=remote_server.get_headers(),
            timeout=timeout_s,
        )

        if r.status_code == 401:
            return False, "Unauthorized (invalid or missing token)"
        if r.status_code == 403:
            return False, "Forbidden"
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        data = r.json()
        if data.get("status") != "ok":
            return False, f"Unexpected response: {data}"

        return True, "OK"

    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def check_konfai_install() -> tuple[bool, dict]:
    """
    Checks that KonfAI dependencies are importable.

    Returns a report dict:
      { ok, missing, errors, versions }
    """
    missing: list[str] = []
    errors: dict[str, str] = {}
    versions: dict[str, str] = {}

    deps = dict(_KONFAI_DEPS)
    for pip_name, import_name in deps.items():
        # best effort version lookup
        try:
            versions[pip_name] = metadata.version(pip_name)
        except metadata.PackageNotFoundError:
            versions[pip_name] = "not installed"
        except Exception:
            versions[pip_name] = "unknown"

        err = _try_import(import_name)
        if err is None:
            continue

        if versions[pip_name] == "not installed":
            missing.append(pip_name)
        else:
            errors[pip_name] = err

    return len(missing) == 0 and len(errors) == 0, {
        "missing": missing,
        "errors": errors,
        "versions": versions,
    }


class KonfAIPackagesError(RuntimeError):
    """Raised when required Python packages for KonfAI are missing/broken."""


def assert_konfai_install() -> None:
    """
    Same as check_konfai_packages(), but raises on failure.
    """
    is_konfai_install, report = check_konfai_install()
    if not is_konfai_install:
        lines = ["KonfAI dependency check failed."]

        if report["missing"]:
            lines.append("\nMissing packages:")
            lines.extend(f"  - {p}" for p in report["missing"])

        if report["errors"]:
            lines.append("\nImport/runtime errors:")
            for p, e in report["errors"].items():
                lines.append(f"  - {p}: {e}")

        raise KonfAIPackagesError("\n".join(lines))
