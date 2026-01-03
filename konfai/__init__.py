import datetime
import os
from importlib import metadata
from pathlib import Path

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


def cuda_visible_devices() -> list[int]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu != ""]
    else:
        import torch

        devices = []
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
    return devices


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
