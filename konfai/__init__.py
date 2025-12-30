import datetime
import os
from pathlib import Path


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
