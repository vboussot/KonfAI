import datetime
import os


def checkpoints_directory() -> str:
    return _get_env("KONFAI_CHECKPOINTS_DIRECTORY")


def path_to_models() -> list[str]:
    return _get_env("KONFAI_MODEL").split(os.pathsep)


def predictions_directory() -> str:
    return _get_env("KONFAI_PREDICTIONS_DIRECTORY")


def evaluations_directory() -> str:
    return _get_env("KONFAI_EVALUATIONS_DIRECTORY")


def statistics_directory() -> str:
    return _get_env("KONFAI_STATISTICS_DIRECTORY")


def setups_directory() -> str:
    return _get_env("KONFAI_SETUPS_DIRECTORY")


def config_file() -> str:
    return _get_env("KONFAI_config_file")


def konfai_state() -> str:
    return _get_env("KONFAI_STATE")


def konfai_root() -> str:
    return _get_env("KONFAI_ROOT")


def cuda_visible_devices() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")


def konfai_nb_cores() -> str:
    return _get_env("KONFAI_NB_CORES")


def current_date() -> str:
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def _get_env(var: str) -> str:
    value = os.environ.get(var)
    if value is None:
        raise RuntimeError(f"Environment variable '{var}' is not set.")
    return value
