import argparse
import importlib.util
import itertools
import json
import os
import random
import re
import resource
import shutil
import socket
import subprocess  # nosec B404
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import closing
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import psutil
import pynvml
import requests
import SimpleITK as sitk  # noqa: N813
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFolder
from packaging.requirements import Requirement
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from konfai import (
    config_file,
    cuda_visible_devices,
    evaluations_directory,
    konfai_state,
    predictions_directory,
    statistics_directory,
)


def description(model, model_ema=None, show_memory: bool = True, train: bool = True) -> str:
    def loss_desc(model):
        return (
            "("
            + " ".join(
                [
                    f"{name}({(network.optimizer.param_groups[0]['lr'] if network.optimizer else 0):.6f}) : "
                    + " ".join(
                        f"{k.split(':')[-1]}({w:.2f}) : {v:.6f}"
                        for (k, v), w in zip(
                            network.measure.get_last_values().items(), network.measure.get_last_weights().values()
                        )
                    )
                    for name, network in model.module.get_networks().items()
                    if network.measure is not None
                ]
            )
            + ")"
        )

    model_loss_desc = loss_desc(model)
    result = ""
    if len(model_loss_desc) > 2:
        result += f"Loss {model_loss_desc} "
    if model_ema is not None:
        model_ema_loss_desc = loss_desc(model_ema)
        if len(model_ema_loss_desc) > 2:
            result += f"Loss EMA {model_ema_loss_desc} "
    result += gpu_info()
    if show_memory:
        result += f" | {get_memory_info()}"
    return result


def get_module(classpath: str, default_classpath: str) -> tuple[str, str]:
    if len(classpath.split(":")) > 1:
        module = ".".join(classpath.split(":")[:-1])
        name = classpath.split(":")[-1]
    else:
        module = (
            default_classpath + ("." if len(classpath.split(".")) > 2 else "") + ".".join(classpath.split(".")[:-1])
        )
        name = classpath.split(".")[-1]
    return module, name.split("/")[0]


def get_cpu_info() -> str:
    return f"CPU ({psutil.cpu_percent(interval=0.5):.2f} %)"


def get_memory_info() -> str:
    return f"Memory ({psutil.virtual_memory().used / 2**30:.2f}G ({psutil.virtual_memory().percent:.2f} %))"


def get_memory() -> float:
    return psutil.virtual_memory().used / 2**30


def memory_forecast(memory_init: float, size: float) -> str:
    current_memory = get_memory()
    forecast = memory_init + ((current_memory - memory_init) * size)
    return f"Memory forecast ({forecast:.2f}G ({forecast / (psutil.virtual_memory().total / 2**30) * 100:.2f} %))"


def gpu_info() -> str:
    if cuda_visible_devices() == "":
        return ""

    devices = [int(i) for i in cuda_visible_devices().split(",")]
    device = devices[0]

    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return ""
    node_name = "Node: {} " + os.environ["SLURMD_NODENAME"] if "SLURMD_NODENAME" in os.environ else ""
    return f"{node_name}GPU({devices}) Memory GPU ({memory.used / 1e9:.2f}G ({memory.used / memory.total * 100:.2f} %))"


def get_max_gpu_memory(device: int | torch.device) -> float:
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return 0
    device = [int(i) for i in cuda_visible_devices().split(",")][device]
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return 0
    return float(memory.total) / (10**9)


def get_gpu_memory(device: int | torch.device) -> float:
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return 0
    device = [int(i) for i in cuda_visible_devices().split(",")][device]
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return 0
    return float(memory.used) / (10**9)


class NeedDevice:

    def __init__(self) -> None:
        super().__init__()
        self.device: torch.device

    def to(self, device: int):
        self.device = get_device(device)


def get_device(device: int):
    return device if torch.cuda.is_available() and device >= 0 else torch.device("cpu")


class State(Enum):
    TRAIN = "TRAIN"
    RESUME = "RESUME"
    TRANSFER_LEARNING = "TRANSFER_LEARNING"
    FINE_TUNING = "FINE_TUNING"
    PREDICTION = "PREDICTION"
    EVALUATION = "EVALUATION"

    def __str__(self) -> str:
        return self.value


def get_patch_slices_from_nb_patch_per_dim(
    patch_size_tmp: list[int],
    nb_patch_per_dim: list[tuple[int, bool]],
    overlap: int | None,
) -> list[tuple[slice, ...]]:
    patch_slices = []
    slices: list[list[slice]] = []
    if overlap is None:
        overlap = 0
    patch_size = []
    i = 0
    for nb in nb_patch_per_dim:
        if nb[1]:
            patch_size.append(1)
        else:
            patch_size.append(patch_size_tmp[i])
            i += 1

    for dim, nb in enumerate(nb_patch_per_dim):
        slices.append([])
        for index in range(nb[0]):
            start = (patch_size[dim] - overlap) * index
            end = start + patch_size[dim]
            slices[dim].append(slice(start, end))
    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))
    return patch_slices


def get_patch_slices_from_shape(
    patch_size: list[int], shape: list[int], overlap_tmp: int | None
) -> tuple[list[tuple[slice, ...]], list[tuple[int, bool]]]:
    if len(shape) != len(patch_size):
        raise DatasetManagerError(
            f"Dimension mismatch: 'patch_size' has {len(patch_size)} dimensions, but 'shape' has {len(shape)}.",
            f"patch_size: {patch_size}",
            f"shape: {shape}",
            "Both must have the same number of dimensions (e.g., 3D patch for 3D volume).",
        )
    patch_slices = []
    nb_patch_per_dim = []
    slices: list[list[slice]] = []
    if overlap_tmp is None:
        size = [np.ceil(a / b) for a, b in zip(shape, patch_size)]
        tmp = np.zeros(len(size), dtype=np.int_)
        for i, s in enumerate(size):
            if s > 1:
                tmp[i] = np.mod(patch_size[i] - np.mod(shape[i], patch_size[i]), patch_size[i]) // (size[i] - 1)
        overlap = tmp
    else:
        overlap = [overlap_tmp if size > 1 else 0 for size in patch_size]

    for dim in range(len(shape)):
        if overlap[dim] >= patch_size[dim]:
            raise ValueError(
                f"Overlap must be less than patch size, got overlap={overlap[dim]}",
                f" ≥ patch_size={patch_size[dim]} at dim={dim}",
            )

    for dim in range(len(shape)):
        slices.append([])
        index = 0
        while True:
            start = (patch_size[dim] - overlap[dim]) * index

            end = start + patch_size[dim]
            if end >= shape[dim]:
                end = shape[dim]
                slices[dim].append(slice(start, end))
                break
            slices[dim].append(slice(start, end))
            index += 1
        nb_patch_per_dim.append((index + 1, patch_size[dim] == 1))

    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))

    return patch_slices, nb_patch_per_dim


def _log_signal_format(array: np.ndarray) -> dict[str, np.ndarray]:
    return {str(i): channel for i, channel in enumerate(array)}


def _log_image_format(array: np.ndarray) -> np.ndarray:
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=0)

    if len(array.shape) == 3 and array.shape[0] != 1:
        array = np.expand_dims(array, axis=0)
    if len(array.shape) == 4:
        array = array[:, array.shape[1] // 2]

    array = array.astype(float)
    b = -np.min(array)
    if (np.max(array) + b) > 0:
        return (array + b) / (np.max(array) + b)
    else:
        return 0 * array


def _log_images_format(array: np.ndarray) -> np.ndarray:
    result = []
    for n in range(array.shape[0]):
        result.append(_log_image_format(array[n]))
    result = np.stack(result, axis=0)
    return result


def _log_video_format(array: np.ndarray) -> np.ndarray:
    result_list = []
    for t in range(array.shape[1]):
        result_list.append(_log_images_format(array[:, t, ...]))
    result = np.stack(result_list, axis=1)

    nb_channel = result.shape[2]
    if nb_channel < 3:
        channel_split = [result[:, :, 0, ...] for i in range(3)]
    else:
        channel_split = np.split(result, 3, axis=0)
    array = np.zeros((result.shape[0], result.shape[1], 3, *list(result.shape[3:])))
    for i, channels in enumerate(channel_split):
        array[:, :, i] = np.mean(channels, axis=0)
    return array


class DataLog(Enum):
    SIGNAL = "SIGNAL"
    IMAGE = "IMAGE"
    IMAGES = "IMAGES"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"

    def __call__(self, tb: SummaryWriter, name: str, layer: torch.Tensor, it: int):
        if self == DataLog.SIGNAL:
            return [
                tb.add_scalars(name, _log_signal_format(layer[b, :, 0]), layer.shape[0] * it + b)
                for b in range(layer.shape[0])
            ]
        elif self == DataLog.IMAGE:
            return tb.add_image(name, _log_image_format(layer[0]), it)
        elif self == DataLog.IMAGES:
            return tb.add_images(name, _log_images_format(layer), it)
        elif self == DataLog.VIDEO:
            return tb.add_video(name, _log_video_format(layer), it)
        elif self == DataLog.AUDIO:
            return tb.add_audio(name, _log_image_format(layer), it)
        else:
            raise ValueError(f"Unsupported DataLog type: {self}")


class Log:
    def __init__(self, name: str, rank: int) -> None:
        if konfai_state() == "PREDICTION":
            path = predictions_directory()
        elif konfai_state() == "EVALUATION":
            path = evaluations_directory()
        else:
            path = statistics_directory()

        self.verbose = os.environ.get("KONFAI_VERBOSE", "True") == "True"
        self.log_path = os.path.join(path, name)
        os.makedirs(self.log_path, exist_ok=True)
        self.rank = rank
        self.file = open(os.path.join(self.log_path, f"log_{rank}.txt"), "w", buffering=1)
        self.stdout_bak = sys.stdout
        self.stderr_bak = sys.stderr
        self._buffered_line = ""

    def __enter__(self):
        self.file.__enter__()
        sys.stdout = cast(TextIO, self)
        sys.stderr = cast(TextIO, self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.__exit__(exc_type, exc_val, exc_tb)
        sys.stdout = self.stdout_bak
        sys.stderr = self.stderr_bak

    def write(self, msg: str):
        if not msg:
            return

        ansi_escape = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
        msg_clean = ansi_escape.sub("", msg)
        if "\r" in msg_clean or "[A" in msg:
            msg_clean = msg_clean.split("\r")[-1].strip()
            self._buffered_line = msg_clean
        else:
            self._buffered_line = msg_clean.strip()

        if self._buffered_line:
            self.file.write(self._buffered_line + "\n")
            self.file.flush()
        if self.verbose and (self.rank == 0 or "KONFAI_CLUSTER" in os.environ):
            if sys.__stdout__ is not None:
                sys.__stdout__.write(msg)
                sys.__stdout__.flush()

    def flush(self):
        self.file.flush()

    def isatty(self):
        return False

    def fileno(self):
        if sys.__stdout__ is None:
            raise RuntimeError("sys.__stdout__ is None, cannot get fileno")
        return sys.__stdout__.fileno()


class TensorBoard:

    def __init__(self, name: str) -> None:
        self.process: subprocess.Popen | None = None
        self.name = name

    def __enter__(self):
        if "KONFAI_TENSORBOARD_PORT" in os.environ:
            tensorboard_exe = shutil.which("tensorboard")
            if tensorboard_exe is None:
                raise RuntimeError("TensorBoard executable not found in PATH.")

            logdir = (
                predictions_directory() if konfai_state() == "PREDICTION" else statistics_directory() + self.name + "/"
            )

            port = os.environ.get("KONFAI_TENSORBOARD_PORT")
            if not port or not port.isdigit():
                raise ValueError("Invalid or missing KONFAI_TENSORBOARD_PORT.")

            command = [
                tensorboard_exe,
                "--logdir",
                logdir,
                "--port",
                port,
                "--bind_all",
            ]
            self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosec B603
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("10.255.255.255", 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = "127.0.0.1"
            finally:
                s.close()
            print(f"Tensorboard : http://{ip}:{os.environ['KONFAI_TENSORBOARD_PORT']}/")
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()


class DistributedObject(ABC):

    def __init__(self, name: str) -> None:
        self.port = find_free_port()
        self.dataloader: list[list[DataLoader]]
        self.manual_seed: int | None = None
        self.name = name
        self.size = 1

    @abstractmethod
    def setup(self, world_size: int):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        cleanup()

    @abstractmethod
    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        pass

    @staticmethod
    def get_measure(
        world_size: int,
        global_rank: int,
        gpu: int,
        models: dict[str, torch.nn.Module],
        n: int,
    ) -> dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]]:
        data = {}
        for label, model in models.items():
            for name, network in model.get_networks().items():
                if network.measure is not None:
                    data[f"{name}{label}"] = (
                        network.measure.format_loss(True, n),
                        network.measure.format_loss(False, n),
                    )
        outputs = synchronize_data(world_size, gpu, data)
        result: dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]] = {}
        if global_rank == 0:
            for output in outputs:
                for k, v in output.items():
                    for t in range(len(v)):
                        for u, n in v[t].items():
                            if k not in result:
                                result[k] = ({}, {})
                            if u not in result[k][t]:
                                result[k][t][u] = (n[0], 0)  # type: ignore[index]
                            result[k][t][u] = (
                                result[k][t][u][0],
                                result[k][t][u][1] + n[1] / world_size,  # type: ignore[index]
                            )
        return result

    def __call__(self, rank: int | None = None) -> None:
        world_size = len(self.dataloader)
        global_rank, local_rank = setup_gpu(world_size, self.port, rank)
        if global_rank is None or local_rank is None:
            return
        with Log(self.name, global_rank):
            if torch.cuda.is_available():
                pynvml.nvmlInit()
            if self.manual_seed is not None:
                np.random.seed(self.manual_seed * world_size + global_rank)
                random.seed(self.manual_seed * world_size + global_rank)
                torch.manual_seed(self.manual_seed * world_size + global_rank)
            torch.backends.cudnn.benchmark = self.manual_seed is None
            torch.backends.cudnn.deterministic = self.manual_seed is not None
            # torch.backends.cuda.matmul.fp32_precision = "tf32"
            # torch.backends.cudnn.conv.fp32_precision = "tf32"
            dataloaders = self.dataloader[global_rank]
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            try:
                self.run_process(world_size, global_rank, local_rank, dataloaders)
            finally:
                cleanup()
                if torch.cuda.is_available():
                    pynvml.nvmlShutdown()


def match_supported(file: Path) -> bool:
    lower = file.name.lower()
    return any(lower.endswith("." + ext) for ext in SUPPORTED_EXTENSIONS)


def list_supported_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if match_supported(path) else []

    files = []
    for f in path.rglob("*"):
        if f.is_file() and match_supported(f):
            files.append(f)
    return files


def setup_apps(
    parser: argparse.ArgumentParser, user_dir: Path, tmp_dir_default: Path
) -> tuple[partial[DistributedObject], Path, Callable[[], None]]:
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    except (ImportError, OSError, ValueError):
        pass

    def get_path(path: Path) -> Path:
        if path.is_absolute():
            result_path = path
        else:
            result_path = user_dir / path
        return result_path

    parser = argparse.ArgumentParser(prog="konfai-apps", description="KonfAI Apps – Apps for Medical AI Models")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------
    # 1) INFERENCE
    # -----------------
    infer_p = subparsers.add_parser("infer", help="Run inference using a KonfAI App.")

    infer_p.add_argument("app", type=str, help="KonfAI App name")
    infer_p.add_argument(
        "-i",
        "--input",
        type=lambda p: get_path(Path(p)),
        required=True,
        help="Input path: either a single volume file OR a dataset directory",
    )
    infer_p.add_argument(
        "-o", "--output", type=lambda p: get_path(Path(p)), default="./Predictions", help="Optional output volume path"
    )
    infer_p.add_argument("--ensemble", type=int, default=0, help="Size of model ensemble")
    infer_p.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations")
    infer_p.add_argument("--mc", type=int, default=0, help="Monte Carlo dropout samples")
    infer_p.add_argument(
        "-g",
        "--gpu",
        type=str,
        default=(os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""),
        help="GPU list (e.g. '0' or '0,1'). Leave empty for CPU.",
    )
    infer_p.add_argument("--cpu", type=int, default=1, help="Number of CPU cores to use when --gpu is empty.")
    infer_p.add_argument(
        "--tmp_dir", type=lambda p: Path(p).absolute(), default=tmp_dir_default, help="Use a temporary directory."
    )
    infer_p.add_argument("--quiet", action="store_false", help="Suppress console output.")

    # -----------------
    # 2) EVALUATION
    # -----------------
    eval_p = subparsers.add_parser("eval", help="Evaluate a KonfAI App using ground-truth labels.")
    eval_p.add_argument("app", type=str, help="KonfAI App name")

    eval_p.add_argument(
        "-i",
        "--input",
        type=lambda p: get_path(Path(p)),
        required=True,
        help="Input path: either a single volume file OR a dataset directory",
    )

    eval_p.add_argument("--gt", type=lambda p: get_path(Path(p)), required=True, help="Ground-truth path")
    eval_p.add_argument("--mask", type=lambda p: get_path(Path(p)), help="Optional evaluation mask path")
    eval_p.add_argument(
        "-o", "--output", type=lambda p: get_path(Path(p)), default="./Evaluations", help="Optional output volume path"
    )
    eval_p.add_argument(
        "--tmp_dir", type=lambda p: Path(p).absolute(), default=tmp_dir_default, help="Use a temporary directory."
    )

    eval_p.add_argument(
        "-g",
        "--gpu",
        type=str,
        default=(os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""),
        help="GPU list (e.g. '0' or '0,1'). Leave empty for CPU.",
    )
    eval_p.add_argument("--cpu", type=int, default=1, help="Number of CPU cores to use when --gpu is empty.")
    eval_p.add_argument("--quiet", action="store_false", help="Suppress console output.")

    # -----------------
    # 3) UNCERTAINTY
    # -----------------
    unc_p = subparsers.add_parser("uncertainty", help="Compute model uncertainty for a KonfAI App.")
    unc_p.add_argument("app", type=str, help="KonfAI App name")

    unc_p.add_argument(
        "-i",
        "--input",
        type=lambda p: get_path(Path(p)),
        required=True,
        help=("Prediction stack: either a single multi-sample prediction file "),
    )

    unc_p.add_argument(
        "-o", "--output", type=lambda p: get_path(Path(p)), default="./Evaluations", help="Optional output volume path"
    )
    unc_p.add_argument(
        "--tmp_dir", type=lambda p: Path(p).absolute(), default=tmp_dir_default, help="Use a temporary directory."
    )

    unc_p.add_argument(
        "-g",
        "--gpu",
        type=str,
        default=(os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""),
        help="GPU list (e.g. '0' or '0,1'). Leave empty for CPU.",
    )
    unc_p.add_argument("--cpu", type=int, default=1, help="Number of CPU cores to use when --gpu is empty.")
    unc_p.add_argument("--quiet", action="store_false", help="Suppress console output.")

    # -----------------
    # 4) Pipeline
    # -----------------
    pipe_p = subparsers.add_parser(
        "pipeline", help="Run inference and optionally evaluation and uncertainty in a single command."
    )

    pipe_p.add_argument("app", type=str, help="KonfAI App name")

    pipe_p.add_argument("-i", "--input", type=str, required=True, help="Input path: volume file or directory.")

    pipe_p.add_argument("--mc", type=int, default=0, help="Number of Monte Carlo dropout samples.")

    pipe_p.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations.")

    pipe_p.add_argument("--ensemble", type=int, default=0, help="Number of models in ensemble.")

    # optional eval
    pipe_p.add_argument("--with-eval", action="store_true", help="Also run evaluation (requires --gt).")

    pipe_p.add_argument("-g", "--gt", type=str, help="Ground-truth path (required when --with-eval is set).")

    # optional uncertainty
    pipe_p.add_argument("--with-uncertainty", action="store_true", help="Also estimate uncertainty using sampling.")

    pipe_p.add_argument("-o", "--output", type=str, help="Output prediction volume (mean prediction).")

    # -----------------
    # 5) FINE-TUNE
    # -----------------
    ft_p = subparsers.add_parser("fine-tune", help="Fine-tune a KonfAI App on a dataset.")
    ft_p.add_argument("app", type=str, help="KonfAI App name")

    ft_p.add_argument("-d", "--dataset", type=str, required=True, help="Path to training dataset")
    ft_p.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")

    parser.add_argument("--version", action="version", version=importlib.metadata.version("konfai"))

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["KONFAI_NB_CORES"] = str(args.cpu)
    os.environ["KONFAI_WORKERS"] = str(0)

    os.environ["KONFAI_CHECKPOINTS_DIRECTORY"] = "./Checkpoints/"
    os.environ["KONFAI_PREDICTIONS_DIRECTORY"] = str(args.output) + "/"
    os.environ["KONFAI_EVALUATIONS_DIRECTORY"] = str(args.output) + "/"
    os.environ["KONFAI_STATISTICS_DIRECTORY"] = "./Statistics/"
    os.environ["KONFAI_SETUPS_DIRECTORY"] = "./Setups/"
    os.environ["KONFAI_OVERWRITE"] = str(True)
    os.environ["KONFAI_CONFIG_MODE"] = "Done"

    os.environ["KONFAI_VERBOSE"] = str(args.quiet)

    model_hf = ModelHF(args.app)

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.chdir(str(args.tmp_dir))
    if str(args.tmp_dir) not in sys.path:
        sys.path.insert(0, str(args.tmp_dir))

    if args.command != "fine-tune":
        from konfai.utils.dataset import Dataset

        dataset = Dataset("Dataset", "mha")
        for idx, file in enumerate(list_supported_files(args.input)):
            dataset.write("Volume", f"P{idx:03d}", sitk.ReadImage(str(file)))

    if args.command == "infer":
        models_path = model_hf.install_inference(args.tta, args.ensemble, args.mc)

        os.environ["KONFAI_ROOT"] = "Predictor"
        os.environ["KONFAI_config_file"] = "Prediction.yml"
        os.environ["KONFAI_MODEL"] = ":".join(models_path)
        os.environ["KONFAI_STATE"] = str(State.PREDICTION)

        def save() -> None:
            if os.path.exists("Predictions"):
                shutil.copytree("Predictions", args.output, dirs_exist_ok=True)

        from konfai.predictor import Predictor

        classname = Predictor
    elif args.command == "eval":

        for idx, file in enumerate(list_supported_files(args.gt)):
            dataset.write("Reference", f"P{idx:03d}", sitk.ReadImage(str(file)))
        if not args.mask:
            names = dataset.get_names("Volume")
            for name in names:
                data, attr = dataset.read_data("Volume", name)
                dataset.write("Mask", name, np.ones_like(data), attr)
        else:
            for idx, file in enumerate(list_supported_files(args.mask)):
                dataset.write("Mask", f"P{idx:03d}", sitk.ReadImage(str(file)))

        model_hf.install_evaluation()
        os.environ["KONFAI_ROOT"] = "Evaluator"

        os.environ["KONFAI_config_file"] = "Evaluation.yml"
        os.environ["KONFAI_STATE"] = str(State.EVALUATION)
        from konfai.evaluator import Evaluator

        def save() -> None:
            if os.path.exists("./Evaluations"):
                shutil.copytree("./Evaluations", args.output, dirs_exist_ok=True)

        classname = Evaluator

    elif args.command == "uncertainty":
        model_hf.install_uncertainty()
        os.environ["KONFAI_ROOT"] = "Evaluator"

        os.environ["KONFAI_config_file"] = "Uncertainty.yml"
        os.environ["KONFAI_STATE"] = str(State.EVALUATION)
        from konfai.evaluator import Evaluator

        def save() -> None:
            if os.path.exists("./Evaluations"):
                shutil.copytree("./Evaluations", args.output, dirs_exist_ok=True)

        classname = Evaluator
    elif args.command == "fine-tune":
        models_path = model_hf.install_fine_tune()
        os.environ["KONFAI_ROOT"] = "Trainer"

        os.environ["KONFAI_config_file"] = "FineTuning.yml"
        os.environ["KONFAI_MODEL"] = ":".join(models_path)
        os.environ["KONFAI_STATE"] = str(State.FINE_TUNING)

        def save() -> None:
            pass

        from konfai.trainer import Trainer

        classname = Trainer
    else:
        raise ValueError(f"Unknown command: {args.command}")

    return partial(classname, config=config_file()), args.tmp_dir, save


def setup(parser: argparse.ArgumentParser) -> DistributedObject:
    # KONFAI arguments
    konfai = parser.add_argument_group("KonfAI arguments")
    konfai.add_argument("type", type=State, choices=list(State))
    konfai.add_argument("-y", action="store_true", help="Accept overwrite")
    konfai.add_argument("-tb", action="store_true", help="Start TensorBoard")
    konfai.add_argument("-c", "--config", type=str, default="None", help="Configuration file location")
    konfai.add_argument(
        "-g",
        "--gpu",
        type=str,
        default=(os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""),
        help="List of GPU",
    )
    konfai.add_argument("-cpu", "--cpu", type=str, default="1", help="Number of cores")
    konfai.add_argument(
        "--num-workers",
        "--num_workers",
        default=0,
        type=int,
        help="Number of workers per DataLoader & GPU",
    )
    konfai.add_argument(
        "-models_dir",
        "--MODELS_DIRECTORY",
        type=str,
        default="./Models/",
        help="Models location",
    )
    konfai.add_argument(
        "-checkpoints_dir",
        "--CHECKPOINTS_DIRECTORY",
        type=str,
        default="./Checkpoints/",
        help="Checkpoints location",
    )
    konfai.add_argument("-model", "--MODEL", type=str, default="", help="URL Model")
    konfai.add_argument(
        "-predictions_dir",
        "--PREDICTIONS_DIRECTORY",
        type=str,
        default="./Predictions/",
        help="Predictions location",
    )
    konfai.add_argument(
        "-evaluation_dir",
        "--EVALUATIONS_DIRECTORY",
        type=str,
        default="./Evaluations/",
        help="Evaluations location",
    )
    konfai.add_argument(
        "-statistics_dir",
        "--STATISTICS_DIRECTORY",
        type=str,
        default="./Statistics/",
        help="Statistics location",
    )
    konfai.add_argument(
        "-setups_dir",
        "--SETUPS_DIRECTORY",
        type=str,
        default="./Setups/",
        help="Setups location",
    )

    konfai.add_argument("-log", action="store_true", help="Enable logging to a file")
    konfai.add_argument("-quiet", action="store_false", help="Suppress console output for a quieter execution")

    konfai.add_argument("--version", action="version", version=importlib.metadata.version("konfai"))
    args = parser.parse_args()
    config = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    os.environ["KONFAI_NB_CORES"] = config["cpu"]

    os.environ["KONFAI_WORKERS"] = str(config["num_workers"])
    os.environ["KONFAI_CHECKPOINTS_DIRECTORY"] = config["CHECKPOINTS_DIRECTORY"]
    os.environ["KONFAI_PREDICTIONS_DIRECTORY"] = config["PREDICTIONS_DIRECTORY"]
    os.environ["KONFAI_EVALUATIONS_DIRECTORY"] = config["EVALUATIONS_DIRECTORY"]
    os.environ["KONFAI_STATISTICS_DIRECTORY"] = config["STATISTICS_DIRECTORY"]
    os.environ["KONFAI_SETUPS_DIRECTORY"] = config["SETUPS_DIRECTORY"]

    os.environ["KONFAI_STATE"] = str(config["type"])

    os.environ["KONFAI_MODEL"] = config["MODEL"]

    os.environ["KONFAI_OVERWRITE"] = str(config["y"])
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    if config["tb"]:
        os.environ["KONFAI_TENSORBOARD_PORT"] = str(find_free_port())

    os.environ["KONFAI_VERBOSE"] = str(config["quiet"])

    if config["config"] == "None":
        if config["type"] is State.PREDICTION:
            os.environ["KONFAI_config_file"] = "Prediction.yml"
        elif config["type"] is State.EVALUATION:
            os.environ["KONFAI_config_file"] = "Evaluation.yml"
        else:
            os.environ["KONFAI_config_file"] = "Config.yml"
    else:
        os.environ["KONFAI_config_file"] = config["config"]
    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if config["type"] is State.PREDICTION:
        from konfai.predictor import Predictor

        os.environ["KONFAI_ROOT"] = "Predictor"
        return Predictor(config=config_file())
    elif config["type"] is State.EVALUATION:
        from konfai.evaluator import Evaluator

        os.environ["KONFAI_ROOT"] = "Evaluator"
        return Evaluator(config=config_file())
    else:
        from konfai.trainer import Trainer

        os.environ["KONFAI_ROOT"] = "Trainer"
        return Trainer(config=config_file())


def setup_gpu(world_size: int, port: int, rank: int | None = None) -> tuple[int | None, int | None]:
    try:
        nodelist = os.getenv("SLURM_JOB_NODELIST")
        if nodelist is None:
            raise RuntimeError("SLURM_JOB_NODELIST is not set.")
        scontrol_path = shutil.which("scontrol")
        if scontrol_path is None:
            raise FileNotFoundError("scontrol not found in PATH")
        host_name = subprocess.check_output(
            [scontrol_path, "show", "hostnames", nodelist], text=True, stderr=subprocess.DEVNULL
        ).strip()  # nosec B603
    except Exception:
        host_name = "localhost"
    if rank is None:
        import submitit

        job_env = submitit.JobEnvironment()
        global_rank = job_env.global_rank
        local_rank = job_env.local_rank
    else:
        global_rank = rank
        local_rank = rank
    if global_rank >= world_size:
        return None, None
    # print("tcp://{}:{}".format(host_name, port))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dist.init_process_group(
            "nccl",
            rank=global_rank,
            init_method=f"tcp://{host_name}:{port}",
            world_size=world_size,
        )
    else:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{host_name}:{port}",
                rank=global_rank,
                world_size=world_size,
            )
    return global_rank, local_rank


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def synchronize_data(world_size: int, gpu: int, data: Any) -> list[Any]:
    if torch.cuda.is_available():
        outputs: list[dict[str, tuple[dict[str, float], dict[str, float]]] | None] = [None] * world_size
        torch.cuda.set_device(gpu)
        dist.all_gather_object(outputs, data)
    else:
        outputs = [data]
    return outputs


def _resample(data: torch.Tensor, size: list[int]) -> torch.Tensor:
    if data.dtype == torch.uint8:
        mode = "nearest"
    elif len(data.shape) < 4:
        mode = "bilinear"
    else:
        mode = "trilinear"
    return (
        F.interpolate(
            data.type(torch.float32).unsqueeze(0),
            size=tuple(reversed(size)),
            mode=mode,
        )
        .squeeze(0)
        .type(data.dtype)
    )


def _affine_matrix(matrix: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.cat((matrix, translation.unsqueeze(0).T), dim=1),
            torch.tensor([[0, 0, 0, 1]]),
        ),
        dim=0,
    )


def _resample_affine(data: torch.Tensor, matrix: torch.Tensor):
    if data.dtype == torch.uint8:
        mode = "nearest"
    else:
        mode = "bilinear"
    return (
        F.grid_sample(
            data.unsqueeze(0).type(torch.float32),
            F.affine_grid(
                matrix[:, :-1, ...].type(torch.float32),
                [1] + list(data.shape),
                align_corners=True,
            ),
            align_corners=True,
            mode=mode,
            padding_mode="reflection",
        )
        .squeeze(0)
        .type(data.dtype)
    )


def download_url(model_name: str, url: str) -> str:
    spec = importlib.util.find_spec("konfai")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("Could not locate 'konfai' package")
    locations = spec.submodule_search_locations
    if not isinstance(locations, list) or not locations:
        raise ImportError("No valid submodule_search_locations found")
    base_path = Path(locations[0]) / "metric" / "models"
    os.makedirs(base_path, exist_ok=True)

    subdirs = Path(model_name).parent
    model_dir = base_path / subdirs
    model_dir.mkdir(exist_ok=True)
    filetmp = model_dir / ("tmp_" + str(Path(model_name).name))
    file = model_dir / Path(model_name).name
    if file.exists():
        return str(file)

    try:
        print(f"[FOCUS] Downloading {model_name} to {file}")
        with requests.get(url + model_name, stream=True, timeout=10) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(filetmp, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {model_name}",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        shutil.copy2(filetmp, file)
        print("Download finished.")
    except Exception as e:
        raise e
    finally:
        if filetmp.exists():
            os.remove(filetmp)
    return str(file)


SUPPORTED_EXTENSIONS = [
    "mha",
    "mhd",  # MetaImage
    "nii",
    "nii.gz",  # NIfTI
    "nrrd",
    "nrrd.gz",  # NRRD
    "gipl",
    "gipl.gz",  # GIPL
    "hdr",
    "img",  # Analyze
    "dcm",  # DICOM (si GDCM activé)
    "tif",
    "tiff",  # TIFF
    "png",
    "jpg",
    "jpeg",
    "bmp",  # 2D formats
    "h5",
    "itk.txt",
    ".fcsv",
    ".xml",
    ".vtk",
    ".npy",
]


class KonfAIError(Exception):

    def __init__(self, type_error: str, *messages: str) -> None:
        super().__init__(
            f"\n[{type_error}] {messages[0]}" + ("\n" if len(messages) > 0 else "") + "\n→\t".join(messages[1:])
        )


class ConfigError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("Config", *message)


class DatasetManagerError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("DatasetManager", *message)


class MeasureError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("Measure", *message)


class TrainerError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("Trainer", *message)


class AugmentationError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("Augmentation", *message)


class EvaluatorError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("Evaluator", *message)


class PredictorError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("Predictor", *message)


class TransformError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("Transform", *message)


class RepositoryHFError(KonfAIError):

    def __init__(self, *message) -> None:
        super().__init__("HF", *message)


def get_available_models_on_hf_repo(repo_id: str) -> list[str]:
    api = HfApi()
    model_names = []
    try:
        tree = api.list_repo_tree(repo_id=repo_id)
    except Exception as e:
        raise RepositoryHFError(f"Unable to access repository '{repo_id}': {e}")
    for entry in tree:
        model_name = entry.path
        if isinstance(entry, RepoFolder) and is_model_repo(repo_id, model_name)[0]:
            model_names.append(model_name)
    return model_names


def is_model_repo(repo_id: str, model_name: str) -> tuple[bool, str, int]:
    """
    Check whether the Hugging Face repository structure is valid for KonfAI.
    Required files:
    - at least one .pt file in the model folder
    - a Prediction.yml file
    - a metadata.json file
    """
    api = HfApi()
    fold_names = []
    found_prediction_file = False
    found_metadata_file = False

    try:
        tree = api.list_repo_tree(repo_id=repo_id, path_in_repo=model_name)
    except Exception as e:
        return False, f"Unable to access repository '{repo_id}': {e}", 0

    for file in tree:
        if file.path.endswith(".pt"):
            fold_names.append(file.path)
        elif file.path.endswith("Prediction.yml"):
            found_prediction_file = True
        elif file.path.endswith("metadata.json"):
            found_metadata_file = True

    if len(fold_names) == 0:
        return False, f"No '.pt' model files were found in '{repo_id}/{model_name}'.", 0

    if not found_prediction_file:
        return False, f"Missing 'Prediction.yml' in '{repo_id}/{model_name}'.", 0

    if not found_metadata_file:
        return False, f"Missing 'metadata.json' in '{repo_id}/{model_name}'.", 0
    return True, "", len(fold_names)


class ModelHF:

    def __init__(self, model_name: str):
        if model_name and len(model_name.split(":")) != 2:
            raise RepositoryHFError(
                f"Invalid model name format in --config: '{model_name}'. "
                "Expected format is 'REPO_ID:NAME', e.g. 'VBoussot/ImpactSynth:MR'."
            )
        self.repo_id, self.model_name = model_name.split(":")
        _, err_message, self._number_of_models = is_model_repo(self.repo_id, self.model_name)
        if err_message:
            raise RepositoryHFError(err_message)
        model_metadata = self._read_metadata()

        required_keys = ["description", "short_description", "tta", "mc_dropout", "display_name"]
        missing = [k for k in required_keys if k not in model_metadata]
        if missing:
            raise RepositoryHFError(f"Missing keys in metadata.json: {', '.join(missing)}")

        self._description = str(model_metadata["description"])
        self._short_description = str(model_metadata["short_description"])

        try:
            self._maximum_tta = int(model_metadata["tta"])
        except Exception:
            raise RepositoryHFError("The field 'tta' must be an integer.")

        try:
            self._mc_dropout = int(model_metadata["mc_dropout"])
        except Exception:
            raise RepositoryHFError("The field 'mc_dropout' must be an integer.")

        self._display_name = str(model_metadata["display_name"])

    def has_capabilities(self) -> tuple[bool, bool]:
        api = HfApi()
        tree = api.list_repo_tree(repo_id=self.repo_id, path_in_repo=self.model_name)
        evaluation_support = False
        uncertainty_support = False

        for file in tree:
            if file.path.endswith("Evaluation.yml"):
                evaluation_support = True
            elif file.path.endswith("Uncertainty.yml"):
                uncertainty_support = True
        return evaluation_support, uncertainty_support

    def _read_metadata(self) -> dict[str, str]:
        metadata_file_path = hf_hub_download(
            repo_id=self.repo_id, filename=f"{self.model_name}/metadata.json", repo_type="model", revision=None
        )  # nosec B615
        with open(metadata_file_path, encoding="utf-8") as f:
            model_metadata = json.load(f)
        return model_metadata

    def set_number_of_augmentation(self, inference_file_path: str, new_value: int) -> None:
        new_value = int(np.clip(new_value, 0, self._maximum_tta))
        yaml = YAML()
        with open(inference_file_path) as f:
            data = yaml.load(f)

        if new_value > 0:
            tmp = data["Predictor"]["Dataset"]["augmentations"]
            if "DataAugmentation_0" in tmp:
                tmp["DataAugmentation_0"]["nb"] = new_value

        else:
            data["Predictor"]["Dataset"]["augmentations"] = {}
        with open(inference_file_path, "w") as f:
            yaml.dump(data, f)

    def download_inference(self, number_of_model: int) -> tuple[list[str], str, list[str]]:
        api = HfApi()
        models_path = []
        codes_path = []
        i = 0
        for filename in api.list_repo_tree(repo_id=self.repo_id, path_in_repo=self.model_name):
            if filename.path.endswith(".pt"):
                i += 1
                if i > number_of_model:
                    continue

            file_path = hf_hub_download(
                repo_id=self.repo_id, filename=filename.path, repo_type="model", revision=None
            )  # nosec B615
            if "Prediction.yml" in filename.path:
                inference_file_path = file_path
            elif filename.path.endswith(".pt"):
                models_path.append(file_path)
            elif "requirements.txt" in filename.path:
                with open(file_path, encoding="utf-8") as f:
                    required_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

                    installed = {
                        dist.metadata["Name"].lower(): dist.version
                        for dist in importlib.metadata.distributions()
                        if dist.metadata.get("Name")
                    }

                    for line in required_lines:
                        req = Requirement(line)
                        name = req.name.lower()
                        installed_version_str = installed.get(name)
                        missing_or_outdated = []
                        if installed_version_str is None:
                            missing_or_outdated.append(line)
                            continue

                        if req.specifier:
                            if not req.specifier.contains(installed_version_str, prereleases=True):
                                missing_or_outdated.append(line)

                    if missing_or_outdated:
                        try:
                            subprocess.check_call(
                                [sys.executable, "-m", "pip", "install", *missing_or_outdated],  # nosec B603
                            )
                        except subprocess.CalledProcessError as e:
                            raise RepositoryHFError(f"Failed to install packages: {e}") from e
            elif "metadata.json" not in filename.path:
                codes_path.append(file_path)
        return models_path, inference_file_path, codes_path

    def download_evaluation(self) -> tuple[str, list[str]]:
        api = HfApi()
        codes_path = []
        for filename in api.list_repo_tree(repo_id=self.repo_id, path_in_repo=self.model_name):
            if "Evaluation.yml" in filename.path:
                evaluation_file_path = hf_hub_download(
                    repo_id=self.repo_id, filename=filename.path, repo_type="model", revision=None
                )  # nosec B615
            elif filename.path.endswith(".py"):
                file_path = hf_hub_download(
                    repo_id=self.repo_id, filename=filename.path, repo_type="model", revision=None
                )  # nosec B615
                codes_path.append(file_path)
        return evaluation_file_path, codes_path

    def download_uncertainty(self) -> tuple[str, list[str]]:
        api = HfApi()
        codes_path = []
        for filename in api.list_repo_tree(repo_id=self.repo_id, path_in_repo=self.model_name):
            if "Uncertainty.yml" in filename.path:
                uncertainty_file_path = hf_hub_download(
                    repo_id=self.repo_id, filename=filename.path, repo_type="model", revision=None
                )  # nosec B615
            elif filename.path.endswith(".py"):
                file_path = hf_hub_download(
                    repo_id=self.repo_id, filename=filename.path, repo_type="model", revision=None
                )  # nosec B615
                codes_path.append(file_path)
        return uncertainty_file_path, codes_path

    def install_inference(
        self, number_of_augmentation: int, number_of_model: int, number_of_mc_dropout: int
    ) -> list[str]:
        if not number_of_model:
            number_of_model = self._number_of_models
        else:
            try:
                number_of_model = int(number_of_model)
            except ValueError:
                raise RepositoryHFError(
                    f"Invalid value provided for '--MODEL': '{number_of_model}'. ",
                    "The value must be an integer (e.g. 1, 2, 3).",
                )
        models_path, inference_file_path, codes_path = self.download_inference(number_of_model)

        shutil.copy2(inference_file_path, "./Prediction.yml")
        self.set_number_of_augmentation("./Prediction.yml", number_of_augmentation)
        for code_path in codes_path:
            if code_path.endswith(".py"):
                shutil.copy2(code_path, "./{}".format(code_path.split("/")[-1]))

        return models_path

    def install_evaluation(self) -> None:
        evaluation_file_path, codes_path = self.download_evaluation()
        shutil.copy2(evaluation_file_path, "./Evaluation.yml")
        for code_path in codes_path:
            if code_path.endswith(".py"):
                shutil.copy2(code_path, "./{}".format(code_path.split("/")[-1]))

    def install_uncertainty(self) -> None:
        uncertainty_file_path, codes_path = self.download_uncertainty()
        shutil.copy2(uncertainty_file_path, "./Uncertainty.yml")
        for code_path in codes_path:
            if code_path.endswith(".py"):
                shutil.copy2(code_path, "./{}".format(code_path.split("/")[-1]))

    def install_fine_tune(self):
        pass

    def get_display_name(self):
        return self._display_name

    def get_maximum_tta(self):
        return self._maximum_tta

    def get_mc_dropout(self):
        return self._mc_dropout

    def get_number_of_models(self):
        return self._number_of_models

    def get_description(self):
        return self._description

    def get_short_description(self):
        return self._short_description
