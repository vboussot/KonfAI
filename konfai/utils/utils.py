import importlib.util
import inspect
import itertools
import json
import os
import random
import re
import shutil
import socket
import subprocess  # nosec B404
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import closing
from enum import Enum
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, ParamSpec, TextIO, TypeVar, cast

import numpy as np
import psutil
import pynvml
import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFolder
from packaging.requirements import Requirement
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from konfai import (
    cuda_visible_devices,
    evaluations_directory,
    konfai_state,
    predictions_directory,
    statistics_directory,
)

P = ParamSpec("P")
R = TypeVar("R")


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


def get_module(classpath: str, default_classpath: str) -> tuple[ModuleType, str]:
    if len(classpath.split(":")) > 1:
        module_name = ".".join(classpath.split(":")[:-1])
        name = classpath.split(":")[-1]
    else:
        module_name = (
            default_classpath + ("." if len(classpath.split(".")) > 2 else "") + ".".join(classpath.split(".")[:-1])
        )
        name = classpath.split(".")[-1]
    os.environ["KONFAI_CONFIG_MODE"] = "Import"
    module = importlib.import_module(module_name)
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    return module, name.split("/")[0]


def get_cpu_info() -> str:
    return f"CPU ({psutil.cpu_percent(interval=0.5):.2f} %)"


def get_memory_info() -> str:
    return f"Memory ({psutil.virtual_memory().used / 2**30:.2f}G ({psutil.virtual_memory().percent:.2f} %))"


def get_memory() -> float:
    return psutil.virtual_memory().used / 2**30


def memory_forecast(memory_init: float, i: int, size: int) -> str:
    current_memory = get_memory()
    forecast = memory_init + ((current_memory - memory_init) * size / i) if i > 0 else memory_init
    return f"Memory forecast ({forecast:.2f}G ({forecast / (psutil.virtual_memory().total / 2**30) * 100:.2f} %))"


def gpu_info() -> str:
    if len(cuda_visible_devices()) == 0:
        return ""

    devices = [int(i) for i in cuda_visible_devices()]
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
    device = cuda_visible_devices()[device]
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
    device = cuda_visible_devices()[device]
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

    if patch_size is None or all(p == 0 for p in patch_size):
        patch_size = shape
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


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class MinimalLog:
    def __init__(self, rank: int = 0) -> None:
        self._stdout_bak = sys.stdout
        self._stderr_bak = sys.stderr
        self._buffered_line = ""
        self.verbose = os.environ.get("KONFAI_VERBOSE", "True") == "True"
        self.rank = rank

    def __enter__(self):
        sys.stdout = cast(TextIO, self)
        sys.stderr = cast(TextIO, self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout_bak
        sys.stderr = self._stderr_bak

    def write(self, msg: str):
        if not msg:
            return
        msg_clean = ANSI_ESCAPE_RE.sub("", msg)
        if "\r" in msg_clean or "[A" in msg:
            msg_clean = msg_clean.split("\r")[-1].strip()
            self._buffered_line = msg_clean
        else:
            self._buffered_line = msg_clean.strip()

        if self.verbose and (self.rank == 0 or "KONFAI_CLUSTER" in os.environ):
            self._stdout_bak.write(msg)
            self._stdout_bak.flush()

    def flush(self):
        self._stdout_bak.flush()

    def fileno(self):
        if sys.__stdout__ is None:
            raise RuntimeError("sys.__stdout__ is None, cannot get fileno")
        return sys.__stdout__.fileno()


class Log(MinimalLog):
    def __init__(self, name: str, rank: int) -> None:
        super().__init__(rank)
        if konfai_state() == "PREDICTION":
            path = predictions_directory()
        elif konfai_state() == "EVALUATION":
            path = evaluations_directory()
        else:
            path = statistics_directory()
        self.log_path = path / name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path / f"log_{rank}.txt", "w", buffering=1)

    def __enter__(self):
        super().__enter__()
        self.file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.file.__exit__(exc_type, exc_val, exc_tb)

    def write(self, msg: str):
        super().write(msg)
        if self._buffered_line:
            self.file.write(self._buffered_line + "\n")
            self.file.flush()

    def flush(self):
        super().flush()
        self.file.flush()


class TensorBoard:

    def __init__(self, name: str) -> None:
        self.process: subprocess.Popen | None = None
        self.name = name

    def __enter__(self):
        if "KONFAI_TENSORBOARD_PORT" in os.environ:
            tensorboard_exe = shutil.which("tensorboard")
            if tensorboard_exe is None:
                raise RuntimeError("TensorBoard executable not found in PATH.")

            logdir = predictions_directory() if konfai_state() == "PREDICTION" else statistics_directory() / self.name

            port = os.environ.get("KONFAI_TENSORBOARD_PORT")
            if not port or not port.isdigit():
                raise ValueError("Invalid or missing KONFAI_TENSORBOARD_PORT.")

            command = [
                tensorboard_exe,
                "--logdir",
                str(logdir),
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


def run_distributed_app(
    func: Callable[..., DistributedObject],
) -> Callable[..., None]:

    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        params = sig.parameters
        kwargs_fun = {k: v for k, v in kwargs.items() if k in params}

        bound = sig.bind_partial(*args, **kwargs_fun)
        bound.apply_defaults()
        gpu = bound.arguments.get("gpu", [])
        cpu = bound.arguments.get("cpu", 1)
        if cpu is None:
            cpu = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu if i >= 0])
        os.environ["KONFAI_OVERWRITE"] = str(bound.arguments.get("overwrite"))

        os.environ["KONFAI_CONFIG_MODE"] = "Done"
        if bound.arguments.get("tensorboard"):
            os.environ["KONFAI_TENSORBOARD_PORT"] = str(find_free_port())

        torch.autograd.set_detect_anomaly(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["KONFAI_VERBOSE"] = str(not bound.arguments.get("quiet"))
        is_cluster = "resubmit" in kwargs
        if is_cluster:
            os.environ["KONFAI_OVERWRITE"] = "True"
            os.environ["KONFAI_CLUSTER"] = "True"
        try:
            with func(*args, **kwargs_fun) as distributed_object:
                with Log(distributed_object.name, 0):
                    if is_cluster:
                        distributed_object.setup(len(gpu) * kwargs["num_nodes"])
                        import submitit

                        executor = submitit.AutoExecutor(folder="./Cluster/")
                        executor.update_parameters(
                            name=kwargs["name"],
                            mem_gb=kwargs["memory"],
                            gpus_per_node=len(gpu),
                            tasks_per_node=len(gpu) // distributed_object.size,
                            cpus_per_task=1,
                            nodes=kwargs["num_nodes"],
                            timeout_min=kwargs["time_limit"],
                        )
                        with TensorBoard(distributed_object.name):
                            executor.submit(distributed_object)
                    else:
                        world_size = len(gpu)
                        if world_size == 0:
                            world_size = cpu
                        distributed_object.setup(world_size)
                        with TensorBoard(distributed_object.name):
                            if world_size > 1:
                                mp.spawn(distributed_object, nprocs=world_size)
                            else:
                                distributed_object(0)

        except KeyboardInterrupt:
            print("\n[KonfAI] Manual interruption (Ctrl+C)")

    return wrapper


def setup_gpu(world_size: int, port: int, rank: int | None = None) -> tuple[int | None, int | None]:
    if os.name == "nt":
        return rank, rank
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
    if dist.is_nccl_available() and torch.cuda.is_available() and len(cuda_visible_devices()):
        torch.cuda.empty_cache()
        dist.init_process_group(
            backend="nccl",
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
    if torch.cuda.is_available() and dist.is_initialized():
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
        torch.nn.functional.interpolate(
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
        torch.nn.functional.grid_sample(
            data.unsqueeze(0).type(torch.float32),
            torch.nn.functional.affine_grid(
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
    "fcsv",
    "xml",
    "vtk",
    "npy",
]


class KonfAIError(Exception):
    TYPE: str | None = None

    def __str__(self) -> str:
        if not self.args:
            return "\n[Error]"

        if isinstance(getattr(self, "TYPE", None), str) and self.TYPE:
            type_error = self.TYPE
            messages = [str(m) for m in self.args]
        else:
            type_error = str(self.args[0])
            messages = [str(m) for m in self.args[1:]]

        if not messages:
            return f"\n[{type_error}]"

        head = f"[{type_error}] {messages[0]}"
        if len(messages) == 1:
            return "\n" + head
        return "\n" + head + "\n→\t" + "\n→\t".join(messages[1:])


class NamedKonfAIError(KonfAIError):
    TYPE: str = "Error"


class EvaluatorError(NamedKonfAIError):
    TYPE = "Evaluator"


class ConfigError(NamedKonfAIError):
    TYPE = "Config"


class DatasetManagerError(NamedKonfAIError):
    TYPE = "DatasetManager"


class MeasureError(NamedKonfAIError):
    TYPE = "Measure"


class TrainerError(NamedKonfAIError):
    TYPE = "Trainer"


class AugmentationError(NamedKonfAIError):
    TYPE = "Augmentation"


class PredictorError(NamedKonfAIError):
    TYPE = "Predictor"


class TransformError(NamedKonfAIError):
    TYPE = "Transform"


class AppRepositoryHFError(NamedKonfAIError):
    TYPE = "Repo Hugging Face"


class AppDirectoryError(NamedKonfAIError):
    TYPE = "Model Directory"


class AppError(NamedKonfAIError):
    TYPE = "Model Config"


class AppMetadataError(NamedKonfAIError):
    TYPE = "Model metadata"


def get_available_models_on_hf_repo(repo_id: str) -> list[str]:
    api = HfApi()
    model_names = []
    try:
        tree = api.list_repo_tree(repo_id=repo_id)
        for entry in tree:
            model_name = entry.path
            if isinstance(entry, RepoFolder) and is_model_repo(repo_id, model_name)[0]:
                model_names.append(model_name)
    except Exception as e:
        raise AppRepositoryHFError(f"Unable to access repository '{repo_id}': {e}")
    return model_names


def is_model_directory(model_path: Path) -> tuple[bool, str, list[str]]:
    checkpoints_name = []
    found_metadata_file = False
    for filename in model_path.glob("*"):
        if filename.name.endswith(".pt"):
            checkpoints_name.append(filename.name)
        elif str(filename).endswith("app.json"):
            found_metadata_file = True

    if not found_metadata_file:
        return False, f"Missing 'app.json' in '{model_path}'.", []
    return True, "", checkpoints_name


def is_model_repo(repo_id: str, model_name: str) -> tuple[bool, str, list[str]]:
    """
    Check whether the Hugging Face repository structure is valid for KonfAI.
    Required files:
    - a app file
    """
    api = HfApi()
    checkpoints_name = []
    found_metadata_file = False

    try:
        tree = api.list_repo_tree(repo_id=repo_id, path_in_repo=model_name)
    except Exception as e:
        return False, f"Unable to access repository '{repo_id}': {e}", []

    for filename in tree:
        if filename.path.endswith(".pt"):
            checkpoints_name.append(Path(filename.path).name)
        elif filename.path.endswith("app.json"):
            found_metadata_file = True

    if not found_metadata_file:
        return False, f"Missing 'app.json' in '{repo_id}/{model_name}'.", []
    return True, "", checkpoints_name


class ModelLoad(ABC):

    def __init__(self, model_name: str, checkpoints_name: list[str]) -> None:
        self._number_of_models = len(checkpoints_name)
        self._checkpoints_name = checkpoints_name
        self._model_name = model_name
        required_keys = ["description", "short_description", "tta", "mc_dropout", "display_name"]
        metadata_file_path = self._download("app.json")
        with open(metadata_file_path, encoding="utf-8") as f:
            model_metadata = json.load(f)

        missing = [k for k in required_keys if k not in model_metadata]
        if missing:
            raise AppMetadataError(f"Missing keys in app.json: {', '.join(missing)}")

        self._description = str(model_metadata["description"])
        self._short_description = str(model_metadata["short_description"])

        try:
            self._maximum_tta = int(model_metadata["tta"])
        except Exception:
            raise AppMetadataError("The field 'tta' must be an integer.")

        try:
            self._mc_dropout = int(model_metadata["mc_dropout"])
        except Exception:
            raise AppMetadataError("The field 'mc_dropout' must be an integer.")

        self._display_name = str(model_metadata["display_name"])
        self._terminology = {}
        if "terminology" in model_metadata:
            self._terminology = {int(k): v for k, v in model_metadata["terminology"].items()}

    def get_checkpoints_name(self):
        return self._checkpoints_name

    def get_display_name(self):
        return self._display_name

    def get_maximum_tta(self):
        return self._maximum_tta

    def get_mc_dropout(self):
        return self._mc_dropout

    def get_terminology(self):
        return self._terminology

    def get_number_of_models(self):
        return self._number_of_models

    def get_description(self):
        return self._description

    def get_short_description(self):
        return self._short_description

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

    @abstractmethod
    def _get_filenames(self):
        raise NotImplementedError()

    @abstractmethod
    def _download(self, filename: str) -> Path:
        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()

    def has_capabilities(self) -> tuple[bool, bool]:
        filenames = self._get_filenames()
        evaluation_support = False
        uncertainty_support = False

        for filename in filenames:
            if filename.endswith("Evaluation.yml"):
                evaluation_support = True
            elif filename.endswith("Uncertainty.yml"):
                uncertainty_support = True
        return evaluation_support, uncertainty_support

    def download_inference(
        self, number_of_model: int, name_of_models: list[str], prediction_file: str
    ) -> tuple[list[Path], Path, list[Path]]:
        filenames = self._get_filenames()
        models_path = []
        codes_path = []
        i = 0
        inference_file_path = None
        filename_filter = [prediction_file, "requirements.txt", "app.json"] + [
            name if name.endswith(".pt") else name + ".pt" for name in name_of_models
        ]
        available_models = []
        for filename in filenames:
            if filename.endswith(".pt"):
                available_models.append(filename)
            if filename not in filename_filter and not filename.endswith(".py"):
                if not len(name_of_models) and filename.endswith(".pt"):
                    i += 1
                    if i > number_of_model:
                        continue
                else:
                    continue
            file_path = self._download(filename)

            if prediction_file in filename:
                inference_file_path = file_path
            elif filename.endswith(".pt"):
                models_path.append(file_path)
            elif "requirements.txt" in filename:
                with open(file_path, encoding="utf-8") as f:
                    required_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    installed = {
                        dist.metadata["Name"].lower(): dist.version
                        for dist in importlib.metadata.distributions()
                        if dist.metadata.get("Name")
                    }
                    missing_or_outdated = []
                    for line in required_lines:
                        req = Requirement(line)
                        name = req.name.lower()
                        installed_version_str = installed.get(name)
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
                            raise AppRepositoryHFError(f"Failed to install packages: {e}") from e
            elif "app.json" not in filename:
                codes_path.append(file_path)
        if inference_file_path is None:
            raise AppError(
                f"Prediction file '{prediction_file}' was not found in the remote archive. "
                f"Available files: {', '.join(filenames)}"
            )
        if len(models_path) == 0:
            raise AppError(
                f"No model was found matching the requested model name(s): {name_of_models}.",
                f"Available models: {available_models}.",
                "Please check that the checkpoint exists.",
            )
        return models_path, inference_file_path, codes_path

    def download_train(self) -> list[Path]:
        filenames = self._get_filenames()
        files_path = []
        for filename in filenames:
            files_path.append(self._download(filename))
        return files_path

    def download_evaluation(self, evaluation_file: str) -> tuple[Path, list[Path]]:
        filenames = self._get_filenames()
        codes_path = []
        evaluation_file_path = None
        for filename in filenames:
            if evaluation_file in filename:
                evaluation_file_path = self._download(filename)
            elif filename.endswith(".py"):
                codes_path.append(self._download(filename))
        if evaluation_file_path is None:
            raise AppError(
                f"Evaluation file '{evaluation_file}' was not found in the remote archive. "
                f"Available files: {', '.join(filenames)}"
            )
        return evaluation_file_path, codes_path

    def download_uncertainty(self, uncertainty_file: str) -> tuple[Path, list[Path]]:
        filenames = self._get_filenames()
        codes_path = []
        uncertainty_file_path = None
        for filename in filenames:
            if uncertainty_file in filename:
                uncertainty_file_path = self._download(filename)
            elif filename.endswith(".py"):
                codes_path.append(self._download(filename))
        if uncertainty_file_path is None:
            raise AppError(
                f"Uncertainty file '{uncertainty_file}' was not found in the remote archive. "
                f"Available files: {', '.join(filenames)}"
            )
        return uncertainty_file_path, codes_path

    def install_inference(
        self,
        number_of_augmentation: int,
        number_of_model: int,
        name_of_models: list[str],
        number_of_mc_dropout: int,
        prediction_file: str,
    ) -> list[Path]:
        if len(name_of_models) == 0 and number_of_model == 0:
            number_of_model = self._number_of_models

        models_path, inference_file_path, codes_path = self.download_inference(
            number_of_model, name_of_models, prediction_file
        )
        shutil.copy2(inference_file_path, prediction_file)
        self.set_number_of_augmentation(prediction_file, number_of_augmentation)
        for code_path in codes_path:
            if code_path.suffix == ".py":
                shutil.copy2(code_path, code_path.name)

        return models_path

    def install_evaluation(self, evaluation_file: str) -> None:
        evaluation_file_path, codes_path = self.download_evaluation(evaluation_file)
        shutil.copy2(evaluation_file_path, evaluation_file)
        for code_path in codes_path:
            if code_path.suffix == ".py":
                shutil.copy2(code_path, code_path.name)

    def install_uncertainty(self, uncertainty_file: str) -> None:
        uncertainty_file_path, codes_path = self.download_uncertainty(uncertainty_file)
        shutil.copy2(uncertainty_file_path, uncertainty_file)
        for code_path in codes_path:
            if code_path.suffix == ".py":
                shutil.copy2(code_path, code_path.name)

    def install_fine_tune(
        self, config_file: str, path: Path, display_name: str, epochs: int, it_validation: int | None
    ) -> list[Path]:
        path.mkdir(parents=True, exist_ok=True)
        src_paths = self.download_train()
        models_path = []

        overwrite_all = None

        def ask_overwrite_cli(dest_path: Path) -> bool:
            """Prompt user in terminal to decide whether to overwrite a file."""
            nonlocal overwrite_all

            # If the user already decided globally, apply it
            if overwrite_all is not None:
                return overwrite_all

            while True:
                print(f"\nFile already exists: {dest_path}")
                choice = input("Overwrite? [y]es / [n]o / [a]ll / [s]kip_all: ").strip().lower()

                if choice == "y":
                    return True
                elif choice == "n":
                    return False
                elif choice == "a":
                    overwrite_all = True
                    return True
                elif choice == "s":
                    overwrite_all = False
                    return False
                else:
                    print("Invalid input. Please choose y / n / a / s.")

        for src in src_paths:
            if src.is_dir():
                for item in src.rglob("*"):
                    rel = item.relative_to(src)
                    dest = path / rel
                    if item.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                        continue

                    # Ensure parent exists
                    dest.parent.mkdir(parents=True, exist_ok=True)

                    # If file exists → prompt in CLI
                    if dest.exists():
                        if not ask_overwrite_cli(dest):

                            continue

                    shutil.copy2(item, dest)

                    # Track model weights
                    if str(item).endswith(".pt"):
                        models_path.append(item)

            elif src.is_file() or src.is_symlink():
                dest = path / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)

                if str(src).endswith(".pt"):
                    models_path.append(src)

                # If file exists → prompt in CLI
                if dest.exists():
                    if not ask_overwrite_cli(dest):
                        continue

                shutil.copy2(src, dest)

        metadata_file = path / "app.json"
        config_file_path = path / config_file
        if not metadata_file.exists():
            raise ConfigError(
                f"Metadata file not found: '{metadata_file}'.",
                "Ensure the metadata file exists and the provided path is correct.",
            )

        if not Path(config_file_path).exists():
            raise ConfigError(
                f"Configuration file not found: '{config_file_path}'.",
                "Ensure the configuration file exists and the provided path is correct.",
            )

        # Load existing metadata
        with open(metadata_file, encoding="utf-8") as f:
            model_metadata = json.load(f)

        # Modify the metadata
        model_metadata["display_name"] = display_name

        # Save back to disk (UTF-8, pretty JSON)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(model_metadata, f, indent=2, ensure_ascii=False)

        yaml = YAML()
        with open(config_file_path) as f:
            data = yaml.load(f)
            data["Trainer"]["epochs"] = epochs
            data["Trainer"]["it_validation"] = it_validation

        with open(config_file_path, "w") as f:
            yaml.dump(data, f)

        return models_path


class ModelDirectory(ModelLoad):

    def __init__(self, model_directory: Path, model_name: str):
        self._model_directory = model_directory
        _, err_message, checkpoints_name = is_model_directory(model_directory / model_name)

        if err_message:
            raise AppDirectoryError(err_message)

        super().__init__(model_name, checkpoints_name)

    def _get_filenames(self) -> list[str]:
        return [filename.name for filename in (self._model_directory / self._model_name).glob("*")]

    def _download(self, filename: str) -> Path:
        return self._model_directory / self._model_name / filename

    def get_name(self) -> str:
        return str(self._model_directory / self._model_name)


class ModelHF(ModelLoad):

    def __init__(self, repo_id: str, model_name: str):
        self._repo_id, self.model_name = repo_id, model_name
        _, err_message, checkpoints_name = is_model_repo(self._repo_id, self.model_name)
        if err_message:
            raise AppRepositoryHFError(err_message)

        super().__init__(self.model_name, checkpoints_name)

    def _get_filenames(self) -> list[str]:
        api = HfApi()
        tree = api.list_repo_tree(repo_id=self._repo_id, path_in_repo=self.model_name)
        return [Path(filename.path).name for filename in tree]

    def _download(self, filename: str) -> Path:
        if not filename.startswith(self._model_name):
            filename = self._model_name + "/" + filename
        file_path = hf_hub_download(
            repo_id=self._repo_id, filename=filename, repo_type="model", revision=None
        )  # nosec B615
        return Path(file_path)

    def get_name(self) -> str:
        return f"{self._repo_id}:{self._model_name}"
