import itertools
import pynvml
import psutil

import numpy as np
import os
import torch

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union

from konfai import CONFIG_FILE, STATISTICS_DIRECTORY, PREDICTIONS_DIRECTORY, DL_API_STATE, CUDA_VISIBLE_DEVICES
import torch.distributed as dist
import argparse
import subprocess
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys

def description(model, modelEMA = None, showMemory: bool = True) -> str:
    values_desc = lambda weights, values: " ".join(["{}({:.2f}) : {:.6f}".format(name.split(":")[-1], weight, value) for (name, value), weight in zip(values.items(), weights.values())])
    model_desc = lambda model : "("+" ".join(["{}({:.6f}) : {}".format(name, network.optimizer.param_groups[0]['lr'] if network.optimizer is not None else 0, values_desc(network.measure.getLastWeights(), network.measure.getLastValues())) for name, network in model.module.getNetworks().items() if network.measure is not None])+")"
    result = "Loss {}".format(model_desc(model))
    if modelEMA is not None:
        result += "Loss EMA {}".format(model_desc(modelEMA))
    result += " "+gpuInfo()
    if showMemory:
        result +=" | {}".format(memoryInfo())
    return result

def _getModule(classpath : str, type : str) -> tuple[str, str]:
    if len(classpath.split("_")) > 1:
        module = ".".join(classpath.split("_")[:-1])
        name = classpath.split("_")[-1] 
    else:
        module = "konfai."+type
        name = classpath
    return module, name

def cpuInfo() -> str:
    return "CPU ({:.2f} %)".format(psutil.cpu_percent(interval=0.5))

def memoryInfo() -> str:
    return "Memory ({:.2f}G ({:.2f} %))".format(psutil.virtual_memory()[3]/2**30, psutil.virtual_memory()[2])

def getMemory() -> float:
    return psutil.virtual_memory()[3]/2**30

def memoryForecast(memory_init : float, i : float, size : float) -> str:
    current_memory = getMemory()
    forecast = memory_init + ((current_memory-memory_init)*size/i) if i > 0 else 0
    return "Memory forecast ({:.2f}G ({:.2f} %))".format(forecast, forecast/(psutil.virtual_memory()[0]/2**30)*100)

def gpuInfo() -> str:
    if CUDA_VISIBLE_DEVICES() == "":
        return ""
    
    devices = [int(i) for i in CUDA_VISIBLE_DEVICES().split(",")]
    device = devices[0]
    
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return ""
    node_name = "Node: {} " +os.environ["SLURMD_NODENAME"] if "SLURMD_NODENAME" in os.environ else ""
    return  "{}GPU({}) Memory GPU ({:.2f}G ({:.2f} %))".format(node_name, devices, float(memory.used)/(10**9), float(memory.used)/float(memory.total)*100)

def getMaxGPUMemory(device : Union[int, torch.device]) -> float:
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return 0
    device = [int(i) for i in CUDA_VISIBLE_DEVICES().split(",")][device]
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return 0
    return float(memory.total)/(10**9)

def getGPUMemory(device : Union[int, torch.device]) -> float:
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return 0
    device = [int(i) for i in CUDA_VISIBLE_DEVICES().split(",")][device]
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return 0
    return float(memory.used)/(10**9)

class NeedDevice(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.device : torch.device
    
    def setDevice(self, device : int):
        self.device = getDevice(device)

def getDevice(device : int):
    return device if torch.cuda.is_available() and device >=0 else torch.device("cpu")

class State(Enum):
    TRAIN = "TRAIN"
    RESUME = "RESUME"
    TRANSFER_LEARNING = "TRANSFER_LEARNING"
    FINE_TUNING = "FINE_TUNING"
    PREDICTION = "PREDICTION"
    EVALUATION = "EVALUATION"
    
    def __str__(self) -> str:
        return self.value

def get_patch_slices_from_nb_patch_per_dim(patch_size_tmp: list[int], nb_patch_per_dim : list[tuple[int, bool]], overlap: Union[int, None]) -> list[tuple[slice]]:
    patch_slices = []
    slices : list[list[slice]] = []
    if overlap is None:
        overlap = 0
    patch_size = []
    i = 0
    for nb in nb_patch_per_dim:
        if nb[1]:
            patch_size.append(1)
        else:
            patch_size.append(patch_size_tmp[i])
            i+=1

    for dim, nb in enumerate(nb_patch_per_dim):
        slices.append([])
        for index in range(nb[0]):
            start = (patch_size[dim]-overlap)*index
            end = start + patch_size[dim]
            slices[dim].append(slice(start,end))
    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))
    return patch_slices

def get_patch_slices_from_shape(patch_size: list[int], shape : list[int], overlap: Union[int, None]) -> tuple[list[tuple[slice]], list[tuple[int, bool]]]:
    if len(shape) != len(patch_size):
        return [tuple([slice(0, s) for s in shape])], [(1, True)]*len(shape)
    
    patch_slices = []
    nb_patch_per_dim = []
    slices : list[list[slice]] = []
    if overlap is None:
        size = [np.ceil(a/b) for a, b in zip(shape, patch_size)]
        tmp = np.zeros(len(size), dtype=np.int_)
        for i, s in enumerate(size):
            if s > 1:
                tmp[i] = np.mod(patch_size[i]-np.mod(shape[i], patch_size[i]), patch_size[i])//(size[i]-1)
        overlap = tmp
    else:
        overlap = [overlap if size > 1 else 0 for size in patch_size]
    
    for dim in range(len(shape)):
        assert overlap[dim] < patch_size[dim],  "Overlap must be less than patch size"

    for dim in range(len(shape)):
        slices.append([])
        index = 0
        while True:
            start = (patch_size[dim]-overlap[dim])*index

            end = start + patch_size[dim]
            if end >= shape[dim]:
                end = shape[dim]
                slices[dim].append(slice(start, end))
                break
            slices[dim].append(slice(start, end))
            index += 1
        nb_patch_per_dim.append((index+1, patch_size[dim] == 1))

    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))
    
    return patch_slices, nb_patch_per_dim

def _logSignalFormat(input : np.ndarray):
    return {str(i): channel for i, channel in enumerate(input)}

def _logImageFormat(input : np.ndarray):
    if len(input.shape) == 2:
        input = np.expand_dims(input, axis=0)

    if len(input.shape) == 3 and input.shape[0] != 1:
        input = np.expand_dims(input, axis=0)

    if len(input.shape) == 4:
        input = input[:, input.shape[1]//2]
    
    if input.dtype == np.uint8:
        return input
        
    input = input.astype(float)
    b = -np.min(input)
    if (np.max(input)+b) > 0:
        return (input+b)/(np.max(input)+b)
    else:
        return 0*input

def _logImagesFormat(input : np.ndarray):
    result = []
    for n in range(input.shape[0]):
        result.append(_logImageFormat(input[n]))
    result = np.stack(result, axis=0)
    return result

def _logVideoFormat(input : np.ndarray):
    result = []
    for t in range(input.shape[1]):
        result.append( _logImagesFormat(input[:, t,...]))
    result = np.stack(result, axis=1)

    nb_channel = result.shape[2]
    if nb_channel < 3:
        channel_split = [result[:, :, 0, ...] for i in range(3)]
    else:
        channel_split = np.split(result, 3, axis=0)
    input = np.zeros((result.shape[0], result.shape[1], 3, *list(result.shape[3:])))
    for i, channels in enumerate(channel_split):
        input[:,:,i] = np.mean(channels, axis=0)
    return input

class DataLog(Enum):
    SIGNAL   = lambda tb, name, layer, it : [tb.add_scalars(name, _logSignalFormat(layer[b, :, 0]), layer.shape[0]*it+b) for b in range(layer.shape[0])],
    IMAGE   = lambda tb, name, layer, it : tb.add_image(name, _logImageFormat(layer[0]), it),
    IMAGES  = lambda tb, name, layer, it : tb.add_images(name, _logImagesFormat(layer), it),
    VIDEO   = lambda tb, name, layer, it : tb.add_video(name, _logVideoFormat(layer), it),
    AUDIO   = lambda tb, name, layer, it : tb.add_audio(name, _logImageFormat(layer), it)

class Log():

    def __init__(self, name: str) -> None:
        path = PREDICTIONS_DIRECTORY() if DL_API_STATE() == "PREDICTION" else STATISTICS_DIRECTORY()
        if not os.path.exists("{}{}".format(path, name)):
            os.makedirs("{}{}".format(path, name))
        self.file = open("{}{}/log.txt".format(path, name), 'w')
        self.stdout_bak = sys.stdout
        self.stderr_bak = sys.stderr
        self.verbose = os.environ["DEEP_LEARNING_VERBOSE"] == "True"

    def __enter__(self):
        self.file.__enter__()
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, type, value, traceback):
        self.file.__exit__(type, value, traceback)
        sys.stdout = self.stdout_bak
        sys.stderr = self.stderr_bak
         
    def write(self, msg):
        if msg.strip() != "":
            self.file.write(msg)
            if self.verbose:
                print(msg, file=sys.__stdout__)

    def flush(self):
        pass
    
class TensorBoard():
    
    def __init__(self, name: str) -> None:
        self.process = None
        self.name = name

    def __enter__(self):
        if "DEEP_LEARNING_TENSORBOARD_PORT" in os.environ:
            command = ["tensorboard", "--logdir", PREDICTIONS_DIRECTORY() if DL_API_STATE() == "PREDICTION" else STATISTICS_DIRECTORY() + self.name + "/", "--port", os.environ["DEEP_LEARNING_TENSORBOARD_PORT"], "--bind_all"]
            self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('10.255.255.255', 1))
                IP = s.getsockname()[0]
            except Exception:
                IP = '127.0.0.1'
            finally:
                s.close()
            print("Tensorboard : http://{}:{}/".format(IP, os.environ["DEEP_LEARNING_TENSORBOARD_PORT"]))
        return self
    
    def __exit__(self, type, value, traceback):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()

class DistributedObject():

    def __init__(self, name: str) -> None:
        self.port = find_free_port()
        self.dataloader : list[list[DataLoader]]
        self.manual_seed: bool = None
        self.name = name
        self.size = 1
    
    @abstractmethod
    def setup(self, world_size: int):
        pass
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @abstractmethod
    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders: list[DataLoader]):
        pass 
    
    def getMeasure(world_size: int, global_rank: int, gpu: int, models: dict[str, torch.nn.Module], n: int) -> dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]]:
        data = {}
        for label, model in models.items():
            for name, network in model.getNetworks().items():
                if network.measure is not None:
                    data["{}{}".format(name, label)] = (network.measure.format(True, n), network.measure.format(False, n))
        outputs = synchronize_data(world_size, gpu, data)
        result = {}
        if global_rank == 0:
            for output in outputs:
                for k, v in output.items():
                    for t in range(len(v)):
                        for u, n in v[t].items():
                            if k not in result:
                                result[k] = ({}, {})
                            if u not in result[k][t]:
                                result[k][t][u] = (n[0], 0)
                            result[k][t][u] = (result[k][t][u][0], result[k][t][u][1]+n[1]/world_size)
        return result

    def __call__(self, rank: Union[int, None] = None) -> None:
        with Log(self.name):
            world_size = len(self.dataloader)
            global_rank, local_rank = setupGPU(world_size, self.port, rank)
            if global_rank is None:
                return
            if torch.cuda.is_available():
                pynvml.nvmlInit()
            if self.manual_seed is not None:
                np.random.seed(self.manual_seed * world_size + global_rank)
                random.seed(self.manual_seed * world_size + global_rank)
                torch.manual_seed(self.manual_seed * world_size + global_rank)
            torch.backends.cudnn.benchmark = self.manual_seed is None
            torch.backends.cudnn.deterministic = self.manual_seed is not None
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            dataloaders = self.dataloader[global_rank]
            if torch.cuda.is_available():    
                torch.cuda.set_device(local_rank)
                
            self.run_process(world_size, global_rank, local_rank, dataloaders)
            if torch.cuda.is_available():
                pynvml.nvmlShutdown()
            cleanup()

def setupAPI(parser: argparse.ArgumentParser) -> DistributedObject:
    # API arguments
    api_args = parser.add_argument_group('API arguments')
    api_args.add_argument("type", type=State, choices=list(State))
    api_args.add_argument('-y', action='store_true', help="Accept overwrite")
    api_args.add_argument('-tb', action='store_true', help='Start TensorBoard')
    api_args.add_argument("-c", "--config", type=str, default="None", help="Configuration file location")
    api_args.add_argument("-g", "--gpu", type=str, default=os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else "", help="List of GPU")
    api_args.add_argument('--num-workers', '--num_workers', default=4, type=int, help='No. of workers per DataLoader & GPU')
    api_args.add_argument("-models_dir", "--MODELS_DIRECTORY", type=str, default="./Models/", help="Models location")
    api_args.add_argument("-checkpoints_dir", "--CHECKPOINTS_DIRECTORY", type=str, default="./Checkpoints/", help="Checkpoints location")
    api_args.add_argument("-model", "--MODEL", type=str, default="", help="URL Model")
    api_args.add_argument("-predictions_dir", "--PREDICTIONS_DIRECTORY", type=str, default="./Predictions/", help="Predictions location")
    api_args.add_argument("-evaluation_dir", "--EVALUATIONS_DIRECTORY", type=str, default="./Evaluations/", help="Evaluations location")
    api_args.add_argument("-statistics_dir", "--STATISTICS_DIRECTORY", type=str, default="./Statistics/", help="Statistics location")
    api_args.add_argument("-setups_dir", "--SETUPS_DIRECTORY", type=str, default="./Setups/", help="Setups location")
    api_args.add_argument('-log', action='store_true', help='Save log')
    api_args.add_argument('-quiet', action='store_false', help='')
    
    
    args = parser.parse_args()
    config = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    os.environ["DL_API_MODELS_DIRECTORY"] = config["MODELS_DIRECTORY"]
    os.environ["DL_API_CHECKPOINTS_DIRECTORY"] = config["CHECKPOINTS_DIRECTORY"]
    os.environ["DL_API_PREDICTIONS_DIRECTORY"] = config["PREDICTIONS_DIRECTORY"]
    os.environ["DL_API_EVALUATIONS_DIRECTORY"] = config["EVALUATIONS_DIRECTORY"]
    os.environ["DL_API_STATISTICS_DIRECTORY"] = config["STATISTICS_DIRECTORY"]
    
    os.environ["DL_API_STATE"] = str(config["type"])
    
    os.environ["DL_API_MODEL"] = config["MODEL"]

    os.environ["DL_API_SETUPS_DIRECTORY"] = config["SETUPS_DIRECTORY"]

    os.environ["DL_API_OVERWRITE"] = "{}".format(config["y"])
    os.environ["DEEP_LEANING_API_CONFIG_MODE"] = "Done"
    if config["tb"]:
        os.environ["DEEP_LEARNING_TENSORBOARD_PORT"] = str(find_free_port())

    os.environ["DEEP_LEARNING_VERBOSE"] = str(config["quiet"])

    if config["config"] == "None":
        if config["type"] is State.PREDICTION:
             os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Prediction.yml"
        elif config["type"] is State.EVALUATION:
            os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Evaluation.yml"
        else:
            os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = "Config.yml"
    else:
        os.environ["DEEP_LEARNING_API_CONFIG_FILE"] = config["config"]
    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    if config["type"] is State.PREDICTION:    
        from konfai.predictor import Predictor
        os.environ["DEEP_LEARNING_API_ROOT"] = "Predictor"
        return Predictor(config=CONFIG_FILE())    
    elif config["type"] is State.EVALUATION:
        from konfai.evaluator import Evaluator
        os.environ["DEEP_LEARNING_API_ROOT"] = "Evaluator"
        return Evaluator(config=CONFIG_FILE())
    else:
        from konfai.trainer import Trainer
        os.environ["DEEP_LEARNING_API_ROOT"] = "Trainer"
        return Trainer(config=CONFIG_FILE())


def setupGPU(world_size: int, port: int, rank: Union[int, None] = None) -> tuple[int , int]:
    try:
        host_name = subprocess.check_output("scontrol show hostnames {}".format(os.getenv('SLURM_JOB_NODELIST')).split()).decode().splitlines()[0]
    except:
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
    print("tcp://{}:{}".format(host_name, port))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dist.init_process_group("nccl", rank=global_rank, init_method="tcp://{}:{}".format(host_name, port), world_size=world_size)
    return global_rank, local_rank

import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    
def cleanup():
    if torch.cuda.is_available():
        dist.destroy_process_group()

def synchronize_data(world_size: int, gpu: int, data: any) -> list[Any]:
    if torch.cuda.is_available():
        outputs: list[dict[str, tuple[dict[str, float], dict[str, float]]]] = [None for _ in range(world_size)]
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
    return F.interpolate(data.type(torch.float32).unsqueeze(0), size=tuple([s for s in reversed(size)]), mode=mode).squeeze(0).type(data.dtype)

def _affine_matrix(matrix: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cat((matrix, translation.unsqueeze(0).T), dim=1), torch.tensor([[0, 0, 0, 1]])), dim=0)

def _resample_affine(data: torch.Tensor, matrix: torch.Tensor):
    if data.dtype == torch.uint8:
        mode = "nearest"
    else:
        mode = "bilinear"
    return F.grid_sample(data.unsqueeze(0).type(torch.float32), F.affine_grid(matrix[:, :-1,...].type(torch.float32), [1]+list(data.shape), align_corners=True), align_corners=True, mode=mode, padding_mode="reflection").squeeze(0).type(data.dtype)
