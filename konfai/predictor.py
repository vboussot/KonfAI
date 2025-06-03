from abc import ABC, abstractmethod
import builtins
import importlib
import shutil
import torch
import tqdm
import os

from konfai import MODELS_DIRECTORY, PREDICTIONS_DIRECTORY, CONFIG_FILE, MODEL, DEEP_LEARNING_API_ROOT
from konfai.utils.config import config
from konfai.utils.utils import State, get_patch_slices_from_nb_patch_per_dim, NeedDevice, _getModule, DistributedObject, DataLog, description
from konfai.utils.dataset import Dataset, Attribute
from konfai.data.dataset import DataPrediction, DatasetIter
from konfai.data.HDF5 import Accumulator, PathCombine
from konfai.network.network import ModelLoader, Network, NetState, CPU_Model
from konfai.data.transform import Transform, TransformLoader

from torch.utils.tensorboard.writer import SummaryWriter
from typing import Union
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import importlib

class OutDataset(Dataset, NeedDevice, ABC):

    def __init__(self, filename: str, group: str, pre_transforms : dict[str, TransformLoader], post_transforms : dict[str, TransformLoader], final_transforms : dict[str, TransformLoader], patchCombine: Union[str, None]) -> None: 
        filename, format = filename.split(":")
        super().__init__(filename, format)
        self.group = group
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self._final_transforms = final_transforms
        self._patchCombine = patchCombine
       
        self.pre_transforms : list[Transform] = []
        self.post_transforms : list[Transform] = []
        self.final_transforms : list[Transform] = []
        self.patchCombine: PathCombine = None

        self.output_layer_accumulator: dict[int, dict[int, Accumulator]] = {}
        self.attributes: dict[int, dict[int, dict[int, Attribute]]] = {}
        self.names: dict[int, str] = {}
        self.nb_data_augmentation = 0

    def load(self, name_layer: str, datasets: list[Dataset]):
        transforms_type = ["pre_transforms", "post_transforms", "final_transforms"]
        for name, _transform_type, transform_type in [(k, getattr(self, "_{}".format(k)), getattr(self, k)) for k in transforms_type]:
            
            if _transform_type is not None:
                for classpath, transform in _transform_type.items():
                    transform = transform.getTransform(classpath, DL_args =  "{}.outsDataset.{}.OutDataset.{}".format(DEEP_LEARNING_API_ROOT(), name_layer, name))
                    transform.setDatasets(datasets)
                    transform_type.append(transform)

        if self._patchCombine is not None:
            module, name = _getModule(self._patchCombine, "data.HDF5")
            self.patchCombine = getattr(importlib.import_module(module), name)(config = None, DL_args =  "{}.outsDataset.{}.OutDataset".format(DEEP_LEARNING_API_ROOT(), name_layer))
    
    def setPatchConfig(self, patchSize: Union[list[int], None], overlap: Union[int, None], nb_data_augmentation: int) -> None:
        if patchSize is not None and overlap is not None:
            if self.patchCombine is not None:
                self.patchCombine.setPatchConfig(patchSize, overlap)
        else:
            self.patchCombine = None
        self.nb_data_augmentation = nb_data_augmentation
    
    def setDevice(self, device: torch.device):
        super().setDevice(device)
        transforms_type = ["pre_transforms", "post_transforms", "final_transforms"]
        for transform_type in [(getattr(self, k)) for k in transforms_type]:
            if transform_type is not None:
                for transform in transform_type:
                    transform.setDevice(device)       

    @abstractmethod
    def addLayer(self, index: int, index_patch: int, layer: torch.Tensor, dataset: DatasetIter):
        pass

    def isDone(self, index: int) -> bool:
        return len(self.output_layer_accumulator[index]) == self.nb_data_augmentation and all([acc.isFull() for acc in self.output_layer_accumulator[index].values()])

    @abstractmethod
    def getOutput(self, index: int, dataset: DatasetIter) -> torch.Tensor:
        pass

    def write(self, index: int, name: str, layer: torch.Tensor):
        super().write(self.group, name, layer.numpy(), self.attributes[index][0][0])
        self.attributes.pop(index)

class OutSameAsGroupDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "Dataset:h5", group: str = "default", sameAsGroup: str = "default", pre_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, final_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, patchCombine: Union[str, None] = None, redution: str = "mean", inverse_transform: bool = True) -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms, final_transforms, patchCombine)
        self.group_src, self.group_dest = sameAsGroup.split(":")
        self.redution = redution
        self.inverse_transform = inverse_transform

    def addLayer(self, index_dataset: int, index_augmentation: int, index_patch: int, layer: torch.Tensor, dataset: DatasetIter):
        if index_dataset not in self.output_layer_accumulator or index_augmentation not in self.output_layer_accumulator[index_dataset]:
            input_dataset = dataset.getDatasetFromIndex(self.group_dest, index_dataset)
            if index_dataset not in self.output_layer_accumulator:
                self.output_layer_accumulator[index_dataset] = {}
                self.attributes[index_dataset] = {}
                self.names[index_dataset] = input_dataset.name
            self.attributes[index_dataset][index_augmentation] = {}

            self.output_layer_accumulator[index_dataset][index_augmentation] = Accumulator(input_dataset.patch.getPatch_slices(index_augmentation), input_dataset.patch.patch_size, self.patchCombine, batch=False)

            for i in range(len(input_dataset.patch.getPatch_slices(index_augmentation))):
                self.attributes[index_dataset][index_augmentation][i] = Attribute(input_dataset.cache_attributes[0])

        for transform in self.pre_transforms:
            layer = transform(self.names[index_dataset], layer, self.attributes[index_dataset][index_augmentation][index_patch])

        if self.inverse_transform:
            for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].post_transforms):
                layer = transform.inverse(self.names[index_dataset], layer, self.attributes[index_dataset][index_augmentation][index_patch])
                
        self.output_layer_accumulator[index_dataset][index_augmentation].addLayer(index_patch, layer)

    def _getOutput(self, index: int, index_augmentation: int, dataset: DatasetIter) -> torch.Tensor:
        layer = self.output_layer_accumulator[index][index_augmentation].assemble()
        name = self.names[index]
        if index_augmentation > 0:
            
            i = 0
            index_augmentation_tmp = index_augmentation-1
            for dataAugmentations in dataset.dataAugmentationsList:
                if index_augmentation_tmp >= i and index_augmentation_tmp < i+dataAugmentations.nb:
                    for dataAugmentation in reversed(dataAugmentations.dataAugmentations):
                        layer = dataAugmentation.inverse(index, index_augmentation_tmp-i, layer)
                    break
                i += dataAugmentations.nb

        for transform in self.post_transforms:
            layer = transform(name, layer, self.attributes[index][index_augmentation][0])
            
        if self.inverse_transform:
            for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].pre_transforms):
                layer = transform.inverse(name, layer, self.attributes[index][index_augmentation][0])
        return layer

    def getOutput(self, index: int, dataset: DatasetIter) -> torch.Tensor:
        result = torch.cat([self._getOutput(index, index_augmentation, dataset).unsqueeze(0) for index_augmentation in self.output_layer_accumulator[index].keys()], dim=0)
        name = self.names[index]
        self.output_layer_accumulator.pop(index)
        dtype = result.dtype

        if self.redution == "mean":
            result = torch.mean(result.float(), dim=0).to(dtype)
        elif self.redution == "median":
            result, _ = torch.median(result.float(), dim=0).to(dtype)
        else:
            raise NameError("Reduction method does not exist (mean, median)")
        for transform in self.final_transforms:
            result = transform(name, result, self.attributes[index][0][0])
        return result

class OutLayerDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "Dataset.h5", group: str = "default", overlap : Union[list[int], None] = None, pre_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, final_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, patchCombine: Union[str, None] = None) -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms, final_transforms, patchCombine)
        self.overlap = overlap
        
    def addLayer(self, index: int, index_patch: int, layer: torch.Tensor, dataset: DatasetIter):
        if index not in self.output_layer_accumulator:
            group = list(dataset.groups.keys())[0]
            patch_slices = get_patch_slices_from_nb_patch_per_dim(list(layer.shape[2:]), dataset.getDatasetFromIndex(group, index).patch.nb_patch_per_dim, self.overlap)
            self.output_layer_accumulator[index] = Accumulator(patch_slices, self.patchCombine, batch=False)
            self.attributes[index] = Attribute()
            self.names[index] = dataset.getDatasetFromIndex(group, index).name
            

        for transform in self.pre_transforms:
            layer = transform(layer, self.attributes[index])
        self.output_layer_accumulator[index].addLayer(index_patch, layer)

    def getOutput(self, index: int, dataset: DatasetIter) -> torch.Tensor:
        layer = self.output_layer_accumulator[index].assemble()
        name = self.names[index]
        for transform in self.post_transforms:
            layer = transform(name, layer, self.attributes[index])

        self.output_layer_accumulator.pop(index)
        return layer

class OutDatasetLoader():

    @config("OutDataset")
    def __init__(self, name_class: str = "OutSameAsGroupDataset") -> None:
        self.name_class = name_class

    def getOutDataset(self, layer_name: str) -> OutDataset:
        return getattr(importlib.import_module("konfai.predictor"), self.name_class)(config = None, DL_args = "Predictor.outsDataset.{}".format(layer_name))

class _Predictor():

    def __init__(self, world_size: int, global_rank: int, local_rank: int, predict_path: str, data_log: Union[list[str], None], outsDataset: dict[str, OutDataset], model: DDP, dataloader_prediction: DataLoader) -> None:
        self.world_size = world_size        
        self.global_rank = global_rank
        self.local_rank = local_rank

        self.model = model
        self.dataloader_prediction = dataloader_prediction
        self.outsDataset = outsDataset

        
        self.it = 0

        self.device = self.model.device
        self.dataset: DatasetIter = self.dataloader_prediction.dataset
        patch_size, overlap = self.dataset.getPatchConfig()
        for outDataset in self.outsDataset.values():
            outDataset.setPatchConfig([size for size in patch_size if size > 1], overlap, np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataset.dataAugmentationsList])+1), 1]))
        self.data_log : dict[str, tuple[DataLog, int]] = {}
        if data_log is not None:
            for data in data_log:
                self.data_log[data.split("/")[0].replace(":", ".")] = (DataLog.__getitem__(data.split("/")[1]).value[0], int(data.split("/")[2]))
        self.tb = SummaryWriter(log_dir = predict_path+"Metric/") if len([network for network in self.model.module.getNetworks().values() if network.measure is not None]) or len(self.data_log) else None
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        if self.tb:
            self.tb.close()

    def getInput(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int, str, bool]]) -> dict[tuple[str, bool], torch.Tensor]:
        return {(k, v[5][0].item()) : v[0] for k, v in data_dict.items()}
    
    @torch.no_grad()
    def run(self):
        self.model.eval()  
        self.model.module.setState(NetState.PREDICTION)
        desc = lambda : "Prediction : {}".format(description(self.model))
        self.dataloader_prediction.dataset.load()
        with tqdm.tqdm(iterable = enumerate(self.dataloader_prediction), leave=False, desc = desc(), total=len(self.dataloader_prediction), disable=self.global_rank != 0 and "DL_API_CLUSTER" not in os.environ) as batch_iter:
            dist.barrier()
            for it, data_dict in batch_iter:
                input = self.getInput(data_dict)
                for name, output in self.model(input, list(self.outsDataset.keys())):
                    self._predict_log(data_dict)
                    outDataset = self.outsDataset[name]
                    for i, (index, patch_augmentation, patch_index) in enumerate([(int(index), int(patch_augmentation), int(patch_index)) for index, patch_augmentation, patch_index in zip(list(data_dict.values())[0][1], list(data_dict.values())[0][2], list(data_dict.values())[0][3])]):
                        outDataset.addLayer(index, patch_augmentation, patch_index, output[i].cpu(), self.dataset)
                        if outDataset.isDone(index):
                            outDataset.write(index, self.dataset.getDatasetFromIndex(list(data_dict.keys())[0], index).name.split("/")[-1], outDataset.getOutput(index, self.dataset))

                batch_iter.set_description(desc())
                self.it += 1

    def _predict_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]):
        measures = DistributedObject.getMeasure(self.world_size, self.global_rank, self.local_rank, {"" : self.model.module}, 1)
        
        if self.global_rank == 0:
            images_log = []
            if len(self.data_log):    
                for name, data_type in self.data_log.items():
                    if name in data_dict:
                        data_type[0](self.tb, "Prediction/{}".format(name), data_dict[name][0][:self.data_log[name][1]].detach().cpu().numpy(), self.it)
                    else:
                        images_log.append(name.replace(":", "."))

            for name, network in self.model.module.getNetworks().items():
                if network.measure is not None:
                    self.tb.add_scalars("Prediction/{}/Loss".format(name), {k : v[1] for k, v in measures["{}{}".format(name, "")][0].items()}, self.it)
                    self.tb.add_scalars("Prediction/{}/Metric".format(name), {k : v[1] for k, v in measures["{}{}".format(name, "")][1].items()}, self.it)
                if len(images_log):
                    for name, layer, _ in self.model.module.get_layers([v.to(0) for k, v in self.getInput(data_dict).items() if k[1]], images_log):
                        self.data_log[name][0](self.tb, "Prediction/{}".format(name), layer[:self.data_log[name][1]].detach().cpu().numpy(), self.it)
        
class Predictor(DistributedObject):

    @config("Predictor")
    def __init__(self, 
                    model: ModelLoader = ModelLoader(),
                    dataset: DataPrediction = DataPrediction(),
                    train_name: str = "name",
                    manual_seed : Union[int, None] = None,
                    gpu_checkpoints: Union[list[str], None] = None,
                    outsDataset: Union[dict[str, OutDatasetLoader], None] = {"default:Default" : OutDatasetLoader()},
                    images_log: list[str] = []) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
        super().__init__(train_name)
        self.manual_seed = manual_seed
        self.dataset = dataset

        self.model = model.getModel(train=False)
        self.it = 0
        self.outsDatasetLoader = outsDataset if outsDataset else {}
        self.outsDataset = {name.replace(":", ".") : value.getOutDataset(name) for name, value in self.outsDatasetLoader.items()}

        self.datasets_filename = []
        self.predict_path = PREDICTIONS_DIRECTORY()+self.name+"/"
        self.images_log = images_log
        for outDataset in self.outsDataset.values():
            self.datasets_filename.append(outDataset.filename)
            outDataset.filename = "{}{}".format(self.predict_path, outDataset.filename)
            
        
        self.gpu_checkpoints = gpu_checkpoints

    def _load(self) -> dict[str, dict[str, torch.Tensor]]:
        if MODEL().startswith("https://"):
            try:
                state_dict = {MODEL().split(":")[1]: torch.hub.load_state_dict_from_url(url=MODEL().split(":")[0], map_location="cpu", check_hash=True)}
            except:
                raise Exception("Model : {} does not exist !".format(MODEL())) 
        else:
            if MODEL() != "":
                path = ""
                name = MODEL() 
            else:
                if self.name.endswith(".pt"):
                    path = MODELS_DIRECTORY()+"/".join(self.name.split("/")[:-1])+"/StateDict/"
                    name = self.name.split("/")[-1]
                else:
                    path = MODELS_DIRECTORY()+self.name+"/StateDict/"
                    name = sorted(os.listdir(path))[-1]
            if os.path.exists(path+name):
                state_dict = torch.load(path+name, weights_only=False)
            else:
                raise Exception("Model : {} does not exist !".format(path+name))
        return state_dict
    
    def setup(self, world_size: int):
        for dataset_filename in self.datasets_filename:
            path = self.predict_path +dataset_filename
            if os.path.exists(path):
                if os.environ["DL_API_OVERWRITE"] != "True":
                    accept = builtins.input("The prediction {} already exists ! Do you want to overwrite it (yes,no) : ".format(path))
                    if accept != "yes":
                        return
                    
            if not os.path.exists(path):
                os.makedirs(path)

        shutil.copyfile(CONFIG_FILE(), self.predict_path+"Prediction.yml")

        self.model.init(autocast=False, state = State.PREDICTION)
        self.model.init_outputsGroup()
        self.model._compute_channels_trace(self.model, self.model.in_channels, None, self.gpu_checkpoints)
        self.model.load(self._load(), init=False)
        
        if len(list(self.outsDataset.keys())) == 0 and len([network for network in self.model.getNetworks().values() if network.measure is not None]) == 0:
            exit(0)

        
        self.size = (len(self.gpu_checkpoints)+1 if self.gpu_checkpoints else 1)
        self.dataloader = self.dataset.getData(world_size//self.size)
        for name, outDataset in self.outsDataset.items():
            outDataset.load(name.replace(".", ":"), list(self.dataset.datasets.values()))
           
           
    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders: list[DataLoader]):
        model = Network.to(self.model, local_rank*self.size)
        model = DDP(model, static_graph=True) if torch.cuda.is_available() else CPU_Model(model)
        with _Predictor(world_size, global_rank, local_rank, self.predict_path, self.images_log, self.outsDataset, model, *dataloaders) as p:
            p.run()

        