from abc import ABC, abstractmethod
import builtins
import importlib
import shutil
import torch
import tqdm
import os

from konfai import MODELS_DIRECTORY, PREDICTIONS_DIRECTORY, CONFIG_FILE, MODEL, KONFAI_ROOT
from konfai.utils.config import config
from konfai.utils.utils import State, get_patch_slices_from_nb_patch_per_dim, NeedDevice, _getModule, DistributedObject, DataLog, description, PredictorError
from konfai.utils.dataset import Dataset, Attribute
from konfai.data.data_manager import DataPrediction, DatasetIter
from konfai.data.patching import Accumulator, PathCombine
from konfai.network.network import ModelLoader, Network, NetState, CPU_Model
from konfai.data.transform import Transform, TransformLoader

from torch.utils.tensorboard.writer import SummaryWriter
from typing import Union
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import importlib
import copy
from collections import defaultdict

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

    def load(self, name_layer: str, datasets: list[Dataset], groups: dict[str, str]):
        transforms_type = ["pre_transforms", "post_transforms", "final_transforms"]
        for name, _transform_type, transform_type in [(k, getattr(self, "_{}".format(k)), getattr(self, k)) for k in transforms_type]:
            
            if _transform_type is not None:
                for classpath, transform in _transform_type.items():
                    transform = transform.getTransform(classpath, DL_args =  "{}.outsDataset.{}.OutDataset.{}".format(KONFAI_ROOT(), name_layer, name))
                    transform.setDatasets(datasets)
                    transform_type.append(transform)

        if self._patchCombine is not None:
            module, name = _getModule(self._patchCombine, "konfai.data.patching")
            self.patchCombine = getattr(importlib.import_module(module), name)(config = None, DL_args =  "{}.outsDataset.{}.OutDataset".format(KONFAI_ROOT(), name_layer))
    
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

class Reduction():

    def __init__(self):
        pass

class Mean(Reduction):

    def __init__(self):
        pass

    def __call__(self, result: torch.Tensor) -> torch.Tensor:
        return torch.mean(result.float(), dim=0)
        
class Median(Reduction):

    def __init__(self):
        pass
    
    def __call__(self, result: torch.Tensor) -> torch.Tensor:
        return torch.median(result.float(), dim=0).values

class OutSameAsGroupDataset(OutDataset):

    @config("OutDataset")
    def __init__(self, dataset_filename: str = "./Dataset:mha", group: str = "default", sameAsGroup: str = "default", pre_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, post_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, final_transforms : dict[str, TransformLoader] = {"default:Normalize": TransformLoader()}, patchCombine: Union[str, None] = None, reduction: str = "mean", inverse_transform: bool = True) -> None:
        super().__init__(dataset_filename, group, pre_transforms, post_transforms, final_transforms, patchCombine)
        self.group_src, self.group_dest = sameAsGroup.split(":")
        self.reduction_classpath = reduction
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


    def load(self, name_layer: str, datasets: list[Dataset], groups: dict[str, str]):
        super().load(name_layer, datasets, groups)
        module, name = _getModule(self.reduction_classpath, "konfai.predictor")
        self.reduction = config("{}.outsDataset.{}.OutDataset.{}".format(KONFAI_ROOT(), name_layer, self.reduction_classpath))(getattr(importlib.import_module(module), name))(config = None)
       
        if self.group_src not in groups.keys():
            raise PredictorError(
                f"Source group '{self.group_src}' not found. Available groups: {list(groups.keys())}."
            )

        if self.group_dest not in groups[self.group_src]:
            raise PredictorError(
                f"Destination group '{self.group_dest}' not found. Available groups: {groups[self.group_src]}."
            )
    
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
        result = self.reduction(result.float()).to(result.dtype)

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

    def __init__(self, world_size: int, global_rank: int, local_rank: int, autocast: bool, predict_path: str, data_log: Union[list[str], None], outsDataset: dict[str, OutDataset], modelComposite: DDP, dataloader_prediction: DataLoader) -> None:
        self.world_size = world_size        
        self.global_rank = global_rank
        self.local_rank = local_rank

        self.modelComposite = modelComposite
        self.dataloader_prediction = dataloader_prediction
        self.outsDataset = outsDataset
        self.autocast = autocast
        
        self.it = 0

        self.device = self.modelComposite.device
        self.dataset: DatasetIter = self.dataloader_prediction.dataset
        patch_size, overlap = self.dataset.getPatchConfig()
        for outDataset in self.outsDataset.values():
            outDataset.setPatchConfig([size for size in patch_size if size > 1], overlap, np.max([int(np.sum([data_augmentation.nb for data_augmentation in self.dataset.dataAugmentationsList])+1), 1]))
        self.data_log : dict[str, tuple[DataLog, int]] = {}
        if data_log is not None:
            for data in data_log:
                self.data_log[data.split("/")[0].replace(":", ".")] = (DataLog.__getitem__(data.split("/")[1]).value[0], int(data.split("/")[2]))
        self.tb = SummaryWriter(log_dir = predict_path+"Metric/") if len([network for network in self.modelComposite.module.getNetworks().values() if network.measure is not None]) or len(self.data_log) else None
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        if self.tb:
            self.tb.close()

    def getInput(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int, str, bool]]) -> dict[tuple[str, bool], torch.Tensor]:
        return {(k, v[5][0].item()) : v[0] for k, v in data_dict.items()}
    
    @torch.no_grad()
    def run(self):
        self.modelComposite.eval()  
        self.modelComposite.module.setState(NetState.PREDICTION)
        desc = lambda : "Prediction : {}".format(description(self.modelComposite))
        self.dataloader_prediction.dataset.load("Prediction")
        with tqdm.tqdm(iterable = enumerate(self.dataloader_prediction), leave=True, desc = desc(), total=len(self.dataloader_prediction), ncols=0) as batch_iter:
            for it, data_dict in batch_iter:
                with torch.amp.autocast('cuda', enabled=self.autocast):
                    input = self.getInput(data_dict)
                    for name, output in self.modelComposite(input, list(self.outsDataset.keys())):
                        self._predict_log(data_dict)
                        outDataset = self.outsDataset[name]
                        for i, (index, patch_augmentation, patch_index) in enumerate([(int(index), int(patch_augmentation), int(patch_index)) for index, patch_augmentation, patch_index in zip(list(data_dict.values())[0][1], list(data_dict.values())[0][2], list(data_dict.values())[0][3])]):
                            outDataset.addLayer(index, patch_augmentation, patch_index, output[i].cpu(), self.dataset)
                            if outDataset.isDone(index):
                                outDataset.write(index, self.dataset.getDatasetFromIndex(list(data_dict.keys())[0], index).name.split("/")[-1], outDataset.getOutput(index, self.dataset))

                    batch_iter.set_description(desc())
                    self.it += 1
        
    def _predict_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]):
        measures = DistributedObject.getMeasure(self.world_size, self.global_rank, self.local_rank, {"" : self.modelComposite.module}, 1)
        
        if self.global_rank == 0:
            images_log = []
            if len(self.data_log):    
                for name, data_type in self.data_log.items():
                    if name in data_dict:
                        data_type[0](self.tb, "Prediction/{}".format(name), data_dict[name][0][:self.data_log[name][1]].detach().cpu().numpy(), self.it)
                    else:
                        images_log.append(name.replace(":", "."))

            for name, network in self.modelComposite.module.getNetworks().items():
                if network.measure is not None:
                    self.tb.add_scalars("Prediction/{}/Loss".format(name), {k : v[1] for k, v in measures["{}{}".format(name, "")][0].items()}, self.it)
                    self.tb.add_scalars("Prediction/{}/Metric".format(name), {k : v[1] for k, v in measures["{}{}".format(name, "")][1].items()}, self.it)
                if len(images_log):
                    for name, layer, _ in self.model.module.get_layers([v.to(0) for k, v in self.getInput(data_dict).items() if k[1]], images_log):
                        self.data_log[name][0](self.tb, "Prediction/{}".format(name), layer[:self.data_log[name][1]].detach().cpu().numpy(), self.it)

class ModelComposite(Network):

    def __init__(self, model: Network, nb_models: int, combine: Reduction):
        super().__init__(model.in_channels, model.optimizer, model.schedulers, model.outputsCriterionsLoader, model.patch, model.nb_batch_per_step, model.init_type, model.init_gain, model.dim)
        self.combine = combine
        for i in range(nb_models):
            self.add_module("Model_{}".format(i), copy.deepcopy(model), in_branch=[0], out_branch=["output_{}".format(i)])

    def load(self, state_dicts : list[dict[str, dict[str, torch.Tensor]]]):
        for i, state_dict in enumerate(state_dicts):
            self["Model_{}".format(i)].load(state_dict, init=False)
            self["Model_{}".format(i)].setName("{}_{}".format(self["Model_{}".format(i)].getName(), i))
            
    def forward(self, data_dict: dict[tuple[str, bool], torch.Tensor], output_layers: list[str] = []) -> list[tuple[str, torch.Tensor]]:
        result = {}
        for name, module in self.items():
            result[name] = module(data_dict, output_layers)
        
        aggregated = defaultdict(list)
        for module_outputs in result.values():
            for key, tensor in module_outputs:
                aggregated[key].append(tensor)

        final_outputs = []
        for key, tensors in aggregated.items():
            final_outputs.append((key, self.combine(torch.stack(tensors, dim=0))))

        return final_outputs
    

class Predictor(DistributedObject):

    @config("Predictor")
    def __init__(self, 
                    model: ModelLoader = ModelLoader(),
                    dataset: DataPrediction = DataPrediction(),
                    combine: str = "mean",
                    train_name: str = "name",
                    manual_seed : Union[int, None] = None,
                    gpu_checkpoints: Union[list[str], None] = None,
                    autocast : bool = False,
                    outsDataset: Union[dict[str, OutDatasetLoader], None] = {"default:Default" : OutDatasetLoader()},
                    images_log: list[str] = []) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            exit(0)
        super().__init__(train_name)
        self.manual_seed = manual_seed
        self.dataset = dataset
        self.combine_classpath = combine
        self.autocast = autocast

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

    def _load(self) -> list[dict[str, dict[str, torch.Tensor]]]:
        model_paths = MODEL().split(":")
        state_dicts = []
        for model_path in model_paths:
            if model_path.startswith("https://"):
                try:
                    state_dicts.append(torch.hub.load_state_dict_from_url(url=model_path, map_location="cpu", check_hash=True))
                except:
                    raise Exception("Model : {} does not exist !".format(model_path)) 
            else:
                if model_path != "":
                    path = ""
                    name = model_path
                else:
                    if self.name.endswith(".pt"):
                        path = MODELS_DIRECTORY()+"/".join(self.name.split("/")[:-1])+"/StateDict/"
                        name = self.name.split("/")[-1]
                    else:
                        path = MODELS_DIRECTORY()+self.name+"/StateDict/"
                        name = sorted(os.listdir(path))[-1]
                if os.path.exists(path+name):
                    state_dicts.append(torch.load(path+name, weights_only=False))
                else:
                    raise Exception("Model : {} does not exist !".format(path+name))
        return state_dicts
    
    def setup(self, world_size: int):
        for dataset_filename in self.datasets_filename:
            path = self.predict_path +dataset_filename
            if os.path.exists(path):
                if os.environ["KONFAI_OVERWRITE"] != "True":
                    accept = builtins.input("The prediction {} already exists ! Do you want to overwrite it (yes,no) : ".format(path))
                    if accept != "yes":
                        return
                    
            if not os.path.exists(path):
                os.makedirs(path)

        shutil.copyfile(CONFIG_FILE(), self.predict_path+"Prediction.yml")

        
        self.model.init(self.autocast, State.PREDICTION, self.dataset.getGroupsDest())
        self.model.init_outputsGroup()
        self.model._compute_channels_trace(self.model, self.model.in_channels, None, self.gpu_checkpoints)
        
        modules = []
        for i,_,_ in self.model.named_ModuleArgsDict():
            modules.append(i)
        for output_group in self.outsDataset.keys():
            if output_group not in modules:
                raise PredictorError("The output group '{}' defined in 'outputsCriterions' does not correspond to any module in the model.".format(output_group),
                    "Available modules: {}".format(modules),
                    "Please check that the name matches exactly a submodule or output of your model architecture."
                )
        module, name = _getModule(self.combine_classpath, "konfai.predictor")
        combine = config("{}.{}".format(KONFAI_ROOT(), self.combine_classpath))(getattr(importlib.import_module(module), name))(config = None)
       
        self.modelComposite = ModelComposite(self.model, len(MODEL().split(":")), combine)
        self.modelComposite.load(self._load())

        if len(list(self.outsDataset.keys())) == 0 and len([network for network in self.modelComposite.getNetworks().values() if network.measure is not None]) == 0:
            exit(0)
        
        self.size = (len(self.gpu_checkpoints)+1 if self.gpu_checkpoints else 1)
        self.dataloader = self.dataset.getData(world_size//self.size)
        for name, outDataset in self.outsDataset.items():
            outDataset.load(name.replace(".", ":"), list(self.dataset.datasets.values()), {src : dest for src, inner in self.dataset.groups_src.items() for dest in inner})
           
           
    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders: list[DataLoader]):
        modelComposite = Network.to(self.modelComposite, local_rank*self.size)
        modelComposite = DDP(modelComposite, static_graph=True) if torch.cuda.is_available() else CPU_Model(modelComposite)
        with _Predictor(world_size, global_rank, local_rank, self.autocast, self.predict_path, self.images_log, self.outsDataset, modelComposite, *dataloaders) as p:
            p.run()

        