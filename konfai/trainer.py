import torch
import torch.optim.adamw
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
from typing import Union
from torch.nn.parallel import DistributedDataParallel as DDP

import shutil
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.swa_utils import AveragedModel
import torch.distributed as dist

from konfai import MODELS_DIRECTORY, CHECKPOINTS_DIRECTORY, STATISTICS_DIRECTORY, SETUPS_DIRECTORY, CONFIG_FILE, MODEL, DATE, DL_API_STATE
from konfai.data.dataset import DataTrain
from konfai.utils.config import config
from konfai.utils.utils import State, DataLog, DistributedObject, description
from konfai.network.network import Network, ModelLoader, NetState, CPU_Model


class _Trainer():

    def __init__(self, world_size: int, global_rank: int, local_rank: int, size: int, train_name: str, data_log: Union[list[str], None] , save_checkpoint_mode: str, epochs: int, epoch: int, autocast: bool, it_validation: Union[int, None], it: int, model: Union[DDP, CPU_Model], modelEMA: AveragedModel, dataloader_training: DataLoader, dataloader_validation: Union[DataLoader, None] = None) -> None:
        self.world_size = world_size        
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.size = size

        self.save_checkpoint_mode = save_checkpoint_mode
        self.train_name = train_name
        self.epochs = epochs
        self.epoch = epoch
        self.model = model
        self.dataloader_training = dataloader_training
        self.dataloader_validation = dataloader_validation
        self.autocast = autocast
        self.modelEMA = modelEMA
        
        self.it_validation = it_validation
        if self.it_validation is None:
            self.it_validation = len(dataloader_training)
        self.it = it
        self.tb = SummaryWriter(log_dir = STATISTICS_DIRECTORY()+self.train_name+"/")
        self.data_log : dict[str, tuple[DataLog, int]] = {}
        if data_log is not None:
            for data in data_log:
                self.data_log[data.split("/")[0].replace(":", ".")] = (DataLog.__getitem__(data.split("/")[1]).value[0], int(data.split("/")[2]))
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        if self.tb is not None:
            self.tb.close()

    def run(self) -> None:
        with tqdm.tqdm(iterable = range(self.epoch, self.epochs), leave=False, total=self.epochs, initial=self.epoch, desc="Progress", disable=self.global_rank != 0) as epoch_tqdm:
            for self.epoch in epoch_tqdm:
                self.dataloader_training.dataset.load()
                self.train()
                self.dataloader_training.dataset.resetAugmentation()
                
    def getInput(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int, str, bool]]) -> dict[tuple[str, bool], torch.Tensor]:
        return {(k, v[5][0].item()) : v[0] for k, v in data_dict.items()}

    def train(self) -> None:
        self.model.train()
        self.model.module.setState(NetState.TRAIN)
        if self.modelEMA is not None:
            self.modelEMA.eval()
            self.modelEMA.module.setState(NetState.TRAIN)

        desc = lambda : "Training : {}".format(description(self.model, self.modelEMA))
        with tqdm.tqdm(iterable = enumerate(self.dataloader_training), desc = desc(), total=len(self.dataloader_training), leave=False, disable=self.global_rank != 0 and "DL_API_CLUSTER" not in os.environ) as batch_iter:
            for _, data_dict in batch_iter:
                with torch.amp.autocast('cuda', enabled=self.autocast):
                    input = self.getInput(data_dict)
                    self.model(input)
                    self.model.module.backward(self.model.module)
                    if self.modelEMA is not None:
                        self.modelEMA.update_parameters(self.model)
                        self.modelEMA.module(input)
                    self.it += 1
                    if (self.it) % self.it_validation == 0:
                        loss = self._train_log(data_dict)

                        if self.dataloader_validation is not None:
                            loss = self._validate()
                        self.model.module.update_lr()
                        self.checkpoint_save(loss)

                batch_iter.set_description(desc()) 
            
                
    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        self.model.module.setState(NetState.PREDICTION)
        if self.modelEMA is not None:
            self.modelEMA.module.setState(NetState.PREDICTION)

        desc = lambda : "Validation : {}".format(description(self.model, self.modelEMA))
        data_dict = None
        self.dataloader_validation.dataset.load()
        with tqdm.tqdm(iterable = enumerate(self.dataloader_validation), desc = desc(), total=len(self.dataloader_validation), leave=False, disable=self.global_rank != 0 and "DL_API_CLUSTER" not in os.environ) as batch_iter:
            for _, data_dict in batch_iter:
                input = self.getInput(data_dict)
                self.model(input)
                if self.modelEMA is not None:
                    self.modelEMA.module(input)

                batch_iter.set_description(desc())
        self.dataloader_validation.dataset.resetAugmentation()
        dist.barrier()
        self.model.train()
        self.model.module.setState(NetState.TRAIN)
        if self.modelEMA is not None:
            self.modelEMA.module.setState(NetState.TRAIN)
        return self._validation_log(data_dict)

    def checkpoint_save(self, loss: float) -> None:
        if self.global_rank == 0:
            path = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
            last_loss = None
            if os.path.exists(path) and os.listdir(path):
                name = sorted(os.listdir(path))[-1]
                state_dict = torch.load(path+name, weights_only=False)
                last_loss = state_dict["loss"]
                if self.save_checkpoint_mode == "BEST":
                    if last_loss >= loss:
                        os.remove(path+name)

            if self.save_checkpoint_mode != "BEST" or (last_loss is None or last_loss >= loss):
                name = DATE()+".pt"
                if not os.path.exists(path):
                    os.makedirs(path)
                
                save_dict = {
                    "epoch": self.epoch,
                    "it": self.it,
                    "loss": loss,
                    "Model": self.model.module.state_dict()}

                if self.modelEMA is not None:
                    save_dict.update({"Model_EMA" : self.modelEMA.module.state_dict()})

                save_dict.update({'{}_optimizer_state_dict'.format(name): network.optimizer.state_dict() for name, network in self.model.module.getNetworks().items() if network.optimizer is not None})
                torch.save(save_dict, path+name)

    @torch.no_grad()
    def _log(self, type_log: str, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> float:
        models: dict[str, Network] = {"" : self.model.module}
        if self.modelEMA is not None:
            models["_EMA"] = self.modelEMA.module
        
        measures = DistributedObject.getMeasure(self.world_size, self.global_rank, self.local_rank*self.size+self.size-1, models, self.it_validation if type_log == "Trainning" else len(self.dataloader_validation))
        
        if self.global_rank == 0:
            images_log = []
            if len(self.data_log):    
                for name, data_type in self.data_log.items():
                    if name in data_dict:
                        data_type[0](self.tb, "{}/{}".format(type_log ,name), data_dict[name][0][:self.data_log[name][1]].detach().cpu().numpy(), self.it)
                    else:
                        images_log.append(name.replace(":", "."))
                        
            for label, model in models.items():
                for name, network in model.getNetworks().items():
                    if network.measure is not None:
                        self.tb.add_scalars("{}/{}/Loss/{}".format(type_log, name, label), {k : v[1] for k, v in measures["{}{}".format(name, label)][0].items()}, self.it)
                        self.tb.add_scalars("{}/{}/Loss_weight/{}".format(type_log, name, label), {k : v[0] for k, v in measures["{}{}".format(name, label)][0].items()}, self.it)

                        self.tb.add_scalars("{}/{}/Metric/{}".format(type_log, name, label), {k : v[1] for k, v in measures["{}{}".format(name, label)][1].items()}, self.it)
                        self.tb.add_scalars("{}/{}/Metric_weight/{}".format(type_log, name, label), {k : v[0] for k, v in measures["{}{}".format(name, label)][1].items()}, self.it)
                
                    if len(images_log):
                        for name, layer, _ in model.get_layers([v.to(0) for k, v in self.getInput(data_dict).items() if k[1]], images_log):
                            self.data_log[name][0](self.tb, "{}/{}{}".format(type_log, name, label), layer[:self.data_log[name][1]].detach().cpu().numpy(), self.it)
                        
            if type_log == "Trainning":
                for name, network in self.model.module.getNetworks().items():
                    if network.optimizer is not None:
                        self.tb.add_scalar("{}/{}/Learning Rate".format(type_log, name), network.optimizer.param_groups[0]['lr'], self.it)
        
        if self.global_rank == 0:
            loss = []
            for name, network in self.model.module.getNetworks().items():
                if network.measure is not None:
                    loss.append(sum([v[1] for v in measures["{}".format(name)][0].values()]))
            return np.mean(loss)
        return None
    
    @torch.no_grad()
    def _train_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> float:
        return self._log("Trainning", data_dict)

    @torch.no_grad()
    def _validation_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> float:
        return self._log("Validation", data_dict)

class Trainer(DistributedObject):

    @config("Trainer")
    def __init__(   self,
                    model : ModelLoader = ModelLoader(),
                    dataset : DataTrain = DataTrain(),
                    train_name : str = "default:name",
                    manual_seed : Union[int, None] = None,
                    epochs: int = 100,
                    it_validation : Union[int, None] = None,
                    autocast : bool = False,
                    gradient_checkpoints: Union[list[str], None] = None,
                    gpu_checkpoints: Union[list[str], None] = None,
                    ema_decay : float = 0,
                    data_log: Union[list[str], None] = None,
                    save_checkpoint_mode: str= "BEST") -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
        super().__init__(train_name)
        self.manual_seed = manual_seed        
        self.dataset = dataset
        self.autocast = autocast
        self.epochs = epochs
        self.epoch = 0
        self.it = 0
        self.it_validation = it_validation
        self.model = model.getModel(train=True)
        self.ema_decay = ema_decay
        self.modelEMA : Union[torch.optim.swa_utils.AveragedModel, None] = None
        self.data_log = data_log
        self.gradient_checkpoints = gradient_checkpoints
        self.gpu_checkpoints = gpu_checkpoints
        self.save_checkpoint_mode = save_checkpoint_mode
        self.config_namefile_src = CONFIG_FILE().replace(".yml", "")
        self.config_namefile = SETUPS_DIRECTORY()+self.name+"/"+self.config_namefile_src.split("/")[-1]+"_"+str(self.it)+".yml"
        self.size = (len(self.gpu_checkpoints)+1 if self.gpu_checkpoints else 1)
    
    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)
        self._save()

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
                path = CHECKPOINTS_DIRECTORY()+self.name+"/"
                if os.listdir(path):
                    name = sorted(os.listdir(path))[-1]
            
            if os.path.exists(path+name):
                state_dict = torch.load(path+name, weights_only=False)
            else:
                raise Exception("Model : {} does not exist !".format(self.name))
            
        if "epoch" in state_dict:
            self.epoch = state_dict['epoch']
        if "it" in state_dict:
            self.it = state_dict['it']
        return state_dict

    def _save(self) -> None:
        path_checkpoint = CHECKPOINTS_DIRECTORY()+self.name+"/"
        path_model = MODELS_DIRECTORY()+self.name+"/"
        if os.path.exists(path_checkpoint) and os.listdir(path_checkpoint):
            for dir in [path_model, "{}Serialized/".format(path_model), "{}StateDict/".format(path_model)]:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            for name in sorted(os.listdir(path_checkpoint)):
                checkpoint = torch.load(path_checkpoint+name, weights_only=False)
                self.model.load(checkpoint, init=False, ema=False)

                torch.save(self.model, "{}Serialized/{}".format(path_model, name))
                torch.save({"Model" : self.model.state_dict()}, "{}StateDict/{}".format(path_model, name))
                
                if self.modelEMA is not None:
                    self.modelEMA.module.load(checkpoint, init=False, ema=True)
                    torch.save(self.modelEMA.module, "{}Serialized/{}".format(path_model, DATE()+"_EMA.pt"))
                    torch.save({"Model_EMA" : self.modelEMA.module.state_dict()}, "{}StateDict/{}".format(path_model, DATE()+"_EMA.pt"))

            os.rename(self.config_namefile, self.config_namefile.replace(".yml", "")+"_"+str(self.it)+".yml")
    
    def _avg_fn(self, averaged_model_parameter, model_parameter, num_averaged):
        return (1-self.ema_decay) * averaged_model_parameter + self.ema_decay * model_parameter
    
    def setup(self, world_size: int):
        state = State._member_map_[DL_API_STATE()]
        if state != State.RESUME and os.path.exists(STATISTICS_DIRECTORY()+self.name+"/"):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = input("The model {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.name))
                if accept != "yes":
                    return
            for directory_path in [STATISTICS_DIRECTORY(), MODELS_DIRECTORY(), CHECKPOINTS_DIRECTORY(), SETUPS_DIRECTORY()]:
                if os.path.exists(directory_path+self.name+"/"):
                    shutil.rmtree(directory_path+self.name+"/")
        
        state_dict = {}
        if state != State.TRAIN:
            state_dict = self._load()

        self.model.init(self.autocast, state)
        self.model.init_outputsGroup()
        self.model._compute_channels_trace(self.model, self.model.in_channels, self.gradient_checkpoints, self.gpu_checkpoints)
        self.model.load(state_dict, init=True, ema=False)
        if self.ema_decay > 0:
            self.modelEMA = AveragedModel(self.model, avg_fn=self._avg_fn)
            if state_dict is not None:
                self.modelEMA.module.load(state_dict, init=False, ema=True)

        if not os.path.exists(SETUPS_DIRECTORY()+self.name+"/"):
            os.makedirs(SETUPS_DIRECTORY()+self.name+"/")
        shutil.copyfile(self.config_namefile_src+".yml", self.config_namefile)
        
        self.dataloader = self.dataset.getData(world_size//self.size)

    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders: list[DataLoader]):
        model = Network.to(self.model, local_rank*self.size)
        model = DDP(model, static_graph=True) if torch.cuda.is_available() else CPU_Model(model)
        if self.modelEMA is not None:
            self.modelEMA.module = Network.to(self.modelEMA.module, local_rank)
        with _Trainer(world_size, global_rank, local_rank, self.size, self.name, self.data_log, self.save_checkpoint_mode, self.epochs, self.epoch, self.autocast, self.it_validation, self.it, model, self.modelEMA, *dataloaders) as t:
            t.run()