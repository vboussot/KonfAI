import torch
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

from konfai import MODELS_DIRECTORY, CHECKPOINTS_DIRECTORY, STATISTICS_DIRECTORY, SETUPS_DIRECTORY, CONFIG_FILE, MODEL, DATE, KONFAI_STATE
from konfai.data.data_manager import DataTrain
from konfai.utils.config import config
from konfai.utils.utils import State, DataLog, DistributedObject, description, TrainerError
from konfai.network.network import Network, ModelLoader, NetState, CPU_Model

class EarlyStoppingBase:

    def __init__(self):
        pass

    def isStopped(self) -> bool:
        return False

    def getScore(self, values: dict[str, float]):
        return sum([i for i in values.values()])
        
    def __call__(self, current_score: float) -> bool:
        return False

class EarlyStopping(EarlyStoppingBase):

    @config("EarlyStopping")
    def __init__(self, monitor: Union[list[str], None] = [], patience: int=10, min_delta: float=0.0, mode: str="min"):
        super().__init__()
        self.monitor = [] if monitor is None else monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def isStopped(self) -> bool:
        return self.early_stop

    def getScore(self, values: dict[str, float]):
        if len(self.monitor) == 0:
            return super().getScore(values)
        for v in self.monitor:
            if v not in values.keys():
                raise TrainerError(
                    "Metric '{}' specified in EarlyStopping.monitor not found in logged values. ",
                    "Available keys: {}. Please check your configuration.".format(v, list(values.keys())))
        return sum([i for v, i in values.items() if v in self.monitor])
        
    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improvement = self.best_score - current_score
        elif self.mode == "max":
            improvement = current_score - self.best_score
        else:
            raise TrainerError("Mode must be 'min' or 'max'.")

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

class _Trainer():

    def __init__(self, world_size: int, global_rank: int, local_rank: int, size: int, train_name: str, early_stopping: EarlyStopping, data_log: Union[list[str], None] , save_checkpoint_mode: str, epochs: int, epoch: int, autocast: bool, it_validation: Union[int, None], it: int, model: Union[DDP, CPU_Model], modelEMA: AveragedModel, dataloader_training: DataLoader, dataloader_validation: Union[DataLoader, None] = None) -> None:
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
        self.early_stopping = EarlyStoppingBase() if early_stopping is None else early_stopping 
        
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
        self.dataloader_training.dataset.load("Train")
        if self.dataloader_validation is not None:
            self.dataloader_validation.dataset.load("Validation")
        with tqdm.tqdm(iterable = range(self.epoch, self.epochs), leave=False, total=self.epochs, initial=self.epoch, desc="Progress") as epoch_tqdm:
            for self.epoch in epoch_tqdm:
                self.train()
                if self.early_stopping.isStopped():
                    break
                self.dataloader_training.dataset.resetAugmentation("Train")
                
    def getInput(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int, str, bool]]) -> dict[tuple[str, bool], torch.Tensor]:
        return {(k, v[5][0].item()) : v[0] for k, v in data_dict.items()}

    def train(self) -> None:
        self.model.train()
        self.model.module.setState(NetState.TRAIN)
        if self.modelEMA is not None:
            self.modelEMA.eval()
            self.modelEMA.module.setState(NetState.TRAIN)

        desc = lambda : "Training : {}".format(description(self.model, self.modelEMA))

        with tqdm.tqdm(iterable = enumerate(self.dataloader_training), desc = desc(), total=len(self.dataloader_training), leave=False, ncols=0) as batch_iter:
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
                        score = self.early_stopping.getScore(loss)
                        self.checkpoint_save(score)
                        if self.early_stopping(score):
                            break
                        

                batch_iter.set_description(desc()) 
            
                
    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        self.model.module.setState(NetState.PREDICTION)
        if self.modelEMA is not None:
            self.modelEMA.module.setState(NetState.PREDICTION)

        desc = lambda : "Validation : {}".format(description(self.model, self.modelEMA))
        data_dict = None
        with tqdm.tqdm(iterable = enumerate(self.dataloader_validation), desc = desc(), total=len(self.dataloader_validation), leave=False, ncols=0) as batch_iter:
            for _, data_dict in batch_iter:
                input = self.getInput(data_dict)
                self.model(input)
                if self.modelEMA is not None:
                    self.modelEMA.module(input)

                batch_iter.set_description(desc())
        self.dataloader_validation.dataset.resetAugmentation("Validation")
        dist.barrier()
        self.model.train()
        self.model.module.setState(NetState.TRAIN)
        if self.modelEMA is not None:
            self.modelEMA.module.setState(NetState.TRAIN)
        return self._validation_log(data_dict)

    def checkpoint_save(self, loss: float) -> None:
        if self.global_rank != 0:
            return

        path = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
        os.makedirs(path, exist_ok=True)

        name = DATE() + ".pt"
        save_path = os.path.join(path, name)

        save_dict = {
        "epoch": self.epoch,
        "it": self.it,
        "loss": loss,
        "Model": self.model.module.state_dict()
        }

        if self.modelEMA is not None:
            save_dict["Model_EMA"] = self.modelEMA.module.state_dict()
        
        save_dict.update({'{}_optimizer_state_dict'.format(name): network.optimizer.state_dict() for name, network in self.model.module.getNetworks().items() if network.optimizer is not None})
        save_dict.update({'{}_it'.format(name): network._it for name, network in self.model.module.getNetworks().items() if network.optimizer is not None})
        save_dict.update({'{}_nb_lr_update'.format(name): network._nb_lr_update for name, network in self.model.module.getNetworks().items() if network.optimizer is not None})
        
        torch.save(save_dict, save_path)

        if self.save_checkpoint_mode == "BEST":
            all_checkpoints = sorted([
                os.path.join(path, f)
                for f in os.listdir(path) if f.endswith(".pt")
            ])
            best_ckpt = None
            best_loss = float('inf')

            for f in all_checkpoints:
                d = torch.load(f, weights_only=False)
                if d.get("loss", float("inf")) < best_loss:
                    best_loss = d["loss"]
                    best_ckpt = f

            for f in all_checkpoints:
                if f != best_ckpt and f != save_path:
                    os.remove(f)

    @torch.no_grad()
    def _log(self, type_log: str, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> dict[str, float]:
        models: dict[str, Network] = {"" : self.model.module}
        if self.modelEMA is not None:
            models["_EMA"] = self.modelEMA.module
        
        measures = DistributedObject.getMeasure(self.world_size, self.global_rank, self.local_rank*self.size+self.size-1, models, self.it_validation if type_log == "Training" else len(self.dataloader_validation))
        
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
                        
            if type_log == "Training":
                for name, network in self.model.module.getNetworks().items():
                    if network.optimizer is not None:
                        self.tb.add_scalar("{}/{}/Learning Rate".format(type_log, name), network.optimizer.param_groups[0]['lr'], self.it)
        
        if self.global_rank == 0:
            loss = {}
            for name, network in self.model.module.getNetworks().items():
                if network.measure is not None:
                    loss.update({k : v[1] for k, v in measures["{}{}".format(name, label)][0].items()})
                    loss.update({k : v[1] for k, v in measures["{}{}".format(name, label)][1].items()})
            return loss
        return None
    
    @torch.no_grad()
    def _train_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> dict[str, float]:
        return self._log("Training", data_dict)

    @torch.no_grad()
    def _validation_log(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int]]) -> dict[str, float]:
        return self._log("Validation", data_dict)

    
class Trainer(DistributedObject):

    @config("Trainer")
    def __init__(   self,
                    model : ModelLoader = ModelLoader(),
                    dataset : DataTrain = DataTrain(),
                    train_name : str = "default:TRAIN_01",
                    manual_seed : Union[int, None] = None,
                    epochs: int = 100,
                    it_validation : Union[int, None] = None,
                    autocast : bool = False,
                    gradient_checkpoints: Union[list[str], None] = None,
                    gpu_checkpoints: Union[list[str], None] = None,
                    ema_decay : float = 0,
                    data_log: Union[list[str], None] = None,
                    early_stopping: Union[EarlyStopping, None] = None,
                    save_checkpoint_mode: str= "BEST") -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            exit(0)
        super().__init__(train_name)
        self.manual_seed = manual_seed        
        self.dataset = dataset
        self.autocast = autocast
        self.epochs = epochs
        self.epoch = 0
        self.early_stopping = early_stopping
        self.it = 0
        self.it_validation = it_validation
        self.model = model.getModel(train=True)
        self.ema_decay = ema_decay
        self.modelEMA : Union[torch.optim.swa_utils.AveragedModel, None] = None
        self.data_log = data_log

        modules = []
        for i,_ in self.model.named_modules():
            modules.append(i)
        for k in self.data_log:
            tmp = k.split("/")[0].replace(":", ".")
            if tmp not in self.dataset.getGroupsDest() and tmp not in modules:
                raise TrainerError( f"Invalid key '{tmp}' in `data_log`.",
                                   f"This key is neither a destination group from the dataset ({self.dataset.getGroupsDest()})",
                                    f"nor a valid module name in the model ({modules}).",
                "Please check your `data_log` configuration — it should reference either a model output or a dataset group.")
            
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
                state_dict = torch.load(path+name, weights_only=False, map_location="cpu")
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
                checkpoint = torch.load(path_checkpoint+name, weights_only=False, map_location='cpu')
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
        state = State._member_map_[KONFAI_STATE()]
        if state != State.RESUME and os.path.exists(CHECKPOINTS_DIRECTORY()+self.name+"/"):
            if os.environ["KONFAI_OVERWRITE"] != "True":
                accept = input("The model {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.name))
                if accept != "yes":
                    return
            for directory_path in [STATISTICS_DIRECTORY(), MODELS_DIRECTORY(), CHECKPOINTS_DIRECTORY(), SETUPS_DIRECTORY()]:
                if os.path.exists(directory_path+self.name+"/"):
                    shutil.rmtree(directory_path+self.name+"/")
        
        state_dict = {}
        if state != State.TRAIN:
            state_dict = self._load()

        self.model.init(self.autocast, state, self.dataset.getGroupsDest())
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
        with _Trainer(world_size, global_rank, local_rank, self.size, self.name, self.early_stopping, self.data_log, self.save_checkpoint_mode, self.epochs, self.epoch, self.autocast, self.it_validation, self.it, model, self.modelEMA, *dataloaders) as t:
            t.run()