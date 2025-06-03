from functools import partial
import importlib
import inspect
import os
from typing import Iterable, Iterator, Callable
from typing_extensions import Self
import torch
from abc import ABC
import numpy as np
from torch._jit_internal import _copy_to_script_wrapper
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from typing import Union
from enum import Enum

from konfai import DEEP_LEARNING_API_ROOT
from konfai.metric.schedulers import Scheduler
from konfai.utils.config import config
from konfai.utils.utils import State, _getModule, getDevice, getGPUMemory
from konfai.data.HDF5 import Accumulator, ModelPatch

class NetState(Enum):
    TRAIN = 0,
    PREDICTION = 1

class Patch_Indexed():

    def __init__(self, patch: ModelPatch, index: int) -> None:
        self.patch = patch
        self.index = index

    def isFull(self) -> bool:
        return len(self.patch.getPatch_slices(0)) == self.index

class OptimizerLoader():
    
    @config("Optimizer")
    def __init__(self, name: str = "AdamW") -> None:
        self.name = name
    
    def getOptimizer(self, key: str, parameter: Iterator[torch.nn.parameter.Parameter]) -> torch.optim.Optimizer:
        torch.optim.AdamW
        return config("{}.Model.{}.Optimizer".format(DEEP_LEARNING_API_ROOT(), key))(getattr(importlib.import_module('torch.optim'), self.name))(parameter, config = None)
        
class SchedulerStep():
    
    @config(None)
    def __init__(self, nb_step : int = 0) -> None:
        self.nb_step = nb_step

class LRSchedulersLoader():
        
    @config("Schedulers")
    def __init__(self, params: dict[str, SchedulerStep] = {"default:ReduceLROnPlateau" : SchedulerStep(0)}) -> None:
        self.params = params

    def getShedulers(self, key: str, optimizer: torch.optim.Optimizer) -> dict[torch.optim.lr_scheduler._LRScheduler, int]:
        shedulers : dict[torch.optim.lr_scheduler._LRScheduler, int] = {}
        for name, step in self.params.items():
            if name:
                shedulers[config("Trainer.Model.{}.Schedulers.{}".format(key, name))(getattr(importlib.import_module('torch.optim.lr_scheduler'), name))(optimizer, config = None)] = step.nb_step
        return shedulers

class SchedulersLoader():
        
    @config("Schedulers")
    def __init__(self, params: dict[str, SchedulerStep] = {"default:Constant" : SchedulerStep(0)}) -> None:
        self.params = params

    def getShedulers(self, key: str) -> dict[torch.optim.lr_scheduler._LRScheduler, int]:
        shedulers : dict[Scheduler, int] = {}
        for name, step in self.params.items():
            if name:    
                shedulers[getattr(importlib.import_module("konfai.metric.schedulers"), name)(config = None, DL_args = key)] = step.nb_step
        return shedulers
    
class CriterionsAttr():
    
    @config()
    def __init__(self, l: SchedulersLoader = SchedulersLoader(), isLoss: bool = True, group: int = 0, stepStart: int = 0, stepStop: Union[int, None] = None, accumulation: bool = False) -> None:
        self.l = l
        self.isTorchCriterion = True
        self.isLoss = isLoss
        self.stepStart = stepStart
        self.stepStop = stepStop
        self.group = group
        self.accumulation = accumulation
        self.sheduler = None
        
class CriterionsLoader():

    @config()
    def __init__(self, criterionsLoader: dict[str, CriterionsAttr] = {"default:torch_nn_CrossEntropyLoss:Dice:NCC": CriterionsAttr()}) -> None:
        self.criterionsLoader = criterionsLoader

    def getCriterions(self, model_classname : str, output_group : str, target_group : str) -> dict[torch.nn.Module, CriterionsAttr]:
        criterions = {}
        for module_classpath, criterionsAttr in self.criterionsLoader.items():
            module, name = _getModule(module_classpath, "metric.measure")
            criterionsAttr.isTorchCriterion = module.startswith("torch")
            criterionsAttr.sheduler = criterionsAttr.l.getShedulers("{}.Model.{}.outputsCriterions.{}.targetsCriterions.{}.criterionsLoader.{}".format(DEEP_LEARNING_API_ROOT(), model_classname, output_group, target_group, module_classpath))
            criterions[config("{}.Model.{}.outputsCriterions.{}.targetsCriterions.{}.criterionsLoader.{}".format(DEEP_LEARNING_API_ROOT(), model_classname, output_group, target_group, module_classpath))(getattr(importlib.import_module(module), name))(config = None)] = criterionsAttr
        return criterions

class TargetCriterionsLoader():

    @config()
    def __init__(self, targetsCriterions : dict[str, CriterionsLoader] = {"default" : CriterionsLoader()}) -> None:
        self.targetsCriterions = targetsCriterions
        
    def getTargetsCriterions(self, output_group : str, model_classname : str) -> dict[str, dict[torch.nn.Module, float]]:
        targetsCriterions = {}
        for target_group, criterionsLoader in self.targetsCriterions.items():
            targetsCriterions[target_group] = criterionsLoader.getCriterions(model_classname, output_group, target_group)
        return targetsCriterions

class Measure():

    class Loss():

        def __init__(self, name: str, output_group: str, target_group: str, group: int, isLoss: bool, accumulation: bool) -> None:
            self.name = name
            self.isLoss = isLoss
            self.accumulation = accumulation
            self.output_group = output_group
            self.target_group = target_group
            self.group = group

            self._loss: list[torch.Tensor] = []
            self._weight: list[float] = []
            self._values: list[float] = []
        
        def resetLoss(self) -> None:
            self._loss.clear()

        def add(self, weight: float, value: torch.Tensor) -> None:
            self._loss.append(value if self.isLoss else value.detach())
            self._values.append(value.item())
            self._weight.append(weight)

        def getLastLoss(self) -> torch.Tensor:
            return self._loss[-1]*self._weight[-1] if len(self._loss) else torch.zeros((1), requires_grad = True)
        
        def getLoss(self) -> torch.Tensor:
            return torch.stack([w*l for w, l in zip(self._weight, self._loss)], dim=0).mean(dim=0) if len(self._loss) else torch.zeros((1), requires_grad = True) 

        def __len__(self) -> int:
            return len(self._loss)
        
    def __init__(self, model_classname : str, outputsCriterions: dict[str, TargetCriterionsLoader]) -> None:
        super().__init__()
        self.outputsCriterions = {}
        for output_group, targetCriterionsLoader in outputsCriterions.items():
            self.outputsCriterions[output_group.replace(":", ".")] = targetCriterionsLoader.getTargetsCriterions(output_group, model_classname)
        self._loss : dict[int, dict[str, Measure.Loss]] = {}

    def init(self, model : torch.nn.Module) -> None:
        outputs_group_rename = {}
        for output_group in self.outputsCriterions.keys():
            for target_group in self.outputsCriterions[output_group]:
                for criterion in self.outputsCriterions[output_group][target_group]:
                    if not self.outputsCriterions[output_group][target_group][criterion].isTorchCriterion:
                        outputs_group_rename[output_group] = criterion.init(model, output_group, target_group)

        outputsCriterions_bak = self.outputsCriterions.copy()
        for old, new in outputs_group_rename.items():
            self.outputsCriterions.pop(old)
            self.outputsCriterions[new] = outputsCriterions_bak[old]
        for output_group in self.outputsCriterions:
            for target_group in self.outputsCriterions[output_group]:
                for criterion, criterionsAttr in self.outputsCriterions[output_group][target_group].items():
                    if criterionsAttr.group not in self._loss:
                        self._loss[criterionsAttr.group] = {}
                    self._loss[criterionsAttr.group]["{}:{}:{}".format(output_group, target_group, criterion.__class__.__name__)] = Measure.Loss(criterion.__class__.__name__, output_group, target_group, criterionsAttr.group, criterionsAttr.isLoss, criterionsAttr.accumulation) 

    def update(self, output_group: str, output : torch.Tensor, data_dict: dict[str, torch.Tensor], it: int, nb_patch: int, training: bool) -> None:
        for target_group in self.outputsCriterions[output_group]:
            target = [data_dict[group].to(output[0].device).detach() for group in target_group.split("/") if group in data_dict]

            for criterion, criterionsAttr in self.outputsCriterions[output_group][target_group].items():
                if it >= criterionsAttr.stepStart and (criterionsAttr.stepStop is None or it <= criterionsAttr.stepStop):
                    scheduler = self.update_scheduler(criterionsAttr.sheduler, it)
                    self._loss[criterionsAttr.group]["{}:{}:{}".format(output_group, target_group, criterion.__class__.__name__)].add(scheduler.get_value(), criterion(output, *target))
                    if training and len(np.unique([len(l) for l in self._loss[criterionsAttr.group].values() if l.accumulation and l.isLoss])) == 1:
                        if criterionsAttr.isLoss:
                            loss = torch.zeros((1), requires_grad = True)
                            for v in [l for l in self._loss[criterionsAttr.group].values() if l.accumulation and l.isLoss]:
                                l = v.getLastLoss()
                                loss = loss.to(l.device)+l
                            loss = loss/nb_patch
                            loss.backward()

    def getLoss(self) -> list[torch.Tensor]:
        loss: dict[int, torch.Tensor] = {}
        for group in self._loss.keys():
            loss[group] = torch.zeros((1), requires_grad = True)
            for v in self._loss[group].values():
                if v.isLoss and not v.accumulation:
                    l = v.getLoss()
                    loss[v.group] = loss[v.group].to(l.device)+l
        return loss.values()

    def resetLoss(self) -> None:
        for group in self._loss.keys():
            for v in self._loss[group].values():
                v.resetLoss()
        
    def getLastValues(self, n: int = 1) -> dict[str, float]:
        result = {}
        for group in self._loss.keys():
            result.update({name : np.nanmean(value._values[-n:] if n > 0 else value._values) for name, value in self._loss[group].items() if n < 0 or len(value._values) >= n})
        return result
    
    def getLastWeights(self, n: int = 1) -> dict[str, float]:
        result = {}
        for group in self._loss.keys():
            result.update({name : np.nanmean(value._weight[-n:] if n > 0 else value._weight) for name, value in self._loss[group].items() if n < 0 or len(value._values) >= n})
        return result

    def format(self, isLoss: bool, n: int) -> dict[str, tuple[float, float]]:
        result = dict()
        for group in self._loss.keys():
            for name, loss in self._loss[group].items():
                if loss.isLoss == isLoss and len(loss._values) >= n:
                    result[name] = (np.nanmean(loss._weight[-n:]), np.nanmean(loss._values[-n:]))
        return result

    def update_scheduler(self, schedulers: dict[Scheduler, int], it: int) -> Scheduler:
        step = 0
        scheduler = None
        for scheduler, value in schedulers.items():
            if value is None or (it >= step  and it < step+value):
                break
            step += value
        if scheduler:
            scheduler.step(it-step)
        return scheduler
    
class ModuleArgsDict(torch.nn.Module, ABC):
   
    class ModuleArgs:

        def __init__(self, in_branch: list[str], out_branch: list[str], pretrained : bool, alias : list[str], requires_grad: Union[bool, None], training: Union[None, bool]) -> None:
            super().__init__()
            self.alias= alias
            self.pretrained = pretrained
            self.in_branch = in_branch
            self.out_branch = out_branch
            self.in_channels = None
            self.in_is_channel = True
            self.out_channels = None
            self.out_is_channel = True
            self.requires_grad = requires_grad
            self.isCheckpoint = False
            self.isGPU_Checkpoint = False
            self.gpu = "cpu"
            self.training = training
            self._isEnd = False

    def __init__(self) -> None:
        super().__init__()
        self._modulesArgs : dict[str, ModuleArgsDict.ModuleArgs] = dict()
        self._training = NetState.TRAIN

    def _addindent(self, s_: str, numSpaces : int):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    def __repr__(self):
        extra_lines = []

        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        child_lines = []
        is_simple_branch = lambda x : len(x) > 1 or x[0] != 0 
        for key, module in self._modules.items():
            mod_str = repr(module)

            mod_str = self._addindent(mod_str, 2)
            desc = ""
            if is_simple_branch(self._modulesArgs[key].in_branch) or is_simple_branch(self._modulesArgs[key].out_branch):
                desc += ", {}->{}".format(self._modulesArgs[key].in_branch, self._modulesArgs[key].out_branch)
            if not self._modulesArgs[key].pretrained:
                desc += ", pretrained=False"
            if self._modulesArgs[key].alias:
                desc += ", alias={}".format(self._modulesArgs[key].alias)
            desc += ", in_channels={}".format(self._modulesArgs[key].in_channels)
            desc += ", in_is_channel={}".format(self._modulesArgs[key].in_is_channel)
            desc += ", out_channels={}".format(self._modulesArgs[key].out_channels)
            desc += ", out_is_channel={}".format(self._modulesArgs[key].out_is_channel)
            desc += ", is_end={}".format(self._modulesArgs[key]._isEnd)
            desc += ", isInCheckpoint={}".format(self._modulesArgs[key].isCheckpoint)
            desc += ", isInGPU_Checkpoint={}".format(self._modulesArgs[key].isGPU_Checkpoint)
            desc += ", requires_grad={}".format(self._modulesArgs[key].requires_grad)
            desc += ", device={}".format(self._modulesArgs[key].gpu)
            
            child_lines.append("({}{}) {}".format(key, desc, mod_str))
            
        lines = extra_lines + child_lines

        desc = ""
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                desc += extra_lines[0]
            else:
                desc += '\n  ' + '\n  '.join(lines) + '\n'

        return "{}({})".format(self._get_name(), desc)
    
    def __getitem__(self, key: str) -> torch.nn.Module:
        module = self._modules[key]
        assert module, "Error {} is None".format(key)
        return module 

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[tuple[str, Union[torch.nn.Module, None]]]:
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[Union[torch.nn.Module, None]]:
        return self._modules.values()

    def add_module(self, name: str, module : torch.nn.Module, in_branch: list[Union[int, str]] = [0], out_branch: list[Union[int, str]] = [0], pretrained : bool = True, alias : list[str] = [], requires_grad: Union[bool, None] = None, training: Union[None, bool] = None) -> None:
        super().add_module(name, module)
        self._modulesArgs[name] = ModuleArgsDict.ModuleArgs([str(value) for value in in_branch], [str(value) for value in out_branch], pretrained, alias, requires_grad, training)
    
    def getMap(self):
        results : dict[str, str] = {}
        for name, moduleArgs in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                if len(moduleArgs.alias):
                    count = {k : 0 for k in set(module.getMap().values())}
                    if len(count):
                        for k, v in module.getMap().items():
                            alias_name = moduleArgs.alias[count[v]]
                            if k == "":
                                results.update({alias_name : name+"."+v})
                            else:
                                results.update({alias_name+"."+k : name+"."+v})
                            count[v]+=1
                    else:
                        for alias in moduleArgs.alias:
                            results.update({alias : name})
                else:
                    results.update({k : name+"."+v for k, v in module.getMap().items()})
            else:
                for alias in moduleArgs.alias:
                    results[alias] = name
        return results

    @staticmethod
    def init_func(module: torch.nn.Module, init_type: str, init_gain: float):
        if not isinstance(module, Network):
            if isinstance(module, ModuleArgsDict):
                module.init(init_type, init_gain)
            elif isinstance(module, torch.nn.modules.conv._ConvNd) or isinstance(module, torch.nn.Linear):
                if init_type == 'normal':
                    torch.nn.init.normal_(module.weight, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(module.weight, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(module.weight, gain=init_gain)
                elif init_type == "trunc_normal":
                    torch.nn.init.trunc_normal_(module.weight, std=init_gain)
                else:
                    raise NotImplementedError('Initialization method {} is not implemented'.format(init_type))
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if module.weight is not None:
                    torch.nn.init.normal_(module.weight, 0.0, std = init_gain)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def init(self, init_type: str, init_gain: float):
        for module in self._modules.values():
            ModuleArgsDict.init_func(module, init_type, init_gain)


    def named_forward(self, *inputs: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:    
        if len(inputs) > 0:
            branchs: dict[str, torch.Tensor] = {}
            for i, sinput in enumerate(inputs):
                branchs[str(i)] = sinput

            out = inputs[0]
            tmp = []
            for name, module in self.items():
                if self._modulesArgs[name].training is None or (not (self._modulesArgs[name].training and self._training == NetState.PREDICTION) and not (not self._modulesArgs[name].training and self._training == NetState.TRAIN)):
                    requires_grad = self._modulesArgs[name].requires_grad
                    if requires_grad is not None and module:
                        module.requires_grad_(requires_grad)
                    for ib in self._modulesArgs[name].in_branch:
                        if ib not in branchs:
                            branchs[ib] = inputs[0]
                    for branchs_key in branchs.keys():
                        if str(self._modulesArgs[name].gpu) != 'cpu' and str(branchs[branchs_key].device) != "cuda:"+self._modulesArgs[name].gpu:
                            branchs[branchs_key] = branchs[branchs_key].to(int(self._modulesArgs[name].gpu))
                    
                    if self._modulesArgs[name].isCheckpoint:
                        out = checkpoint(module, *[branchs[i] for i in self._modulesArgs[name].in_branch], use_reentrant=True)
                        for ob in self._modulesArgs[name].out_branch:
                            branchs[ob] = out
                        yield name, out
                    else:
                        if isinstance(module, ModuleArgsDict):
                            for k, out in module.named_forward(*[branchs[i] for i in self._modulesArgs[name].in_branch]):
                                for ob in self._modulesArgs[name].out_branch:
                                    if ob in module._modulesArgs[k.split(".")[0].replace(";accu;", "")].out_branch:
                                        tmp.append(ob)
                                        branchs[ob] = out
                                yield name+"."+k, out
                            for ob in self._modulesArgs[name].out_branch:
                                if ob not in tmp:
                                    branchs[ob] = out
                        elif isinstance(module, torch.nn.Module):
                            out = module(*[branchs[i] for i in self._modulesArgs[name].in_branch])                    
                            for ob in self._modulesArgs[name].out_branch:
                                branchs[ob] = out
                            yield name, out
            del branchs

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        v = input
        for k, v in self.named_forward(*input):
            pass
        return v

    def named_parameters(self, pretrained: bool = False, recurse=False) -> Iterator[tuple[str, torch.nn.parameter.Parameter]]:
        for name, moduleArgs in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                for k, v in module.named_parameters(pretrained = pretrained):
                    yield name+"."+k, v
            elif isinstance(module, torch.nn.Module):
                if not pretrained or not moduleArgs.pretrained:
                    if moduleArgs.training is None or moduleArgs.training:
                        for k, v in module.named_parameters():
                            yield name+"."+k, v

    def parameters(self, pretrained: bool = False):
        for _, v in self.named_parameters(pretrained = pretrained):
            yield v
    
    def named_ModuleArgsDict(self) -> Iterator[tuple[str, Self, ModuleArgs]]:
        for name, module in self._modules.items():
            yield name, module, self._modulesArgs[name]
            if isinstance(module, ModuleArgsDict):
                for k, v, u in module.named_ModuleArgsDict():
                    yield name+"."+k, v, u

    def _requires_grad(self, keys: list[str]):
        keys = keys.copy()
        for name, module, args in self.named_ModuleArgsDict():
            requires_grad = args.requires_grad
            if requires_grad is not None:
                module.requires_grad_(requires_grad)
            if name in keys:
                keys.remove(name)
                if len(keys) == 0:
                    break

class OutputsGroup(list):

    def __init__(self, measure: Measure) -> None:
        self.layers: dict[str, torch.Tensor] = {}
        self.measure = measure

    def addLayer(self, name: str, layer: torch.Tensor):
        self.layers[name] = layer

    def isDone(self):
        return len(self) == len(self.layers)
    
    def clear(self):
        self.layers.clear()

 

class Network(ModuleArgsDict, ABC):

    def _apply_network(self, name_function : Callable[[Self], str], networks: list[str], key: str, function: Callable, *args, **kwargs) -> dict[str, object]:
        results : dict[str, object] = {}
        for module in self.values():
            if isinstance(module, Network):
                if name_function(module) not in networks:
                    networks.append(name_function(module))
                    for k, v in module._apply_network(name_function, networks, key+"."+name_function(module), function, *args, **kwargs).items():
                        results.update({name_function(self)+"."+k : v})
        if len([param.name for param in list(inspect.signature(function).parameters.values()) if param.name == "key"]):
            function = partial(function, key=key)

        results[name_function(self)] = function(self, *args, **kwargs)
        return results
    
    def _function_network(t : bool = False):
        def _function_network_d(function : Callable):
            def new_function(self : Self, *args, **kwargs) -> dict[str, object]:
                return self._apply_network(lambda network: network._getName() if t else network.getName(), [], self.getName(), function, *args, **kwargs)
            return new_function
        return _function_network_d

    def __init__(   self,
                    in_channels : int = 1,
                    optimizer: Union[OptimizerLoader, None] = None, 
                    schedulers: Union[LRSchedulersLoader, None] = None, 
                    outputsCriterions: Union[dict[str, TargetCriterionsLoader], None] = None,
                    patch : Union[ModelPatch, None] = None,
                    nb_batch_per_step : int = 1,
                    init_type : str = "normal",
                    init_gain : float = 0.02,
                    dim : int = 3) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.in_channels = in_channels
        self.optimizerLoader  = optimizer
        self.optimizer : Union[torch.optim.Optimizer, None] = None

        self.LRSchedulersLoader  = schedulers
        self.schedulers : Union[dict[torch.optim.lr_scheduler._LRScheduler, int], None] = None

        self.outputsCriterionsLoader = outputsCriterions
        self.measure : Union[Measure, None] = None

        self.patch = patch

        self.nb_batch_per_step = nb_batch_per_step
        self.init_type  = init_type
        self.init_gain  = init_gain
        self.dim = dim
        self._it = 0
        self.outputsGroup : list[OutputsGroup]= []

    @_function_network(True)
    def state_dict(self) -> dict[str, OrderedDict]:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        destination._metadata[""] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, "", False)
        for name, module in self._modules.items():
            if module is not None:
                if not isinstance(module, Network):   
                    module.state_dict(destination=destination, prefix="" + name + '.', keep_vars=False)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, "", local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination
    
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        error_msgs: list[str] = []

        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata 

        def load(module: torch.nn.Module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    if not isinstance(child, Network):
                        if isinstance(child, torch.nn.modules.conv._ConvNd) or isinstance(module, torch.nn.Linear):

                            current_size = child.weight.shape[0]
                            last_size = state_dict[prefix + name+".weight"].shape[0]

                            if current_size != last_size:
                                print("Warning: The size of '{}' has changed from {} to {}. Please check for potential impacts".format(prefix + name, last_size, current_size))
                                ModuleArgsDict.init_func(child, self.init_type, self.init_gain)

                                with torch.no_grad():
                                    child.weight[:last_size] = state_dict[prefix + name+".weight"]
                                    if child.bias is not None:
                                        child.bias[:last_size] = state_dict[prefix + name+".bias"]
                                return
                        load(child, prefix + name + '.')

        load(self)
        del load

        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        
    def apply(self, fn: Callable[[torch.nn.Module], None]) -> None:
        for module in self.children():
            if not isinstance(module, Network):
                module.apply(fn)
        fn(self)
        
    @_function_network(True)
    def load(self, state_dict : dict[str, dict[str, torch.Tensor]], init: bool = True, ema : bool =False):
        if init:
            self.apply(partial(ModuleArgsDict.init_func, init_type=self.init_type, init_gain=self.init_gain))
        name = "Model" + ("_EMA" if ema else "")
        if name in state_dict:
            model_state_dict_tmp = {k.split(".")[-1] : v for k, v in state_dict[name].items()}[self._getName()]
            map = self.getMap()
            model_state_dict : OrderedDict[str, torch.Tensor] = OrderedDict()
            
            for alias in model_state_dict_tmp.keys():
                prefix = ".".join(alias.split(".")[:-1])
                alias_list = [(".".join(prefix.split(".")[:len(i.split("."))]), v) for i, v in map.items() if prefix.startswith(i)]

                if len(alias_list):
                    for a, b in alias_list:
                        model_state_dict[alias.replace(a, b)] = model_state_dict_tmp[alias]
                        break
                else:
                    model_state_dict[alias] = model_state_dict_tmp[alias]
            self.load_state_dict(model_state_dict)
            
        if "{}_optimizer_state_dict".format(name) in state_dict and self.optimizer:
            self.optimizer.load_state_dict(state_dict['{}_optimizer_state_dict'.format(name)])
        self.initialized()

    def _compute_channels_trace(self, module : ModuleArgsDict, in_channels : int, gradient_checkpoints: Union[list[str], None], gpu_checkpoints: Union[list[str], None], name: Union[str, None] = None, in_is_channel = True, out_channels : Union[int, None] = None, out_is_channel = True) -> tuple[int, bool, int, bool]:
        
        for k1, v1 in module.items():
            if isinstance(v1, ModuleArgsDict):
                for t in module._modulesArgs[k1].out_branch:
                    last = None
                    for k2, _ in v1.items():
                        if t in v1._modulesArgs[k2].out_branch:
                            last = k2
                    if last is not None:
                        v1._modulesArgs[last]._isEnd = True
                    else:
                        v1._modulesArgs[k2]._isEnd = True

        for k, v in module.items():
            if hasattr(v, "in_channels"):
                if v.in_channels:
                    in_channels = v.in_channels
            if hasattr(v, "in_features"):
                if v.in_features:
                    in_channels = v.in_features
            key = name+"."+k if name else k

            if gradient_checkpoints:
                if key in gradient_checkpoints:
                    module._modulesArgs[k].isCheckpoint = True
            
            if gpu_checkpoints:
                if key in gpu_checkpoints:
                    module._modulesArgs[k].isGPU_Checkpoint = True
           
            module._modulesArgs[k].in_channels = in_channels
            module._modulesArgs[k].in_is_channel = in_is_channel
            
            if isinstance(v, ModuleArgsDict):
                in_channels, in_is_channel, out_channels, out_is_channel = self._compute_channels_trace(v, in_channels, gradient_checkpoints, gpu_checkpoints, key, in_is_channel, out_channels, out_is_channel)
            
            if v.__class__.__name__ == "ToChannels":
                out_is_channel = True
            
            if v.__class__.__name__ == "ToFeatures":
                out_is_channel = False

            if hasattr(v, "out_channels"):
                if v.out_channels:
                    out_channels = v.out_channels
            if hasattr(v, "out_features"):
                if v.out_features:
                    out_channels = v.out_features

            module._modulesArgs[k].out_channels = out_channels
            module._modulesArgs[k].out_is_channel = out_is_channel

            in_channels = out_channels
            in_is_channel = out_is_channel
            
        return in_channels, in_is_channel, out_channels, out_is_channel

    @_function_network()
    def init(self, autocast : bool, state : State, key: str) -> None: 
        if self.outputsCriterionsLoader:
            self.measure = Measure(key, self.outputsCriterionsLoader)
            self.measure.init(self)
        if state != State.PREDICTION:
            self.scaler = torch.amp.GradScaler("cuda", enabled=autocast)
            if self.optimizerLoader:
                self.optimizer = self.optimizerLoader.getOptimizer(key, self.parameters(state == State.TRANSFER_LEARNING))
                self.optimizer.zero_grad()

            if self.LRSchedulersLoader and self.optimizer:
                self.schedulers = self.LRSchedulersLoader.getShedulers(key, self.optimizer)
    
    def initialized(self):
        pass

    def named_forward(self, *inputs: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        if self.patch:
            self.patch.load(inputs[0].shape[2:])
            accumulators: dict[str, Accumulator] = {}

            patchIterator = self.patch.disassemble(*inputs)
            buffer = []
            for i, patch_input in enumerate(patchIterator):
                for (name, output_layer) in super().named_forward(*patch_input):
                    yield "{}{}".format(";accu;", name), output_layer
                    buffer.append((name.split(".")[0], output_layer))
                    if len(buffer) == 2:
                        if buffer[0][0] != buffer[1][0]:
                            if self._modulesArgs[buffer[0][0]]._isEnd:
                                if buffer[0][0] not in accumulators:
                                    accumulators[buffer[0][0]] = Accumulator(self.patch.getPatch_slices(), self.patch.patch_size, self.patch.patchCombine)
                                accumulators[buffer[0][0]].addLayer(i, buffer[0][1])
                        buffer.pop(0)
                if self._modulesArgs[buffer[0][0]]._isEnd:
                    if buffer[0][0] not in accumulators:
                        accumulators[buffer[0][0]] = Accumulator(self.patch.getPatch_slices(), self.patch.patch_size, self.patch.patchCombine)
                    accumulators[buffer[0][0]].addLayer(i, buffer[0][1])
            for name, accumulator in accumulators.items():
                yield name, accumulator.assemble()
        else:
            for (name, output_layer) in super().named_forward(*inputs):
                yield name, output_layer


    def get_layers(self, inputs : list[torch.Tensor], layers_name: list[str]) -> Iterator[tuple[str, torch.Tensor, Union[Patch_Indexed, None]]]:
        layers_name = layers_name.copy()
        output_layer_accumulator : dict[str, Accumulator] = {}
        output_layer_patch_indexed : dict[str, Patch_Indexed] = {}
        it = 0
        debug = "DL_API_DEBUG" in os.environ
        for (nameTmp, output_layer) in self.named_forward(*inputs):
            name = nameTmp.replace(";accu;", "")
            if debug:
                if "DL_API_DEBUG_LAST_LAYER" in os.environ:
                    os.environ["DL_API_DEBUG_LAST_LAYER"] = "{}|{}:{}:{}".format(os.environ["DL_API_DEBUG_LAST_LAYER"], name, getGPUMemory(output_layer.device), str(output_layer.device).replace("cuda:", ""))
                else:
                    os.environ["DL_API_DEBUG_LAST_LAYER"] = "{}:{}:{}".format(name, getGPUMemory(output_layer.device), str(output_layer.device).replace("cuda:", ""))
            it += 1
            if name in layers_name or nameTmp in layers_name:
                if ";accu;" in nameTmp:
                    if name not in output_layer_patch_indexed:
                        networkName = nameTmp.split(".;accu;")[-2].split(".")[-1] if ".;accu;" in nameTmp  else nameTmp.split(";accu;")[-2].split(".")[-1]
                        module = self
                        network = None
                        if networkName == "":
                            network = module
                        else:
                            for n in name.split("."):
                                module = module[n]
                                if isinstance(module, Network) and n == networkName:
                                    network = module
                                    break
                        
                        if network and network.patch:
                            output_layer_patch_indexed[name] = Patch_Indexed(network.patch, 0)

                    if name not in output_layer_accumulator:
                        output_layer_accumulator[name] = Accumulator(output_layer_patch_indexed[name].patch.getPatch_slices(0), output_layer_patch_indexed[name].patch.patch_size, output_layer_patch_indexed[name].patch.patchCombine)
                    
                    if nameTmp in layers_name:
                        output_layer_accumulator[name].addLayer(output_layer_patch_indexed[name].index, output_layer)
                        output_layer_patch_indexed[name].index += 1
                        if output_layer_accumulator[name].isFull():
                            output_layer = output_layer_accumulator[name].assemble()
                            output_layer_accumulator.pop(name)
                            output_layer_patch_indexed.pop(name)
                            layers_name.remove(nameTmp)
                            yield nameTmp, output_layer, None

                if name in layers_name:
                    if ";accu;" in nameTmp:
                        yield name, output_layer, output_layer_patch_indexed[name]
                        output_layer_patch_indexed[name].index += 1
                        if output_layer_patch_indexed[name].isFull():
                            output_layer_patch_indexed.pop(name)
                            layers_name.remove(name)
                    else:
                        layers_name.remove(name)
                        yield name, output_layer, None

            if not len(layers_name):
                break
    


    def init_outputsGroup(self):
        metric_tmp = {network.measure : network.measure.outputsCriterions.keys() for network in self.getNetworks().values() if network.measure}
        for k, v in metric_tmp.items():
            for a in v:
                outputs_group = OutputsGroup(k)
                outputs_group.append(a)
                for targetsGroup in k.outputsCriterions[a].keys():
                    if ":" in targetsGroup:
                        outputs_group.append(targetsGroup.replace(":", "."))

                self.outputsGroup.append(outputs_group)

    def forward(self, data_dict: dict[tuple[str, bool], torch.Tensor], output_layers: list[str] = []) -> list[tuple[str, torch.Tensor]]:
        if not len(self.outputsGroup) and not len(output_layers):
            return []

        self.resetLoss()
        results = []
        measure_output_layers = set()
        for outputs_group in self.outputsGroup:
             for name in outputs_group:
                measure_output_layers.add(name)
        for name, layer, patch_indexed in self.get_layers([v for k, v in data_dict.items() if k[1]], list(set(list(measure_output_layers)+output_layers))):
            
            outputs_group = [outputs_group for outputs_group in self.outputsGroup if name in outputs_group]
            if len(outputs_group) > 0:
                if patch_indexed is None:
                    targets = {k[0] : v for k, v in data_dict.items()}
                    nb = 1
                else:
                    targets = {k[0] : patch_indexed.patch.getData(v, patch_indexed.index, 0, False) for k, v in data_dict.items()}
                    nb = patch_indexed.patch.getSize(0)

                for output_group in outputs_group:
                    output_group.addLayer(name, layer)
                    if output_group.isDone():
                        targets.update({k.replace(".", ":"): v for k, v in output_group.layers.items() if k != output_group[0]})
                        output_group.measure.update(output_group[0], output_group.layers[output_group[0]], targets, self._it, nb, self.training)
                        output_group.clear()
            if name in output_layers:
                results.append((name, layer))
        return results

    @_function_network()
    def resetLoss(self):
        if self.measure:
            self.measure.resetLoss()

    @_function_network()
    def backward(self, model: Self):
        if self.measure:
            if self.scaler and self.optimizer:
                model._requires_grad(list(self.measure.outputsCriterions.keys()))
                for loss in self.measure.getLoss():
                    self.scaler.scale(loss / self.nb_batch_per_step).backward()
                    if self._it % self.nb_batch_per_step == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                self._it += 1

    @_function_network()
    def update_lr(self):
        step = 0
        scheduler = None
        if self.schedulers:
            for scheduler, value in self.schedulers.items():
                if value is None or (self._it >= step  and self._it < step+value):
                    break
                step += value
        if scheduler:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                if self.measure:
                    print(sum(self.measure.getLastValues(0).values()))
                    scheduler.step(sum(self.measure.getLastValues(0).values()))
            else:
                scheduler.step()     

    @_function_network()
    def getNetworks(self) -> Self:
        return self

    def to(module : ModuleArgsDict, device: int):
        if "device" not in os.environ:
            os.environ["device"] = str(device)
        for k, v in module.items():
            if module._modulesArgs[k].gpu == "cpu":
                if module._modulesArgs[k].isGPU_Checkpoint:
                    os.environ["device"] = str(int(os.environ["device"])+1)
                module._modulesArgs[k].gpu = str(getDevice(int(os.environ["device"])))
                if isinstance(v, ModuleArgsDict):
                    v = Network.to(v, int(os.environ["device"]))
                else:
                    v = v.to(getDevice(int(os.environ["device"])))
        return module
                
    def getName(self) -> str:
        return self.__class__.__name__
    
    def setName(self, name: str) -> Self:
        self.name = name
        return self
    
    def _getName(self) -> str:
        return self.name
    
    def setState(self, state: NetState):
        for module in self.modules():
            if isinstance(module, ModuleArgsDict):
                module._training = state

class ModelLoader():

    @config("Model")
    def __init__(self, classpath : str = "default:segmentation.UNet") -> None:
        self.module, self.name = _getModule(classpath.split(".")[-1] if len(classpath.split(".")) > 1 else classpath, ".".join(classpath.split(".")[:-1]) if len(classpath.split(".")) > 1 else "")
        
    def getModel(self, train : bool = True, DL_args: Union[str, None] = None, DL_without=["optimizer", "schedulers", "nb_batch_per_step", "init_type", "init_gain"]) -> Network:
        if not DL_args:
            DL_args="{}.Model".format(DEEP_LEARNING_API_ROOT())
        model = partial(getattr(importlib.import_module(self.module), self.name), config = None, DL_args=DL_args)
        if not train: 
            model = partial(model, DL_without = DL_without)
        return model()

class CPU_Model():

    def __init__(self, model: Network) -> None:
        self.module = model
        self.device = torch.device('cpu')

    def train(self):
        self.module.train()
    
    def eval(self):
        self.module.eval()
    
    def __call__(self, data_dict: dict[tuple[str, bool], torch.Tensor], output_layers: list[str] = []) -> list[tuple[str, torch.Tensor]]:
        return self.module(data_dict, output_layers)