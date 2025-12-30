import importlib
import inspect
import os
from abc import ABC
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from enum import Enum
from functools import partial
from typing import Any

try:
    from typing import Self  # Python ≥ 3.11
except ImportError:
    from typing_extensions import Self  # Python ≤ 3.10

import numpy as np
import torch
from torch._jit_internal import _copy_to_script_wrapper
from torch.utils.checkpoint import checkpoint

from konfai import konfai_root
from konfai.data.patching import Accumulator, ModelPatch
from konfai.metric.schedulers import Scheduler
from konfai.utils.config import apply_config, config
from konfai.utils.utils import MeasureError, State, TrainerError, get_device, get_gpu_memory, get_module


class NetState(Enum):
    TRAIN = (0,)
    PREDICTION = 1


class PatchIndexed:

    def __init__(self, patch: ModelPatch, index: int) -> None:
        self.patch = patch
        self.index = index

    def is_full(self) -> bool:
        return len(self.patch.get_patch_slices(0)) == self.index


class OptimizerLoader:

    def __init__(self, name: str = "AdamW") -> None:
        self.name = name

    def get_optimizer(self, key: str, parameter: Iterator[torch.nn.parameter.Parameter]) -> torch.optim.Optimizer:
        return apply_config(f"{konfai_root()}.Model.{key}.optimizer")(
            getattr(importlib.import_module("torch.optim"), self.name)
        )(parameter)


class LRSchedulersLoader:

    def __init__(self, nb_step: int = 0) -> None:
        self.nb_step = nb_step

    def getschedulers(
        self, key: str, scheduler_classname: str, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        for m in ["torch.optim.lr_scheduler", "konfai.metric.schedulers"]:
            module, name = get_module(scheduler_classname, m)
            if hasattr(module, name):
                return apply_config(f"{konfai_root()}.Model.{key}.schedulers.{scheduler_classname}")(
                    getattr(module, name)
                )(optimizer)
        raise TrainerError(
            f"Unknown scheduler {scheduler_classname}, tried importing from: 'torch.optim.lr_scheduler' and "
            "'konfai.metric.schedulers', but no valid match was found. "
            "Check your YAML config or scheduler name spelling."
        )


class LossSchedulersLoader:

    def __init__(self, nb_step: int = 0) -> None:
        self.nb_step = nb_step

    def getschedulers(self, key: str, scheduler_classname: str) -> torch.optim.lr_scheduler._LRScheduler:
        return apply_config(f"{key}.{scheduler_classname}")(
            getattr(importlib.import_module("konfai.metric.schedulers"), scheduler_classname)
        )()


class CriterionsAttr:

    def __init__(
        self,
        schedulers: dict[str, LossSchedulersLoader] = {"default|Constant": LossSchedulersLoader(0)},
        is_loss: bool = True,
        group: int = 0,
        start: int = 0,
        stop: int | None = None,
        accumulation: bool = False,
    ) -> None:
        self.schedulersLoader = schedulers
        self.isTorchCriterion = True
        self.is_loss = is_loss
        self.start = start
        self.stop = stop
        self.group = group
        self.accumulation = accumulation
        self.schedulers: dict[Scheduler, int] = {}


class CriterionsLoader:

    def __init__(
        self,
        criterions_loader: dict[str, CriterionsAttr] = {"default|torch:nn:CrossEntropyLoss|Dice|NCC": CriterionsAttr()},
    ) -> None:
        self.criterions_loader = criterions_loader

    def get_criterions(
        self, model_classname: str, output_group: str, target_group: str
    ) -> dict[torch.nn.Module, CriterionsAttr]:
        criterions = {}
        for module_classpath, criterions_attr in self.criterions_loader.items():
            module, name = get_module(module_classpath, "konfai.metric.measure")
            criterions_attr.isTorchCriterion = module.__name__.startswith("torch")
            for (
                scheduler_classname,
                schedulers,
            ) in criterions_attr.schedulersLoader.items():
                criterions_attr.schedulers[
                    schedulers.getschedulers(
                        f"{konfai_root()}.Model.{model_classname}.outputs_criterions.{output_group}"
                        f".targets_criterions.{target_group}"
                        f".criterions_loader.{module_classpath}.schedulers",
                        scheduler_classname,
                    )
                ] = schedulers.nb_step
            criterions[
                apply_config(
                    f"{konfai_root()}.Model.{model_classname}.outputs_criterions."
                    f"{output_group}.targets_criterions.{target_group}."
                    f"criterions_loader.{module_classpath}"
                )(getattr(module, name))()
            ] = criterions_attr
        return criterions


class TargetCriterionsLoader:

    def __init__(
        self,
        targets_criterions: dict[str, CriterionsLoader] = {"Labels": CriterionsLoader()},
    ) -> None:
        self.targets_criterions = targets_criterions

    def get_targets_criterions(
        self, output_group: str, model_classname: str
    ) -> dict[str, dict[torch.nn.Module, CriterionsAttr]]:
        targets_criterions = {}
        for target_group, criterions_loader in self.targets_criterions.items():
            targets_criterions[target_group] = criterions_loader.get_criterions(
                model_classname, output_group, target_group
            )
        return targets_criterions


class Measure:

    class Loss:

        def __init__(
            self,
            name: str,
            output_group: str,
            target_group: str,
            group: int,
            is_loss: bool,
            accumulation: bool,
        ) -> None:
            self.name = name
            self.is_loss = is_loss
            self.accumulation = accumulation
            self.output_group = output_group
            self.target_group = target_group
            self.group = group

            self._loss: list[torch.Tensor] = []
            self._weight: list[float] = []
            self._values: list[float] = []

        def reset_loss(self) -> None:
            self._loss.clear()

        def add(self, weight: float, value: torch.Tensor | tuple[torch.Tensor, float]) -> None:
            if isinstance(value, tuple):
                loss_value, true_value = value
            else:
                loss_value = value
                true_value = value.item()

            self._loss.append(loss_value if self.is_loss else loss_value.detach())
            self._values.append(true_value)
            self._weight.append(weight)

        def get_last_loss(self) -> torch.Tensor:
            return self._loss[-1] * self._weight[-1] if len(self._loss) else torch.zeros((1), requires_grad=True)

        def get_loss(self) -> torch.Tensor:
            return (
                torch.stack([w * loss_value for w, loss_value in zip(self._weight, self._loss)], dim=0).mean(dim=0)
                if len(self._loss)
                else torch.zeros((1), requires_grad=True)
            )

        def __len__(self) -> int:
            return len(self._loss)

    def __init__(
        self,
        model_classname: str,
        outputs_criterions_loader: dict[str, TargetCriterionsLoader],
    ) -> None:
        super().__init__()
        self.outputs_criterions: dict[str, dict[str, dict[torch.nn.Module, CriterionsAttr]]] = {}
        for output_group, target_criterions_loader in outputs_criterions_loader.items():
            self.outputs_criterions[output_group.replace(":", ".")] = target_criterions_loader.get_targets_criterions(
                output_group, model_classname
            )
        self._loss: dict[int, dict[str, Measure.Loss]] = {}

    def init(self, model: torch.nn.Module, group_dest: list[str]) -> None:
        outputs_group_rename = {}

        modules = []
        for i, _, _ in model.named_module_args_dict():
            modules.append(i)

        for output_group in self.outputs_criterions.keys():
            if output_group.replace(";accu;", "") not in modules:
                raise MeasureError(
                    f"The output group '{output_group}' defined in 'outputs_criterions' "
                    "does not correspond to any module in the model.",
                    f"Available modules: {modules}",
                    "Please check that the name matches exactly a submodule or output of your model architecture.",
                )
            for target_group in self.outputs_criterions[output_group]:
                for target_group_tmp in target_group.split(";"):
                    if target_group_tmp not in group_dest:
                        raise MeasureError(
                            f"The target_group {target_group_tmp} defined in "
                            "'outputs_criterions.{output_group}.targets_criterions'"
                            " was not found in the available destination groups.",
                            "This target_group is expected for loss or metric computation, "
                            "but was not loaded in 'group_dest'.",
                            f"Please make sure that the group {target_group_tmp} is defined in "
                            "Dataset:groups_src:...:groups_dest: {target_group_tmp} "
                            "and correctly loaded from the dataset.",
                        )
                for criterion in self.outputs_criterions[output_group][target_group]:
                    if not self.outputs_criterions[output_group][target_group][criterion].isTorchCriterion:
                        outputs_group_rename[output_group] = criterion.init(model, output_group, target_group)

        outputs_criterions_bak = self.outputs_criterions.copy()
        for old, new in outputs_group_rename.items():
            self.outputs_criterions.pop(old)
            self.outputs_criterions[new] = outputs_criterions_bak[old]
        for output_group in self.outputs_criterions:
            for target_group in self.outputs_criterions[output_group]:
                for criterion, criterions_attr in self.outputs_criterions[output_group][target_group].items():
                    if criterions_attr.group not in self._loss:
                        self._loss[criterions_attr.group] = {}
                    self._loss[criterions_attr.group][
                        f"{output_group}:{target_group}:{criterion.__class__.__name__}"
                    ] = Measure.Loss(
                        criterion.__class__.__name__,
                        output_group,
                        target_group,
                        criterions_attr.group,
                        criterions_attr.is_loss,
                        criterions_attr.accumulation,
                    )

    def update(
        self,
        output_group: str,
        output: torch.Tensor,
        data_dict: dict[str, torch.Tensor],
        it: int,
        nb_patch: int,
        training: bool,
    ) -> None:
        for target_group in self.outputs_criterions[output_group]:
            target = [
                data_dict[group].to(output[0].device).detach()
                for group in target_group.split(";")
                if group in data_dict
            ]

            for criterion, criterions_attr in self.outputs_criterions[output_group][target_group].items():
                if it >= criterions_attr.start and (criterions_attr.stop is None or it <= criterions_attr.stop):
                    scheduler = self.update_scheduler(criterions_attr.schedulers, it)
                    self._loss[criterions_attr.group][
                        f"{output_group}:{target_group}:{criterion.__class__.__name__}"
                    ].add(scheduler.get_value(), criterion(output, *target))
                    if (
                        training
                        and len(
                            np.unique(
                                [
                                    len(loss_value)
                                    for loss_value in self._loss[criterions_attr.group].values()
                                    if loss_value.accumulation and loss_value.is_loss
                                ]
                            )
                        )
                        == 1
                    ):
                        if criterions_attr.is_loss:
                            loss = torch.zeros((1), requires_grad=True)
                            for v in [
                                loss_value
                                for loss_value in self._loss[criterions_attr.group].values()
                                if loss_value.accumulation and loss_value.is_loss
                            ]:
                                loss_value = v.get_last_loss()
                                loss = loss.to(loss_value.device) + loss_value
                            loss = loss / nb_patch
                            loss.backward()

    def get_loss(self) -> list[torch.Tensor]:
        loss: dict[int, torch.Tensor] = {}
        for group in self._loss.keys():
            loss[group] = torch.zeros((1), requires_grad=True)
            for v in self._loss[group].values():
                if v.is_loss and not v.accumulation:
                    loss_value = v.get_loss()
                    loss[v.group] = loss[v.group].to(loss_value.device) + loss_value
        return list(loss.values())

    def reset_loss(self) -> None:
        for group in self._loss.keys():
            for v in self._loss[group].values():
                v.reset_loss()

    def get_last_values(self, n: int = 1) -> dict[str, float]:
        result = {}
        for group in self._loss.keys():
            result.update(
                {
                    name: np.nanmean(value._values[-n:] if n > 0 else value._values)
                    for name, value in self._loss[group].items()
                    if n < 0 or len(value._values) >= n
                }
            )
        return result

    def get_last_weights(self, n: int = 1) -> dict[str, float]:
        result = {}
        for group in self._loss.keys():
            result.update(
                {
                    name: np.nanmean(value._weight[-n:] if n > 0 else value._weight)
                    for name, value in self._loss[group].items()
                    if n < 0 or len(value._values) >= n
                }
            )
        return result

    def format_loss(self, is_loss: bool, n: int) -> dict[str, tuple[float, float]]:
        result = {}
        for group in self._loss.keys():
            for name, loss in self._loss[group].items():
                if loss.is_loss == is_loss and len(loss._values) >= n:
                    result[name] = (
                        np.nanmean(loss._weight[-n:]),
                        np.nanmean(loss._values[-n:]),
                    )
        return result

    def update_scheduler(self, schedulers: dict[Scheduler, int], it: int) -> Scheduler:
        step = 0
        _scheduler = None
        for _scheduler, value in schedulers.items():
            if value is None or (it >= step and it < step + value):
                break
            step += value
        if _scheduler:
            _scheduler.step(it - step)
        if _scheduler is None:
            raise NameError(
                f"No scheduler found for iteration {it}. "
                f"Available steps were: {list(schedulers.values())}. "
                f"Check your configuration."
            )
        return _scheduler


class ModuleArgsDict(torch.nn.Module, ABC):

    class ModuleArgs:

        def __init__(
            self,
            in_branch: list[str],
            out_branch: list[str],
            pretrained: bool,
            alias: list[str],
            requires_grad: bool | None,
            training: None | bool,
        ) -> None:
            super().__init__()
            self.alias = alias
            self.pretrained = pretrained
            self.in_branch = in_branch
            self.out_branch = out_branch
            self.in_channels: int | None = None
            self.in_is_channel: bool = True
            self.out_channels: int | None = None
            self.out_is_channel: bool = True
            self.requires_grad = requires_grad
            self.isCheckpoint = False
            self.isGPU_Checkpoint = False
            self.gpu = "cpu"
            self.training = training
            self._isEnd = False

    def __init__(self) -> None:
        super().__init__()
        self._modulesArgs: dict[str, ModuleArgsDict.ModuleArgs] = {}
        self._training = NetState.TRAIN

    def _addindent(self, s_: str, num_spaces: int):
        s = s_.split("\n")
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * " ") + line for line in s]
        return first + "\n" + "\n".join(s)

    def __repr__(self):
        extra_lines = []

        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []

        def is_simple_branch(x):
            return len(x) > 1 or x[0] != 0

        for key, module in self._modules.items():
            mod_str = repr(module)

            mod_str = self._addindent(mod_str, 2)
            desc = ""
            if is_simple_branch(self._modulesArgs[key].in_branch) or is_simple_branch(
                self._modulesArgs[key].out_branch
            ):
                desc += f", {self._modulesArgs[key].in_branch}->{self._modulesArgs[key].out_branch}"
            if not self._modulesArgs[key].pretrained:
                desc += ", pretrained=False"
            if self._modulesArgs[key].alias:
                desc += f", alias={self._modulesArgs[key].alias}"
            desc += f", in_channels={self._modulesArgs[key].in_channels}"
            desc += f", in_is_channel={self._modulesArgs[key].in_is_channel}"
            desc += f", out_channels={self._modulesArgs[key].out_channels}"
            desc += f", out_is_channel={self._modulesArgs[key].out_is_channel}"
            desc += f", is_end={self._modulesArgs[key]._isEnd}"
            desc += f", isInCheckpoint={self._modulesArgs[key].isCheckpoint}"
            desc += f", isInGPU_Checkpoint={self._modulesArgs[key].isGPU_Checkpoint}"
            desc += f", requires_grad={self._modulesArgs[key].requires_grad}"
            desc += f", device={self._modulesArgs[key].gpu}"

            child_lines.append(f"({key}{desc}) {mod_str}")

        lines = extra_lines + child_lines

        desc = ""
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                desc += extra_lines[0]
            else:
                desc += "\n  " + "\n  ".join(lines) + "\n"

        return f"{self._get_name()}({desc})"

    def __getitem__(self, key: str) -> torch.nn.Module:
        module = self._modules[key]
        if not module:
            raise ValueError(f"Module '{key}' is None or missing in self._modules")
        return module

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[tuple[str, torch.nn.Module | None]]:
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[torch.nn.Module | None]:
        return self._modules.values()

    def add_module(
        self,
        name: str,
        module: torch.nn.Module,
        in_branch: Sequence[int | str] = [0],
        out_branch: Sequence[int | str] = [0],
        pretrained: bool = True,
        alias: list[str] = [],
        requires_grad: bool | None = None,
        training: None | bool = None,
    ) -> None:
        super().add_module(name, module)
        self._modulesArgs[name] = ModuleArgsDict.ModuleArgs(
            [str(value) for value in in_branch],
            [str(value) for value in out_branch],
            pretrained,
            alias,
            requires_grad,
            training,
        )

    def get_mapping(self):
        results: dict[str, str] = {}
        for name, module_args in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                if len(module_args.alias):
                    count = dict.fromkeys(set(module.get_mapping().values()), 0)
                    if len(count):
                        for k, v in module.get_mapping().items():
                            alias_name = module_args.alias[count[v]]
                            if k == "":
                                results.update({alias_name: name + "." + v})
                            else:
                                results.update({alias_name + "." + k: name + "." + v})
                            count[v] += 1
                    else:
                        for alias in module_args.alias:
                            results.update({alias: name})
                else:
                    results.update({k: name + "." + v for k, v in module.get_mapping().items()})
            else:
                for alias in module_args.alias:
                    results[alias] = name
        return results

    @staticmethod
    def init_func(module: torch.nn.Module, init_type: str, init_gain: float):
        if not isinstance(module, Network):
            if isinstance(module, ModuleArgsDict):
                module.init(init_type, init_gain)
            elif isinstance(module, torch.nn.modules.conv._ConvNd) or isinstance(module, torch.nn.Linear):
                if init_type == "normal":
                    torch.nn.init.normal_(module.weight, 0.0, init_gain)
                elif init_type == "xavier":
                    torch.nn.init.xavier_normal_(module.weight, gain=init_gain)
                elif init_type == "kaiming":
                    torch.nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    torch.nn.init.orthogonal_(module.weight, gain=init_gain)
                elif init_type == "trunc_normal":
                    torch.nn.init.trunc_normal_(module.weight, std=init_gain)
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if module.weight is not None:
                    torch.nn.init.normal_(module.weight, 0.0, std=init_gain)
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
                if self._modulesArgs[name].training is None or (
                    not (self._modulesArgs[name].training and self._training == NetState.PREDICTION)
                    and not (not self._modulesArgs[name].training and self._training == NetState.TRAIN)
                ):
                    requires_grad = self._modulesArgs[name].requires_grad
                    if requires_grad is not None and module:
                        module.requires_grad_(requires_grad)
                    for ib in self._modulesArgs[name].in_branch:
                        if ib not in branchs:
                            branchs[ib] = inputs[0]
                    for branchs_key in branchs.keys():
                        if (
                            str(self._modulesArgs[name].gpu) != "cpu"
                            and str(branchs[branchs_key].device) != "cuda:" + self._modulesArgs[name].gpu
                        ):
                            branchs[branchs_key] = branchs[branchs_key].to(int(self._modulesArgs[name].gpu))

                    if self._modulesArgs[name].isCheckpoint:
                        out = checkpoint(
                            module,
                            *[branchs[i] for i in self._modulesArgs[name].in_branch],
                            use_reentrant=True,
                        )
                        for ob in self._modulesArgs[name].out_branch:
                            branchs[ob] = out
                        yield name, out
                    else:
                        if isinstance(module, ModuleArgsDict):
                            for k, out in module.named_forward(
                                *[branchs[i] for i in self._modulesArgs[name].in_branch]
                            ):
                                for ob in self._modulesArgs[name].out_branch:
                                    if ob in module._modulesArgs[k.split(".")[0].replace(";accu;", "")].out_branch:
                                        tmp.append(ob)
                                        branchs[ob] = out
                                yield name + "." + k, out
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
        _v = input
        for _, _v in self.named_forward(*input):
            pass
        return _v

    def named_parameters(
        self, pretrained: bool = False, recurse=False
    ) -> Iterator[tuple[str, torch.nn.parameter.Parameter]]:
        for name, module_args in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                for k, v in module.named_parameters(pretrained=pretrained):
                    yield name + "." + k, v
            elif isinstance(module, torch.nn.Module):
                if not pretrained or not module_args.pretrained:
                    if module_args.training is None or module_args.training:
                        for k, v in module.named_parameters():
                            yield name + "." + k, v

    def parameters(self, pretrained: bool = False):
        for _, v in self.named_parameters(pretrained=pretrained):
            yield v

    def named_module_args_dict(self) -> Iterator[tuple[str, Self, ModuleArgs]]:
        for name, module in self._modules.items():
            yield name, module, self._modulesArgs[name]
            if isinstance(module, ModuleArgsDict):
                for k, v, u in module.named_module_args_dict():
                    yield name + "." + k, v, u

    def _requires_grad(self, keys: list[str]):
        keys = keys.copy()
        for name, module, args in self.named_module_args_dict():
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

    def add_layer(self, name: str, layer: torch.Tensor):
        self.layers[name] = layer

    def is_done(self):
        return len(self) == len(self.layers)

    def clear(self):
        self.layers.clear()


class Network(ModuleArgsDict, ABC):

    def _apply_network(
        self,
        name_function: Callable[[Self], str],
        networks: list[str],
        key: str,
        function: Callable,
        *args,
        **kwargs,
    ) -> dict[str, object]:
        results: dict[str, object] = {}
        for module in self.values():
            if isinstance(module, Network):
                if name_function(module) not in networks:
                    networks.append(name_function(module))
                    for k, v in module._apply_network(
                        name_function,
                        networks,
                        key + "." + name_function(module),
                        function,
                        *args,
                        **kwargs,
                    ).items():
                        results.update({name_function(self) + "." + k: v})
        if len([param.name for param in list(inspect.signature(function).parameters.values()) if param.name == "key"]):
            function = partial(function, key=key)

        results[name_function(self)] = function(self, *args, **kwargs)
        return results

    def _function_network():  # type: ignore[misc]
        def _function_network_d(function: Callable):
            def new_function(self: Self, *args, **kwargs) -> dict[str, object]:
                return self._apply_network(
                    lambda network: network.get_name(),
                    [],
                    self.get_name(),
                    function,
                    *args,
                    **kwargs,
                )

            return new_function

        return _function_network_d

    def __init__(
        self,
        in_channels: int = 1,
        optimizer: OptimizerLoader | None = None,
        schedulers: dict[str, LRSchedulersLoader] | None = None,
        outputs_criterions: dict[str, TargetCriterionsLoader] | None = None,
        patch: ModelPatch | None = None,
        nb_batch_per_step: int = 1,
        init_type: str = "normal",
        init_gain: float = 0.02,
        dim: int = 3,
    ) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.in_channels = in_channels
        self.optimizerLoader = optimizer
        self.optimizer: torch.optim.Optimizer | None = None

        self.lr_schedulers_loader = schedulers
        self.schedulers: dict[torch.optim.lr_scheduler._LRScheduler, int] = {}

        self.outputs_criterions_loader = outputs_criterions
        self.measure: Measure | None = None

        self.patch = patch

        self.nb_batch_per_step = nb_batch_per_step
        self.init_type = init_type
        self.init_gain = init_gain
        self.dim = dim
        self._it = 0
        self._nb_lr_update = 0
        self.outputsGroup: list[OutputsGroup] = []

    @_function_network()
    def state_dict(self) -> dict[str, OrderedDict]:
        destination: OrderedDict[str, Any] = OrderedDict()
        local_metadata = {"version": self._version}
        # destination["_metadata"] = OrderedDict({"": local_metadata})
        self._save_to_state_dict(destination, "", False)
        for name, module in self._modules.items():
            if module is not None:
                if not isinstance(module, Network):
                    module.state_dict(destination=destination, prefix="" + name + ".", keep_vars=False)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, "", local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        error_msgs: list[str] = []

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict["_metadata"] = metadata

        def load(module: torch.nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    if not isinstance(child, Network):
                        if isinstance(child, torch.nn.modules.conv._ConvNd) or isinstance(module, torch.nn.Linear):

                            current_size = child.weight.shape[0]
                            last_size = state_dict[prefix + name + ".weight"].shape[0]

                            if current_size != last_size:
                                print(
                                    f"Warning: The size of '{prefix + name}' has changed from {last_size}"
                                    f" to {current_size}. Please check for potential impacts"
                                )
                                ModuleArgsDict.init_func(child, self.init_type, self.init_gain)

                                with torch.no_grad():
                                    child.weight[:last_size] = state_dict[prefix + name + ".weight"]
                                    if child.bias is not None:
                                        child.bias[:last_size] = state_dict[prefix + name + ".bias"]
                                return
                        load(child, prefix + name + ".")

        load(self)

        if len(unexpected_keys) > 0:
            formatted_keys = ", ".join(f'"{k}"' for k in unexpected_keys)
            error_msgs.insert(
                0,
                f"Unexpected key(s) in state_dict: {formatted_keys}.",
            )
        if len(missing_keys) > 0:
            formatted_keys = ", ".join(f'"{k}"' for k in missing_keys)
            error_msgs.insert(
                0,
                f"Missing key(s) in state_dict: {formatted_keys}.",
            )

        if len(error_msgs) > 0:
            formatted_errors = "\n\t".join(error_msgs)
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{formatted_errors}",
            )

    def apply(self, fn: Callable[[torch.nn.Module], None]) -> None:
        for module in self.children():
            if not isinstance(module, Network):
                module.apply(fn)
        fn(self)

    @_function_network()
    def load(
        self,
        state_dict: dict[str, dict[str, torch.Tensor] | int],
        init: bool = True,
        ema: bool = False,
    ):
        if init:
            self.apply(
                partial(
                    ModuleArgsDict.init_func,
                    init_type=self.init_type,
                    init_gain=self.init_gain,
                )
            )
        name = "Model"
        if ema:
            if name + "_EMA" in state_dict:
                name += "_EMA"
        if name in state_dict:
            value = state_dict[name]
            model_state_dict_tmp = {}
            if isinstance(value, dict):
                model_state_dict_tmp = {k.split(".")[-1]: v for k, v in value.items()}[self.get_name()]
            modules_name = self.get_mapping()
            model_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

            for alias in model_state_dict_tmp.keys():
                prefix = ".".join(alias.split(".")[:-1])
                alias_list = [
                    (".".join(prefix.split(".")[: len(i.split("."))]), v)
                    for i, v in modules_name.items()
                    if prefix.startswith(i)
                ]

                if len(alias_list):
                    for a, b in alias_list:
                        model_state_dict[alias.replace(a, b)] = model_state_dict_tmp[alias]
                        break
                else:
                    model_state_dict[alias] = model_state_dict_tmp[alias]
            self.load_state_dict(model_state_dict)
        if f"{self.get_name()}_optimizer_state_dict" in state_dict and self.optimizer:
            last_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.load_state_dict(state_dict[f"{self.get_name()}_optimizer_state_dict"])
            self.optimizer.param_groups[0]["lr"] = last_lr
        if f"{self.get_name()}_it" in state_dict:
            _it = state_dict.get(f"{self.get_name()}_it")
            if isinstance(_it, int):
                self._it = _it
        if f"{self.get_name()}_nb_lr_update" in state_dict:
            _nb_lr_update = state_dict.get(f"{self.get_name()}_nb_lr_update")
            if isinstance(_nb_lr_update, int):
                self._nb_lr_update = _nb_lr_update

        for scheduler in self.schedulers:
            if scheduler.last_epoch == -1:
                scheduler.last_epoch = self._nb_lr_update
        self.initialized()

    def _compute_channels_trace(
        self,
        module: ModuleArgsDict,
        in_channels: int,
        gradient_checkpoints: list[str] | None,
        gpu_checkpoints: list[str] | None,
        name: str | None = None,
        in_is_channel: bool = True,
        out_channels: int | None = None,
        out_is_channel: bool = True,
    ) -> tuple[int, bool, int | None, bool]:

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
            key = name + "." + k if name else k

            if gradient_checkpoints:
                if key in gradient_checkpoints:
                    module._modulesArgs[k].isCheckpoint = True

            if gpu_checkpoints:
                if key in gpu_checkpoints:
                    module._modulesArgs[k].isGPU_Checkpoint = True

            module._modulesArgs[k].in_channels = in_channels
            module._modulesArgs[k].in_is_channel = in_is_channel

            if isinstance(v, ModuleArgsDict):
                in_channels, in_is_channel, out_channels, out_is_channel = self._compute_channels_trace(
                    v,
                    in_channels,
                    gradient_checkpoints,
                    gpu_checkpoints,
                    key,
                    in_is_channel,
                    out_channels,
                    out_is_channel,
                )

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

            in_channels = out_channels if out_channels is not None else in_channels
            in_is_channel = out_is_channel

        return in_channels, in_is_channel, out_channels, out_is_channel

    @_function_network()
    def init(self, autocast: bool, state: State, group_dest: list[str], key: str) -> None:
        if self.outputs_criterions_loader:
            self.measure = Measure(key, self.outputs_criterions_loader)
            self.measure.init(self, group_dest)
        if self.patch is not None:
            self.patch.init(f"{konfai_root()}.Model.{key}.Patch")
        if state != State.PREDICTION:
            self.scaler = torch.amp.GradScaler("cuda", enabled=autocast)
            if self.optimizerLoader:
                self.optimizer = self.optimizerLoader.get_optimizer(key, self.parameters(False))
                self.optimizer.zero_grad()

            if self.lr_schedulers_loader and self.optimizer:
                for schedulers_classname, schedulers in self.lr_schedulers_loader.items():
                    self.schedulers[schedulers.getschedulers(key, schedulers_classname, self.optimizer)] = (
                        schedulers.nb_step
                    )

    def initialized(self):
        pass

    def named_forward(self, *inputs: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        if self.patch:
            self.patch.load(inputs[0].shape[2:])
            accumulators: dict[str, Accumulator] = {}

            patch_iterator = self.patch.disassemble(*inputs)
            buffer = []
            for i, patch_input in enumerate(patch_iterator):
                for name, output_layer in super().named_forward(*patch_input):
                    yield f";accu;{name}", output_layer
                    buffer.append((name.split(".")[0], output_layer))
                    if len(buffer) == 2:
                        if buffer[0][0] != buffer[1][0]:
                            if self._modulesArgs[buffer[0][0]]._isEnd:
                                if buffer[0][0] not in accumulators:
                                    accumulators[buffer[0][0]] = Accumulator(
                                        self.patch.get_patch_slices(),
                                        self.patch.patch_size,
                                        self.patch.patch_combine,
                                    )
                                accumulators[buffer[0][0]].add_layer(i, buffer[0][1])
                        buffer.pop(0)
                if self._modulesArgs[buffer[0][0]]._isEnd:
                    if buffer[0][0] not in accumulators:
                        accumulators[buffer[0][0]] = Accumulator(
                            self.patch.get_patch_slices(),
                            self.patch.patch_size,
                            self.patch.patch_combine,
                        )
                    accumulators[buffer[0][0]].add_layer(i, buffer[0][1])
            for name, accumulator in accumulators.items():
                yield name, accumulator.assemble()
        else:
            for name, output_layer in super().named_forward(*inputs):
                yield name, output_layer

    def get_layers(
        self, inputs: list[torch.Tensor], layers_name: list[str]
    ) -> Iterator[tuple[str, torch.Tensor, PatchIndexed | None]]:
        layers_name = layers_name.copy()
        output_layer_accumulator: dict[str, Accumulator] = {}
        output_layer_patch_indexed: dict[str, PatchIndexed] = {}
        it = 0
        debug = "KONFAI_DEBUG" in os.environ
        for name_tmp, output_layer in self.named_forward(*inputs):
            name = name_tmp.replace(";accu;", "")
            if debug:
                if "KONFAI_DEBUG_LAST_LAYER" in os.environ:
                    os.environ["KONFAI_DEBUG_LAST_LAYER"] = f"{os.environ['KONFAI_DEBUG_LAST_LAYER']}|{name}:"
                    f"{get_gpu_memory(output_layer.device)}:"
                    f"{str(output_layer.device).replace('cuda:', '')}"
                else:
                    os.environ["KONFAI_DEBUG_LAST_LAYER"] = (
                        f"{name}:{get_gpu_memory(output_layer.device)}:{str(output_layer.device).replace('cuda:', '')}"
                    )
            it += 1
            if name in layers_name or name_tmp in layers_name:
                if ";accu;" in name_tmp:
                    if name not in output_layer_patch_indexed:
                        network_name = (
                            name_tmp.split(".;accu;")[-2].split(".")[-1]
                            if ".;accu;" in name_tmp
                            else name_tmp.split(";accu;")[-2].split(".")[-1]
                        )
                        module = self
                        network = None
                        if network_name == "":
                            network = module
                        else:
                            for n in name.split("."):
                                module = module[n]
                                if isinstance(module, Network) and n == network_name:
                                    network = module
                                    break

                        if network and network.patch:
                            output_layer_patch_indexed[name] = PatchIndexed(network.patch, 0)

                    if name not in output_layer_accumulator:
                        output_layer_accumulator[name] = Accumulator(
                            output_layer_patch_indexed[name].patch.get_patch_slices(0),
                            output_layer_patch_indexed[name].patch.patch_size,
                            output_layer_patch_indexed[name].patch.patch_combine,
                        )

                    if name_tmp in layers_name:
                        output_layer_accumulator[name].add_layer(output_layer_patch_indexed[name].index, output_layer)
                        output_layer_patch_indexed[name].index += 1
                        if output_layer_accumulator[name].is_full():
                            output_layer = output_layer_accumulator[name].assemble()
                            output_layer_accumulator.pop(name)
                            output_layer_patch_indexed.pop(name)
                            layers_name.remove(name_tmp)
                            yield name_tmp, output_layer, None

                if name in layers_name:
                    if ";accu;" in name_tmp:
                        yield name, output_layer, output_layer_patch_indexed[name]
                        output_layer_patch_indexed[name].index += 1
                        if output_layer_patch_indexed[name].is_full():
                            output_layer_patch_indexed.pop(name)
                            layers_name.remove(name)
                    else:
                        layers_name.remove(name)
                        yield name, output_layer, None

            if not len(layers_name):
                break

    def init_outputs_group(self):
        metric_tmp = {
            network.measure: network.measure.outputs_criterions.keys()
            for network in self.get_networks().values()
            if network.measure
        }
        for k, v in metric_tmp.items():
            for a in v:
                outputs_group = OutputsGroup(k)
                outputs_group.append(a)
                for targets_group in k.outputs_criterions[a].keys():
                    if ":" in targets_group:
                        outputs_group.append(targets_group.replace(":", "."))

                self.outputsGroup.append(outputs_group)

    def forward(
        self,
        data_dict: dict[tuple[str, bool], torch.Tensor],
        output_layers: list[str] = [],
    ) -> list[tuple[str, torch.Tensor]]:
        if not len(self.outputsGroup) and not len(output_layers):
            return []

        self.reset_loss()
        results = []
        measure_output_layers = set()
        for _outputs_group in self.outputsGroup:
            for name in _outputs_group:
                measure_output_layers.add(name)
        for name, layer, patch_indexed in self.get_layers(
            [v for k, v in data_dict.items() if k[1]],
            list(set(list(measure_output_layers) + output_layers)),
        ):

            outputs_group = [outputs_group for outputs_group in self.outputsGroup if name in outputs_group]
            if len(outputs_group) > 0:
                if patch_indexed is None:
                    targets = {k[0]: v for k, v in data_dict.items()}
                    nb = 1
                else:
                    targets = {
                        k[0]: patch_indexed.patch.get_data(v, patch_indexed.index, 0, False)
                        for k, v in data_dict.items()
                    }
                    nb = patch_indexed.patch.get_size(0)

                for output_group in outputs_group:
                    output_group.add_layer(name, layer)
                    if output_group.is_done():
                        targets.update(
                            {k.replace(".", ":"): v for k, v in output_group.layers.items() if k != output_group[0]}
                        )
                        output_group.measure.update(
                            output_group[0],
                            output_group.layers[output_group[0]],
                            targets,
                            self._it,
                            nb,
                            self.training,
                        )
                        output_group.clear()
            if name in output_layers:
                results.append((name, layer))
        return results

    @_function_network()
    def reset_loss(self):
        if self.measure:
            self.measure.reset_loss()

    @_function_network()
    def backward(self, model: Self):
        if self.measure:
            if self.scaler and self.optimizer:
                model._requires_grad(list(self.measure.outputs_criterions.keys()))
                for loss in self.measure.get_loss():
                    self.scaler.scale(loss / self.nb_batch_per_step).backward()

                    if self._it % self.nb_batch_per_step == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                self._it += 1

    @_function_network()
    def update_lr(self):
        self._nb_lr_update += 1
        step = 0
        _scheduler = None
        for _scheduler, value in self.schedulers.items():
            if value is None or (self._nb_lr_update >= step and self._nb_lr_update < step + value):
                break
            step += value
        if _scheduler:
            if _scheduler.__class__.__name__ == "ReduceLROnPlateau":
                if self.measure:
                    _scheduler.step(sum(self.measure.get_last_values(0).values()))
            else:
                _scheduler.step()

    @_function_network()
    def get_networks(self) -> Self:
        return self

    @staticmethod
    def to(module: ModuleArgsDict, device: int):
        if "device" not in os.environ:
            os.environ["device"] = str(device)
        for k, v in module.items():
            if module._modulesArgs[k].gpu == "cpu":
                if module._modulesArgs[k].isGPU_Checkpoint:
                    os.environ["device"] = str(int(os.environ["device"]) + 1)
                module._modulesArgs[k].gpu = str(get_device(int(os.environ["device"])))
                if isinstance(v, ModuleArgsDict):
                    v = Network.to(v, int(os.environ["device"]))
                else:
                    v = v.to(get_device(int(os.environ["device"])))
        if isinstance(module, Network):
            if module.optimizer is not None:
                for state in module.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(get_device(int(os.environ["device"])))
        return module

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str) -> Self:
        self.name = name
        return self

    def set_state(self, state: NetState):
        for module in self.modules():
            if isinstance(module, ModuleArgsDict):
                module._training = state


class MinimalModel(Network):

    def __init__(
        self,
        model: Network,
        optimizer: OptimizerLoader = OptimizerLoader(),
        schedulers: dict[str, LRSchedulersLoader] = {"default|StepLR": LRSchedulersLoader(0)},
        outputs_criterions: dict[str, TargetCriterionsLoader] = {"default": TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        dim: int = 3,
        nb_batch_per_step=1,
        init_type="normal",
        init_gain=0.02,
    ):
        super().__init__(
            1,
            optimizer,
            schedulers,
            outputs_criterions,
            patch,
            nb_batch_per_step,
            init_type,
            init_gain,
            dim,
        )
        self.add_module("Model", model)


@config("Model")
class ModelLoader:

    def __init__(self, classpath: str = "default|segmentation.UNet.UNet") -> None:
        self.classpath = classpath

    def get_model(
        self,
        train: bool = True,
        konfai_args: str | None = None,
        konfai_without=[
            "optimizer",
            "schedulers",
            "nb_batch_per_step",
            "init_type",
            "init_gain",
        ],
    ) -> Network:
        module, name = get_module(self.classpath, "konfai.models")

        if not konfai_args:
            konfai_args = f"{konfai_root()}.Model"
        konfai_args += "." + name

        model = apply_config(konfai_args)(getattr(module, name))(konfai_without=konfai_without if not train else [])
        if not isinstance(model, Network):
            model = apply_config(konfai_args)(partial(MinimalModel, model))(
                konfai_without=konfai_without + ["model"] if not train else []
            )
            model.set_name(name)
        return model


class Model:

    def __init__(self, model: Network) -> None:
        self.module = model

    def train(self):
        self.module.train()

    def eval(self):  # noqa: A003
        self.module.eval()

    def __call__(
        self,
        data_dict: dict[tuple[str, bool], torch.Tensor],
        output_layers: list[str] = [],
    ) -> list[tuple[str, torch.Tensor]]:
        return self.module(data_dict, output_layers)
