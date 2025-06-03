import os
from torch.utils.data import DataLoader
import torch
import tqdm
import numpy as np
import json
import shutil
import builtins
import importlib
from konfai import EVALUATIONS_DIRECTORY, PREDICTIONS_DIRECTORY, DEEP_LEARNING_API_ROOT, CONFIG_FILE
from konfai.utils.config import config
from konfai.utils.utils import _getModule, DistributedObject, synchronize_data
from konfai.data.dataset import DataMetric

class CriterionsAttr():

    @config()
    def __init__(self) -> None:
        pass  

class CriterionsLoader():

    @config()
    def __init__(self, criterionsLoader: dict[str, CriterionsAttr] = {"default:torch_nn_CrossEntropyLoss:Dice:NCC": CriterionsAttr()}) -> None:
        self.criterionsLoader = criterionsLoader

    def getCriterions(self, output_group : str, target_group : str) -> dict[torch.nn.Module, CriterionsAttr]:
        criterions = {}
        for module_classpath, criterionsAttr in self.criterionsLoader.items():
            module, name = _getModule(module_classpath, "metric.measure")
            criterions[config("{}.metrics.{}.targetsCriterions.{}.criterionsLoader.{}".format(DEEP_LEARNING_API_ROOT(), output_group, target_group, module_classpath))(getattr(importlib.import_module(module), name))(config = None)] = criterionsAttr
        return criterions

class TargetCriterionsLoader():

    @config()
    def __init__(self, targetsCriterions : dict[str, CriterionsLoader] = {"default" : CriterionsLoader()}) -> None:
        self.targetsCriterions = targetsCriterions
        
    def getTargetsCriterions(self, output_group : str) -> dict[str, dict[torch.nn.Module, float]]:
        targetsCriterions = {}
        for target_group, criterionsLoader in self.targetsCriterions.items():
            targetsCriterions[target_group] = criterionsLoader.getCriterions(output_group, target_group)
        return targetsCriterions

class Statistics():

    def __init__(self, filename: str) -> None:
        self.measures: dict[str, dict[str, float]] = {}
        self.filename = filename

    def add(self, values: dict[str, float], name_dataset: str) -> None:
        for name, value in values.items():
            if name_dataset not in self.measures:
                self.measures[name_dataset] = {}
            self.measures[name_dataset][name] = value
            
    def getStatistic(values: list[float]) -> dict[str, float]:
        return {"max": np.max(values), "min": np.min(values), "std": np.std(values), "25pc": np.percentile(values, 25), "50pc": np.percentile(values, 50), "75pc": np.percentile(values, 75), "mean": np.mean(values), "count": len(values)}
    
    def write(self, outputs: list[dict[str, any]]) -> None:
        measures = {}
        for output in outputs:
            measures.update(output)
        result = {}
        result["case"] = {}
        for name, v in measures.items():
            for metric_name, value in v.items():
                if metric_name not in result["case"]:
                    result["case"][metric_name] = {}
                result["case"][metric_name][name] = value

        result["aggregates"] = {}
        tmp: dict[str, list[float]] = {}
        for name, v in measures.items():
            for metric_name, value in v.items():
                if metric_name not in tmp:
                    tmp[metric_name] = []
                tmp[metric_name].append(v[metric_name])
        for metric_name, values in tmp.items():
            result["aggregates"][metric_name] = Statistics.getStatistic(values)

        with open(self.filename, "w") as f:
            f.write(json.dumps(result, indent=4))
            
class Evaluator(DistributedObject):

    @config("Evaluator")
    def __init__(self, train_name: str = "default:name", metrics: dict[str, TargetCriterionsLoader] = {"default": TargetCriterionsLoader()}, dataset : DataMetric = DataMetric(),) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
        super().__init__(train_name)
        self.metric_path = EVALUATIONS_DIRECTORY()+self.name+"/"
        self.predict_path = PREDICTIONS_DIRECTORY()+self.name+"/"
        self.metricsLoader = metrics
        self.dataset = dataset
        self.metrics = {k: v.getTargetsCriterions(k) for k, v in self.metricsLoader.items()}
        self.statistics_train = Statistics(self.metric_path+"Metric_TRAIN.json")
        self.statistics_validation = Statistics(self.metric_path+"Metric_VALIDATION.json")

    def update(self, data_dict: dict[str, tuple[torch.Tensor, str]], statistics : Statistics) -> dict[str, float]:
        result = {}
        for output_group in self.metrics:
            for target_group in self.metrics[output_group]:
                targets = [data_dict[group][0] for group in target_group.split("/") if group in data_dict]
                name = data_dict[output_group][1][0]
                for metric in self.metrics[output_group][target_group]:
                    result["{}:{}:{}".format(output_group, target_group, metric.__class__.__name__)] = metric(data_dict[output_group][0], *targets).item()
        statistics.add(result, name)
        return result
    
    def setup(self, world_size: int):
        if os.path.exists(self.metric_path):
            if os.environ["DL_API_OVERWRITE"] != "True":
                accept = builtins.input("The metric {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.name))
                if accept != "yes":
                    return
           
            if os.path.exists(self.metric_path):
                shutil.rmtree(self.metric_path) 

        if not os.path.exists(self.metric_path):
            os.makedirs(self.metric_path)
        metric_namefile_src = CONFIG_FILE().replace(".yml", "")
        shutil.copyfile(metric_namefile_src+".yml", "{}{}.yml".format(self.metric_path, metric_namefile_src))

        self.dataloader = self.dataset.getData(world_size)

    def run_process(self, world_size: int, global_rank: int, gpu: int, dataloaders: list[DataLoader]):
        description = lambda measure : "Metric TRAIN : {} ".format(" | ".join("{}: {:.2f}".format(k, v) for k, v in measure.items()) if measure is not None else "")
        with tqdm.tqdm(iterable = enumerate(dataloaders[0]), leave=False, desc = description(None), total=len(dataloaders[0])) as batch_iter:
            for _, data_dict in batch_iter:
                batch_iter.set_description(description(self.update({k: (v[0], v[4]) for k,v in data_dict.items()}, self.statistics_train)))
        outputs = synchronize_data(world_size, gpu, self.statistics_train.measures)
        if global_rank == 0:
            self.statistics_train.write(outputs)
        if len(dataloaders) == 2:
            description = lambda measure : "Metric VALIDATION : {} ".format(" | ".join("{}: {:.2f}".format(k, v) for k, v in measure.items()) if measure is not None else "")
            with tqdm.tqdm(iterable = enumerate(dataloaders[1]), leave=False, desc = description(None), total=len(dataloaders[1])) as batch_iter:
                for _, data_dict in batch_iter:
                    batch_iter.set_description(description(self.update({k: (v[0], v[4]) for k,v in data_dict.items()}, self.statistics_validation)))
            outputs = synchronize_data(world_size, gpu, self.statistics_validation.measures)
            if global_rank == 0:
                self.statistics_validation.write(outputs)