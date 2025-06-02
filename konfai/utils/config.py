import os
import ruamel.yaml
import inspect 
import collections
from copy import deepcopy
from typing import Union
import torch

from konfai import CONFIG_FILE

yaml = ruamel.yaml.YAML()

class ConfigError(Exception):

    def __init__(self, message : str = "The config only supports types : config(Object), int, str, bool, float, list[int], list[str], list[bool], list[float], dict[str, Object]") -> None:
        self.message = message
        super().__init__(self.message)

class Config():

    def __init__(self, filename, key) -> None:
        self.filename = filename
        self.keys = key.split(".")

    def __enter__(self):
        if not os.path.exists(self.filename):
            result = input("Create a new config file ? [no,yes,interactive] : ")
            if result in ["yes", "interactive"]:
                os.environ["DEEP_LEANING_API_CONFIG_MODE"] = "interactive" if result == "interactive" else "default"
            else:
                exit(0)
            with open(self.filename, "w") as f:
                pass
                    
        self.yml = open(self.filename, 'r')
        self.data = yaml.load(self.yml)
        if self.data == None:
            self.data = {}
        
        self.config = self.data

        for key in self.keys:
            if self.config == None or key not in self.config:
                self.config = {key : {}}
            
            self.config = self.config[key]
        return self

    def createDictionary(self, data, keys, i) -> dict:
        if keys[i] not in data:
            data = {keys[i]: data}
        if i == 0:
            return data
        else:
            i -= 1
            return self.createDictionary(data, keys, i)
    
    def merge(self, dict1, dict2) -> dict:
        result = deepcopy(dict1)

        for key, value in dict2.items():
            if isinstance(value, collections.abc.Mapping):
                result[key] = self.merge(result.get(key, {}), value)
            else:
                if not dict2[key] == None:
                    result[key] = deepcopy(dict2[key])
        return result


    def __exit__(self, type, value, traceback) -> None:
        self.yml.close()
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] == "remove":
            if os.path.exists(CONFIG_FILE()):
                os.remove(CONFIG_FILE())
            return
        with open(self.filename, 'r') as yml:
            data = yaml.load(yml)
            if data == None:
                data = {}
        with open(self.filename, 'w') as yml:
            yaml.dump(self.merge(data, self.createDictionary(self.config, self.keys, len(self.keys)-1)), yml)
  
    @staticmethod
    def _getInput(name : str, default : str) ->  str:
        try:
            return input("{} [{}]: ".format(name, ",".join(default.split(":")[1:]) if len(default.split(":")) else ""))
        except:
            result = input("\nKeep a default configuration file ? (yes,no) : ")
            if result == "yes":
                os.environ["DEEP_LEANING_API_CONFIG_MODE"] = "default"
            else:
                os.environ["DEEP_LEANING_API_CONFIG_MODE"] = "remove"
                exit(0)
        return default.split(":")[1] if len(default.split(":")) > 1 else default

    @staticmethod
    def _getInputDefault(name : str, default : Union[str, None], isList : bool = False) -> Union[list[Union[str, None]], str, None]:
        if isinstance(default, str) and (default == "default" or (len(default.split(":")) > 1 and default.split(":")[0] == "default")):
            if os.environ["DEEP_LEANING_API_CONFIG_MODE"] == "interactive":
                if isList:
                    list_tmp = []
                    key_tmp = "OK"
                    while key_tmp != "!" and os.environ["DEEP_LEANING_API_CONFIG_MODE"] == "interactive":
                        key_tmp = Config._getInput(name, default)
                        if key_tmp != "!":
                            if key_tmp == "":
                                key_tmp = default.split(":")[1] if len(default.split(":")) > 1 else default
                            list_tmp.append(key_tmp)
                    return list_tmp
                else:
                    value = Config._getInput(name, default)
                    if value == "":
                        return default.split(":")[1] if len(default.split(":")) > 1 else default 
                    else: 
                        return value
            else:
                default = default.split(":")[1] if len(default.split(":")) > 1 else default
        return  [default] if isList else default

    def getValue(self, name, default) -> object:  
        if name in self.config and self.config[name] is not None:
            value = self.config[name]
            if value == None:
                value = default
            value_config = value
        else:
            value = Config._getInputDefault(name, default if default != inspect._empty else None)
            
            value_config = value
            if type(value_config) == tuple:
                value_config = list(value)
                
            if type(value_config) == list:
                list_tmp = [] 
                for key in value_config:
                    list_tmp.extend(Config._getInputDefault(name, key, isList=True))

                value = list_tmp
                value_config = list_tmp

            if type(value) == dict:
                key_tmp = []
                
                value_config = {}
                dict_value = {}
                for key in value:
                    key_tmp.extend(Config._getInputDefault(name, key, isList=True))
                for key in key_tmp:
                    if key in value:
                        value_tmp = value[key]
                    else:
                        value_tmp = next(v for k,v in value.items() if "default" in k)

                    value_config[key] = None
                    dict_value[key] = value_tmp
                value = dict_value
        if isinstance(self.config, str):
            os.environ['DEEP_LEARNING_API_CONFIG_VARIABLE'] = "True"
            return None

        self.config[name] = value_config if value_config is not None else "None"
        if value == "None":
            value = None
        return value
                    
def config(key : Union[str, None] = None):
    def decorator(function):
        def new_function(*args, **kwargs):
            if "config" in kwargs:
                filename =  kwargs["config"]
                if filename == None:
                    filename = os.environ['DEEP_LEARNING_API_CONFIG_FILE']
                else:
                    os.environ['DEEP_LEARNING_API_CONFIG_FILE'] = filename
                key_tmp =  kwargs["DL_args"]+("."+key if key is not None else "") if "DL_args" in kwargs else key
                without =  kwargs["DL_without"] if "DL_without" in kwargs else []
                os.environ['DEEP_LEARNING_API_CONFIG_PATH'] = key_tmp
                with Config(filename, key_tmp) as config:
                    os.environ['DEEP_LEARNING_API_CONFIG_VARIABLE'] = "False"
                    kwargs = {} 
                    for param in list(inspect.signature(function).parameters.values())[len(args):]:
                        annotation = param.annotation
                        if str(annotation).startswith("typing.Union") or str(annotation).startswith("typing.Optional"):
                            for i in annotation.__args__:
                                annotation = i
                                break
                        if param.name in without:
                            continue
                        if not annotation == inspect._empty:
                            if annotation not in [int, str, bool, float, torch.Tensor]:
                                if str(annotation).startswith("list") or str(annotation).startswith("tuple") or str(annotation).startswith("typing.Tuple"):
                                    if annotation.__args__[0] in [int, str, bool, float]:
                                        values = config.getValue(param.name, param.default)
                                        kwargs[param.name] = values
                                    else:
                                        raise ConfigError()
                                elif str(annotation).startswith("dict"):
                                    if annotation.__args__[0] == str:
                                        values = config.getValue(param.name, param.default)
                                        if values is not None and annotation.__args__[1] not in [int, str, bool, float]:
                                            kwargs[param.name] = {value : annotation.__args__[1](config = filename, DL_args = key_tmp+"."+param.name+"."+value) for value in values}
                                        else:
                                            kwargs[param.name] = values
                                    else: 
                                        raise ConfigError()
                                else:
                                    kwargs[param.name] = annotation(config = filename, DL_args = key_tmp)
                                    if os.environ['DEEP_LEARNING_API_CONFIG_VARIABLE'] == "True":
                                        os.environ['DEEP_LEARNING_API_CONFIG_VARIABLE'] = "False"
                                        kwargs[param.name] = None
                            else:
                                kwargs[param.name] = config.getValue(param.name, param.default)
                        elif param.name != "self":
                            kwargs[param.name] = config.getValue(param.name, param.default)
            result = function(*args, **kwargs)
            return result
        return new_function
    return decorator
