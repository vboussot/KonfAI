import collections
import inspect
import os
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import ruamel.yaml
import torch

from konfai import config_file
from konfai.utils.utils import ConfigError

yaml = ruamel.yaml.YAML()


class Config:

    def __init__(self, key: str) -> None:
        self.filename = Path(os.environ["KONFAI_config_file"])
        self.keys = key.split(".")

    def __enter__(self):
        if not self.filename.exists():
            result = input("Create a new config file ? [no,yes,interactive] : ")
            if result in ["yes", "interactive"]:
                os.environ["KONFAI_CONFIG_MODE"] = "interactive" if result == "interactive" else "default"
                open(self.filename, "w").close()
            else:
                exit(0)

        self.yml = open(self.filename)
        self.data = yaml.load(self.yml)
        if self.data is None:
            self.data = {}

        self.config = self.data

        for key in self.keys:
            if self.config is None or key not in self.config:
                self.config = {key: {}}

            self.config = self.config[key]
        return self

    def create_dictionary(self, data, keys, i) -> dict:
        if keys[i] not in data:
            data = {keys[i]: data}
        if i == 0:
            return data
        else:
            i -= 1
            return self.create_dictionary(data, keys, i)

    def merge(self, dict1, dict2) -> dict:
        result = deepcopy(dict1)

        for key, value in dict2.items():
            if isinstance(value, collections.abc.Mapping):
                result[key] = self.merge(result.get(key, {}), value)
            else:
                if dict2[key] is not None:
                    result[key] = deepcopy(dict2[key])
        return result

    def __exit__(self, exc_type, value, traceback) -> None:
        self.yml.close()
        if os.environ["KONFAI_CONFIG_MODE"] == "remove":
            if os.path.exists(config_file()):
                os.remove(config_file())
            return
        with open(self.filename) as yml:
            data = yaml.load(yml)
            if data is None:
                data = {}
        with open(self.filename, "w") as yml:
            yaml.dump(
                self.merge(
                    data,
                    self.create_dictionary(self.config, self.keys, len(self.keys) - 1),
                ),
                yml,
            )

    @staticmethod
    def _get_input(name: str, default: str) -> str:
        try:
            return input(f"{name} [{','.join(default.split(':')[1:]) if ':' in default else ''}]: ")
        except Exception:
            result = input("\nKeep a default configuration file ? (yes,no) : ")
            if result == "yes":
                os.environ["KONFAI_CONFIG_MODE"] = "default"
            else:
                os.environ["KONFAI_CONFIG_MODE"] = "remove"
                exit(0)
        return default.split("|")[1] if len(default.split("|")) > 1 else default

    @staticmethod
    def _get_input_default(name: str, default: str | None, is_list: bool = False) -> list[str | None] | str | None:
        if isinstance(default, str) and (
            default == "default" or (len(default.split("|")) > 1 and default.split("|")[0] == "default")
        ):
            if os.environ["KONFAI_CONFIG_MODE"] == "interactive":
                if is_list:
                    list_tmp: list[str | None] = []
                    key_tmp = "OK"
                    while (key_tmp != "!" and key_tmp != " ") and os.environ["KONFAI_CONFIG_MODE"] == "interactive":
                        key_tmp = Config._get_input(name, default)
                        if key_tmp != "!" and key_tmp != " ":
                            if key_tmp == "":
                                key_tmp = default.split("|")[1] if len(default.split("|")) > 1 else default
                            list_tmp.append(key_tmp)
                    return list_tmp
                else:
                    value = Config._get_input(name, default)
                    if value == "":
                        return default.split("|")[1] if len(default.split("|")) > 1 else default
                    else:
                        return value
            else:
                default = default.split("|")[1] if len(default.split("|")) > 1 else default
        return [default] if is_list else default

    def get_value(self, name, default) -> object:
        if name in self.config and self.config[name] is not None:
            value = self.config[name]
            if value is None:
                value = default
            value_config = value
        else:
            value = Config._get_input_default(name, default if default != inspect._empty else None)

            value_config = value
            if isinstance(value_config, tuple):
                value_config = list(value)

            if isinstance(value_config, list):
                list_tmp = []
                for key in value_config:
                    res = Config._get_input_default(name, key, is_list=True)
                    if isinstance(res, list):
                        list_tmp.extend(res)
                    else:
                        list_tmp.append(str(res))

                value = list_tmp
                value_config = list_tmp

            if isinstance(value, dict):
                key_tmp = []

                value_config = {}
                dict_value = {}
                for key in value:
                    res = Config._get_input_default(name, key, is_list=True)
                    if isinstance(res, list):
                        key_tmp.extend(res)
                    else:
                        key_tmp.append(str(res))
                for key in key_tmp:
                    if key in value:
                        value_tmp = value[key]
                    else:
                        value_tmp = next(v for k, v in value.items() if "default" in k)

                    value_config[key] = None
                    dict_value[key] = value_tmp
                value = dict_value
        if isinstance(self.config, str):
            os.environ["KONFAI_CONFIG_VARIABLE"] = "True"
            return None

        self.config[name] = value_config if value_config is not None else "None"
        if value == "None":
            value = None
        return value


def config(key: str | None = None):
    def decorator(function):
        function._key = key
        return function

    return decorator


def apply_config(konfai_args: str | None = None):
    def decorator(function):
        def new_function(*args, **kwargs):
            key = getattr(function, "_key", None)
            key_tmp = konfai_args + ("." + key if key is not None else "") if konfai_args is not None else key
            if (
                "KONFAI_config_file" in os.environ
                and "KONFAI_CONFIG_MODE" in os.environ
                and os.environ["KONFAI_CONFIG_MODE"] != "Import"
                and key_tmp is not None
            ):
                os.environ["KONFAI_CONFIG_PATH"] = key_tmp
                without = kwargs["konfai_without"] if "konfai_without" in kwargs else []
                with Config(key_tmp) as config:
                    os.environ["KONFAI_CONFIG_VARIABLE"] = "False"
                    kwargs = {}
                    for param in list(inspect.signature(function).parameters.values())[len(args) :]:
                        if param.name in without:
                            continue

                        annotation = param.annotation
                        if annotation == "int":
                            annotation = int
                        if annotation == "float":
                            annotation = float
                        if annotation == "bool":
                            annotation = bool
                        # --- support Literal ---
                        if get_origin(annotation) is Literal:
                            allowed_values = get_args(annotation)
                            default_value = param.default if param.default != inspect._empty else allowed_values[0]
                            value = config.get_value(param.name, f"default|{default_value}")
                            if value not in allowed_values:
                                raise ConfigError(
                                    f"Invalid value '{value}' for parameter '{param.name} "
                                    f"expected one of: {allowed_values}."
                                )
                            kwargs[param.name] = value
                            continue
                        if (
                            str(annotation).startswith("typing.Union")
                            or str(annotation).startswith("typing.Optional")
                            or get_origin(annotation) is types.UnionType
                        ):
                            for i in annotation.__args__:
                                annotation = i
                                break

                        if not annotation == inspect._empty:
                            if annotation not in [int, str, bool, float, torch.Tensor]:
                                if (
                                    str(annotation).startswith("list")
                                    or str(annotation).startswith("tuple")
                                    or str(annotation).startswith("typing.Tuple")
                                    or str(annotation).startswith("typing.List")
                                    or str(annotation).startswith("typing.Sequence")
                                ):
                                    elem_type = annotation.__args__[0]
                                    values = config.get_value(param.name, param.default)
                                    if getattr(elem_type, "__origin__", None) is Union:
                                        valid_types = elem_type.__args__
                                        result = []
                                        for v in values:
                                            for t in valid_types:
                                                try:
                                                    if t == torch.Tensor and not isinstance(v, torch.Tensor):
                                                        v = torch.tensor(v)
                                                    result.append(t(v) if t != torch.Tensor else v)
                                                    break
                                                except Exception:
                                                    raise ValueError("Merde")
                                        kwargs[param.name] = result

                                    elif annotation.__args__[0] in [
                                        int,
                                        str,
                                        bool,
                                        float,
                                    ]:
                                        values = config.get_value(param.name, param.default)
                                        kwargs[param.name] = values
                                    else:
                                        raise ConfigError(
                                            "Config: The config only supports types : config(Object), int, str, bool,"
                                            " float, list[int], list[str], list[bool], list[float], dict[str, Object]"
                                        )
                                elif str(annotation).startswith("dict"):
                                    if annotation.__args__[0] is str:
                                        values = config.get_value(param.name, param.default)
                                        if values is not None and annotation.__args__[1] not in [
                                            int,
                                            str,
                                            bool,
                                            float,
                                            Any,
                                        ]:
                                            try:
                                                kwargs[param.name] = {
                                                    value: apply_config(str(key_tmp) + "." + param.name + "." + value)(
                                                        annotation.__args__[1]
                                                    )()
                                                    for value in values
                                                }
                                            except ValueError as e:
                                                raise ValueError(e)
                                            except Exception as e:
                                                raise ConfigError(f"{values} {e}")
                                        else:

                                            kwargs[param.name] = values
                                    else:
                                        raise ConfigError(
                                            "Config: The config only supports types : config(Object), int, str, bool,"
                                            " float, list[int], list[str], list[bool], list[float], dict[str, Object]"
                                        )
                                else:
                                    try:
                                        kwargs[param.name] = apply_config(key_tmp)(annotation)()
                                    except Exception as e:
                                        raise ConfigError(
                                            f"Failed to instantiate {param.name} with type {annotation}, error {e} "
                                        )

                                    if os.environ["KONFAI_CONFIG_VARIABLE"] == "True":
                                        os.environ["KONFAI_CONFIG_VARIABLE"] = "False"
                                        kwargs[param.name] = None
                            else:
                                kwargs[param.name] = config.get_value(param.name, param.default)
                        elif param.name != "self":
                            kwargs[param.name] = config.get_value(param.name, param.default)
            result = function(*args, **kwargs)
            return result

        return new_function

    return decorator
