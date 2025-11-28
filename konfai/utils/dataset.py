import ast
import copy
import csv
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from lxml import etree  # nosec B410

from konfai import current_date


class Attribute(dict[str, Any]):

    def __init__(self, attributes: dict[str, Any] | None = None) -> None:
        super().__init__()
        attributes = attributes or {}
        for k, v in attributes.items():
            super().__setitem__(copy.deepcopy(k), copy.deepcopy(v))

    def __getitem__(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return str(super().__getitem__(f"{key}_{i - 1}"))
        else:
            raise NameError(f"{key} not in cache_attribute")

    def __setitem__(self, key: str, value: Any) -> None:
        if "_" not in key:
            i = len([k for k in super().keys() if k.startswith(key)])
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace("\n", "")
            super().__setitem__(f"{key}_{i}", result)
        else:
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace("\n", "")
            super().__setitem__(key, result)

    def pop(self, key: str, default: Any = None) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return super().pop(f"{key}_{i - 1}")
        else:
            raise NameError(f"{key} not in cache_attribute")

    def get_np_array(self, key) -> np.ndarray:
        return np.fromstring(self[key][1:-1], sep=" ", dtype=np.double)

    def get_tensor(self, key) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key)).to(torch.float32)

    def pop_np_array(self, key):
        return np.fromstring(self.pop(key)[1:-1], sep=" ", dtype=np.double)

    def pop_tensor(self, key) -> torch.Tensor:
        return torch.tensor(self.pop_np_array(key))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(k.startswith(key) for k in super().keys())

    def is_info(self, key: str, value: str) -> bool:
        return key in self and self[key] == value


def is_an_image(attributes: Attribute):
    return "Origin" in attributes and "Spacing" in attributes and "Direction" in attributes


def data_to_image(data: np.ndarray, attributes: Attribute) -> sitk.Image:
    if not is_an_image(attributes):
        raise NameError("Data is not an image")
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i + 1 for i in range(len(data.shape) - 1)] + [0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    for k, v in attributes.items():
        if v and len(v):
            image.SetMetaData(k, v)
    image.SetOrigin(attributes.get_np_array("Origin").tolist())
    image.SetSpacing(attributes.get_np_array("Spacing").tolist())
    image.SetDirection(attributes.get_np_array("Direction").tolist())
    return image


def image_to_data(image: sitk.Image) -> tuple[np.ndarray, Attribute]:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    for k in image.GetMetaDataKeys():
        attributes[k] = image.GetMetaData(k)
    data = sitk.GetArrayFromImage(image)

    if image.GetNumberOfComponentsPerPixel() == 1:
        data = np.expand_dims(data, 0)
    else:
        data = np.transpose(data, (len(data.shape) - 1, *list(range(len(data.shape) - 1))))
    return data, attributes


def get_infos(filename: str | Path) -> tuple[list[int], Attribute]:
    attributes = Attribute()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(str(filename))
    file_reader.ReadImageInformation()
    attributes["Origin"] = np.asarray(file_reader.GetOrigin())
    attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
    attributes["Direction"] = np.asarray(file_reader.GetDirection())
    for k in file_reader.GetMetaDataKeys():
        attributes[k] = file_reader.GetMetaData(k)
    size = list(file_reader.GetSize())
    if len(size) == 3:
        size = list(reversed(size))
    size = [file_reader.GetNumberOfComponents()] + size
    return size, attributes


def read_landmarks(filename: str) -> np.ndarray | None:
    data = None
    with open(filename, newline="") as csvfile:
        reader = csv.reader(filter(lambda row: row[0] != "#", csvfile))
        lines = list(reader)
        data = np.zeros((len(list(lines)), 3), dtype=np.double)
        for i, row in enumerate(lines):
            data[i] = np.array(row[1:4], dtype=np.double)
        csvfile.close()
    return data


class Dataset:

    class AbstractFile(ABC):

        @abstractmethod
        def __init__(self) -> None:
            pass

        @abstractmethod
        def __enter__(self):
            pass

        @abstractmethod
        def __exit__(self, exc_type, value, traceback):
            pass

        @abstractmethod
        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            pass

        @abstractmethod
        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            pass

        @abstractmethod
        def get_names(self, group: str) -> list[str]:
            pass

        @abstractmethod
        def get_group(self) -> list[str]:
            pass

        @abstractmethod
        def is_exist(self, group: str, name: str | None = None) -> bool:
            pass

        @abstractmethod
        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            pass

    class H5File(AbstractFile):

        def __init__(self, filename: str, read: bool) -> None:
            self.h5: h5py.File | None = None
            self.filename = filename
            if not self.filename.endswith(".h5"):
                self.filename += ".h5"
            self.read = read

        def __enter__(self):
            if self.read:
                self.h5 = h5py.File(self.filename, "r")
            else:
                if not os.path.exists(self.filename):
                    if len(self.filename.split("/")) > 1 and not os.path.exists(
                        "/".join(self.filename.split("/")[:-1])
                    ):
                        os.makedirs("/".join(self.filename.split("/")[:-1]))
                    self.h5 = h5py.File(self.filename, "w")
                else:
                    self.h5 = h5py.File(self.filename, "r+")
                self.h5.attrs["Date"] = current_date()
            self.h5.__enter__()
            return self.h5

        def __exit__(self, exc_type, value, traceback):
            if self.h5 is not None:
                self.h5.close()

        def file_to_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
            dataset = self._get_dataset(groups, name)
            data = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(data)
            return data, Attribute({k: str(v) for k, v in dataset.attrs.items()})

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            if self.h5 is None:
                return
            if attributes is None:
                attributes = Attribute()
            if isinstance(data, sitk.Image):
                data, attributes_tmp = image_to_data(data)
                attributes.update(attributes_tmp)
            elif isinstance(data, sitk.Transform):
                transforms = []
                if isinstance(data, sitk.CompositeTransform):
                    for i in range(data.GetNumberOfTransforms()):
                        transforms.append(data.GetNthTransform(i))
                else:
                    transforms.append(data)
                datas = []
                for i, transform in enumerate(transforms):
                    if isinstance(transform, sitk.Euler3DTransform):
                        transform_type = "Euler3DTransform_double_3_3"
                    if isinstance(transform, sitk.AffineTransform):
                        transform_type = "AffineTransform_double_3_3"
                    if isinstance(transform, sitk.BSplineTransform):
                        transform_type = "BSplineTransform_double_3_3"
                    attributes[f"{i}:Transform"] = transform_type
                    attributes[f"{i}:FixedParameters"] = transform.GetFixedParameters()

                    datas.append(np.asarray(transform.GetParameters()))
                data = np.asarray(datas)

            h5_group = self.h5
            if len(name.split("/")) > 1:
                group = "/".join(name.split("/")[:-1])
                if group not in self.h5:
                    self.h5.create_group(group)
                h5_group = self.h5[group]

            name = name.split("/")[-1]
            if name in h5_group:
                del h5_group[name]

            dataset = h5_group.create_dataset(name, data=data, dtype=data.dtype, chunks=None)
            dataset.attrs.update({k: str(v) for k, v in attributes.items()})

        def is_exist(self, group: str, name: str | None = None) -> bool:
            if self.h5 is not None:
                if group in self.h5:
                    if isinstance(self.h5[group], h5py.Dataset):
                        return True
                    elif name is not None:
                        return name in self.h5[group]
                    else:
                        return False
            return False

        def get_names(self, groups: str, h5_group: h5py.Group = None) -> list[str]:
            names = []
            if h5_group is None:
                h5_group = self.h5
            group = groups.split("/")[0]
            if group == "":
                names = [
                    dataset.name.split("/")[-1] for dataset in h5_group.values() if isinstance(dataset, h5py.Dataset)
                ]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[k]))
            else:
                if group in h5_group:
                    names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[group]))
            return names

        def get_group(self) -> list[str]:
            return list(self.h5.keys()) if self.h5 is not None else []

        def _get_dataset(self, groups: str, name: str, h5_group: h5py.Group = None) -> h5py.Dataset:
            if h5_group is None:
                h5_group = self.h5
            if groups != "":
                group = groups.split("/")[0]
            else:
                group = ""
            result = None
            if group == "":
                if name in h5_group:
                    result = h5_group[name]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        result_tmp = self._get_dataset("/".join(groups.split("/")[1:]), name, h5_group[k])
                        if result_tmp is not None:
                            result = result_tmp
            else:
                if group in h5_group:
                    result_tmp = self._get_dataset("/".join(groups.split("/")[1:]), name, h5_group[group])
                    if result_tmp is not None:
                        result = result_tmp
            return result

        def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
            dataset = self._get_dataset(groups, name)
            return (
                dataset.shape,
                Attribute({k: str(v) for k, v in dataset.attrs.items()}),
            )

    class SitkFile(AbstractFile):

        def __init__(self, filename: str, read: bool, file_format: str) -> None:
            self.filename = filename
            self.read = read
            self.file_format = file_format

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            attributes = Attribute()
            if os.path.exists(f"{self.filename}{name}.{self.file_format}"):
                image = sitk.ReadImage(f"{self.filename}{name}.{self.file_format}")
                data, attributes_tmp = image_to_data(image)
                attributes.update(attributes_tmp)
            elif os.path.exists(f"{self.filename}{name}.itk.txt"):
                data = sitk.ReadTransform(f"{self.filename}{name}.itk.txt")
                transforms = []
                if isinstance(data, sitk.CompositeTransform):
                    for i in range(data.GetNumberOfTransforms()):
                        transforms.append(data.GetNthTransform(i))
                else:
                    transforms.append(data)
                datas = []
                for i, transform in enumerate(transforms):
                    if isinstance(transform, sitk.Euler3DTransform):
                        transform_type = "Euler3DTransform_double_3_3"
                    if isinstance(transform, sitk.AffineTransform):
                        transform_type = "AffineTransform_double_3_3"
                    if isinstance(transform, sitk.BSplineTransform):
                        transform_type = "BSplineTransform_double_3_3"
                    attributes[f"{i}:Transform"] = transform_type
                    attributes[f"{i}:FixedParameters"] = transform.GetFixedParameters()

                    datas.append(np.asarray(transform.GetParameters()))

                max_len = max(len(v) for v in datas)

                padded_datas = np.array([np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in datas])

                data = np.asarray(padded_datas)
            elif os.path.exists(f"{self.filename}{name}.fcsv"):
                data = read_landmarks(f"{self.filename}{name}.fcsv")
            elif os.path.exists(f"{self.filename}{name}.xml"):
                with open(f"{self.filename}{name}.xml", "rb") as xml_file:
                    result = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()  # nosec B320
                    xml_file.close()
                    return result
            elif os.path.exists(f"{self.filename}{name}.vtk"):
                import vtk

                vtk_reader = vtk.vtkPolyDataReader()
                vtk_reader.SetFileName(f"{self.filename}{name}.vtk")
                vtk_reader.Update()
                data = []
                points = vtk_reader.GetOutput().GetPoints()
                num_points = points.GetNumberOfPoints()
                for i in range(num_points):
                    data.append(list(points.GetPoint(i)))
                data = np.asarray(data)
            elif os.path.exists(f"{self.filename}{name}.npy"):
                data = np.load(f"{self.filename}{name}.npy")
            return data, attributes

        def is_vtk_polydata(self, obj):
            try:
                import vtk

                return isinstance(obj, vtk.vtkPolyData)
            except ImportError:
                return False

        def __enter__(self):
            pass

        def __exit__(self, exc_type, value, traceback):
            pass

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            if attributes is None:
                attributes = Attribute()
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
            if isinstance(data, sitk.Image):
                for k, v in attributes.items():
                    if v and len(v):
                        data.SetMetaData(k, v)
                sitk.WriteImage(data, f"{self.filename}{name}.{self.file_format}")
            elif isinstance(data, sitk.Transform):
                sitk.WriteTransform(data, f"{self.filename}{name}.itk.txt")
            elif self.is_vtk_polydata(data):
                import vtk

                vtk_writer = vtk.vtkPolyDataWriter()
                vtk_writer.SetFileName(f"{self.filename}{name}.vtk")
                vtk_writer.SetInputData(data)
                vtk_writer.Write()
            elif is_an_image(attributes):
                self.data_to_file(name, data_to_image(data, attributes), attributes)
            elif len(data.shape) == 2 and data.shape[1] == 3 and data.shape[0] > 0:
                data = np.round(data, 4)
                with open(f"{self.filename}{name}.fcsv", "w") as f:
                    f.write(
                        "# Markups fiducial file version = 4.6\n# CoordinateSystem = 0\n#"
                        " columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n",
                    )
                    for i in range(data.shape[0]):
                        f.write(
                            "vtkMRMLMarkupsFiducialNode_"
                            + str(i + 1)
                            + ","
                            + str(data[i, 0])
                            + ","
                            + str(data[i, 1])
                            + ","
                            + str(data[i, 2])
                            + ",0,0,0,1,1,1,0,F-"
                            + str(i + 1)
                            + ",,vtkMRMLScalarVolumeNode1\n"
                        )
                    f.close()
            elif "path" in attributes:
                if os.path.exists(f"{self.filename}{name}.xml"):
                    with open(f"{self.filename}{name}.xml", "rb") as xml_file:
                        root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()  # nosec B320
                        xml_file.close()
                else:
                    root = etree.Element(name)
                node = root
                path = attributes["path"].split(":")

                for node_name in path:
                    node_tmp = node.find(node_name)
                    if node_tmp is None:
                        node_tmp = etree.SubElement(node, node_name)
                        node.append(node_tmp)
                    node = node_tmp
                if attributes is not None:
                    for attribute_tmp in attributes.keys():
                        attribute = "_".join(attribute_tmp.split("_")[:-1])
                        if attribute != "path":
                            node.set(attribute, attributes[attribute])
                if data.size > 0:
                    node.text = ", ".join(
                        map(str, data.flatten())
                    )  # np.array2string(data, separator=',')[1:-1].replace('\n','')
                with open(f"{self.filename}{name}.xml", "wb") as f:
                    f.write(etree.tostring(root, pretty_print=True, encoding="utf-8"))
                    f.close()
            else:
                np.save(f"{self.filename}{name}.npy", data)

        def is_exist(self, group: str, name: str | None = None) -> bool:
            return (
                os.path.exists(f"{self.filename}{group}.{self.file_format}")
                or os.path.exists(f"{self.filename}{group}.itk.txt")
                or os.path.exists(f"{self.filename}{group}.fcsv")
                or os.path.exists(f"{self.filename}{group}.npy")
            )

        def get_names(self, group: str) -> list[str]:
            raise NotImplementedError()

        def get_group(self):
            raise NotImplementedError()

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            attributes = Attribute()
            if os.path.exists(f"{self.filename}{group if group is not None else ''}{name}.{self.file_format}"):
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName(f"{self.filename}{group if group is not None else ''}{name}.{self.file_format}")
                file_reader.ReadImageInformation()
                attributes["Origin"] = np.asarray(file_reader.GetOrigin())
                attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
                attributes["Direction"] = np.asarray(file_reader.GetDirection())
                for k in file_reader.GetMetaDataKeys():
                    attributes[k] = file_reader.GetMetaData(k)
                size = list(file_reader.GetSize())
                if len(size) == 3:
                    size = list(reversed(size))
                size = [file_reader.GetNumberOfComponents()] + size
            else:
                data, attributes = self.file_to_data(group if group is not None else "", name)
                size = data.shape
            return size, attributes

    class File:

        def __init__(self, filename: str, read: bool, file_format: str) -> None:
            self.filename = filename
            self.read = read
            self.file: "Dataset.AbstractFile" | None = None
            self.file_format = file_format

        def __enter__(self) -> "Dataset.AbstractFile":
            if self.file_format == "h5":
                self.file = Dataset.H5File(self.filename, self.read)
            else:
                self.file = Dataset.SitkFile(self.filename + "/", self.read, self.file_format)
            self.file.__enter__()
            return self.file

        def __exit__(self, exc_type, value, traceback):
            if self.file is not None:
                self.file.__exit__(exc_type, value, traceback)

    def __init__(self, filename: str | Path, file_format: str) -> None:
        if file_format != "h5" and not str(filename).endswith("/"):
            filename = f"{filename}/"
        self.is_directory = str(filename).endswith("/")
        self.filename = str(filename)
        self.file_format = file_format

    def write(
        self,
        group: str,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ):
        if attributes is None:
            attributes = Attribute()
        if self.is_directory:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                sub_directory = "/".join(s_group[:-1])
                name = f"{sub_directory}/{name}"
                group = s_group[-1]
            with Dataset.File(f"{self.filename}{name}", False, self.file_format) as file:
                file.data_to_file(group, data, attributes)
        else:
            with Dataset.File(self.filename, False, self.file_format) as file:
                file.data_to_file(f"{group}/{name}", data, attributes)

    def read_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
        if not os.path.exists(self.filename):
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        False,
                        self.file_format,
                    ) as file:
                        result = file.file_to_data("", group)
        else:
            with Dataset.File(self.filename, False, self.file_format) as file:
                result = file.file_to_data(groups, name)
        return result

    def read_transform(self, group: str, name: str) -> sitk.Transform:
        if not os.path.exists(self.filename):
            raise NameError(f"Dataset {self.filename} not found")
        transform_parameters, attribute = self.read_data(group, name)
        transforms_type = [v for k, v in attribute.items() if k.endswith(":Transform_0")]
        transforms = []
        for i, transform_type in enumerate(transforms_type):
            if transform_type == "Euler3DTransform_double_3_3":
                transform = sitk.Euler3DTransform()
            if transform_type == "AffineTransform_double_3_3":
                transform = sitk.AffineTransform(3)
            if transform_type == "BSplineTransform_double_3_3":
                transform = sitk.BSplineTransform(3)
            transform.SetFixedParameters(ast.literal_eval(attribute[f"{i}:FixedParameters"]))
            transform.SetParameters(tuple(transform_parameters[i]))
            transforms.append(transform)
        return sitk.CompositeTransform(transforms) if len(transforms) > 1 else transforms[0]

    def read_image(self, group: str, name: str):
        data, attribute = self.read_data(group, name)
        return data_to_image(data, attribute)

    def get_size(self, group: str) -> int:
        return len(self.get_names(group))

    def is_group_exist(self, group: str) -> bool:
        return self.get_size(group) > 0

    def is_dataset_exist(self, group: str, name: str) -> bool:
        return name in self.get_names(group)

    def _get_sub_directories(self, groups: str, sub_directory: str = ""):
        group = groups.split("/")[0]
        sub_directories = []
        if len(groups.split("/")) == 1:
            sub_directories.append(sub_directory)
        elif group == "*":
            for k in os.listdir(f"{self.filename}{sub_directory}"):
                if not os.path.isfile(f"{self.filename}{sub_directory}{k}"):
                    sub_directories.extend(
                        self._get_sub_directories(
                            "/".join(groups.split("/")[1:]),
                            f"{sub_directory}{k}/",
                        )
                    )
        else:
            sub_directory = f"{sub_directory}{group}/"
            if os.path.exists(f"{self.filename}{sub_directory}"):
                sub_directories.extend(self._get_sub_directories("/".join(groups.split("/")[1:]), sub_directory))
        return sub_directories

    def get_names(self, groups: str, index: list[int] | None = None) -> list[str]:
        names = []
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}"):
                    for name in sorted(os.listdir(f"{self.filename}{sub_directory}")):
                        if os.path.isfile(f"{self.filename}{sub_directory}{name}") or self.file_format != "h5":
                            with Dataset.File(
                                f"{self.filename}{sub_directory}{name}",
                                True,
                                self.file_format,
                            ) as file:
                                if file.is_exist(group):
                                    names.append(name.replace(".h5", "") if self.file_format == "h5" else name)
        else:
            with Dataset.File(self.filename, True, self.file_format) as file:
                names = file.get_names(groups)
        return [name for i, name in enumerate(sorted(names)) if index is None or i in index]

    def get_group(self):
        if self.is_directory:
            groups_set = set()
            for root, _, files in os.walk(self.filename):
                for file in files:
                    path = os.path.relpath(os.path.join(root, file.split(".")[0]), self.filename)
                    parts = path.split("/")
                    if len(parts) >= 2:
                        del parts[-2]
                    groups_set.add("/".join(parts))
            groups = list(groups_set)
        else:
            with Dataset.File(self.filename, True, self.file_format) as dataset_file:
                groups = dataset_file.get_group()
        return list(groups)

    def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                    ) as file:
                        result = file.get_infos("", group)
        else:
            with Dataset.File(self.filename, True, self.file_format) as file:
                result = file.get_infos(groups, name)
        return result
