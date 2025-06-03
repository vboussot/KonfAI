import SimpleITK as sitk
import h5py
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Union
import copy
import torch
import os

from lxml import etree
import csv
from konfai import DATE

class Plot():

    def __init__(self, root: etree.ElementTree) -> None:
        self.root = root

    def _explore(root, result, label):
        if len(root) == 0:
            for attribute in root.attrib:
                result["attrib:"+label+":"+attribute] = root.attrib[attribute]
            if root.text is not None:
                result[label] = np.fromstring(root.text, sep = ",").astype('double')
        else:
            for node in root:
                Plot._explore(node, result, label+":"+node.tag)

    def getNodes(root, path = None, id = None):
        nodes = []
        if path != None:
            path = path.split(":")
            for node_name in path:
                node = root.find(node_name)
                if node != None:
                    root = node
                else:
                    break
        if id != None:
            for node in root.findall(".//"+id):
                nodes.append(node)
        else:
            nodes.append(root)                
        return nodes

    def read(root, path = None, id = None):
        result = dict()
        for node in Plot.getNodes(root, path, id):
            Plot._explore(node, result, etree.ElementTree(root).getpath(node))    
        return result

    def _extract(self, ids = [], patients = []):
        result = dict()
        if len(patients) == 0:
            if len(ids) == 0:
                result.update(Plot.read(self.root,None, None))
            else:
                for id in ids:
                    result.update(Plot.read(self.root, None, id))
        else:
            for path in patients:
                if len(ids) == 0:
                    result.update(Plot.read(self.root, path, None))
                else:
                    for id in ids:
                        result.update(Plot.read(self.root, path, id))
        return result

    def getErrors(self, ids = [], patients = []):
        results = self._extract(ids=ids, patients=patients)
        errors = {k: v for k, v in results.items() if not k.startswith("attrib:")}
        results : dict[str, dict[str, np.ndarray]]= {}
        for key, error in errors.items():
            patient = key.replace("/",":").split(":")[2]
            k = key.replace("/",":").split(":")[-1]
            err = np.linalg.norm(error.reshape(int(error.shape[0]/3),3), ord=2, axis=1)
            if patient not in results:
                results[patient] = {k : err}
            else:
                results[patient].update({k : err})
        return results
    
    def statistic_attrib(self, ids = [], patients = [], type: str = "HD95Mean"):
        results = self._extract(ids=ids, patients=patients)
        
        errors = {k.replace("attrib:", ""): float(v) for k, v in results.items() if type in k}
        
        values = {key : np.array([]) for key in ids}
        for key, error in errors.items():
            k = key.replace("/",":").split(":")[-2]
            values[k] = np.append(values[k], error)
        
        for k in values:
            values[k] = np.mean(values[k])
        print(values)
        return values
    
    def statistic_parameter(self, ids = [], patients = []):
        results = self._extract(ids=ids, patients=patients)
        errors = {k.replace("attrib:", "").replace(":Time", "") : np.load("./Results/{}/{}.npy".format(k.split("/")[3].split(":")[0], k.split("/")[2])) for k in results.keys()}
        
        norms = {key : np.array([]) for key in ids}
        max = 0
        for key, error in errors.items():
            if max < int(error.shape[0]/3): 
                max = int(error.shape[0]/3)
        
        for key, error in errors.items():
            k = key.replace("/",":").split(":")[-1]
            norms[k] = np.append(norms[k], np.linalg.norm(error.reshape(int(error.shape[0]/3),3), ord=2, axis=1))
            v = np.linalg.norm(error.reshape(int(error.shape[0]/3),3), ord=2, axis=1)

            print(key, "{} {} {} {} {}".format(np.round(np.mean(v), 2), np.round(np.std(v), 2), np.round(np.quantile(v, 0.25), 2), np.round(np.quantile(v, 0.5), 2), np.round(np.quantile(v, 0.75), 2))) 
        results = {}
        for key, values in norms.items():
            if key == "Rigid":
                results.update({key : values})
            else:    
                try:
                    name = "{}".format("_".join(key.split("_"))[:-1])
                    it = int(key.split("_")[-1])
                except:
                    name = "{}".format(key.split("-")[0])
                    it = int(key.split("-")[-1])
                
                if name in results:
                    results[name].update({it : values})
                else:
                    results.update({name : {it : values}})
        
        r = []
        for key, values in norms.items():
            #r.append("{} $\pm$ {}".format(np.round(np.mean(values), 2), np.round(np.std(values), 2))) 
            r.append("{} {} {}".format(np.round(np.quantile(values, 0.25), 2), np.round(np.quantile(values, 0.5),2), np.round(np.quantile(values, 0.75), 2)))
            #r.append("{} $\pm$ {}".format(np.round(np.quantile(values, 0.5), 2), np.round(np.quantile(values, 0.75)-np.quantile(values, 0.25), 2)))
        print(" & ".join(r))

    def statistic(self, ids = [], patients = []):
        results = self._extract(ids=ids, patients=patients)
        #errors = {k.replace("attrib:", "").replace(":Time", "") : np.load("./Dataset/{}/{}.npy".format(k.split("/")[3].split(":")[0], k.split("/")[2])) for k in results.keys()}
        errors = {k: v for k, v in results.items() if not k.startswith("attrib:")}
        print(errors)
        norms = {key : np.array([]) for key in ids}
        max = 0
        for key, error in errors.items():
            if max < int(error.shape[0]/3): 
                max = int(error.shape[0]/3)
        
        for key, error in errors.items():
            k = key.replace("/",":").split(":")[-1]
            norms[k] = np.append(norms[k], np.linalg.norm(error.reshape(int(error.shape[0]/3),3), ord=2, axis=1))
            v = np.linalg.norm(error.reshape(int(error.shape[0]/3),3), ord=2, axis=1)
            print(key, (np.mean(v), np.std(v), np.quantile(v, 0.25), np.quantile(v, 0.5), np.quantile(v, 0.75)) )
        results = {}
        """for key, values in norms.items():
            if key == "Rigid":
                results.update({key : values})
            else:    
                try:
                    name = "{}".format("_".join(key.split("_"))[:-1])
                    it = int(key.split("_")[-1])
                except:
                    name = "{}".format(key.split("-")[0])
                    it = int(key.split("-")[-1])
                
                if name in results:
                    results[name].update({it : values})
                else:
                    results.update({name : {it : values}})"""


        print({key: (np.mean(values), np.std(values), np.quantile(values, 0.25), np.quantile(values, 0.5), np.quantile(values, 0.75)) for key, values in norms.items()})
        return results
    
    def plot(self, ids = [], patients = [], labels = [], colors = None):

        import matplotlib.pyplot as pyplot
        results = self._extract(ids=ids, patients=patients)

        attrs = {k: v for k, v in results.items() if k.startswith("attrib:")}
        errors = {k: v for k, v in results.items() if not k.startswith("attrib:")}

        patients = set()
        max = 0
        for key, error in errors.items():
            patients.add(key.replace("/",":").split(":")[2])
            if max < int(error.shape[0]/3): 
                max = int(error.shape[0]/3)
        patients = sorted(patients)

        norms = {patient : np.array([]) for patient in patients}
        markups = {patient : np.array([]) for patient in patients}
        series = list()
        for key, error in errors.items():
            patient = key.replace("/",":").split(":")[2]
            markup = np.full((max,3), np.nan)
            markup[0:int(error.shape[0]/3), :] = error.reshape(int(error.shape[0]/3),3)

            markups[patient] = np.append(markups[patient], markup)
            norms[patient] = np.append(norms[patient], np.linalg.norm(markup, ord=2, axis=1))

        if len(labels) == 0:
            labels = list(set([k.split("/")[-1] for k in errors.keys()]))

        for label in labels:
            series = series+[label]*max
        import pandas as pd
        df = pd.DataFrame(dict([(k,pd.Series(v)) for k, v in norms.items()]))
        df['Categories'] = pd.Series(series)
        
        bp = df.boxplot(by='Categories', color="black", figsize=(12,8), notch=True,layout=(1,len(patients)), fontsize=18, rot=0, patch_artist = True, return_type='both',  widths=[0.5]*len(labels))

        color_pallet = {"b" : "paleturquoise", "g" : "lightgreen"}
        if colors == None:
            colors = ["b"] * len(patients)
        pyplot.suptitle('')
        it_1 = 0
        for index, (ax,row)  in bp.items():
            ax.set_xlabel('')
            ax.set_ylim(ymin=0)
            ax.set_ylabel("TRE (mm)", fontsize=18)
            ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,15,20,25])  # Set label locations.
            for i,object in enumerate(row["boxes"]):
                object.set_edgecolor("black")
                object.set_facecolor(color_pallet[colors[i]])
                object.set_alpha(0.7)
                object.set_linewidth(1.0)
            
            for i,object in enumerate(row["medians"]):
                object.set_color("indianred")
                xy = object.get_xydata()
                object.set_linewidth(2.0)
                it_1+=1
        return self
    
    def show(self):
        import matplotlib.pyplot as pyplot
        pyplot.show()

class Attribute(dict[str, Any]):

    def __init__(self, attributes : dict[str, Any] = {}) -> None:
        super().__init__()
        for k, v in attributes.items():
            super().__setitem__(copy.deepcopy(k), copy.deepcopy(v))
    
    def __getitem__(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return str(super().__getitem__("{}_{}".format(key, i-1)))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def __setitem__(self, key: str, value: Any) -> None:
        if "_" not in key:
            i = len([k for k in super().keys() if k.startswith(key)])
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace('\n', '')
            super().__setitem__("{}_{}".format(key, i), result)
        else:
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace('\n', '')
            super().__setitem__(key, result)

    def pop(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return super().pop("{}_{}".format(key, i-1))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def get_np_array(self, key) -> np.ndarray:
        return np.fromstring(self[key][1:-1], sep=" ", dtype=np.double)
    
    def get_tensor(self, key) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key)).to(torch.float32)
    
    def pop_np_array(self, key):
        return np.fromstring(self.pop(key)[1:-1], sep=" ", dtype=np.double)
    
    def pop_tensor(self, key) -> torch.Tensor:
        return torch.tensor(self.pop_np_array(key))
    
    def __contains__(self, key: str) -> bool:
        return len([k for k in super().keys() if k.startswith(key)]) > 0
    
    def isInfo(self, key: str, value: str) -> bool:
        return key in self and self[key] == value

def isAnImage(attributes: Attribute):
    return "Origin" in attributes and "Spacing" in attributes and "Direction" in attributes

def data_to_image(data : np.ndarray, attributes: Attribute) -> sitk.Image:
    if not isAnImage(attributes):
        raise NameError("Data is not an image")
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i+1 for i in range(len(data.shape)-1)]+[0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    image.SetOrigin(attributes.get_np_array("Origin").tolist())
    image.SetSpacing(attributes.get_np_array("Spacing").tolist())
    image.SetDirection(attributes.get_np_array("Direction").tolist())
    return image

def image_to_data(image: sitk.Image) -> tuple[np.ndarray, Attribute]:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    data = sitk.GetArrayFromImage(image)

    if image.GetNumberOfComponentsPerPixel() == 1:
        data = np.expand_dims(data, 0)
    else:
        data = np.transpose(data, (len(data.shape)-1, *[i for i in range(len(data.shape)-1)]))
    return data, attributes

class Dataset():

    class AbstractFile(ABC):

        def __init__(self) -> None:
            pass
        
        def __enter__(self):
            pass

        def __exit__(self, type, value, traceback):
            pass

        @abstractmethod
        def file_to_data(self):
            pass

        @abstractmethod
        def data_to_file(self):
            pass

        @abstractmethod
        def getNames(self, group: str) -> list[str]:
            pass

        @abstractmethod
        def isExist(self, group: str, name: Union[str, None] = None) -> bool:
            pass
        
        @abstractmethod
        def getInfos(self, group: Union[str, None], name: str) -> tuple[list[int], Attribute]:
            pass

    class H5File(AbstractFile):

        def __init__(self, filename: str, read: bool) -> None:
            self.h5: Union[h5py.File, None] = None
            self.filename = filename
            if not self.filename.endswith(".h5"):
                self.filename += ".h5"
            self.read = read

        def __enter__(self):
            args = {}
            if self.read:
                self.h5 = h5py.File(self.filename, 'r', **args)
            else:
                if not os.path.exists(self.filename):
                    if len(self.filename.split("/")) > 1 and not os.path.exists("/".join(self.filename.split("/")[:-1])):
                        os.makedirs("/".join(self.filename.split("/")[:-1]))
                    self.h5 = h5py.File(self.filename, 'w', **args)
                else: 
                    self.h5 = h5py.File(self.filename, 'r+', **args)
                self.h5.attrs["Date"] = DATE()
            self.h5.__enter__()
            return self.h5
    
        def __exit__(self, type, value, traceback):
            if self.h5 is not None:
                self.h5.close()
        
        def file_to_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
            dataset = self._getDataset(groups, name)
            data = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(data)
            return data, Attribute({k : str(v) for k, v in dataset.attrs.items()})

        def data_to_file(self, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Union[Attribute, None] = None) -> None:
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
                    attributes["{}:Transform".format(i)] = transform_type
                    attributes["{}:FixedParameters".format(i)] = transform.GetFixedParameters()

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
            dataset.attrs.update({k : str(v) for k, v in attributes.items()})
        
        def isExist(self, group: str, name: Union[str, None] = None) -> bool:
            if group in self.h5:
                if isinstance(self.h5[group], h5py.Dataset):
                    return True
                elif name is not None:
                    return name in self.h5[group]
                else:
                    return False
            return False

        def getNames(self, groups: str, h5_group: h5py.Group = None) -> list[str]:
            names = []
            if h5_group is None:
                h5_group = self.h5
            group = groups.split("/")[0]
            if group == "":
                names = [dataset.name.split("/")[-1] for dataset in h5_group.values() if isinstance(dataset, h5py.Dataset)]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        names.extend(self.getNames("/".join(groups.split("/")[1:]), h5_group[k]))
            else:
                if group in h5_group:
                    names.extend(self.getNames("/".join(groups.split("/")[1:]), h5_group[group]))
            return names
        
        def _getDataset(self, groups: str, name: str, h5_group: h5py.Group = None) -> h5py.Dataset:
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
                        result_tmp = self._getDataset("/".join(groups.split("/")[1:]), name, h5_group[k])
                        if result_tmp is not None:
                            result = result_tmp
            else:
                if group in h5_group:
                    result_tmp = self._getDataset("/".join(groups.split("/")[1:]), name, h5_group[group])
                    if result_tmp is not None:
                        result = result_tmp
            return result
        
        def getInfos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
            dataset = self._getDataset(groups, name)
            return (dataset.shape, Attribute({k : str(v) for k, v in dataset.attrs.items()}))
        
    class SitkFile(AbstractFile):

        def __init__(self, filename: str, read: bool, format: str) -> None:
            self.filename = filename
            self.read = read
            self.format = format
     
        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            attributes = Attribute()
            if os.path.exists("{}{}.{}".format(self.filename, name, self.format)):
                image = sitk.ReadImage("{}{}.{}".format(self.filename, name, self.format))
                data, attributes_tmp = image_to_data(image)
                attributes.update(attributes_tmp)
            elif os.path.exists("{}{}.itk.txt".format(self.filename, name)): 
                data = sitk.ReadTransform("{}{}.itk.txt".format(self.filename, name))
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
                    attributes["{}:Transform".format(i)] = transform_type
                    attributes["{}:FixedParameters".format(i)] = transform.GetFixedParameters()

                    datas.append(np.asarray(transform.GetParameters()))
                data = np.asarray(datas)
            elif os.path.exists("{}{}.fcsv".format(self.filename, name)):
                with open("{}{}.fcsv".format(self.filename, name), newline="") as csvfile:
                    reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
                    lines = list(reader)
                    data = np.zeros((len(list(lines)), 3), dtype=np.double)
                    for i, row in enumerate(lines):
                        data[i] = np.array(row[1:4], dtype=np.double)
                    csvfile.close()
            elif os.path.exists("{}{}.xml".format(self.filename, name)):
                with open("{}{}.xml".format(self.filename, name), 'rb') as xml_file:
                    result = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()
                    xml_file.close()
                    return result
            elif os.path.exists("{}{}.vtk".format(self.filename, name)):
                import vtk
                vtkReader = vtk.vtkPolyDataReader()
                vtkReader.SetFileName("{}{}.vtk".format(self.filename, name))
                vtkReader.Update()
                data = []
                points = vtkReader.GetOutput().GetPoints()
                num_points = points.GetNumberOfPoints()
                for i in range(num_points):
                    data.append(list(points.GetPoint(i)))
                data = np.asarray(data)
            elif os.path.exists("{}{}.npy".format(self.filename, name)):
                data = np.load("{}{}.npy".format(self.filename, name))
            return data, attributes
        
        def is_vtk_polydata(self, obj):
            try:
                import vtk
                return isinstance(obj, vtk.vtkPolyData)
            except ImportError:
                return False
                
        def data_to_file(self, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Attribute = Attribute()) -> None:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
            if isinstance(data, sitk.Image):
                for k, v in attributes.items():
                    data.SetMetaData(k, v)
                sitk.WriteImage(data, "{}{}.{}".format(self.filename, name, self.format))
            elif isinstance(data, sitk.Transform):
                sitk.WriteTransform(data, "{}{}.itk.txt".format(self.filename, name))
            elif self.is_vtk_polydata(data):
                import vtk
                vtkWriter = vtk.vtkPolyDataWriter()
                vtkWriter.SetFileName("{}{}.vtk".format(self.filename, name))
                vtkWriter.SetInputData(data)
                vtkWriter.Write()
            elif isAnImage(attributes):   
                self.data_to_file(name, data_to_image(data, attributes), attributes)
            elif (len(data.shape) == 2 and data.shape[1] == 3 and data.shape[0] > 0):
                data = np.round(data, 4)
                with open("{}{}.fcsv".format(self.filename, name), 'w') as f:
                    f.write("# Markups fiducial file version = 4.6\n# CoordinateSystem = 0\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
                    for i in range(data.shape[0]):
                        f.write("vtkMRMLMarkupsFiducialNode_"+str(i+1)+","+str(data[i, 0])+","+str(data[i, 1])+","+str(data[i, 2])+",0,0,0,1,1,1,0,F-"+str(i+1)+",,vtkMRMLScalarVolumeNode1\n")
                    f.close()
            elif "path" in attributes:
                if os.path.exists("{}{}.xml".format(self.filename, name)):
                    with open("{}{}.xml".format(self.filename, name), 'rb') as xml_file:
                        root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()
                        xml_file.close()
                else:
                    root = etree.Element(name)
                node = root
                path = attributes["path"].split(':')

                for node_name in path:
                    node_tmp = node.find(node_name)
                    if node_tmp == None:
                        node_tmp = etree.SubElement(node, node_name)
                        node.append(node_tmp)
                    node = node_tmp
                if attributes != None:
                    for attribute_tmp in attributes.keys():
                        attribute = "_".join(attribute_tmp.split("_")[:-1])
                        if attribute != "path":
                            node.set(attribute, attributes[attribute])
                if data.size > 0:
                    node.text = ", ".join(map(str, data.flatten())) #np.array2string(data, separator=',')[1:-1].replace('\n','')
                with open("{}{}.xml".format(self.filename, name), 'wb') as f:
                    f.write(etree.tostring(root, pretty_print=True, encoding='utf-8'))
                    f.close()
            else:
                np.save("{}{}.npy".format(self.filename, name), data)
                 
        def isExist(self, group: str, name: Union[str, None] = None) -> bool:
            return os.path.exists("{}{}.{}".format(self.filename, group, self.format)) or os.path.exists("{}{}.itk.txt".format(self.filename, group)) or os.path.exists("{}{}.fcsv".format(self.filename, group)) or os.path.exists("{}{}.npy".format(self.filename, group))
        
        def getNames(self, group: str) -> list[str]:
            raise NotImplementedError()

        def getInfos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            attributes = Attribute()
            if os.path.exists("{}{}{}.{}".format(self.filename, group if group is not None else "", name, self.format)):
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName("{}{}{}.{}".format(self.filename, group if group is not None else "", name, self.format))
                file_reader.ReadImageInformation()
                attributes["Origin"] = np.asarray(file_reader.GetOrigin())
                attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
                attributes["Direction"] = np.asarray(file_reader.GetDirection())
                for k in file_reader.GetMetaDataKeys():
                    attributes[k] = file_reader.GetMetaData(k)
                size = list(file_reader.GetSize())
                if len(size) == 3:
                    size = list(reversed(size))
                size = [file_reader.GetNumberOfComponents()]+size
            else:
                data, attributes = self.file_to_data(group if group is not None else "", name)
                size = data.shape
            return tuple(size), attributes

    class File(ABC):

        def __init__(self, filename: str, read: bool, format: str) -> None:
            self.filename = filename
            self.read = read
            self.file = None
            self.format = format

        def __enter__(self):
            if self.format == "h5":
                self.file = Dataset.H5File(self.filename, self.read)
            else:
                self.file = Dataset.SitkFile(self.filename+"/", self.read, self.format)
            self.file.__enter__()
            return self.file

        def __exit__(self, type, value, traceback):
            self.file.__exit__(type, value, traceback)

    def __init__(self, filename : str, format: str) -> None:
        if format != "h5" and not filename.endswith("/"):
            filename = "{}/".format(filename)
        self.is_directory = filename.endswith("/") 
        self.filename = filename
        self.format = format
        
    def write(self, group : str, name : str, data : Union[sitk.Image, sitk.Transform, np.ndarray], attributes : Attribute = Attribute()):
        if self.is_directory:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                subDirectory = "/".join(s_group[:-1])
                name = "{}/{}".format(subDirectory, name)
                group = s_group[-1]
            with Dataset.File("{}{}".format(self.filename, name), False, self.format) as file:
                file.data_to_file(group, data, attributes)
        else:
            with Dataset.File(self.filename, False, self.format) as file:
                file.data_to_file("{}/{}".format(group, name), data, attributes)
    
    def readData(self, groups : str, name : str) -> tuple[np.ndarray, Attribute]:
        if not os.path.exists(self.filename):
            raise NameError("Dataset {} not found".format(self.filename))
        if self.is_directory:
            for subDirectory in self._getSubDirectories(groups):
                group = groups.split("/")[-1]
                if os.path.exists("{}{}{}{}".format(self.filename, subDirectory, name, ".h5" if self.format == "h5" else "")):
                    with Dataset.File("{}{}{}".format(self.filename, subDirectory, name), False, self.format) as file:
                        result = file.file_to_data("", group)
        else:
            with Dataset.File(self.filename, False, self.format) as file:
                result = file.file_to_data(groups, name)
        return result
    
    def readTransform(self, group : str, name : str) -> sitk.Transform:
        if not os.path.exists(self.filename):
            raise NameError("Dataset {} not found".format(self.filename))
        transformParameters, attribute = self.readData(group, name)
        transforms_type = [v for k, v in attribute.items() if k.endswith(":Transform_0")]
        transforms = []
        for i, transform_type in enumerate(transforms_type):
            if transform_type == "Euler3DTransform_double_3_3":
                transform = sitk.Euler3DTransform()
            if transform_type == "AffineTransform_double_3_3":
                transform = sitk.AffineTransform(3)
            if transform_type == "BSplineTransform_double_3_3":
                transform = sitk.BSplineTransform(3)
            transform.SetFixedParameters(eval(attribute["{}:FixedParameters".format(i)]))
            transform.SetParameters(tuple(transformParameters[i]))
            transforms.append(transform)
        return sitk.CompositeTransform(transforms) if len(transforms) > 1 else transforms[0]

    def readImage(self, group : str, name : str):
         data, attribute = self.readData(group, name)
         return data_to_image(data, attribute)
            
    def getSize(self, group: str) -> int:
        return len(self.getNames(group))
    
    def isGroupExist(self, group: str) -> bool:
        return self.getSize(group) > 0
    
    def isDatasetExist(self, group: str, name: str) -> bool:
        return name in self.getNames(group)
    
    def _getSubDirectories(self, groups: str, subDirectory: str = ""):
        group = groups.split("/")[0]
        subDirectories = []
        if len(groups.split("/")) == 1:
            subDirectories.append(subDirectory)
        elif group == "*":
            for k in os.listdir("{}{}".format(self.filename, subDirectory)):
                if not os.path.isfile("{}{}{}".format(self.filename, subDirectory, k)):
                    subDirectories.extend(self._getSubDirectories("/".join(groups.split("/")[1:]), "{}{}/".format(subDirectory , k)))
        else:
            subDirectory = "{}{}/".format(subDirectory, group)
            if os.path.exists("{}{}".format(self.filename, subDirectory)):
                subDirectories.extend(self._getSubDirectories("/".join(groups.split("/")[1:]), subDirectory))
        return subDirectories

    def getNames(self, groups: str, index: Union[list[int], None] = None, subDirectory: str = "") -> list[str]:
        names = []
        if self.is_directory:
            for subDirectory in self._getSubDirectories(groups):
                group = groups.split("/")[-1]
                if os.path.exists("{}{}".format(self.filename, subDirectory)):
                    for name in sorted(os.listdir("{}{}".format(self.filename, subDirectory))):
                        if os.path.isfile("{}{}{}".format(self.filename, subDirectory, name)) or self.format != "h5":
                            with Dataset.File("{}{}{}".format(self.filename, subDirectory, name), True, self.format) as file:
                                if file.isExist(group):
                                    names.append(name.replace(".h5", "") if self.format == "h5" else name)
        else:
            with Dataset.File(self.filename, True, self.format) as file:
                names = file.getNames(groups)
        return [name for i, name in enumerate(names) if index is None or i in index]
    
    def getInfos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
        if self.is_directory:
            for subDirectory in self._getSubDirectories(groups):
                group = groups.split("/")[-1]
                if os.path.exists("{}{}{}{}".format(self.filename, subDirectory, name, ".h5" if self.format == "h5" else "")):
                    with Dataset.File("{}{}{}".format(self.filename, subDirectory, name), True, self.format) as file:
                        result = file.getInfos("", group)      
        else:
            with Dataset.File(self.filename, True, self.format) as file:
                result = file.getInfos(groups, name)
        return result