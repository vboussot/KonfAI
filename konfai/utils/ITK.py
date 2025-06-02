import SimpleITK as sitk
from typing import Union
import numpy as np
import torch
import scipy
import torch.nn.functional as F
from konfai.utils.utils import _resample

def _openTransform(transform_files: dict[Union[str, sitk.Transform], bool], image: sitk.Image= None) -> list[sitk.Transform]:
    transforms: list[sitk.Transform] = []

    for transform_file, invert in transform_files.items():
        if isinstance(transform_file, str):
            transform = sitk.ReadTransform(transform_file+".itk.txt")
        else:
            transform = transform_file
        if transform.GetName() == "TranslationTransform":
            transform = sitk.TranslationTransform(transform)
            if invert:
                transform = sitk.TranslationTransform(transform.GetInverse())
        elif transform.GetName() == "Euler3DTransform":
            transform = sitk.Euler3DTransform(transform)
            if invert:
                transform = sitk.Euler3DTransform(transform.GetInverse())
        elif transform.GetName() == "VersorRigid3DTransform":
            transform = sitk.VersorRigid3DTransform(transform)
            if invert:
                transform = sitk.VersorRigid3DTransform(transform.GetInverse())
        elif transform.GetName() == "AffineTransform":
            transform = sitk.AffineTransform(transform)
            if invert:
                transform = sitk.AffineTransform(transform.GetInverse())
        elif transform.GetName() == "DisplacementFieldTransform":
            if invert:
                transformToDisplacementFieldFilter = sitk.TransformToDisplacementFieldFilter()
                transformToDisplacementFieldFilter.SetReferenceImage(image)
                displacementField = transformToDisplacementFieldFilter.Execute(transform)
                iterativeInverseDisplacementFieldImageFilter = sitk.IterativeInverseDisplacementFieldImageFilter()
                iterativeInverseDisplacementFieldImageFilter.SetNumberOfIterations(20)
                inverseDisplacementField = iterativeInverseDisplacementFieldImageFilter.Execute(displacementField)
                transform = sitk.DisplacementFieldTransform(inverseDisplacementField)
            transforms.append(transform)
        else:
            transform = sitk.BSplineTransform(transform)
            if invert:
                transformToDisplacementFieldFilter = sitk.TransformToDisplacementFieldFilter()
                transformToDisplacementFieldFilter.SetReferenceImage(image)
                displacementField = transformToDisplacementFieldFilter.Execute(transform)
                iterativeInverseDisplacementFieldImageFilter = sitk.IterativeInverseDisplacementFieldImageFilter()
                iterativeInverseDisplacementFieldImageFilter.SetNumberOfIterations(20)
                inverseDisplacementField = iterativeInverseDisplacementFieldImageFilter.Execute(displacementField)
                transform = sitk.DisplacementFieldTransform(inverseDisplacementField)
        transforms.append(transform)
    if len(transforms) == 0:
        transforms.append(sitk.Euler3DTransform())
    return transforms

def _openRigidTransform(transform_files: dict[Union[str, sitk.Transform], bool]) -> tuple[np.ndarray, np.ndarray]:
    transforms = _openTransform(transform_files)
    matrix_result = np.identity(3)
    translation_result = np.array([0,0,0])

    for transform in transforms:
        if hasattr(transform, "GetMatrix"):
            matrix = np.linalg.inv(np.array(transform.GetMatrix(), dtype=np.double).reshape((3,3)))
            translation = -np.asarray(transform.GetTranslation(), dtype=np.double)
            center = np.asarray(transform.GetCenter(), dtype=np.double)
        else:
            matrix = np.eye(len(transform.GetOffset()))
            translation = -np.asarray(transform.GetOffset(), dtype=np.double)
            center = np.asarray([0]*len(transform.GetOffset()), dtype=np.double)
        
        translation_center = np.linalg.inv(matrix).dot(matrix.dot(translation-center)+center)
        translation_result = np.linalg.inv(matrix_result).dot(translation_center)+translation_result
        matrix_result = matrix.dot(matrix_result)
    return np.linalg.inv(matrix_result), -translation_result

def composeTransform(transform_files : dict[Union[str, sitk.Transform], bool], image : sitk.Image = None) -> None:#sitk.CompositeTransform:
    transforms = _openTransform(transform_files, image)
    result = sitk.CompositeTransform(transforms)
    return result

def flattenTransform(transform_files: dict[Union[str, sitk.Transform], bool]) -> sitk.AffineTransform:
    [matrix, translation] = _openRigidTransform(transform_files)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix.flatten())
    transform.SetTranslation(translation)
    return transform

def apply_to_image_RigidTransform(image: sitk.Image, transform_files: dict[Union[str, sitk.Transform], bool]) -> sitk.Image:
    [matrix, translation] = _openRigidTransform(transform_files)
    matrix = np.linalg.inv(matrix)
    translation = -translation
    data = sitk.GetArrayFromImage(image)
    result = sitk.GetImageFromArray(data)
    result.SetDirection(matrix.dot(np.array(image.GetDirection()).reshape((3,3))).flatten())
    result.SetOrigin(matrix.dot(np.array(image.GetOrigin())+translation))
    result.SetSpacing(image.GetSpacing())
    return result

def apply_to_data_Transform(data: np.ndarray, transform_files: dict[Union[str, sitk.Transform], bool]) -> sitk.Image:
    transforms = composeTransform(transform_files)
    result = np.copy(data)
    _LPS = lambda matrix: np.array([-matrix[0], -matrix[1], matrix[2]], dtype=np.double)
    for i in range(data.shape[0]):
        result[i, :] =  _LPS(transforms.TransformPoint(np.asarray(_LPS(data[i, :]), dtype=np.double)))
    return result

def resampleITK(image_reference : sitk.Image, image : sitk.Image, transform_files : dict[Union[str, sitk.Transform], bool], mask = False, defaultPixelValue: Union[float, None] = None, torch_resample : bool = False) -> sitk.Image:
    if torch_resample:
        input = torch.tensor(sitk.GetArrayFromImage(image)).unsqueeze(0)
        vectors = [torch.arange(0, s) for s in input.shape[1:]]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        transformToDisplacementFieldFilter = sitk.TransformToDisplacementFieldFilter()        
        transformToDisplacementFieldFilter.SetReferenceImage(image)
        transformToDisplacementFieldFilter.SetNumberOfThreads(16)
        new_locs = grid + torch.tensor(sitk.GetArrayFromImage(transformToDisplacementFieldFilter.Execute(composeTransform(transform_files, image)))).unsqueeze(0).permute(0, 4, 1, 2, 3)
        shape = new_locs.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        result_data = F.grid_sample(input.unsqueeze(0).float(), new_locs.float(), align_corners=True, padding_mode="border", mode="nearest" if input.dtype == torch.uint8 else "bilinear").squeeze(0)
        result_data = result_data.type(torch.uint8) if input.dtype == torch.uint8 else result_data
        result = sitk.GetImageFromArray(result_data.squeeze(0).numpy())
        result.CopyInformation(image_reference)
        return result
    else:
        return sitk.Resample(image, image_reference, composeTransform(transform_files, image), sitk.sitkNearestNeighbor if mask else sitk.sitkBSpline, (defaultPixelValue if defaultPixelValue is not None else (0 if mask else int(np.min(sitk.GetArrayFromImage(image))))))

def parameterMap_to_transform(path_src: str) -> Union[sitk.Transform, list[sitk.Transform]]:
    transform = sitk.ReadParameterFile("{}.0.txt".format(path_src))
    format = lambda x: np.array([float(i) for i in x])

    if transform["Transform"][0] == "EulerTransform":
        result = sitk.Euler3DTransform()
        parameters = format(transform["TransformParameters"])
        fixedParameters = format(transform["CenterOfRotationPoint"])+[0]
    elif transform["Transform"][0] == "AffineTransform":
        result = sitk.AffineTransform(3)
        parameters = format(transform["TransformParameters"])
        fixedParameters = format(transform["CenterOfRotationPoint"])+[0]
    elif transform["Transform"][0] == "BSplineStackTransform":
        parameters = format(transform["TransformParameters"])
        GridSize = format(transform["GridSize"])
        GridOrigin = format(transform["GridOrigin"])
        GridSpacing = format(transform["GridSpacing"])
        GridDirection = format(transform["GridDirection"]).reshape((3,3)).T.flatten() 
        fixedParameters = np.concatenate([GridSize, GridOrigin, GridSpacing, GridDirection])

        nb = int(format(transform["Size"])[-1])
        sub = int(np.prod(GridSize))*3
        results = []
        for i in range(nb):
            result = sitk.BSplineTransform(3)
            sub_parameters = parameters[i*sub:(i+1)*sub]
            result.SetFixedParameters(fixedParameters)
            result.SetParameters(sub_parameters)
            results.append(result)
        return results
    elif transform["Transform"][0] == "AffineLogStackTransform":
        parameters = format(transform["TransformParameters"])
        fixedParameters = format(transform["CenterOfRotationPoint"])+[0]

        nb = int(transform["NumberOfSubTransforms"][0])
        sub = 12
        results = []
        for i in range(nb):
            result = sitk.AffineTransform(3)
            sub_parameters = parameters[i*sub:(i+1)*sub]

            result.SetFixedParameters(fixedParameters)
            result.SetParameters(np.concatenate([scipy.linalg.expm(sub_parameters[:9].reshape((3,3))).flatten(), sub_parameters[-3:]]))
            results.append(result)
        return results
    elif transform["Transform"][0] == "BSplineTransform":
        result = sitk.BSplineTransform(3)
        
        parameters = format(transform["TransformParameters"])
        GridSize = format(transform["GridSize"])
        GridOrigin = format(transform["GridOrigin"])
        GridSpacing = format(transform["GridSpacing"])
        GridDirection = np.array(format(transform["GridDirection"])).reshape((3,3)).T.flatten() 
        fixedParameters = np.concatenate([GridSize, GridOrigin, GridSpacing, GridDirection])
    else:
        raise NameError("Transform {} doesn't exist".format(transform["Transform"][0]))
    result.SetFixedParameters(fixedParameters)
    result.SetParameters(parameters)
    return result

def resampleIsotropic(image: sitk.Image, spacing : list[float] = [1., 1., 1.]) -> sitk.Image:
    resize_factor = [y/x for x,y in zip(spacing, image.GetSpacing())]
    result = sitk.GetImageFromArray(_resample(torch.tensor(sitk.GetArrayFromImage(image)).unsqueeze(0), [int(size*factor) for size, factor in zip(image.GetSize(), resize_factor)]).squeeze(0).numpy())
    result.SetDirection(image.GetDirection())
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing(spacing)
    return result

def resampleResize(image: sitk.Image, size : list[int] = [100,512,512]):
    result =  sitk.GetImageFromArray(_resample(torch.tensor(sitk.GetArrayFromImage(image)).unsqueeze(0), size).squeeze(0).numpy())
    result.SetDirection(image.GetDirection())
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing([x/y*z for x,y,z in zip(image.GetSize(), size, image.GetSpacing())])
    return result

def box_with_mask(mask: sitk.Image, label: list[int], dilatations: list[int]) -> np.ndarray:

    dilatations = [int(np.ceil(d/s)) for d, s in zip(dilatations, reversed(mask.GetSpacing()))]

    data = sitk.GetArrayFromImage(mask)
    border = np.where(np.isin(sitk.GetArrayFromImage(mask), label))
    box = []
    for w, dilatation, s in zip(border, dilatations, data.shape):
        box.append([max(np.min(w)-dilatation, 0), min(np.max(w)+dilatation, s)])
    box = np.asarray(box)
    return box

def crop_with_mask(image: sitk.Image, box: np.ndarray) -> sitk.Image:
    data = sitk.GetArrayFromImage(image)
    
    for i, w in enumerate(box):
        data = np.delete(data, slice(w[1], data.shape[i]), i)
        data = np.delete(data, slice(0, w[0]), i)
    
    origin = np.asarray(image.GetOrigin())
    matrix = np.asarray(image.GetDirection()).reshape((len(origin), len(origin)))
    origin = origin.dot(matrix)
    for i, w in enumerate(box):
        origin[-i-1] += w[0]*np.asarray(image.GetSpacing())[-i-1]
    origin = origin.dot(np.linalg.inv(matrix))

    result = sitk.GetImageFromArray(data)
    result.SetOrigin(origin)
    result.SetSpacing(image.GetSpacing())
    result.SetDirection(image.GetDirection())
    return result

def formatMaskLabel(mask: sitk.Image, labels: list[tuple[int, int]]) -> sitk.Image:
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)

    for label_old, label_new in labels:
        result_data[np.where(data == label_old)] = label_new

    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)
    return result

def getFlatLabel(mask: sitk.Image, labels: Union[None, list[int]] = None) -> sitk.Image:
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)
    if labels is not None:
        for label in labels:
            result_data[np.where(data == label)] = 1
    else:
        result_data[np.where(data > 0)] = 1
    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)        
    return result

def clipAndCast(image : sitk.Image, min: float, max: float, dtype: np.dtype) -> sitk.Image:
    data = sitk.GetArrayFromImage(image)
    data[np.where(data > max)] = max
    data[np.where(data < min)] = min
    result = sitk.GetImageFromArray(data.astype(dtype))
    result.CopyInformation(image)
    return result
