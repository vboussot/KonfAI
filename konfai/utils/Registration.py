import SimpleITK as sitk
from typing import Union
import numpy as np
import sys
import scipy

def parameterMap_to_transform(path_src: str) -> Union[sitk.Transform, list[sitk.Transform]]:
    transform = sitk.ReadParameterFile(path_src)
    format = lambda x: [float(i) for i in x]
    dimension = int(transform["FixedImageDimension"][0])

    if transform["Transform"][0] == "EulerTransform":
        if dimension == 2:
            result = sitk.Euler2DTransform()
        else:
            result = sitk.Euler3DTransform()
        parameters = format(transform["TransformParameters"])
        fixedParameters = format(transform["CenterOfRotationPoint"])+[0]
    elif transform["Transform"][0] == "TranslationTransform":
        result = sitk.TranslationTransform(dimension)
        parameters = format(transform["TransformParameters"])
        fixedParameters = []
    elif transform["Transform"][0] == "AffineTransform":
        result = sitk.AffineTransform(dimension)
        parameters = format(transform["TransformParameters"])
        fixedParameters = format(transform["CenterOfRotationPoint"])+[0]
    elif transform["Transform"][0] == "BSplineStackTransform":
        parameters = format(transform["TransformParameters"])
        GridSize = format(transform["GridSize"])
        GridOrigin = format(transform["GridOrigin"])
        GridSpacing = format(transform["GridSpacing"])
        GridDirection =  np.asarray(format(transform["GridDirection"])).reshape((dimension, dimension)).T.flatten() 
        fixedParameters = np.concatenate([GridSize, GridOrigin, GridSpacing, GridDirection])

        nb = int(format(transform["Size"])[-1])
        sub = int(np.prod(GridSize))*dimension
        results = []
        for i in range(nb):
            result = sitk.BSplineTransform(dimension)
            sub_parameters = np.asarray(parameters[i*sub:(i+1)*sub])
            result.SetFixedParameters(fixedParameters)
            result.SetParameters(sub_parameters)
            results.append(result)
        return results
    elif transform["Transform"][0] == "AffineLogStackTransform":
        parameters = format(transform["TransformParameters"])
        fixedParameters = format(transform["CenterOfRotationPoint"])+[0]

        nb = int(transform["NumberOfSubTransforms"][0])
        sub = dimension*4
        results = []
        for i in range(nb):
            result = sitk.AffineTransform(dimension)
            sub_parameters = np.asarray(parameters[i*sub:(i+1)*sub])

            result.SetFixedParameters(fixedParameters)
            result.SetParameters(np.concatenate([scipy.linalg.expm(sub_parameters[:dimension*dimension].reshape((dimension,dimension))).flatten(), sub_parameters[-dimension:]]))
            results.append(result)
        return results
    else:
        result = sitk.BSplineTransform(dimension)
        
        parameters = format(transform["TransformParameters"])
        GridSize = format(transform["GridSize"])
        GridOrigin = format(transform["GridOrigin"])
        GridSpacing = format(transform["GridSpacing"])
        GridDirection = np.array(format(transform["GridDirection"])).reshape((dimension,dimension)).T.flatten() 
        fixedParameters = np.concatenate([GridSize, GridOrigin, GridSpacing, GridDirection])

    result.SetFixedParameters(fixedParameters)
    result.SetParameters(parameters)
    return result

if __name__ == "__main__":
    out_path = sys.argv[1]
    finename = sys.argv[2]
    finename_dest = sys.argv[3]
    transform = parameterMap_to_transform("{}/{}".format(out_path, finename))
    sitk.WriteTransform(transform, "{}/{}".format(out_path, finename_dest))

def getFlatLabel(mask: sitk.Image, labels: list[int]) -> sitk.Image:
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)

    for label in labels:
        result_data[data == label] = 1

    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)        
    return result

def rampFilterHistogram(image: sitk.Image, rampStart: float, rampEnd: float) -> sitk.Image:
    imageData = sitk.GetArrayFromImage(image)
    filter = np.logical_and(imageData > rampStart, imageData < rampEnd)
    rampWidth = rampEnd - rampStart
    imageData[filter] = (1/rampWidth) * (imageData[filter] - rampStart) * imageData[filter]
    filteredImage = sitk.GetImageFromArray(imageData)
    filteredImage.CopyInformation(image)
    return filteredImage

def elastic_registration(   fixed_image : sitk.Image,
                            moving_image : sitk.Image,
                            fixed_mask : Union[sitk.Image, None],
                            moving_mask : Union[sitk.Image, None],
                            name_parameterMap : str,
                            outputDir: str) -> sitk.Transform:
        labels = np.unique(sitk.GetArrayFromImage(fixed_mask))
        fixed_mask = getFlatLabel(fixed_mask, labels[1:])
        moving_mask = getFlatLabel(moving_mask, labels[1:])

        fixed_mask.CopyInformation(fixed_image)
        moving_mask.CopyInformation(moving_image)

        fixed_mask_dillated = sitk.BinaryDilate(fixed_mask, [5,5,5])
        moving_mask_dillated = sitk.BinaryDilate(moving_mask, [5,5,5])
        
        fixed_image = sitk.Mask(fixed_image, fixed_mask)
        moving_image = sitk.Mask(moving_image, moving_mask)


        minGradientMagnitude = 50
        fixed_image_gradient = rampFilterHistogram(sitk.VectorMagnitude(sitk.Gradient(fixed_image)), 0, minGradientMagnitude)
        moving_image_gradient = rampFilterHistogram(sitk.VectorMagnitude(sitk.Gradient(moving_image)), 0, minGradientMagnitude)

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_image_gradient)
        elastixImageFilter.AddFixedImage(fixed_image)
        elastixImageFilter.AddFixedImage(fixed_image)

        if fixed_mask is not None:
            elastixImageFilter.SetFixedMask(fixed_mask)
            elastixImageFilter.AddFixedMask(fixed_mask_dillated)
            elastixImageFilter.AddFixedMask(fixed_mask_dillated)


        elastixImageFilter.SetMovingImage(moving_image_gradient)
        elastixImageFilter.AddMovingImage(moving_image)
        elastixImageFilter.AddMovingImage(moving_image)
        
        if moving_mask is not None:
            elastixImageFilter.SetMovingMask(moving_mask)
            elastixImageFilter.AddMovingMask(moving_mask_dillated)
            elastixImageFilter.AddMovingMask(moving_mask_dillated)

        elastixImageFilter.SetParameterMap(sitk.ReadParameterFile("{}.txt".format(name_parameterMap)))
        elastixImageFilter.LogToConsoleOn()
        elastixImageFilter.SetOutputDirectory(outputDir)
        
        elastixImageFilter.Execute()
        
        transform = parameterMap_to_transform("{}TransformParameters".format(outputDir))
        
        return transform

def registration(   fixed_image : sitk.Image,
                    moving_image : sitk.Image,
                    fixed_mask : Union[sitk.Image, None],
                    moving_mask : Union[sitk.Image, None],
                    name_parameterMap : str,
                    outputDir: str) -> sitk.Transform:
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_image)
        if fixed_mask is not None:
            elastixImageFilter.SetFixedMask(fixed_mask)

        elastixImageFilter.SetMovingImage(moving_image)
        if moving_mask is not None:
            elastixImageFilter.SetMovingMask(moving_mask)

        elastixImageFilter.SetParameterMap(sitk.ReadParameterFile("{}.txt".format(name_parameterMap)))
        elastixImageFilter.LogToConsoleOn()
        elastixImageFilter.SetOutputDirectory(outputDir)
        elastixImageFilter.Execute()
        
        transform = parameterMap_to_transform("{}TransformParameters".format(outputDir))
        
        return transform

def registration_groupewise(images_1: sitk.Image, masks: sitk.Image, images_2: sitk.Image, name_parameterMap : str, output_dir: str):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(images_1)
    elastixImageFilter.SetMovingImage(images_1)
    elastixImageFilter.SetFixedMask(masks)
    elastixImageFilter.SetMovingMask(masks)
    
    if images_2 is not None:
        elastixImageFilter.AddFixedImage(images_2)
        elastixImageFilter.AddMovingImage(images_2)
        #elastixImageFilter.AddFixedImage(images_2)
        #elastixImageFilter.AddMovingImage(images_2)
    
    elastixImageFilter.SetParameterMap(sitk.ReadParameterFile("{}.txt".format(name_parameterMap)))
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.SetOutputDirectory(output_dir)
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.Execute()

    transforms = parameterMap_to_transform("{}TransformParameters".format(output_dir))
    return transforms