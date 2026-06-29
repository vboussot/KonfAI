import pytest
import torch

from konfai.data.transform import ResampleToResolution, ResampleToShape
from konfai.models.classification.convNeXt import LayerScaler
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError


def test_resample_to_resolution_transform_shape_raises_without_spacing() -> None:
    # The missing-Spacing guard used to build a TransformError and silently drop it.
    transform = ResampleToResolution(spacing=[1.0, 1.0, 1.0])
    with pytest.raises(TransformError, match="Spacing"):
        transform.transform_shape("CT", "case", [10, 10, 10], Attribute())


def test_resample_to_shape_transform_shape_needs_no_spacing_but_checks_dims() -> None:
    # ResampleToShape.transform_shape does not use Spacing; it just resolves the target shape.
    transform = ResampleToShape(shape=[10, 20, 30])
    assert [int(x) for x in transform.transform_shape("CT", "case", [5, 6, 7], Attribute())] == [10, 20, 30]
    with pytest.raises(TransformError, match="dimensions do not match"):
        transform.transform_shape("CT", "case", [5, 6], Attribute())


def test_layer_scaler_broadcasts_in_2d_and_3d() -> None:
    # gamma was hardcoded [C, 1, 1] (2D); it now sizes to `dim` so 3D broadcasts.
    scaler_3d = LayerScaler(init_value=1e-6, dimensions=4, dim=3)
    assert scaler_3d(torch.randn(2, 4, 5, 6, 7)).shape == (2, 4, 5, 6, 7)
    scaler_2d = LayerScaler(init_value=1e-6, dimensions=4, dim=2)
    assert scaler_2d(torch.randn(2, 4, 6, 7)).shape == (2, 4, 6, 7)
