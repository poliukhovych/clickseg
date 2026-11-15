import numpy as np
import pytest
from PIL import Image

from clickseg.images import overlay_mask, read_rgb, validate_point


def test_xy_order_for_rectangular_image():
    image = np.zeros((20, 80, 3), dtype=np.uint8)
    validate_point(image, 70, 10)
    with pytest.raises(ValueError):
        validate_point(image, 10, 70)


@pytest.mark.parametrize("point", [(-1, 2), (80, 2), (2, 20), (float("nan"), 1)])
def test_out_of_bounds_and_nonfinite_clicks(point):
    with pytest.raises(ValueError):
        validate_point(np.zeros((20, 80, 3), dtype=np.uint8), *point)


def test_exif_rotation_defines_coordinate_space(tmp_path):
    image = Image.new("RGB", (80, 20))
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    assert read_rgb(path).shape == (80, 20, 3)


def test_overlay_does_not_contaminate_model_input():
    image = np.zeros((20, 80, 3), dtype=np.uint8)
    original = image.copy()
    mask = np.ones((20, 80), dtype=bool)
    overlay_mask(image, mask, (70, 10))
    np.testing.assert_array_equal(image, original)
