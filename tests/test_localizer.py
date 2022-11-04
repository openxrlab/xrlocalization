import os

import pytest

from xrloc.localizer import Localizer
from xrloc.utils.image import read_image

map_path = 'tests/data/map'
image_dir = 'tests/data/query'


def test_init_localizer():
    loc = Localizer(map_path=map_path)
    assert loc.database is not None
    with pytest.raises(
            ValueError,
            match='Not found database under map: {}'.format(image_dir)):
        Localizer(map_path=image_dir)


def test_extract_features():
    loc = Localizer(map_path=map_path)
    image = read_image(os.path.join(image_dir, '000001.jpg'))
    keypoints, _ = loc.extract_features(image)
    assert keypoints.shape[0] == 2
    # assert descriptors.shape[0] == 512
