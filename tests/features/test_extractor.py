import numpy as np
import torch

import pytest

from xrloc.features.extractor import Extractor
from xrloc.features.netvlad import NetVLAD
from xrloc.features.d2net import D2Net


def test_init():
    with pytest.raises(ValueError, match='Not support the extractor test'):
        Extractor('test')
    model = Extractor('d2net')
    assert isinstance(model.extractor, D2Net)
    if torch.cuda.is_available():
        assert model.device == 'cuda'
    else:
        assert model.device == 'cpu'
    model = Extractor('netvlad')
    assert isinstance(model.extractor, NetVLAD)


def test_extract():
    width, height, channel = 640, 480, 3
    image = np.ones([width, height, channel])
    model = Extractor('netvlad')
    data = model.extract(image)
    assert data.shape[0] == 4096
    image = np.ones([width, height, 3])
    model = Extractor('d2net')
    data = model.extract(image)
    assert 'keypoints' in data


def test_back_to_origin_size():
    keypoints = np.array([1, 2, 3, 4]).reshape(2, 2)
    factor = 0.5
    new_kpt = Extractor.back_to_origin_size(keypoints, factor)
    assert (new_kpt == np.array([[2.5, 4.5], [6.5, 8.5]])).all()


def test_make_model():
    assert Extractor.make_model('netvlad') == NetVLAD
    assert Extractor.make_model('d2net') == D2Net
