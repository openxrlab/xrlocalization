import numpy as np

import torch
import pytest

from xrloc.retrieval.image_database import ImageDatabase
from xrloc.features.extractor import Extractor


def test_init():
    idb = ImageDatabase()
    assert idb.feature_data == []
    assert idb.index_to_image_id == []
    assert idb.names == []
    assert idb.is_built == False
    assert idb.size == 0
    assert idb.extractor is None
    if torch.cuda.is_available():
        assert idb.device == 'cuda'
    else:
        assert idb.device == 'cpu'


def test_add_feature():
    idb = ImageDatabase()
    feature, image_id = np.ones(4096), 1
    idb.add_feature(feature, image_id=image_id, image_name='')
    assert idb.size == 1
    assert idb.index_to_image_id == [1]
    assert idb.names == ['']


def test_set_extractor():
    idb = ImageDatabase()
    extractor = Extractor('netvlad')
    idb.set_image_extractor(extractor)
    assert idb.extractor == extractor


def test_create():
    idb = ImageDatabase()
    with pytest.raises(ValueError, match='Image database is empty'):
        idb.create()
    idb.is_built = True
    with pytest.raises(ValueError, match='Image database have been built'):
        idb.create()
    idb.is_built = False
    feature, image_id = np.ones(4096), 1
    idb.add_feature(feature, image_id)
    idb.create()
    assert idb.is_built == True


def test_retrieve():
    idb = ImageDatabase()
    with pytest.raises(ValueError, match='Image database have not been built'):
        idb.image_retrieve(np.ones(4096), 1, '')
    feature, image_id = np.ones(4096), 1
    idb.add_feature(feature, image_id)
    idb.create()
    assert idb.retrieve(np.ones(4096), 1) == np.array([1])
    assert idb.retrieve(np.ones(4096), 1, ret_name=True) == ['']


def test_image_retrieve():
    width, height, channel = 640, 480, 3
    image = np.ones([width, height, channel])
    idb = ImageDatabase('tests/data/map/database.bin')
    idb.create()
    with pytest.raises(
            ValueError,
            match='Image global feature extractor have not been set'):
        idb.image_retrieve(image, 1)
    idb.set_image_extractor(Extractor('netvlad'))
    idb.is_built = False
    with pytest.raises(ValueError, match='Image database have not been built'):
        idb.image_retrieve(image, 1)
    idb.is_built = True
    image_ids = idb.image_retrieve(image, 0)
    assert len(image_ids) == 0
    image_ids = idb.image_retrieve(image, -1)
    assert len(image_ids) == idb.size
    image_ids = idb.image_retrieve(image, 10)
    assert len(image_ids) == 10


def test_save_binary():
    idb = ImageDatabase('tests/data/map/database.bin')
    idb.save_binary('tests/data/map/database_save_test.bin')
    idb.create()
    with pytest.raises(ValueError, match='Image database have been built'):
        idb.save_binary('tests/data/map/database_save_test.bin')


def test_load_binary():
    idb = ImageDatabase('tests/data/map/database.bin')
    assert len(idb.feature_data) == 231
    assert len(idb.index_to_image_id) == 231
    assert idb.size == 231
