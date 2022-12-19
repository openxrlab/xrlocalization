import pytest
import numpy as np

from xrloc.matchers.nearest_neighbor import NearestNeighborMatcher


def test_init():
    nnm = NearestNeighborMatcher()
    assert 'ratio' in nnm.config.keys()
    assert nnm.config['ratio'] == 0.9
    assert 'dis_thres' in nnm.config.keys()
    assert nnm.config['dis_thres'] == 0.9
    assert 'cross_check' in nnm.config.keys()
    assert nnm.config['cross_check'] == True

    config = {'ratio': 0.8, 'dis_thres': 0.8, 'cross_check': False}
    nnm = NearestNeighborMatcher(config)
    assert 'ratio' in nnm.config.keys()
    assert nnm.config['ratio'] == 0.8
    assert 'dis_thres' in nnm.config.keys()
    assert nnm.config['dis_thres'] == 0.8
    assert 'cross_check' in nnm.config.keys()
    assert nnm.config['cross_check'] == False


def test_nearest_neigbbor_matcher():
    nnm = NearestNeighborMatcher()
    data = {}
    with pytest.raises(ValueError, match='2d_descriptors not exist in input'):
        nnm(data)
    data = {
        '3d_descriptors': np.ones([256, 1000]),
    }
    with pytest.raises(ValueError, match='2d_descriptors not exist in input'):
        nnm(data)
    data = {
        '2d_descriptors': np.ones([256, 500]),
    }
    with pytest.raises(ValueError, match='3d_descriptors not exist in input'):
        nnm(data)
