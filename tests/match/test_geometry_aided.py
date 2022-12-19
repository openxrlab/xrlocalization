import numpy as np

import torch
import pytest

from xrloc.matchers.geometry_aided import GeometryAidedMatcher


def test_geometry_aided_matcher_init():
    matcher = GeometryAidedMatcher()
    assert matcher.config == {
        'k': 3,
        'ratio': 0.7,
        'dis_thres': 0.9,
        'geo_prior': 0
    }
    config = {'k': 4, 'ratio': 0.9, 'dis_thres': 0.8, 'geo_prior': 0.1}
    matcher = GeometryAidedMatcher(config)
    assert matcher.config == config
    if torch.cuda.is_available():
        assert matcher.device == 'cuda'
    else:
        assert matcher.device == 'cpu'


def test_match():
    model = GeometryAidedMatcher()
    num_point2ds, num_point3ds = 100, 500
    data = {
        '2d_points': np.ones([2, num_point2ds]),
        '2d_descriptors': np.ones([256, num_point2ds]),
        '3d_points': np.ones([3, num_point3ds]),
        '3d_descriptors': np.ones([256, num_point3ds]),
        'width': 640,
        'height': 480,
    }
    del data['2d_points']
    with pytest.raises(ValueError, match='2d_points not exist in input'):
        model(data)
    data['2d_points'] = np.ones([2, num_point2ds])
    del data['2d_descriptors']
    with pytest.raises(ValueError, match='2d_descriptors not exist in input'):
        model(data)
    data['2d_descriptors'] = np.ones([256, num_point2ds])
    data['3d_points'] = np.ones([3, model.config['k'] - 1])
    data['3d_descriptors'] = np.ones([256, model.config['k'] - 1])
    matches, sims = model(data)
    assert matches.shape == (2, 0)
    assert sims.shape == (0, )
    data['3d_points'] = np.ones([3, num_point3ds])
    data['3d_descriptors'] = np.ones([256, num_point3ds])
    assert matches.shape == (2, 0)
    assert sims.shape == (0, )


def test_knn_ratio_match():
    num_point2ds, num_point3ds = 100, 500
    descriptors1 = np.ones([256, num_point2ds])
    descriptors2 = np.ones([128, num_point3ds])
    with pytest.raises(Exception):
        GeometryAidedMatcher.knn_ratio_match(descriptors1, descriptors2)

    descriptors2 = np.ones([256, num_point3ds])
    with pytest.raises(Exception):
        GeometryAidedMatcher.knn_ratio_match(descriptors1, descriptors2, k=0)

    with pytest.raises(Exception):
        GeometryAidedMatcher.knn_ratio_match(descriptors1, descriptors2, k=501)

    descriptors1 /= 16.0
    descriptors2 /= 16.0

    matches, sims = GeometryAidedMatcher.knn_ratio_match(descriptors1,
                                                         descriptors2,
                                                         k=1,
                                                         ratio=0,
                                                         thres=2)
    assert matches.shape[1] == num_point2ds

    matches, sims = GeometryAidedMatcher.knn_ratio_match(descriptors1,
                                                         descriptors2,
                                                         k=2,
                                                         ratio=-1.,
                                                         thres=2)
    assert matches.shape[1] == num_point2ds * 2
