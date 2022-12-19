import numpy as np

import pytest

from xrloc.matchers.matcher import Matcher
from xrloc.matchers.nearest_neighbor import NearestNeighborMatcher


def test_init_nn():
    matcher = Matcher('nn')
    assert 'ratio' in matcher.config.keys()
    assert matcher.config['ratio'] == False
    assert 'dis_thres' in matcher.config.keys()
    assert matcher.config['dis_thres'] == False
    assert 'cross_check' in matcher.config.keys()
    assert matcher.config['cross_check'] == False
    assert isinstance(matcher.model, NearestNeighborMatcher)


def test_init_nn_cross():
    matcher = Matcher('nn+cross')
    assert 'ratio' in matcher.config.keys()
    assert matcher.config['ratio'] == False
    assert 'dis_thres' in matcher.config.keys()
    assert matcher.config['dis_thres'] == False
    assert 'cross_check' in matcher.config.keys()
    assert matcher.config['cross_check']
    assert isinstance(matcher.model, NearestNeighborMatcher)


def test_init_nn_ratio_cross():
    matcher = Matcher('nn+ratio+cross')
    assert 'ratio' in matcher.config.keys()
    assert matcher.config['ratio'] == 0.85
    assert 'dis_thres' in matcher.config.keys()
    assert matcher.config['dis_thres'] == False
    assert 'cross_check' in matcher.config.keys()
    assert matcher.config['cross_check']
    assert isinstance(matcher.model, NearestNeighborMatcher)


def test_init_nn_ratio_distance_cross():
    matcher = Matcher('nn+ratio+distance+cross')
    assert 'ratio' in matcher.config.keys()
    assert matcher.config['ratio'] == 0.85
    assert 'dis_thres' in matcher.config.keys()
    assert matcher.config['dis_thres'] == 0.9
    assert 'cross_check' in matcher.config.keys()
    assert matcher.config['cross_check']
    assert isinstance(matcher.model, NearestNeighborMatcher)


def test_init_invalid_mathcer():
    with pytest.raises(ValueError, match='Not support the extractor test'):
        Matcher('test')


# def test_init_gam_matcher():
# matcher = Matcher('gam')
# assert isinstance(matcher.model, GeometryAidedMatcher)


def test_matching():
    data = {
        '2d_descriptors': np.ones([256, 500]),
        '3d_descriptors': np.ones([256, 1000]),
    }
    matcher = Matcher('nn')
    matches, sims = matcher.match(data)
    assert matches.shape == (2, 500)
    assert sims.shape == (500, )
    matcher = Matcher('nn+cross')
    matches, sims = matcher.match(data)
    assert matches.shape == (2, 1)
    assert sims.shape == (1, )
    assert sims[0] == 256
    matcher = Matcher('nn+ratio+cross')
    matches, sims = matcher.match(data)
    assert matches.shape == (2, 0)
    assert sims.shape == (0, )
    matcher = Matcher('nn+ratio+distance+cross')
    matches, sims = matcher.match(data)
    assert matches.shape == (2, 0)
    assert sims.shape == (0, )
