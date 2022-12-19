import numpy as np

import torch
import pytest

from xrloc.matchers.bmnet import PointCN, hungarian_pooling, BipartiteMatchingNet


def test_pointcn_init():
    in_channels, out_channels = 32, 64
    model = PointCN(in_channels, out_channels, False)
    assert not model.attention
    assert not hasattr(model, 'att')

    model = PointCN(in_channels, out_channels, True)
    assert model.attention
    assert hasattr(model, 'att')


def test_pointcn_forward():
    in_channels, out_channels, N = 32, 64, 128
    model = PointCN(in_channels, out_channels, False)
    input = torch.ones([1, in_channels, N])
    output = model(input)
    assert output.shape == (1, out_channels, N)

    model = PointCN(in_channels, out_channels, True)
    input = torch.ones([1, in_channels, N])
    output = model(input)
    assert output.shape == (1, out_channels, N)


def test_hungarian_pooling():
    N, cols, rows = 100, 100, 100
    weights = torch.ones([1, N])
    edges = torch.from_numpy(np.array([np.arange(N), np.arange(N)]))
    with pytest.raises(ValueError, match='Input weights must be 1 dims!'):
        hungarian_pooling(weights, edges, cols, rows)
    weights = torch.ones([N])
    with pytest.raises(ValueError,
                       match='Rows and cols must be larger than 0!'):
        hungarian_pooling(weights, edges, -1, -1)
    assignment = hungarian_pooling(weights, edges, cols, rows)
    assert assignment.size(0) == N


def test_bipartite_graph_matching_net_init():
    model = BipartiteMatchingNet()
    assert model.config == {
        'channels0': [2, 32, 64] + [128] * 5,
        'channels1': [3, 32, 64] + [128] * 5,
        'channels2': [256] * 18,
    }
    assert hasattr(model, 'unet')
    assert hasattr(model, 'vnet')
    assert hasattr(model, 'enet')
    assert hasattr(model, 'hungarian_pooling')
