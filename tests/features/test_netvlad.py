import numpy as np

import torch

from xrloc.features.netvlad import NetVLAD


def test_init():
    net = NetVLAD()
    assert hasattr(net, 'whiten')
    config = {'model_name': 'VGG16-NetVLAD-Pitts30K', 'whiten': False}
    net = NetVLAD(config)
    assert not hasattr(net, 'whiten')


def test_forward():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = NetVLAD().to(device)
    width, height = 640, 480
    image = np.ones([3, width, height], dtype=np.float32)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    data = net(image)
    assert 'global_descriptor' in data
    assert data['global_descriptor'].shape[1] == 4096
    config = {'model_name': 'VGG16-NetVLAD-Pitts30K', 'whiten': False}
    net = NetVLAD(config).to(device)
    data = net(image)
    assert 'global_descriptor' in data
    assert data['global_descriptor'].shape[1] == 32768
