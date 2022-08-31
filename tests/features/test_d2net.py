import numpy as np

import torch

from xrloc.features.d2net import D2Net


def test_init():
    net = D2Net()
    assert net.config == {
        'model_name': 'd2_tf.pth',
        'multiscale': False,
    }
    config = {
        'model_name': 'd2_tf.pth',
        'multiscale': True,
    }
    net = D2Net(config)
    assert net.config == config


def test_forward():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = D2Net().to(device)
    width, height = 640, 480
    image = np.ones([3, width, height], dtype=np.float32)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    # data = net(image)
    # assert 'keypoints' in data
    # assert 'scores' in data
    # assert 'descriptors' in data
    # config = {
    #     'model_name': 'd2_tf.pth',
    #     'multiscale': True,
    # }
    # net = D2Net(config).to(device)
    # width, height = 640, 480
    # image = np.ones([3, width, height], dtype=np.float32)
    # image = torch.from_numpy(image).unsqueeze(0).to(device)
    # data = net(image)
    # assert 'keypoints' in data
    # assert 'scores' in data
    # assert 'descriptors' in data
