# Warp https://github.com/mihaidusmanu/d2-net
import os
import sys
import torch
import torch.nn as nn

from xrloc.utils.miscs import get_parent_dir, download_model

sys.path.append(get_parent_dir(__file__) + '/../3rdparty/d2net')
from lib.model_test import D2Net as D2N
from lib.pyramid import process_multiscale


class D2Net(nn.Module):
    default_config = {
        'model_name': 'd2_tf.pth',
        'multiscale': False,
    }

    def __init__(self, config=default_config):
        super().__init__()
        self.config = {**self.default_config, **config}

        model_dir = get_parent_dir(__file__) + '/../models/'
        url = os.path.join('https://dsmn.ml/files/d2-net',
                           self.config['model_name'])

        download_model(url, model_dir, self.config['model_name'])
        model_path = os.path.join(model_dir, self.config['model_name'])

        self.model = D2N(model_file=model_path, use_relu=True, use_cuda=False)

    def forward(self, image):
        # https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/extractors/d2net.py
        image = torch.flip(image, dims=[1])  # RBG -> BGR
        norm = image.new_tensor([103.939, 116.779, 123.68])
        image = (image * 255 - norm.view(1, 3, 1, 1))
        if self.config['multiscale']:
            keypoints, scores, descriptors = process_multiscale(
                image, self.model)
        else:
            keypoints, scores, descriptors = process_multiscale(image,
                                                                self.model,
                                                                scales=[1])
        keypoints = keypoints[:, [1, 0]]  # (x, y) and remove the scale

        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors.T)[None],
        }
