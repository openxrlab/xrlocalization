import os
import torch

import torch.nn as nn

from xrloc.utils.miscs import download_model, get_parent_dir
from scipy.optimize import linear_sum_assignment


class PointCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super(PointCN, self).__init__()
        self.shot_cut = None
        self.attention = attention
        if out_channels != in_channels:
            self.shot_cut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.encoder = nn.Sequential(
            nn.InstanceNorm1d(in_channels, eps=1e-3, affine=True),
            nn.BatchNorm1d(in_channels, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(out_channels, eps=1e-3, affine=True),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(out_channels, eps=1e-3, affine=True),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            # nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False))
        if self.attention:
            self.att = nn.Conv1d(out_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.encoder(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        if self.attention:
            out = out * torch.sigmoid(self.att(out))
        return out


class EncodingBlock(nn.Module):
    def __init__(self, channels: list, attention=False):
        super(EncodingBlock, self).__init__()
        self.num_layers = len(channels)
        for i in range(1, self.num_layers):
            setattr(self, 'block_%d' % i,
                    PointCN(channels[i - 1], channels[i], attention))

    def forward(self, input):
        out = input
        for i in range(1, self.num_layers):
            out = getattr(self, 'block_%d' % i)(out)
        return out


class ConcatenationLayer(torch.nn.Module):
    def __init__(self):
        super(ConcatenationLayer, self).__init__()

    def forward(self, in0, in1, matches):
        num_matches = matches.shape[1]
        num_channels0, num_channels1 = in0.shape[1], in1.shape[1]
        edges_feature_0 = torch.zeros(1, num_channels0, num_matches).cuda()
        edges_feature_1 = torch.zeros(1, num_channels1, num_matches).cuda()
        edges_feature_0[0] = torch.index_select(in0[0],
                                                dim=1,
                                                index=matches[0])
        edges_feature_1[0] = torch.index_select(in1[0],
                                                dim=1,
                                                index=matches[1])
        edge_feature = torch.cat((edges_feature_0, edges_feature_1), dim=1)
        return edge_feature


class HungarianPooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, matches, rows, cols):
        probs = torch.sigmoid(weights)
        probs, edges = probs.cpu(), matches.cpu()
        cost_matrix = torch.zeros(rows, cols)
        cost_matrix[edges[0], edges[1]] = probs
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        htb = torch.zeros([rows, cols], dtype=torch.int64)
        htb[row_ind, col_ind] = 1
        assignments = htb[edges[0], edges[1]]
        assignments = assignments.cuda()
        ctx.save_for_backward(assignments)
        return assignments


def hungarian_pooling(weights, edges, rows, cols):
    if len(weights.shape) != 1:
        raise ValueError('Input weights must be 1 dims!')
    if rows <= 0 or cols <= 0:
        raise ValueError('Rows and cols must be larger than 0!')
    return HungarianPooling.apply(weights, edges, rows, cols)


def normalize_point2Ds(x, w, h):
    one = x.new_tensor(1)
    size = torch.stack([one * w, one * h]).unsqueeze(1)
    center = size / 2
    scale = size.max() * 1.2
    return (x - center) / scale


def normalize_point3Ds(x):
    scale = (x.max() - x.min()) * 1.2
    return (x - torch.mean(x)) / scale


class BipartiteMatchingNet(nn.Module):
    default_config = {
        'channels0': [2, 32, 64] + [128] * 5,
        'channels1': [3, 32, 64] + [128] * 5,
        'channels2': [256] * 18,
    }

    def __init__(self, config=default_config):
        super(BipartiteMatchingNet, self).__init__()
        self.config = {**self.default_config, **config}
        self.unet = EncodingBlock(channels=self.config['channels0'],
                                  attention=False)
        self.vnet = EncodingBlock(channels=self.config['channels1'],
                                  attention=False)
        self.enet = EncodingBlock(channels=self.config['channels2'],
                                  attention=False)
        self.cat = ConcatenationLayer()
        self.hungarian_pooling = hungarian_pooling\
            # if self.config[

        # 'hpooling'] else None
        self.conv = nn.Conv1d(self.config['channels2'][-1], 1, kernel_size=1)

        url = 'https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrlocalization/weights/bmnet.pth'
        model_dir = get_parent_dir(__file__) + '/../models/'

        model_name = 'bmnet.pth'
        download_model(url, model_dir, model_name)
        model_path = os.path.join(model_dir, model_name)

        self.load_state_dict(
            torch.load(model_path, map_location='cpu')['state_dict'])

    def forward(self, data):
        ps0 = normalize_point2Ds(data['PA'], data['W'], data['H']).unsqueeze(0)
        ps1 = normalize_point3Ds(data['PB']).unsqueeze(0)
        matches = data['MA']

        ufeature = self.unet(ps0)
        vfeature = self.vnet(ps1)
        efeature = self.cat(ufeature, vfeature, matches)

        efeature = self.enet(efeature)

        weights = self.conv(efeature).squeeze(1).squeeze(0)

        M, N = ps0.shape[2], ps1.shape[2]
        assignments = self.hungarian_pooling(weights, matches, M, N)

        return weights, assignments
