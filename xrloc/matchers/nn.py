import torch
import numpy as np


def nearest_neighbor(descriptors1,
                     descriptors2,
                     ratio_thres=None,
                     dis_thres=None,
                     cross_check=False):
    """Find the nearest neighbor
    Args:
        descriptors1 (np.array): Query descriptors
        descriptors2 (np.array): Train descriptors
        ratio_thres (float | None): If not none, perform ratio test
        dis_thres (float | None): If not none, perform distance filter
        cross_check (bool): If true, perform cross check
    """
    assert descriptors1.shape[0] == descriptors2.shape[0]
    if descriptors1.shape[1] == 0 or descriptors2.shape[1] == 0:
        return np.zeros((3, 0))
    descriptors1, descriptors2 = descriptors1.cpu(), descriptors2.cpu()
    similarity = torch.matmul(descriptors1.t(), descriptors2)
    sims, ind1 = similarity.topk(2 if ratio_thres else 1, dim=1, largest=True)
    dists = torch.sqrt(2 * (1 - sims.clamp(-1, 1)))

    mask = torch.ones(dists.shape[0], dtype=torch.bool)
    if ratio_thres:
        mask = mask & (dists[:, 0] / dists[:, 1] < ratio_thres)
    sims, dists, ind1 = sims[:, 0], dists[:, 0], ind1[:, 0]
    if dis_thres:
        mask = mask & (dists < dis_thres)
    if cross_check:
        _, ind0 = similarity.topk(1, dim=0, largest=True)
        mask = mask & (torch.arange(ind1.shape[0]) == ind0[0, ind1])

    ind0 = torch.arange(ind1.shape[0])[mask].to(torch.long)
    ind1, sims = ind1[mask].to(torch.long), sims[mask].to(torch.float)
    matches = torch.stack([ind0, ind1], dim=1).permute([1, 0])
    return matches, sims


class NN:
    """Find the nearest neighbor.

    Args:
        config (dict): Refer to default_config
    """
    default_config = {'ratio': 0.9, 'dis_thres': 0.9, 'cross_check': True}

    def __init__(self, config=default_config):
        self.config = config

    def __call__(self, data):
        required_keys = ['query_descs', 'train_descs']
        for key in required_keys:
            if key not in data:
                raise ValueError(key + ' not exist in input')
        matches, scores = nearest_neighbor(data['query_descs'], data['train_descs'],
                                self.config['ratio'], self.config['dis_thres'],
                                self.config['cross_check'])
        return {
            'matches': matches,
            'scores': scores
        }
