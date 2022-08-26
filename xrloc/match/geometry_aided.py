import logging
import torch

import numpy as np

from xrloc.match.bmnet import BipartiteMatchingNet


class GeometryAidedMatcher(object):
    default_config = {'k': 3, 'ratio': 0.7, 'dis_thres': 0.9, 'geo_prior': 0}

    def __init__(self, config=default_config):
        self.config = {**self.default_config, **config}
        self.model = BipartiteMatchingNet()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval().to(self.device)

    def __call__(self, data: dict):
        required_keys = [
            '2d_points', '2d_descriptors', '3d_points', '3d_descriptors',
            'width', 'height'
        ]

        for key in required_keys:
            if key not in data:
                raise ValueError(key + " not exist in input")

        if data['3d_descriptors'].shape[1] < self.config['k']:
            return np.zeros((2, 0), dtype=int), np.zeros(0, dtype=np.float32)

        matches, sims = self.knn_ratio_match(data['2d_descriptors'],
                                             data['3d_descriptors'],
                                             self.config['k'],
                                             self.config['ratio'],
                                             self.config['dis_thres'])
        logging.info('k:{0}, ratio:{1}, thres:{2}'.format(
            self.config['k'], self.config['ratio'], self.config['dis_thres']))
        logging.info('Knn ratio match size: {0}'.format(matches.shape[1]))

        p2ds, p3ds, edges = self.generate_bipartite_graph(
            data['2d_points'], data['3d_points'], matches)
        if edges.shape[1] < 2 or p2ds.shape[1] < 2 or p3ds.shape[1] < 2:
            return np.zeros((2, 0), dtype=int), np.zeros(0, dtype=np.float32)

        priors, mask = self.find_maximum_matching(p2ds, p3ds, edges,
                                                  data['width'],
                                                  data['height'])
        matches = matches[:, (mask == 1).__and__(
            priors > self.config['geo_prior'])]
        priors = priors[(mask == 1).__and__(priors > self.config['geo_prior'])]
        return matches, priors

    @staticmethod
    def generate_bipartite_graph(point2ds, point3ds, matches):
        point2d_ids, point3d_ids = list(set(matches[0])), list(set(matches[1]))
        point2ds, point3ds = point2ds[:, point2d_ids], point3ds[:, point3d_ids]
        point2d_ids_inv = dict(
            zip(point2d_ids, [i for i in range(len(point2d_ids))]))
        edge_1 = np.array(
            [point2d_ids_inv[point2d_id] for point2d_id in matches[0]])
        point3d_ids_inv = dict(
            zip(point3d_ids, [i for i in range(len(point3d_ids))]))
        edge_2 = np.array(
            [point3d_ids_inv[point3d_id] for point3d_id in matches[1]])
        edges = np.array([edge_1, edge_2])
        return point2ds, point3ds, edges

    @torch.no_grad()
    def find_maximum_matching(self, point2ds, point3ds, edges, width, height):
        data = {
            'W': width,
            'H': height,
            'PA': torch.from_numpy(point2ds.copy()).to(torch.float32).cuda(),
            'PB': torch.from_numpy(point3ds.copy()).to(torch.float32).cuda(),
            'MA': torch.from_numpy(edges).to(torch.long).cuda()
        }
        weights, masks = self.model(data)
        priors = torch.sigmoid(weights) + 1e-6
        return priors.detach().cpu().numpy(), masks.detach().cpu().numpy()

    @staticmethod
    def knn_ratio_match(descriptors1, descriptors2, k=4, ratio=0.9, thres=0.7):
        assert descriptors1.shape[0] == descriptors2.shape[0]
        assert 1 <= k < descriptors2.shape[1]
        descriptors1 = torch.from_numpy(descriptors1).to(torch.float32).cuda()
        descriptors2 = torch.from_numpy(descriptors2).to(torch.float32).cuda()
        scores = torch.einsum('dm,dn->mn', descriptors1, descriptors2)
        sims, inds1 = scores.topk(k, dim=1, largest=True)
        dists = torch.sqrt(2 * (1 - sims.clamp(-1, 1)))

        # Filter by ratio and thres
        mask = torch.ones((dists.shape[0], k), dtype=torch.bool).cuda()
        mask[:, 0] = (dists[:, 0] < 1)
        for i in range(1, k):
            mask[:, i] = (
                (dists[:, 0] / (dists[:, i]+1e-6)) > ratio) & (dists[:, i] < thres)
        mask = mask.reshape(-1)

        # Compose match
        inds0 = torch.arange(0, inds1.shape[0]).unsqueeze(1).expand(
            [inds1.shape[0], k]).cuda()
        inds0, inds1 = inds0.reshape(-1), inds1.reshape(-1)
        matches = torch.stack([inds0, inds1],
                              dim=1).to(torch.long).permute([1, 0])
        sims = sims.reshape(-1)
        return matches[:, mask].cpu().numpy(), sims[mask].cpu().numpy()
