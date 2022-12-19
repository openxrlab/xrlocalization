import logging
import os

import numpy as np
import xrloc.map.covisible as covisible

# from solver import prior_guided_pnp
from xrprimer.data_structure import VectorDouble
from xrprimer.ops import prior_guided_pnp

from xrloc.features.extractor import Extractor
from xrloc.map.reconstruction import Reconstruction
from xrloc.matchers.matcher import Matcher
from xrloc.retrieval.image_database import ImageDatabase
from xrloc.retrieval.pairs_database import PairsDatabase
from xrloc.utils.miscs import head_logging, config_logging


class Localizer(object):
    """Hieracihcal Localization
    Args:
        config (dict): Configuration
    """
    default_config = {
        'mode': '2D3D',
        'local_feature': 'd2net',
        'global_feature': 'netvlad',
        'matcher': 'gam',
        'coarse': 'cluster',
        'retrieval_num': 20,
        'scene_size': 20,
        'max_inlier': 100,
        'max_scene_num': 2
    }

    def __init__(self, map_path, config=default_config):
        head_logging('XRLocalization')
        self.config = config
        database_path = os.path.join(map_path, 'database.bin')
        
        if os.path.exists(database_path):
            self.database = ImageDatabase(database_path)
            self.database.create()
        else:
            pairs = [name for name in os.listdir(map_path) 
                     if name.startswith('pairs-query')]
            if len(pairs) == 0:
                raise ValueError(
                    'Not found database under map: {}'.format(map_path))
            else:
                self.pairs = PairsDatabase(os.path.join(map_path, pairs[0]))

        self.reconstruction = Reconstruction(map_path)
        
        if hasattr(self, 'database'):
            self.gextractor = Extractor(self.config['global_feature'])
        self.lextractor = Extractor(self.config['local_feature'])

        self.matcher = Matcher(self.config['matcher'])

        config_logging(self.config)

        if self.config['mode'] == '2D2D' and self.config['matcher'] == 'gam':
            raise ValueError('Loc mode {} is not compatible with matcher {}'.format(
                self.config['mode'], self.config['matcher']))

        head_logging('Init Success')

    def coarse_localize(self, image_ids):
        """Coarse localization phase."""
        if self.config['coarse'] == 'cluster':
            scenes = covisible.covisible_clustering(image_ids,
                                                    self.reconstruction)
        elif self.config['coarse'] == 'sr':
            scenes = covisible.scene_retrieval(image_ids, self.reconstruction,
                                               self.config['scene_size'])
        elif self.config['coarse'] == 'none':
            scenes = [image_ids]
        else:
            raise ValueError('Not support coarse loc: {}'.format(
                self.config['coarse']))
        return scenes

    def geo_localize(self, data):
        """Perform geo localization
        Args:
            data (np.array or str): image data or image name
        Returns:
            list(int): image ids
        """
        if hasattr(self, 'database'):
            image_feature = self.gextractor.extract(data)
            image_ids = self.database.retrieve(image_feature,
                                            self.config['retrieval_num'])
            return image_ids
        elif hasattr(self, 'pairs') and isinstance(data, str):
            image_names = self.pairs.image_retrieve(data, self.config['retrieval_num'])
            image_ids = [self.reconstruction.name_to_id(name) for name in image_names]
            return np.array([id for id in image_ids if id != -1])

    def feature_match_2d3d(self, query_points, query_point_descriptors,
                           train_points, train_point_descriptors, width, height):
        """Feature matching phase."""
        query_feat = {
            'shape': np.array([height, width]),
            'points': query_points,
            'descs': query_point_descriptors,
        }
        train_feat = {
            'points': train_points,
            'descs': train_point_descriptors,
        }
        pred = self.matcher.match(query_feat, train_feat)
        return pred['matches'], pred['scores']

    def establish_correspondences_2d2d(self, query_feat, image_ids):
        '''Establish 2D-3D correspondences depend on 2D2D matching
        '''
        logging.info('Scene size: {0}'.format(len(image_ids)))
        match_indices = np.ones(len(query_feat['points']), dtype=int)*-1
        match_priors = np.zeros(len(query_feat['points']))
        for image_id in image_ids:
            ref_image = self.reconstruction.image_at(image_id)
            ref_feat = {
                'points': ref_image.xys,
                'descs': self.reconstruction.point3d_features(ref_image.point3D_ids),
                'scores': np.ones(len(ref_image.xys)),
                'shape': np.array([600, 600]) # TODO
            }
            pred = self.matcher.match(query_feat, ref_feat)
            matches, scores = pred['matches'], pred['scores']
            
            reserve_matches = matches[:, scores > match_priors[matches[0]]]
            reserve_scores = scores[scores > match_priors[matches[0]]]
            if len(reserve_scores) > 0:
                match_indices[reserve_matches[0]] = ref_image.point3D_ids[reserve_matches[1]]
                match_priors[reserve_matches[0]] = reserve_scores

            if len(match_priors[match_indices != -1]) > 400:
                break

        point3d_ids = match_indices[match_indices != -1]
        points3d = self.reconstruction.point3d_coordinates(
            point3d_ids)
        points2d = query_feat['points'][match_indices != -1]
        priors = match_priors[match_indices != -1]
        logging.info('Match number: {0}'.format(len(priors)))
        return points2d, points3d, priors


    def establish_correspondences_2d3d(self, feat, image_ids):
        '''Establish 2D-3D correspondences depend on 2D3D matching
        '''
        point3d_ids = self.reconstruction.visible_points(image_ids)
        point3ds = self.reconstruction.point3d_coordinates(
            point3d_ids)
        point3d_descs = self.reconstruction.point3d_features(
            point3d_ids)
        logging.info('3d points size: {0}'.format(
            point3d_descs.shape[1]))

        # Matching
        matches, priors = self.feature_match_2d3d(feat['points'],
                                                  feat['descs'],
                                                  point3ds,
                                                  point3d_descs,
                                                  feat['shape'][1], # width
                                                  feat['shape'][0])
        logging.info('Match number: {0}'.format(matches.shape[1]))
        points2d = feat['points'][matches[0]]
        points3d = point3ds[matches[1]]
        return points2d, points3d, priors

    def prior_guided_pose_estimation(self, point2Ds, point3Ds, priors, camera):
        """Pose estimation phase."""
        point2Ds = point2Ds.transpose().astype('float32').copy()
        point3Ds = point3Ds.transpose().astype('float32').copy()
        priors = priors.astype('float32').copy()
        params = VectorDouble(camera[3])
        camera_config = {'model_name': camera[0], 'params': params}
        ransac_config = {
            'error_thres': 12,
            'inlier_ratio': 0.01,
            'confidence': 0.9999,
            'max_iter': 10000,
            'local_optimal': True
        }
        return prior_guided_pnp(point2Ds, point3Ds, priors, camera_config,
                                ransac_config)

    def refine_localize(self, image, camera, ref_image_ids):
        """Perform localization
        Args:
            image (np.array): RGB & WHC
            camera (tuple): (model, width, height, params)
        """
        feat = self.lextractor.extract(image)
        logging.info('Local feature number: {0}'.format(
            feat['points'].shape[0]))

        scenes = self.coarse_localize(ref_image_ids)
        logging.info('Coarse location number: {0}'.format(len(scenes)))

        best_ret = {
            'ninlier': 0, 'qvec': np.array([1, 0, 0, 0]),
            'tvec': np.array([0, 0, 0]), 'mask': None
        }
        for i, image_ids in enumerate(scenes[:self.config['max_scene_num']]):
            # Establish 2D-3D correspondences
            if self.config['mode'] == '2D3D':
                points2d, points3d, priors = self.establish_correspondences_2d3d(
                    feat, image_ids)
            elif self.config['mode'] == '2D2D':
                points2d, points3d, priors = self.establish_correspondences_2d2d(
                    feat, image_ids)

            if len(priors) < 3: continue

            # Pose estimation
            ret = self.prior_guided_pose_estimation(points2d, points3d, priors,
                                                    camera)
            logging.info('Inlier number: {0}'.format(ret['ninlier']))

            # Return
            if ret['ninlier'] > best_ret['ninlier']:
                best_ret = ret
            if best_ret['ninlier'] > self.config['max_inlier']:
                break
        return best_ret