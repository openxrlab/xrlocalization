import logging
import os

import numpy as np
import xrloc.map.covisible as covisible

# from solver import prior_guided_pnp
from xrprimer.data_structure import VectorDouble
from xrprimer.ops import prior_guided_pnp

from xrloc.features.extractor import Extractor
from xrloc.map.reconstruction import Reconstruction
from xrloc.match.matcher import Matcher
from xrloc.retrieval.image_database import ImageDatabase
from xrloc.utils.miscs import head_logging, config_logging


class Localizer(object):
    """Hieracihcal Localization
    Args:
        config (dict): Configuration
    """
    default_config = {
        'local_feature': 'd2net',
        'global_feature': 'netvlad',
        'matcher': 'nn',
        'coarse': 'sr',
        'retrieval_num': 20,
        'scene_size': 20,
        'max_inlier': 100,
        'max_scene_num': 2
    }

    def __init__(self, map_path, config=default_config):
        head_logging('XRLocalization')
        self.config = config
        database_path = os.path.join(map_path, 'database.bin')
        if not os.path.exists(database_path):
            raise ValueError(
                'Not found database under map: {}'.format(map_path))
        self.reconstruction = Reconstruction(map_path)
        self.database = ImageDatabase(database_path)
        self.database.create()

        self.gextractor = Extractor(self.config['global_feature'])
        self.lextractor = Extractor(self.config['local_feature'])

        self.matcher = Matcher(self.config['matcher'])
        config_logging(self.config)
        head_logging('Init Success')

    def extract_features(self, image):
        """Extract local feature."""
        data = self.lextractor.extract(image)
        return data['keypoints'], data['descriptors']

    def coarse_localize(self, image):
        """Coarse localization phase."""
        image_feature = self.gextractor.extract(image)
        image_ids = self.database.retrieve(image_feature,
                                           self.config['retrieval_num'])

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

    def feature_match(self, query_points, query_point_descriptors,
                      train_points, train_point_descriptors, width, height):
        """Feature matching phase."""
        data = {
            'width': width,
            'height': height,
            '2d_points': query_points,
            '2d_descriptors': query_point_descriptors,
            '3d_points': train_points,
            '3d_descriptors': train_point_descriptors,
        }
        matches, priors = self.matcher.match(data)
        return matches, priors

    def prior_guided_pose_estimation(self, point2Ds, point3Ds, priors, camera):
        """Pose estimation phase."""
        point2Ds = point2Ds.astype('float32').copy()
        point3Ds = point3Ds.astype('float32').copy()
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

    def localize(self, image, camera):
        """Perform localization
        Args:
            image (np.array): RGB & WHC
            camera (tuple): (model, width, height, params)
        """
        width, height = camera[1], camera[2]
        point2ds_coordinates, point2d_descriptors = self.extract_features(
            image)
        logging.info('Local feature number: {0}'.format(
            point2ds_coordinates.shape[1]))

        scenes = self.coarse_localize(image)
        logging.info('Coarse location number: {0}'.format(len(scenes)))

        best_ret = {
            'ninlier': 0,
            'qvec': np.array([1, 0, 0, 0]),
            'tvec': np.array([0, 0, 0]),
            'mask': None
        }
        for i, image_ids in enumerate(scenes[:self.config['max_scene_num']]):
            point3d_ids = self.reconstruction.visible_points(image_ids)
            point3ds_coordinates = self.reconstruction.point3d_coordinates(
                point3d_ids)
            point3ds_descriptors = self.reconstruction.point3d_features(
                point3d_ids)
            logging.info('3d points size: {0}'.format(
                point3ds_descriptors.shape[1]))

            # Matching
            matches, priors = self.feature_match(point2ds_coordinates,
                                                 point2d_descriptors,
                                                 point3ds_coordinates,
                                                 point3ds_descriptors, width,
                                                 height)
            logging.info('Match number: {0}'.format(matches.shape[1]))

            # Pose estimation
            points2d = point2ds_coordinates[:, matches[0]]
            points3d = point3ds_coordinates[:, matches[1]]
            ret = self.prior_guided_pose_estimation(points2d, points3d, priors,
                                                    camera)
            logging.info('Inlier number: {0}'.format(ret['ninlier']))

            # Return
            if ret['ninlier'] > best_ret['ninlier']:
                best_ret = ret
            if best_ret['ninlier'] > self.config['max_inlier']:
                break
        return best_ret
