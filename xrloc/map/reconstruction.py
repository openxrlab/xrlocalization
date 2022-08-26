# Copyright (c) OpenXRLab. All rights reserved.
import os

import numpy as np

from xrloc.map.read_write_model import read_images_binary
from xrloc.map.read_write_model import read_points3d_binary
from xrloc.map.read_write_model import read_point3d_feature_binary


class Reconstruction(object):
    """Reconstruction used as map for xrloc

    Args:
          map_path  (str) : Directory including images.bin, points3D.bin,
                features.bin
    """

    def __init__(self, map_path):
        images_bin_path = os.path.join(map_path, 'images.bin')
        if not os.path.exists(images_bin_path):
            raise ValueError("Invalid path: {}".format(images_bin_path))

        points3d_bin_path = os.path.join(map_path, 'points3D.bin')
        if not os.path.exists(points3d_bin_path):
            raise ValueError("Invalid path: {}".format(points3d_bin_path))

        features_bin_path = os.path.join(map_path, 'features.bin')
        if not os.path.exists(features_bin_path):
            raise ValueError("Invalid path: {}".format(features_bin_path))

        self.images = read_images_binary(images_bin_path)
        self.point3ds = read_points3d_binary(points3d_bin_path)
        self.features = read_point3d_feature_binary(features_bin_path)

    def covisible_images(self, image_id, num_covisble_point=1):
        """Get co-visible images

        Args:
            image_id (int): Image id
            num_covisble_point (int): The number of co-visible 3D point

        Returns:
            list[int]: Co-visible image ids
        """
        covisible_images_to_num_points = {}
        point3d_ids = self.images[image_id].point3D_ids
        for point3d_id in point3d_ids:
            if point3d_id == -1:
                continue
            image_ids = self.point3ds[point3d_id].image_ids
            for id in image_ids:
                if id in covisible_images_to_num_points:
                    covisible_images_to_num_points[id] += 1
                else:
                    covisible_images_to_num_points[id] = 1

        covisible_pairs = [(id, covisible_images_to_num_points[id])
                           for id in covisible_images_to_num_points]

        covisible_pairs = sorted(covisible_pairs,
                                 key=lambda k: k[1],
                                 reverse=True)

        image_ids = [
            id for id, num_point in covisible_pairs
            if num_point >= num_covisble_point and id != image_id
        ]

        return [image_id] + image_ids

    def visible_points(self, image_ids):
        """Get visible 3D point ids for given image id list

        Args:
            image_ids (list[int]): The image id list

        Returns:
            np.array(int64): The 3D point ids that are visible for
                input images
        """
        set_point3d_ids = set()
        for id in image_ids:
            point3d_ids = self.images[id].point3D_ids
            valid_point3d_ids = point3d_ids[point3d_ids != -1]
            set_point3d_ids.update(valid_point3d_ids)
        mp_point3d_ids = np.array(list(set_point3d_ids))
        return mp_point3d_ids

    def point3d_at(self, point3d_id):
        """Get 3D point coordinate

        Args:
            point3d_id (int): The 3D point id

        Returns:
            np.array(float): 3D point coordinate
        """
        return self.point3ds[point3d_id]

    def image_at(self, image_id):
        """Get image

        Args:
            image_id (int): The image id

        Returns:
            Image: The image with id == image_id
        """
        return self.images[image_id]

    def point3d_coordinates(self, point3d_ids):
        """Get the coordinates of multi 3D points

        Args:
            point3d_ids (array[int]): The point 3D ids

        Returns:
            np.array(float, 3*N): The coordinates
        """
        coordinates = np.array([
            self.point3ds[point3d_id].xyz for point3d_id in point3d_ids
        ]).transpose()
        return coordinates

    def point3d_features(self, point3d_ids):
        """Get the descriptors of multi 3D points

        Args:
            point3d_ids (array[int]): The point 3D ids

        Returns:
            np.array(float, dim*N): The descriptors
        """
        features = np.array([
            self.features[point3d_id] for point3d_id in point3d_ids
        ]).transpose()
        return features
