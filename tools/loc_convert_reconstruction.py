import os
import h5py
import struct
import logging
import argparse
import collections

import numpy as np

from tqdm import tqdm
from xrloc.map.read_write_model import Image, Point3D
from xrloc.map.read_write_model import read_images_binary, read_points3d_binary
from xrloc.map.read_write_model import write_images_binary, write_points3d_binary
from recon_read_write_data import read_features_binary


class ConvertReconstruction2LocMap(object):
    def __init__(self, feature_file_path, reconstruction_path):

        image_bin_path = os.path.join(reconstruction_path, 'images.bin')
        point3d_bin_path = os.path.join(reconstruction_path, 'points3D.bin')
        self.images = read_images_binary(image_bin_path)
        self.point3Ds = read_points3d_binary(point3d_bin_path)
        self.id2image_names = dict(
            zip(self.images.keys(),
                [self.images[key].name for key in self.images.keys()]))
        if feature_file_path.endswith('.h5'):
            self.features = h5py.File(feature_file_path, 'r')
            self.cache_features()
        else:
            self.features = read_features_binary(feature_file_path)

    def cache_features(self):
        logging.info('Begin cache all descriptors ...')
        Descriptors = collections.namedtuple('Descriptors', ['descriptors'])
        tmp_feature = self.features
        self.features = {}
        for image_id, image_name in tqdm(self.id2image_names.items()):
            self.features[image_id] = \
                Descriptors(descriptors=tmp_feature[image_name]['descriptors'].__array__())

    def simplify_image_bin(self):
        self.simplify_image_id_to_map_table = {}
        self.simplify_images = {}
        for image_id in tqdm(self.images):
            image = self.images[image_id]
            new_indices = np.nonzero(image.point3D_ids != -1)[0]
            if len(new_indices) == 0: continue
            old_to_new_p2d_indices = dict(
                zip(new_indices, [i for i in range(len(new_indices))]))
            retain_xys = image.xys[image.point3D_ids != -1]
            retain_point3D_ids = image.point3D_ids[image.point3D_ids != -1]
            self.simplify_image_id_to_map_table[
                image.id] = old_to_new_p2d_indices
            self.simplify_images[image.id] = Image(
                id=image.id,
                qvec=image.qvec,
                tvec=image.tvec,
                camera_id=image.camera_id,
                name=image.name,
                xys=retain_xys,
                point3D_ids=retain_point3D_ids)

    def change_points3D_tracks(self):
        self.changed_point3ds = {}
        for point3d_id in tqdm(self.point3Ds):
            point3d = self.point3Ds[point3d_id]
            changed_point2D_idxs = []
            for i, image_id in enumerate(point3d.image_ids):
                changed_point2d_id = self.simplify_image_id_to_map_table[
                    image_id][point3d.point2D_idxs[i]]
                changed_point2D_idxs.append(changed_point2d_id)
            self.changed_point3ds[point3d_id] = Point3D(
                id=point3d_id,
                xyz=point3d.xyz,
                rgb=point3d.rgb,
                error=point3d.error,
                image_ids=point3d.image_ids,
                point2D_idxs=np.array(changed_point2D_idxs))

    def compute_mean_descriptor(self):
        # Get feature dim
        self.feature_dim = 0
        for image_id in self.features:
            self.feature_dim = self.features[image_id].descriptors.shape[0]
            if self.feature_dim != 0: break

        self.descriptors = {}
        for point3d_id in tqdm(self.point3Ds):
            point3d = self.point3Ds[point3d_id]
            mean = np.zeros([self.feature_dim])
            for i, image_id in enumerate(point3d.image_ids):
                point2d_id = point3d.point2D_idxs[i]
                descriptor = self.features[image_id].descriptors[:, point2d_id]
                mean += descriptor
            mean /= float(len(point3d.image_ids))
            mean /= np.linalg.norm(mean)
            self.descriptors[point3d_id] = mean

    def save_features_bin_file(self, path):
        with open(path, 'wb') as file:
            file.write(struct.pack('<Q', len(self.descriptors)))
            file.write(struct.pack('<Q', self.feature_dim))
            for point3D_id in tqdm(self.descriptors):
                file.write(struct.pack('<Q', point3D_id))
                file.write(struct.pack('<I', 1))
                descriptor = self.descriptors[point3D_id]
                dfmat = '<{0}f'.format(self.feature_dim)
                file.write(struct.pack(dfmat, *descriptor))

    def convert_to_mean_descriptor_map(self, path):
        if not os.path.exists(path): os.mkdir(path)
        logging.info('Compute mean descriptor ...')
        self.compute_mean_descriptor()
        logging.info('Simplify images bin ...')
        self.simplify_image_bin()
        logging.info('Change points3D track ...')
        self.change_points3D_tracks()
        logging.info('Save xrloc map ...')
        self.save_features_bin_file(os.path.join(path, 'features.bin'))
        write_images_binary(self.simplify_images,
                            os.path.join(path, 'images.bin'))
        write_points3d_binary(self.changed_point3ds,
                              os.path.join(path, 'points3D.bin'))


def main(feature_path, model_path, output_path):
    crm = ConvertReconstruction2LocMap(feature_path, model_path)
    crm.convert_to_mean_descriptor_map(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    main(args.feature_path, args.model_path, args.output_path)
