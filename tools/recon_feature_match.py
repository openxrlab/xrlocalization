import os
import argparse

import numpy as np
import cv2 as cv
from tqdm import tqdm
from xrloc.map.read_write_model import read_points3d_binary, read_images_binary

from xrloc.matchers.matcher import Matcher
from recon_read_write_data import read_features_binary
from recon_read_write_data import ImageFeatureMatch, write_matches_binary
from recon_read_write_data import pairid_to_imagepair, imagepair_to_pairid


def generate_image_pair_depend_recon(images, point3ds, image_pairs_num=5):
    """Generate image pair depend on points3D.bin
    Args:
        images (dict): images.bin
        point3ds (dict): points3D.bin
        covisible_number_thres (int): Image pairs with at least
            covisible_number_thres are used
    Returns:
        List: image pairs
    """
    covisible_image_ids = {}
    for image_id in images:
        covisible_images_to_num_points = {}
        point3D_ids = images[image_id].point3D_ids
        for point3D_id in point3D_ids:
            if point3D_id == -1:
                continue
            image_ids = point3ds[point3D_id].image_ids
            for id in image_ids:
                if id == image_id:
                    continue
                if id in covisible_images_to_num_points:
                    covisible_images_to_num_points[id] += 1
                else:
                    covisible_images_to_num_points[id] = 1
        covisible_pairs = [(id, covisible_images_to_num_points[id])
                           for id in covisible_images_to_num_points]
        covisible_pairs = sorted(covisible_pairs,
                                 key=lambda k: k[1],
                                 reverse=True)
        covisible_image_ids[image_id] = [
            id for id, num_point in covisible_pairs[:image_pairs_num]
        ]

    pairid_set = set()
    for image_id1 in covisible_image_ids:
        image_ids = covisible_image_ids[image_id1]
        for image_id2 in image_ids:
            pairid = imagepair_to_pairid(image_id1, image_id2)
            pairid_set.add(pairid)
    pairs = [pairid_to_imagepair(pair_id) for pair_id in pairid_set]
    return pairs


def feature_match(matcher, feat1, feat2):
    '''2D-2D feature match
    ''' 
    query = {
        'points': feat1.point2ds,
        'descs': feat1.descriptors,
        'scores': np.ones(len(feat1.point2ds)), #TODO
        'shape': np.array([feat1.height, feat1.width])
    }
    train = {
        'points': feat2.point2ds,
        'descs': feat2.descriptors,
        'scores': np.ones(len(feat2.point2ds)), #TODO
        'shape': np.array([feat2.height, feat2.width])
    }
    
    pred = matcher.match(query, train)
    return pred['matches'], pred['scores']


def filter_match_by_fundamental(keypoints1, keypoints2, matches, thres=12.):
    """Fundamental filter
    Args:
        keypoints1 (np.array, 2*N): image1 keypoints
        keypoints2 (np.array, 2*M): image2 keypoints
        matches (np.array, 2*K): matched keypoint index
        thres (float): error thres
    Returns:
        np.array (K): mask
    """
    cvmatch = [
        cv.DMatch(matches[0, i], matches[1, i], 1)
        for i in range(matches.shape[1])
    ]
    points1 = np.float32([keypoints1[m.queryIdx] for m in cvmatch])
    points2 = np.float32([keypoints2[m.trainIdx] for m in cvmatch])
    _, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, thres,
                                    0.99)
    return mask[:, 0]


def main(recon_path,
         feature_bin_path,
         match_bin_path,
         image_pairs_num=20,
         f_verify_thres=12):
    """Create image database depend on images.bin
    Args:
        recon_path (str): Path to recon dir including images.bin
        and points3D.bin
        feature_bin_path (str): Path to features.bin path for read
        match_bin_path (str): Path to matches.bin path for write
        covis: The number of co-visible 3D points
    """
    image_local_features = read_features_binary(feature_bin_path)

    images = read_images_binary(os.path.join(recon_path, 'images.bin'))
    point3ds = read_points3d_binary(os.path.join(recon_path, 'points3D.bin'))
    image_pairs = generate_image_pair_depend_recon(images, point3ds,
                                                   image_pairs_num)

    matcher = Matcher('nn')

    raw_image_feature_matches = {}
    for image_id1, image_id2 in tqdm(image_pairs):

        if image_id1 not in image_local_features or \
                image_id2 not in image_local_features:
            continue

        matches, sims = feature_match(matcher,
                                      image_local_features[image_id1],
                                      image_local_features[image_id2])

        if f_verify_thres > 0:
            mask = filter_match_by_fundamental(
                image_local_features[image_id1].point2ds,
                image_local_features[image_id2].point2ds, matches,
                f_verify_thres)
            if mask is None:
                continue
            matches, sims = matches[:, mask == 1], sims[mask == 1]

        if len(matches) == 0:
            continue

        pairid = imagepair_to_pairid(image_id1, image_id2)
        raw_image_feature_matches[pairid] = ImageFeatureMatch(
            image_id1=image_id1, image_id2=image_id2, matches=matches)

    write_matches_binary(raw_image_feature_matches, match_bin_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon_path', type=str, required=True)
    parser.add_argument('--feature_bin_path', type=str, required=True)
    parser.add_argument('--match_bin_path', type=str, required=True)
    parser.add_argument('--pair_num', type=int, required=False, default=20)
    args = parser.parse_args()

    main(args.recon_path, args.feature_bin_path, args.match_bin_path,
         args.pair_num)
