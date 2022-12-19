import collections
import struct
import numpy as np

ImageLocalFeature = collections.namedtuple(
    'ImageLocalFeature',
    ['id', 'name', 'width', 'height', 'point2ds', 'descriptors'])

ImageFeatureMatch = collections.namedtuple(
    'ImageFeatureMatch', ['image_id1', 'image_id2', 'matches'])


def imagepair_to_pairid(image_id1, image_id2):
    """Image pair to pair id
    Args:
        image_id1:
        image_id2:
    Returns:
        pair id
    """
    if image_id1 > image_id2:
        return image_id2 * 2147483647 + image_id1
    else:
        return image_id1 * 2147483647 + image_id2


def pairid_to_imagepair(pair_id):
    """Image pair to pair id
    Args:
        pair_id
    Returns:
        image_id1, image_id2
    """
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return int(image_id1), int(image_id2)


def read_features_binary(path_to_image_features_bin):
    """Read features binary little endian
    Args:
        path_to_image_features_bin: Path to file
    Return:
        ImageLocalFeature dict
    """
    features = {}
    with open(path_to_image_features_bin, 'rb') as file:
        num_images = struct.unpack('<Q', file.read(8))[0]
        feature_dim = struct.unpack('<I', file.read(4))[0]
        for i in range(num_images):
            id = struct.unpack('<I', file.read(4))[0]
            name = ''
            ch = struct.unpack('<c', file.read(1))[0]
            while ch != b'\0':
                name += str(ch, encoding='utf-8')
                ch = struct.unpack('<c', file.read(1))[0]
            width = struct.unpack('<Q', file.read(8))[0]
            height = struct.unpack('<Q', file.read(8))[0]
            num_keypoints = struct.unpack('<Q', file.read(8))[0]
            point2ds = np.zeros((num_keypoints, 2))
            for j in range(num_keypoints):
                point2ds[j, 0] = struct.unpack('<d', file.read(8))[0]
                point2ds[j, 1] = struct.unpack('<d', file.read(8))[0]
            descriptors = np.zeros((feature_dim, num_keypoints))
            for j in range(num_keypoints):
                descriptors[:, j] = np.array(
                    struct.unpack('<{}f'.format(feature_dim),
                                  file.read(4 * feature_dim)))

            features[id] = ImageLocalFeature(id=id,
                                             name=name,
                                             width=width,
                                             height=height,
                                             point2ds=point2ds,
                                             descriptors=descriptors)
    return features


def write_features_binary(features, path_to_image_features_bin):
    """Write features binary little endian
    Args:
        features: ImageLocalFeature dict
        path_to_image_features_bin: Path to file
    """
    feature_dim = 0
    for image_id in features:
        feature_dim = features[image_id].descriptors.shape[0]
        if feature_dim != 0:
            break
    with open(path_to_image_features_bin, 'wb') as file:
        file.write(struct.pack('<Q', len(features)))
        file.write(struct.pack('<I', feature_dim))
        for image_id in features:
            feature = features[image_id]
            file.write(struct.pack('<I', feature.id))
            file.write(
                struct.pack('<' + str(len(feature.name)) + 's',
                            bytes(feature.name, encoding='utf-8')))
            file.write(struct.pack('<c', b'\0'))
            file.write(struct.pack('<Q', feature.width))
            file.write(struct.pack('<Q', feature.height))
            file.write(struct.pack('<Q', len(feature.point2ds)))
            for j in range(len(feature.point2ds)):
                file.write(struct.pack('<d', feature.point2ds[j, 0]))
                file.write(struct.pack('<d', feature.point2ds[j, 1]))
            for j in range(feature.descriptors.shape[1]):
                file.write(struct.pack('<{}f'.format(feature_dim),
                                       *feature.descriptors[:, j]))


def read_matches_binary(path_matches_bin):
    """Read matches binary little endian
    Args:
        path_matches_bin: Path to file
    Return:
        ImageFeatureMatch dict
    """
    feature_matches = {}
    with open(path_matches_bin, 'rb') as file:
        num_image_pairs = struct.unpack('<Q', file.read(8))[0]
        for i in range(num_image_pairs):
            image_id1 = struct.unpack('<I', file.read(4))[0]
            image_id2 = struct.unpack('<I', file.read(4))[0]
            num_matches = struct.unpack('<Q', file.read(8))[0]
            matches = []
            for j in range(num_matches):
                point_id1 = struct.unpack('<I', file.read(4))[0]
                point_id2 = struct.unpack('<I', file.read(4))[0]
                matches.append((point_id1, point_id2))
            pairid = imagepair_to_pairid(image_id1, image_id2)
            feature_matches[pairid] = ImageFeatureMatch(image_id1=image_id1,
                                                        image_id2=image_id2,
                                                        matches=matches)
    return feature_matches


def write_matches_binary(feature_matches, path_matches_bin):
    """Write matches binary little endian
    Args:
        feature_matches: ImageFeatureMatch dict
        path_matches_bin: Path to file
    """
    with open(path_matches_bin, 'wb') as file:
        num_image_pairs = len(feature_matches)
        file.write(struct.pack('<Q', num_image_pairs))
        for pairid in feature_matches:
            match = feature_matches[pairid]
            file.write(struct.pack('<I', match.image_id1))
            file.write(struct.pack('<I', match.image_id2))
            num_matches = match.matches.shape[1]
            file.write(struct.pack('<Q', num_matches))
            for i in range(num_matches):
                file.write(struct.pack('<I', match.matches[0, i]))
                file.write(struct.pack('<I', match.matches[1, i]))
