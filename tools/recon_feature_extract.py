import os
import argparse
from tqdm import tqdm
from xrloc.utils.image import read_image
from xrloc.features.extractor import Extractor
from xrloc.map.read_write_model import read_images_binary
from recon_read_write_data import ImageLocalFeature, write_features_binary


def main(image_dir,
         image_bin_path,
         feature_bin_path,
         extractor_name='d2net'):
    """Create image database depend on images.bin
    Args:
        image_dir (str): Path to image directory
        image_bin_path (str): Path to images.bin for read
        feature_bin_path (str): Path to features.bin path for save
        extractor_name (str): Extractor name
    """
    if not os.path.exists(image_bin_path):
        raise ValueError('File not exist: {}'.format(image_bin_path))
    if extractor_name not in Extractor.support_extractors:
        raise ValueError('Not support extractor: {}'.format(extractor_name))

    images = read_images_binary(image_bin_path)
    model = Extractor(extractor_name)
    features = {}

    for image_id in tqdm(images):
        image = images[image_id]
        image_name = image.name
        image_path = os.path.join(image_dir, image_name)
        image = read_image(image_path)
        width, height = image.shape[1], image.shape[0]
        data = model.extract(image)

        features[image_id] = ImageLocalFeature(id=image_id,
                                               name=image_name,
                                               width=width,
                                               height=height,
                                               point2ds=data['keypoints'],
                                               descriptors=data['descriptors'])

    write_features_binary(features, feature_bin_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--image_bin_path', type=str, required=True)
    parser.add_argument('--feature_bin_path', type=str, required=True)
    parser.add_argument('--extractor',
                        type=str,
                        required=False,
                        default='d2net',
                        choices=list(Extractor.support_extractors))
    args = parser.parse_args()

    main(args.image_dir, args.image_bin_path, args.feature_bin_path,
         args.extractor)
