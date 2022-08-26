import argparse
import os
from xrloc.features.extractor import Extractor
from xrloc.utils.image import read_image, show_image, write_image
from xrloc.utils.miscs import glob_images
from xrloc.utils.viz import draw_keypoint


def main(name, image_path, verb=True, save_path=''):
    image_paths = []
    if os.path.isfile(image_path):
        image_paths.append(image_path)
    else:
        image_paths = glob_images(image_path)
    if save_path != '':
        os.makedirs(save_path, exist_ok=True)
    feature = Extractor(name)
    index = 0
    for image_path in image_paths:
        image = read_image(image_path)
        # while True:
        res = feature.extract(image)
        image = draw_keypoint(image, res['keypoints'])
        if verb:
            show_image(name, image)
        if save_path != '':
            write_image(str(index) + '.jpg', image)
        index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--verb', action='store_true')
    parser.add_argument('--save_path', type=str, required=False, default='')
    parser.add_argument('--extractor',
                        type=str,
                        required=False,
                        default='d2net',
                        choices=list(Extractor.support_extractors))
    args = parser.parse_args()

    main(args.extractor, args.image_path, args.verb, args.save_path)
