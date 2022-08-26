import logging
import time
import argparse
import json

from xrloc.localizer import Localizer
from xrloc.utils.dataset import read_ios_logger_query_dataset
from xrloc.utils.image import read_image


def main(query_path, map_path, json_path=None):
    dataset = read_ios_logger_query_dataset(query_path)
    logging.info('Dataset size: {}'.format(len(dataset)))

    if json_path is not None:
        with open(json_path, 'r') as file:
            config = json.load(file)
        loc = Localizer(map_path, config)
    else:
        loc = Localizer(map_path)

    beg_time = time.time()
    for i, (image_name, camera) in enumerate(dataset):
        logging.info(
            '======================[{0} / {1}]======================'.format(
                i + 1, len(dataset)))
        logging.info('Image path: {0}'.format(image_name))

        image = read_image(image_name)

        # Perform Loc
        ret = loc.localize(image, camera)

        pose = [str(p) for p in (list(ret['qvec']) + list(ret['tvec']))]
        logging.info('Solved Pose:' + ' '.join(pose))
        logging.info('Final inlier num:{0}'.format(ret['ninlier']))

    avg_time = (time.time() - beg_time) / len(dataset)
    logging.info('Log avg time: {0} ms'.format(avg_time))

    logging.info('End loc ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, required=True)
    parser.add_argument('--map_path', type=str, required=True)
    parser.add_argument('--json', type=str, required=False, default=None)
    args = parser.parse_args()
    main(args.query_path, args.map_path, args.json)
