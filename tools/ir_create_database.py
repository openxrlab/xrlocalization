import os
import argparse
from tqdm import tqdm
from xrloc.utils.image import read_image
from xrloc.features.extractor import Extractor
from xrloc.retrieval.image_database import ImageDatabase
from xrloc.map.read_write_model import read_images_binary
from xrloc.utils.miscs import glob_images


def main(image_dir, database_path, image_bin_path='', extractor_name='netvlad'):
    """Create image database depend on images.bin
    Args:
        image_dir (str): Path to image directory
        database_path (str): Path to database path for save
        image_bin_path (str): Path to images.bin for read
        extractor_name (str): Extractor name
    """
    if extractor_name not in Extractor.support_extractors:
        raise ValueError('Not support extractor: {}'.format(extractor_name))

    database = ImageDatabase()
    model = Extractor(extractor_name)
    database.set_image_extractor(model)

    if image_bin_path != '':
        images = read_images_binary(image_bin_path)
        dataset = [(images[image_id].name, image_id) for image_id in images]
    else:
        names = glob_images(image_dir, relative=True)
        dataset = [(name, i) for i, name in enumerate(names)]

    for image_name, image_id in tqdm(dataset):
        image = read_image(os.path.join(image_dir, image_name))
        database.add_image(image, image_id, image_name)

    # if not os.path.exists(database_path):
    #     os.makedirs(database_path)
    # database.save_binary(os.path.join(database_path, 'database.bin'))
    database.save_binary(database_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--image_bin_path', type=str, required=False, default='')
    parser.add_argument('--extractor',
                        type=str,
                        required=False,
                        default='netvlad',
                        choices=list(Extractor.support_extractors))
    args = parser.parse_args()

    main(args.image_dir,  args.database_path, args.image_bin_path,
         args.extractor)
