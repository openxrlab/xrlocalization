import os
import torch
import argparse
from tqdm import tqdm
from xrloc.utils.image import read_image
from xrloc.features.extractor import Extractor
from xrloc.retrieval.image_database import ImageDatabase
from xrloc.utils.miscs import glob_images


def eliminate_duplicated_pairs(pairs):
    """Create image database depend on images.bin
    Args:
        pairs (list[pair]): Duplicated pair list
    Returns:
        list[pair]
    """
    pair_set, left_pairs = set(), list()
    for q, d in pairs:
        pair = q + d if q < d else d + q
        if pair not in pair_set:
            left_pairs.append((q, d))
            pair_set.add(pair)
    return left_pairs


def main(database_path,
         save_path,
         image_dir='',
         retrieve_num=20,
         keep_pairs=False,
         extractor_name='netvlad'):
    """Create image database depend on images.bin
    Args:
        image_dir (str): Path to image directory
        save_path (str): Path to save retrieval pairs results
        database_path (str): Path to database path for save
        retrieve_num (int): The number of retrieved image
        extractor_name (str): Extractor name
    """
    if extractor_name not in Extractor.support_extractors:
        raise ValueError('Not support extractor: {}'.format(extractor_name))

    database = ImageDatabase(database_path)
    database.create()

    model = Extractor(extractor_name)
    database.set_image_extractor(model)

    if image_dir != '':
        query = ImageDatabase()
        query.set_image_extractor(model)
        image_names = glob_images(image_dir, relative=True)
        image_names = [(i, name) for i, name in enumerate(image_names)]
        for i, image_name in tqdm(image_names):
            image = read_image(os.path.join(image_dir, image_name))
            query.add_image(image, i, image_name)
        query.create()
    else:
        query = database
        retrieve_num += 1

    retrieve_num = retrieve_num if retrieve_num < database.size else\
        database.size
    sim = torch.einsum('md,nd->mn', query.feature_data, database.feature_data)
    indices = torch.topk(sim, retrieve_num, dim=1).indices.cpu().numpy()

    pairs = []
    for i, query_name in enumerate(query.names):
        pairs += [(query_name, database.names[j]) for j in indices[i]
                  if query_name != database.names[j]]

    if not keep_pairs:
        pairs = eliminate_duplicated_pairs(pairs)

    print('Found {} image pairs'.format(len(pairs)))
    with open(save_path, 'w') as file:
        file.write('\n'.join(' '.join([q, d]) for q, d in pairs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=False, default='')
    parser.add_argument('--retrieve_num', type=int, required=False, default=20)
    parser.add_argument('--keep_pairs', action='store_true')
    parser.add_argument('--extractor',
                        type=str,
                        required=False,
                        default='netvlad',
                        choices=list(Extractor.support_extractors))
    args = parser.parse_args()

    main(args.database_path, args.save_path, args.image_dir, args.retrieve_num,
         args.keep_pairs, args.extractor)
