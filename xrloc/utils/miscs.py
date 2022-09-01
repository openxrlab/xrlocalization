import os
import logging
import time
import subprocess

import numpy as np

from pathlib import Path
from functools import wraps


def create_directory_if_not_exist(path):
    """Create directory
    Args:
        path (str): If the dir of given path not exist,
            create it
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_parent_dir(file, level=1):
    """Get parent directory with given level
    Args:
        file (str): File path
        level (int): Up level
    Returns:
        str: Path to given level
    """
    if level <= 0:
        raise ValueError(
            'Level should be larger than 0, but given {}'.format(level))
    parent_dir = os.path.dirname(file)
    for i in range(level):
        parent_dir = os.path.join(parent_dir, '..')
    return os.path.abspath(parent_dir)


def is_image(path):
    """Is a regular image file."""
    name = os.path.basename(path).lower()
    postfixs = ['.bmp', '.png', '.jpg', '.jpeg']
    for pt in postfixs:
        if name.endswith(pt):
            return True
    return False


def glob_images(root, relative=False):
    """Find all image in a directory
    Args:
        root (str): Path to dir
    Returns:
        list[str]: All image paths
    """
    image_paths = []

    def __glob_images__(path, image_paths):
        if os.path.isdir(path):
            for file in os.listdir(path):
                next_path = os.path.join(path, file)
                __glob_images__(next_path, image_paths)
        elif is_image(path):
            image_paths.append(path)

    __glob_images__(root, image_paths)

    if relative:
        image_paths = [
            str(Path(path).relative_to(root)) for path in image_paths
        ]

    return sorted(image_paths)


def load_retrieval_pairs(path):
    """Load retrieval image pairs
    Args:
        path (str): Path to retrieval pair
    Returns:
        dict : return
    """
    ret = {}
    with open(path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        items = line.split(' ')
        query, dbname = items[0], items[1]
        if query in ret:
            ret[query].append(dbname)
        else:
            ret[query] = [dbname]
    return ret


def download_model(url, model_dir, model_name):
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        return
    print('Downloading the {} model from {}.'.format(model_name, url))
    command = ['wget', '--no-check-certificate', url, '-O', model_name]
    subprocess.run(command, check=True)
    os.makedirs(model_dir, exist_ok=True)
    command = ['mv', model_name, model_dir]
    subprocess.run(command, check=True)


def head_logging(info: str, width=50):
    logging.info('=' * width)
    left = '*' * int((width - len(info) - 2) / 2) + ' '
    right = ' ' + '*' * int((width - len(info) - 2) / 2)
    logging.info(left + info + right)
    logging.info('=' * width)


def config_logging(config: dict):
    max_len = max([len(key) for key in config.keys()])
    for key in config:
        val = config[key]
        key = str(key)
        logging.info('{0}{1}:{2}'.format(key, ' ' * (max_len - len(key)), val))


def qvec2rotmat(qvec):
    return np.array([[
        1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
        2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
        2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
    ],
                     [
                         2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
                         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
                     ],
                     [
                         2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
                     ]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [
        Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0
    ], [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def invert_pose(qvec, tvec):
    tvec = -np.dot(qvec2rotmat(qvec).transpose(), tvec)
    qvec = [qvec[0], -qvec[1], -qvec[2], -qvec[3]]
    return qvec, tvec


def count_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_time = end_time - start_time
        print('{0} time elapse (ms): {1:.1f}'.format(func.__name__,
                                                     duration_time * 1000))
        return result

    return wrapper
