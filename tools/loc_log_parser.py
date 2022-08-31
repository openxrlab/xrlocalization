import argparse
import collections
import numpy as np

QImage = collections.namedtuple('QImage',
                                ['name', 'qvec', 'tvec', 'num_inlier'])


def extract_value(line):
    return line.split(':')[-1].strip()


def extract_values(line):
    return line.split(']')[-1].split(':')[-1].strip().split(' ')


def xrloc_log_parse(path_to_log):
    qimages = []
    with open(path_to_log, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if line.find('Image path') != -1:
            image = QImage(name='', qvec=None, tvec=None, num_inlier=0)
            image_path = extract_value(line)
            image = image._replace(name=image_path)
            qimages.append(image)
        elif line.find('Solved Pose') != -1:
            pose = [float(p) for p in extract_values(line)]
            qimages[-1] = qimages[-1]._replace(qvec=np.array(pose[0:4]),
                                               tvec=np.array(pose[4:7]))
        elif line.find('Inlier num') != -1:
            num_inlier = int(extract_value(line))
            qimages[-1] = qimages[-1]._replace(num_inlier=num_inlier)
    return qimages


def main(log_paths):
    for log_path in log_paths:
        qimages = xrloc_log_parse(log_path)
        # print(qimages)
        # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=str, nargs='+', required=True)
    args = parser.parse_args()
    main(args.logs)
