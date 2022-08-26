import os
import argparse


def main(feature_bin_path, match_bin_path, camera_bin_path, output_path):
    """Create image database depend on images.bin
    Args:
        feature_bin_path (str): Path to features.bin for read
        match_bin_path (str): Path to matches.bin path for read
        camera_bin_path (str): Path to camera.bin path for read
        output_path (str):  Path to new reconstruction
    """
    if not os.path.exists(feature_bin_path):
        raise ValueError('File not exist: {}'.format(feature_bin_path))
    if not os.path.exists(match_bin_path):
        raise ValueError('File not exist: {}'.format(match_bin_path))
    if not os.path.exists(camera_bin_path):
        raise ValueError('File not exist: {}'.format(camera_bin_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cmd = 'recon_point_triangulate {} {} {} {}'.format(feature_bin_path,
                                                       match_bin_path,
                                                       camera_bin_path,
                                                       output_path)

    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_bin_path', type=str, required=True)
    parser.add_argument('--match_bin_path', type=str, required=True)
    parser.add_argument('--camera_bin_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    main(args.feature_bin_path, args.match_bin_path, args.camera_bin_path,
         args.output_path)
