import numpy as np

# from solver import prior_guided_pnp
from xrprimer.data_structure import VectorDouble
from xrprimer.ops import prior_guided_pnp


def test_pose_estimation():
    correspondences_num = 100
    point2ds = np.random.random([2, correspondences_num]).astype('float32')
    point3ds = np.random.random([3, correspondences_num]).astype('float32')
    priors = np.ones(correspondences_num).astype('float32')
    # params = np.array([1465., 1465., 955., 689.])
    params = VectorDouble([1465., 1465., 955., 689.])
    camera_config = {'model_name': 'PINHOLE', 'params': params}
    ransac_config = {
        'error_thres': 12,
        'inlier_ratio': 0.01,
        'confidence': 0.9999,
        'max_iter': 10000,
        'local_optimal': True
    }
    ret = prior_guided_pnp(point2ds, point3ds, priors, camera_config,
                           ransac_config)
    assert 'ninlier' in ret
    assert 'qvec' in ret
    assert 'tvec' in ret
    assert 'mask' in ret
    ret['ninlier'] == correspondences_num
