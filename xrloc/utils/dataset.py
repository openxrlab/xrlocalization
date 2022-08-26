import os
import numpy as np


def read_ios_logger_query_dataset(path, relative=True):
    """Read iOS logger query dataset
    Args:
        path (str): Path to query.txt
        relative (bool): If true, the name is relative path
    Returns:
        list[tuple]: Query data items with camera
    """
    results = []
    with open(path, 'r') as f:
        raw_data = f.readlines()
    for data in raw_data:
        data = data.strip('\n').split(' ')
        name, camera_model, width, height = data[:4]
        params = np.array(data[4:], float)
        camera = (camera_model, int(width), int(height), params)
        if relative:
            name = os.path.join(os.path.dirname(path), name)
        results.append((name, camera))
    return results
