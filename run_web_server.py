import cv2
import argparse
import numpy as np
from types import SimpleNamespace
import base64
import json
from flask import Flask
from flask import request

from xrloc.localizer import Localizer
from xrloc.utils.miscs import invert_pose

# loc_server should define localize_image interface, which
# take image and intrinsic as input, intrinsic is a dict like
# {
#     "model": 'PINHOLE',
#     "width": 640,
#     "height": 480,
#     "params": [384.35386606462447, 384.9560729180638,
#               319.28590839603237, 239.87334366520707] # [fx, fy, cx, cy]
# }
# and it return
# {
#     "qvec": list
#     "tvec": list
#     "ninlier": int
# }
loc_server = None

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return '<h1>Hello</h1>'


@app.route('/loc', methods=['POST'])
def loc():
    print('Begin loc')
    if type(request.json) is dict:
        j = request.json
    else:
        j = json.loads(request.json)
    print('Load done')
    print(j.keys())

    image = base64.b64decode(j['image'].encode('ascii'))
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)

    intrinsic = j['intrinsic']
    camera = (intrinsic['model'], intrinsic['width'], intrinsic['height'],
              intrinsic['params'])

    global loc_server
    
    ref_image_ids = loc_server.geo_localize(image)
    output = loc_server.refine_localize(image, camera, ref_image_ids)

    # pose in Twc
    # qvec should be [qw, qx, qy, qz]
    if output['ninlier'] > 0:
        output['qvec'], output['tvec'] = invert_pose(output['qvec'],
                                                     output['tvec'])

    ret = {
        'qvec': list(output['qvec']),
        'tvec': list(output['tvec']),
        'ninlier': output['ninlier']
    }

    return json.dumps(ret)


def dict_to_sns(d):
    d = SimpleNamespace(**d)
    for k in d.__dict__:
        q = d.__getattribute__(k)
        if type(d.__getattribute__(k)) is dict:
            d.__setattr__(k, dict_to_sns(d.__getattribute__(k)))
    return d


def main(map_path, port, json_path=None, host=None):
    """Run localization web server
    Args:
        map_path (str): Path to localization map
        port (int): Port
    """
    global loc_server
    if json_path is not None:
        with open(json_path, 'r') as file:
            config = json.load(file)
        loc_server = Localizer(map_path, config)
    else:
        loc_server = Localizer(map_path)

    app.run(port=port, host=host)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--json', type=str, required=False, default=None)
    parser.add_argument('--host', type=str, required=False, default='0.0.0.0')
    args = parser.parse_args()
    main(args.map_path, args.port, args.json, args.host)
