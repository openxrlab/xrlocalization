# Add new local feature extractor
In this part, we introduce how to add a new feature extractor to XRLocalization.
The XRLocalization has provided a global feature `netvlad` and a local feature
`d2net`. If you intend to use other feature like `superpoint` in your experiment,
a few steps should be taken as follow.

**0.** Clone the code and place it in the 3rdparty folder
```commandline
cd xrlocalization/3rdparty
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
```

**1.** Create a python file in `xrlocalization/xrloc/features`
```commandline
vim superpoint.py
```

**2.** Create a warpper class in `superpoint.py`

 ```python
import sys
import torch.nn as nn

from xrloc.utils.miscs import get_parent_dir

sys.path.append(get_parent_dir(__file__) + '/../3rdparty')
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint as SP


class SuperPoint(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config=default_config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.model = SP(self.config)

    def forward(self, image):
        input = {'image': image}
        return self.model(input)
```
Note that the lowercase form of the class name should be the same as the file name.

**3.** Register a default config in `xrlocalization/xrloc/features/extractor.py`

Add a default config in `support_extractors` like this.
```json
support_extractors = {
    //...
    'superpoint': {
        'image_size': 640,
        'gray_image': True,
        'model': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
        },
    },
}
```

Now, you can use `superpoint` in your experiments. Note that
`superpoint` is not permitted for commercial usage.
This project does not include `superpoint` code. If you use `superpoint`,
you need to download it by yourself, and please abide by its license.
