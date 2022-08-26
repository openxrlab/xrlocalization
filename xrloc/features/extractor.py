import inspect
import torch
import numpy as np
from xrloc import features
from xrloc.utils.image import image_resize
from xrloc.utils.image import convert_gray_image
from xrloc.utils.miscs import count_time


class Extractor:
    """Currently support extractor including local and global
    """
    support_extractors = {
        'd2net': {
            'image_size': 1600,
            'gray_image': False,
            'model': {
                'model_name': 'd2_tf.pth',
                'multiscale': False,
            },
        },
        'netvlad': {
            'image_size': 1024,
            'gray_image': False,
            'model': {
                'model_name': 'VGG16-NetVLAD-Pitts30K',
                'whiten': True,
            }
        },
    }
    """General extractor for local and global feature
    Args:
        name (str): The name of extractor supported in support_extractors
    """
    def __init__(self, name):
        if name not in self.support_extractors.keys():
            raise ValueError('Not support the extractor {}'. \
                             format(name))
        self.config = self.support_extractors[name]
        model = self.make_model(name)
        self.extractor = model(self.config['model'])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor.eval().to(self.device)

    # @count_time
    @torch.no_grad()
    def extract(self, image):
        """Extract feature for given image
        Args:
            image (np.array): Required RGB & HWC & Scalar[0-255] image
        Returns:
            Dict: Depends on specific extractor
        """
        image, factor = image_resize(image, self.config['image_size'])
        if self.config['gray_image']:
            image = convert_gray_image(image)
            image = image[None]
        else:
            if len(image.shape) == 2 or image.shape[-1] != 3:
                raise ValueError('Channel incorrect')
            image = image.transpose((2, 0, 1))  # HWC to CHW
        image = image.astype(np.float32) / 255.

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        data = self.extractor(image)
        data = {k: v[0].cpu().numpy() for k, v in data.items()}
        if 'keypoints' in data:
            data['keypoints'] = data['keypoints'].transpose()
            if factor != 1:
                data['keypoints'] = self.back_to_origin_size(
                    data['keypoints'], factor)
        if 'global_descriptor' in data:
            data = data['global_descriptor']
        return data

    @staticmethod
    def back_to_origin_size(keypoints, factor):
        """Back to origin size only for local feature
        Args:
            keypoints (array): Extracted keypoints
            factor: The value calculated at resize image
        Returns:
            Array: Resize keypoints
        """
        return (keypoints + .5) / factor - .5

    @staticmethod
    def make_model(name):
        """Make model class depend on given name
        Args:
            name (str): Model name
        Returns:
            Model Class
        """
        module_path = '{0}.{1}'.format(features.__name__, name)
        module = __import__(module_path, fromlist=[''])
        classes = inspect.getmembers(module, inspect.isclass)
        classes = [c for c in classes if c[1].__module__ == module_path]
        classes = [c[1] for c in classes if c[0].lower() == name.lower()]
        assert len(classes) == 1
        return classes[0]
