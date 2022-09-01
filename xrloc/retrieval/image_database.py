import pickle
import torch
import numpy as np


class ImageDatabase(object):
    """Image database: each image is represented by a fix-len vector.

    Args:
        path_to_database (str, optional): Path to image database
    """
    def __init__(self, path_to_database=None):
        self.feature_data = []
        self.index_to_image_id = []
        self.names = []
        self.is_built = False
        self.size = 0
        self.extractor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if path_to_database is not None:
            self.load_binary(path_to_database)

    def add_feature(self, image_feature, image_id, image_name=''):
        """Add image feature to image database that have not created
        Args:
            image_feature (torch.Tensor(float)): Embedding vector of image
            image_id (int): Id of image
            image_name (str): Image name
        """
        if self.is_built:
            raise ValueError('Image database have been built')
        self.feature_data.append(image_feature)
        self.index_to_image_id.append(image_id)
        self.names.append(image_name)
        self.size += 1

    def add_image(self, image, image_id, image_name=''):
        """Add image feature to image database that have not created
        Args:
            image (array): Opencv image
            image_id (int): Id of image
            image_name (str): Image name
        """
        if self.extractor is None:
            raise ValueError(
                'Image global feature extractor have not been set')
        if self.is_built:
            raise ValueError('Image database have been built')
        image_feature = self.extractor.extract(image)
        self.add_feature(image_feature, image_id, image_name)

    def set_image_extractor(self, extractor):
        """Set image extractor for image database
        Args:
            extractor (GlobalFeature): Image global feature extractor
        """
        self.extractor = extractor

    def create(self):
        """Create image database for image retrieval."""
        if self.is_built:
            raise ValueError('Image database have been built')
        if self.size == 0:
            raise ValueError('Image database is empty')
        self.feature_data = torch.from_numpy(np.array(
            self.feature_data)).squeeze(1).to(self.device)
        self.index_to_image_id = torch.from_numpy(
            np.array(self.index_to_image_id)).to(self.device)
        self.names = np.array(self.names)
        self.is_built = True

    def retrieve(self, image_feature, k, ret_name=False):
        """Retrieve top k similar images form image database
        Args:
            image_feature (np.array): Embedding vector of query image
            k (int): The number of retrieved image
            ret_name: Return image name (True) or image id (False)
        Returns:
            array[int]: The image ids of retrieved top k images
        """
        if not self.is_built:
            raise ValueError('Image database have not been built')
        if k == 0:
            return np.array([])
        if k < 0 or k > self.size:
            k = self.size
        image_feature = torch.from_numpy(image_feature).to(self.device)
        dvec = (-torch.matmul(self.feature_data, image_feature.t()) + 1) / 2.0
        idxs = torch.argsort(dvec)
        if ret_name:
            return self.names[idxs.cpu().numpy().astype(int)]
        else:
            image_ids = self.index_to_image_id[idxs[:k]]
            return image_ids.cpu().numpy()

    def image_retrieve(self, image, k, ret_name=False):
        """Retrieve top k similar images form image database
        Args:
            image (np.array): Query image
            k (int): The number of retrieved image
            ret_name: Return image name (True) or image id (False)
        Returns:
            array[int]: The image ids of retrieved top k images
        """
        if not self.is_built:
            raise ValueError('Image database have not been built')
        if self.extractor is None:
            raise ValueError(
                'Image global feature extractor have not been set')
        if k == 0:
            return np.array([])
        if k < 0 or k > self.size:
            k = self.size
        image_feature = self.extractor.extract(image)
        return self.retrieve(image_feature, k, ret_name)

    def save_binary(self, path_to_database):
        """Save image database
        Args:
            path_to_database (str): Path to image database
        """
        if self.is_built:
            raise ValueError('Image database have been built')
        data = {
            'features': self.feature_data,
            'index': self.index_to_image_id,
            'name': self.names,
            'size': self.size
        }
        # Always save
        # miscs.create_directory_if_not_exist(path_to_database)
        with open(path_to_database, 'wb') as file:
            pickle.dump(data, file, 2)

    def load_binary(self, path_to_database):
        """Load image database
        Args:
            path_to_database (str): Path to image database
        """
        with open(path_to_database, 'rb') as file:
            data = pickle.load(file)
        self.feature_data = data['features']
        self.index_to_image_id = data['index']
        if 'name' in data:  # Compatible with older versions
            self.names = data['name']
        self.size = data['size']
