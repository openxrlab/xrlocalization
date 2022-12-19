import pickle
import numpy as np


class PairsDatabase(object):
    '''
    '''
    def __init__(self, path_to_database=None):
        self.query_name_to_db_names = {}
        self.load_pairs(path_to_database)
        # print(self.query_name_to_db_names)


    def image_retrieve(self, image_name, k):
        """Retrieve top k similar images form image database
        Args:
            image (np.array): Query image
            k (int): The number of retrieved image
        Returns:
            array[int]: The image ids of retrieved top k images
        """
        if image_name not in self.query_name_to_db_names:
            return []
        if k == 0:
            return []
        return self.query_name_to_db_names[image_name][:k]


    def load_pairs(self, path_to_database):
        """Load image database
        Args:
            path_to_database (str): Path to image database
        """
        with open(path_to_database, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line[0] == '#': continue

            query_name, db_name = line.strip().split(' ')
            if query_name in self.query_name_to_db_names:
                self.query_name_to_db_names[query_name].append(db_name)
            else:
                self.query_name_to_db_names[query_name] = [db_name]
