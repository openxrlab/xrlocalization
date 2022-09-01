from xrloc.match.geometry_aided import GeometryAidedMatcher
from xrloc.match.nearest_neighbor import NearestNeighborMatcher
"""Currently support matcher
"""
support_matchers = {
    'nn': {
        'ratio': False,
        'dis_thres': False,
        'cross_check': False
    },
    'nn+cross': {
        'ratio': False,
        'dis_thres': False,
        'cross_check': True
    },
    'nn+ratio+cross': {
        'ratio': 0.85,
        'dis_thres': False,
        'cross_check': True
    },
    'nn+ratio+distance+cross': {
        'ratio': 0.85,
        'dis_thres': 0.9,
        'cross_check': True
    },
    'gam': {
        'k': 3,
        'ratio': 0.7,
        'dis_thres': 0.9,
        'geo_prior': 0
    }
}


class Matcher(object):
    """General matcher
    Args:
        name (str): The name of matcher supported in support_extractors
    """
    def __init__(self, name):
        if name not in support_matchers.keys():
            raise ValueError('Not support the extractor {}'. \
                             format(name))
        self.config = support_matchers[name]
        if name.startswith('nn'):
            self.model = NearestNeighborMatcher(self.config)
        elif name == 'gam':
            self.model = GeometryAidedMatcher(self.config)

    def match(self, data):
        """Perform feature matching
        Args:
            data (dict): Input data
        Returns:
            Dict: Match result
        """
        return self.model(data)
