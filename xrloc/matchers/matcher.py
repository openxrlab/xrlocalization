import inspect
import torch
from xrloc import matchers

"""Currently support matcher
"""
support_matchers = {
    'nn': {
        'ratio': False,
        'dis_thres': False,
        'cross_check': True
    },
    'gam': {
        'k': 3,
        'ratio': 0.7,
        'dis_thres': 0.9,
        'geo_prior': 0
    },
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = self.make_model(name)
        self.matcher = model(self.config)

    def match(self, query, train):
        """Perform feature matching
        Args:
            data (dict): Input data
        Returns:
            Dict: Match result
        """
        data = {}
        for k, v in query.items():
            if k == 'shape':
                data["query_"+k] = torch.from_numpy(v).to(self.device)
            else:
                data["query_"+k] = torch.from_numpy(v).float().to(self.device)
        
        for k, v in train.items():
            if k == 'shape':
                data["train_"+k] = torch.from_numpy(v).to(self.device)
            else:
                data["train_"+k] = torch.from_numpy(v).float().to(self.device)
        
        output =  self.matcher(data)
        output = {k: v.cpu().numpy() for k, v in output.items()}
        return output


    @staticmethod
    def make_model(name):
        """Make model class depend on given name
        Args:
            name (str): Model name
        Returns:
            Model Class
        """
        module_path = '{0}.{1}'.format(matchers.__name__, name)
        module = __import__(module_path, fromlist=[''])
        classes = inspect.getmembers(module, inspect.isclass)
        classes = [c for c in classes if c[1].__module__ == module_path]
        classes = [c[1] for c in classes if c[0].lower() == name.lower()]
        assert len(classes) == 1
        return classes[0]