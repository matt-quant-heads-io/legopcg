import json


class Config(dict):
    #TODO: when porting to control-pcgrl this will be hydra construct
    """Config class which contains data, train and model hyperparameters"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

    
    def __init__(self, data, train, model):
        super().__init__()
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg))
        return cls(params['data'], params['train'], params['model'])