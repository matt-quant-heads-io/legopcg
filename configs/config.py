import json

class Config(dict):
    """Config class which contains data, train and model hyperparameters"""
    
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