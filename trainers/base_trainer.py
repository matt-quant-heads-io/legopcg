from abc import ABC, abstractmethod
import json

from configs.config import Config
import configs


class BaseTrainer(ABC):
    """Abstract Trainer class that is inherited to all models"""

    def __init__(self, config_id):
        self.config = Config.from_json(configs.CONFIGS_MAP[config_id])

    @abstractmethod
    def get_id(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError
