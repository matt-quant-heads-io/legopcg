from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    def __init__():
        pass

    @abstractmethod
    def get_trainer_id(self):
        raise NotImplementedError

    @staticmethod
    def get_trainer_id():
        return "podcnn_trainer"
