from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self):
        pass

    @abstractmethod
    def get_trainer_id(self):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError
