from abc import ABC, abstractmethod


class BaseTrajectoryGenerator(ABC):
    def __init__():
        pass

    @abstractmethod
    def get_traj_generator_id(self):
        raise NotImplementedError

    @staticmethod
    def get_traj_generator_id():
        raise NotImplementedError
