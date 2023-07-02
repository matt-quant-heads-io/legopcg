import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .podcnn_dataloader import PoDCNNDataLoader

DATALOADERS_MAP = {PoDCNNDataLoader.get_trainer_id(): PoDCNNDataLoader}
