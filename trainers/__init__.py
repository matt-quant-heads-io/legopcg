import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .podcnn_trainer import PoDCNNTrainer

TRAINERS_MAP = {
    PoDCNNTrainer.get_id(): PoDCNNTrainer
}
