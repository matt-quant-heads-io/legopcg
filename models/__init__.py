import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .podcnn_model import PoDCNNModel
from .lego_model_3d import LegoModel3D
from .cmamae_model import CMAMAEModel

MODELS_MAP = {
    PoDCNNModel.get_trainer_id(): PoDCNNModel,
    "lego3d_model": LegoModel3D,
    "cmamae_model": CMAMAEModel,
}
