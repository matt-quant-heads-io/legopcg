import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .pod_cnn import PoDCNNModel
from .lego_model_3d import LegoModel3D
from .cmamae_model import CMAMAEModel
from .piecewise_model import LegoModelPiecewise

MODELS_MAP = {
    "podcnn_model": PoDCNNModel,
    "lego3d_model" : LegoModel3D,
    "cmamae_model": CMAMAEModel,
    "piecewise_model": LegoModelPiecewise
}