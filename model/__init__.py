import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .pod_cnn import PoDCNN
from .lego_model_3d import LegoModel3D

MODELS_MAP = {
    "podcnn_model": PoDCNN,
    "lego3d_model" : LegoModel3D
}