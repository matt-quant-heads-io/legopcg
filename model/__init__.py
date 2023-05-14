import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


from .lego_model_3d import LegoModel3D
from .piecewise_model import LegoModelPiecewise

MODELS_MAP = {
    "lego3d_model" : LegoModel3D,
    "piecewise_model": LegoModelPiecewise
}