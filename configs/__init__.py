import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .pod_cnn_config import PODCNN_CONFIG
from .lego_3d_config import LEGO3D_CONFIG

CONFIGS_MAP = {
    "podcnn_config": PODCNN_CONFIG,
    "lego3d_config" : LEGO3D_CONFIG
}

