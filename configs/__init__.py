import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .pod_cnn_config import PODCNN_CONFIG

CONFIGS_MAP = {
    "podcnn_config": PODCNN_CONFIG
}

