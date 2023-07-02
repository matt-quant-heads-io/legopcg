import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .podcnn_trainer_config import PODCNN_TRAINER_CONFIG
from .lego_3d_config import LEGO3D_CONFIG
from .cmamae_config import CMAMAE_CONFIG

# from .pod_trajectory_generator_config import POD_TRAJECTORY_GENERATOR_CONFIG


CONFIGS_MAP = {
    "podcnn_trainer": PODCNN_TRAINER_CONFIG,
    "lego3d_config": LEGO3D_CONFIG,
    "cmamae_config": CMAMAE_CONFIG,
    # "pod_trajectory_generator_config": POD_TRAJECTORY_GENERATOR_CONFIG
}
