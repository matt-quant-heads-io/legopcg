import argparse
import os

import configs
import model as models
import utils.utils as ut

from gym_pcgrl.envs.reps.piecewise_lego_rep import PiecewiseRepresentation


config = configs.CONFIGS_MAP['piecewise_config']
model = models.MODELS_MAP['piecewise_model'](config)

model.test("/home/maria/dev/lego/legopcg/logs/test/")



