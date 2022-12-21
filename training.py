from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

from LegoPCGEnv import LegoPCGEnv
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: LegoPCGEnv()])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000)
model.save("legoAgent")
obs = env.reset()
