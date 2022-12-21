import os, datetime
import map_encodings
import time
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

from utils import write_curr_obs_to_dir_path

from utils import createGIF

from LegoPCGEnv import LegoPCGEnv

NUM_TRIALS = 2
NUM_SUCCESS = NUM_FAILS = 0
GENERATED_MAPS_PATH = 'playable_maps'
ANIMATION_PATH = 'animations'

if os.path.isdir(ANIMATION_PATH):
    shutil.rmtree(ANIMATION_PATH)
os.mkdir(ANIMATION_PATH)

agent = PPO.load("legoAgent")
env = DummyVecEnv([lambda: LegoPCGEnv()])
curr_obs = env.reset()


for trial in range(NUM_TRIALS):
    time.sleep(2)
    timeStamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    dir_path = f'{ANIMATION_PATH}/{timeStamp}'
    os.mkdir(dir_path)
    curr_step_num = 0

    
    while True:
        write_curr_obs_to_dir_path(curr_obs[0], dir_path, curr_step_num)       
        
        action, _state = agent.predict(curr_obs)
        
        curr_obs, _, is_finished, info = env.step(action) 
        
        curr_step_num += 1
    
        if is_finished:
            
            if info[0]['solved']:
                NUM_SUCCESS += 1
                
            else:
                NUM_FAILS += 1
            curr_obs = env.reset()
            # print(curr_obs)
            break

createGIF()



