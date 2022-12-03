import os
import datetime as dt

from utils import write_curr_obs_to_dir_path



# load in trained model

# define environment instance
    # random starting obs (state) 
NUM_TRIALS = 10000
NUM_SUCCESS = NUM_FAILS = 0
GENERATED_MAPS_PATH = 'playable_maps'
ANIMATION_PATH = 'animations'


agent = torch.load('path_to_model')
env = LegoPCGEnv()
curr_obs = env.reset()

for trial in range(NUM_TRIALS):
    dir_path = f'{ANIMATION_PATH}/{dt.datetime.now()}'
    os.mkdir(dir_path)
    curr_step_num = 0
    while True:
        # TODO: need a function in utils that takes dir_path, 
        # reads all char maps in dir, and writes corresponding .dat files
        # to lego_structures creats gif animation of the trial
        write_curr_obs_to_dir_path(dir_path, curr_step_num)       
        
        action = agent.predict(curr_obs)
        curr_obs, _, is_finished, info = env.step(action) #TODO: impl info['solved']

        # For animating the agent's actions ovwr trial
        curr_step_num += 1

        if is_finished:
            if info['solved']:
                NUM_SUCCESS += 1
                str_map = get_string_map_from_onehot(curr_obs) #returns ["wall", ...]
                char_map = get_char_map_from_str(str_map) # see char_map_example.txt
                save_map(char_map, path)
            else:
                NUM_FAILS += 1
            curr_obs = env.reset()
            break
            


print(f"Percentage playable levels: {NUM_SUCCESS/NUM_TRIALS}")


