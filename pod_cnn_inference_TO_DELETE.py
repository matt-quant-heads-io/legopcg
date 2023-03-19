import os
import model
#from stable_baselines import PPO2

import time
# from utils import make_vec_envs
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from pod_data_generator_test_file_TO_DELETE import get_model_obs, calculate_hamming_distance
import random

from decimal import Decimal as D

actions_map = {'recte4.dat': 0, '30602.dat': 1, '43719.dat': 2, '3021.dat': 3, '3710.dat': 4, '2412a.dat': 5, '44674.dat': 6, '4589.dat': 7, '4081a.dat': 8, '3795.dat': 9, '3022.dat': 10, '4600.dat': 11, '6014.dat': 12, '6015.dat': 13, '6157.dat': 14, '32028.dat': 15, '30027a.dat': 16, '30028.dat': 17, '51719.dat': 18, '51011.dat': 19, '50951.dat': 20, '2432.dat': 21, '48183.dat': 22, '2540.dat': 23, '3023.dat': 24, '50947.dat': 25, '3031.dat': 26, '3034.dat': 27, '54200.dat': 28, '6141.dat': 29, '3020.dat': 30, '30603.dat': 31, '41854.dat': 32, '3937.dat': 33, '3938.dat': 34, '2412b.dat': 35, '3839a.dat': 36}
rev_actions_map = {0: 'recte4.dat', 1: '30602.dat', 2: '43719.dat', 3: '3021.dat', 4: '3710.dat', 5: '2412a.dat', 6: '44674.dat', 7: '4589.dat', 8: '4081a.dat', 9: '3795.dat', 10: '3022.dat', 11: '4600.dat', 12: '6014.dat', 13: '6015.dat', 14: '6157.dat', 15: '32028.dat', 16: '30027a.dat', 17: '30028.dat', 18: '51719.dat', 19: '51011.dat', 20: '50951.dat', 21: '2432.dat', 22: '48183.dat', 23: '2540.dat', 24: '3023.dat', 25: '50947.dat', 26: '3031.dat', 27: '3034.dat', 28: '54200.dat', 29: '6141.dat', 30: '3020.dat', 31: '30603.dat', 32: '41854.dat', 33: '3937.dat', 34: '3938.dat', 35: '2412b.dat', 36: '3839a.dat'}


# def gen_random_start_state(obs_size):
#     num_parts = 0
#     obs = np.zeros((obs_size**3))
#     for idx, tile in enumerate(obs):
#         model_size = random.choice([0,1,2])
#         if model_size == 0:
#             act = random.choice([0]*74 + [i for i in range(1,37)])
#         elif model_size == 1:
#             act = random.choice([0]*18 + [i for i in range(1,37)])
#         else:
#             act = random.choice([0] + [i for i in range(1,37)])
#         if act != 0:
#             num_parts += 1

#         obs[idx] = act

#     return obs.reshape((obs_size,obs_size,obs_size)), num_parts


def gen_random_start_state(obs_size):
    num_parts = 0
    obs = np.zeros((obs_size**3))
    for idx, tile in enumerate(obs):
        act = random.choice([0]*4 + [i for i in range(1,37)])
        if act != 0:
            num_parts += 1

        obs[idx] = act

    return obs.reshape((obs_size,obs_size,obs_size)), num_parts




def get_closest_goal_map_path(paths, start_map, obs_size):
    goals_obs = []

    for path in paths:
        print(f"path {path}")
        goal_obs = np.zeros(shape=(obs_size**3,))
        with open(path, "r") as f:
            for idx, line in enumerate(f.readlines()):
                if line.endswith('\n'):
                    clean_line = line[:-1]
                else:
                    clean_line = line

                block_type = clean_line.split(' ')[-1]
                goal_obs[idx] = actions_map[block_type]

        goals_obs.append(goal_obs)

    closest_goal = goals_obs[0]
    min_dist = math.inf
    min_idx = 0
    for idx, goal_obs in enumerate(goals_obs):
        dist = calculate_hamming_distance(goal_obs, start_map)
        if dist < min_dist:
            min_dist = dist
            closest_goal = goal_obs
            min_idx = idx

    return min_dist, closest_goal.reshape((obs_size,obs_size,obs_size)), paths[min_idx]


def to_mpd(mpd_template_filepath, mpd_abs_output_path, map):
        file_lines = []
        with open(mpd_template_filepath, "r") as fx:
            for line in  fx.readlines():
                clean_line = ""
                if line.endswith('\n'):
                    clean_line = line[:-1]
                else:
                    clean_line = line

                file_lines.append(' '.join(clean_line.split(' ')[:-1]))

        for j, act in enumerate(map.flatten()):
            block_type = rev_actions_map[int(act)]
            file_lines[j] += f" {block_type}"


        clean_mpd = '\n'.join(file_lines)

        with open(mpd_abs_output_path, "w") as f:
            f.write(clean_mpd)


def transform_to_oh(obs, pad, pad_value, size, x, y, z):
    ohs = []
    print(f"obs: {obs.shape}")
    obs = get_model_obs(obs, pad, pad_value, size, x, y, z)
    for i, val in enumerate(obs.flatten()):
        new_val = [0]*37
        new_val[int(val)] = 1
        ohs.append(new_val)
    return np.array(ohs).reshape((size,size,size,37))



num_trials = 10
obs_size = 3
size = obs_size*2
crop_size = size
pad = crop_size//2
pad_value = 0
agent = keras.models.load_model(f'{os.path.dirname(os.path.abspath(__file__))}/saved_models/racers_obs_5.h5')



generated_output_path = f"{os.path.dirname(os.path.abspath(__file__))}/data/generated/racers"
racers_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/goals/racers'
mpd_output_path_root = f"{os.path.dirname(os.path.abspath(__file__))}/data/generated/racers_detailed_trajectories"

paths = []
for file in os.listdir(racers_path):
    paths.append(f"{racers_path}/{file}")


for num_parts_targs in [i for i in range(18, 28)]: 
    mpd_output_path_target_level = f"{mpd_output_path_root}/target_{num_parts_targs}_trials"
    if not os.path.exists(mpd_output_path_root):
        os.makedirs(mpd_output_path_root)

    for i in range(1, num_trials+1):
        start_map, curr_num_pars = gen_random_start_state(obs_size)
        min_dist, _, mpd_path_to_use = get_closest_goal_map_path(paths, start_map, obs_size)

        mpd_output_path_trial_level = f"{mpd_output_path_target_level}/trial_{i}"
        if not os.path.exists(mpd_output_path_trial_level):
            os.makedirs(mpd_output_path_trial_level)

        states = []
        x, y, z = 0, 0, 0
        for step in range((obs_size**3)*2):
            mpd_output_path_step_level = f"{mpd_output_path_trial_level}/step_{step}.mpd"
            new_map = start_map.copy()
            map_oh = transform_to_oh(new_map, pad, pad_value, size, x, y, z)
            move_diff_oh = [0]*3
            if curr_num_pars < num_parts_targs:
                move_diff_idx = 2
            elif curr_num_pars == num_parts_targs:
                move_diff_idx = 1
            else:
                move_diff_idx = 0
            move_diff_oh[move_diff_idx] = 1
            
            act = np.argmax(agent.predict(x={'input_1':np.array([map_oh]), 'input_2':np.column_stack((move_diff_oh))}, steps=1))
            prior_to_change = new_map[y][x][z]

            if prior_to_change == 0 and act != 0:
                curr_num_pars += 1
            elif prior_to_change != 0 and act == 0:
                curr_num_pars -= 1

            new_map[y][x][z] = act
            

            start_map = new_map

            to_mpd(mpd_path_to_use, mpd_output_path_step_level, start_map)

            x += 1
            if x >= obs_size:
                x = 0
                z += 1
                if z >= obs_size:
                    z = 0
                    y += 1
                    if y >= obs_size:
                        y = 0

        states.append(start_map)
        potential_min_dist, _, potential_mpd_path_to_use = get_closest_goal_map_path(paths, start_map, obs_size)
        if potential_min_dist < min_dist:
            min_dist = potential_min_dist
            mpd_path_to_use = potential_mpd_path_to_use 



        file_lines = []
        with open(mpd_path_to_use, "r") as fx:
            for line in  fx.readlines():
                clean_line = ""
                if line.endswith('\n'):
                    clean_line = line[:-1]
                else:
                    clean_line = line

                file_lines.append(' '.join(clean_line.split(' ')[:-1]))


        for j, act in enumerate(start_map.flatten()):
            block_type = rev_actions_map[int(act)]
            file_lines[j] += f" {block_type}"


        clean_mpd = '\n'.join(file_lines)

        similarity = int(((float(obs_size**3)-min_dist) / float(obs_size**3))*100.0)
        mpd_abs_output_path = f"{os.path.dirname(os.path.abspath(__file__))}/data/generated/racers_detailed_trajectories/trial_{i}_target_{num_parts_targs}_actual_{curr_num_pars}_similarity_{similarity}.mpd"
        to_mpd(mpd_path_to_use, mpd_abs_output_path, start_map)
           


            


"""

TODO

/Users/matt/legopcg/data/generated/vintage_car/states_trail_1.txt contains state (structure) at each step.
    Need to take each row (i.e. state and plug into the .mpd file) output a final file
"""





