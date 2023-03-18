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
from order_mpd import get_model_obs, calculate_hamming_distance
import random

from decimal import Decimal as D


def gen_random_start_state():
    num_parts = 0
    obs = np.zeros((125))
    for idx, tile in enumerate(obs):
        model_size = random.choice([0,1,2])
        if model_size == 0:
            act = random.choice([0]*74 + [i for i in range(1,37)])
        elif model_size == 1:
            act = random.choice([0]*18 + [i for i in range(1,37)])
        else:
            act = random.choice([0] + [i for i in range(1,37)])
        if act != 0:
            num_parts += 1

        obs[idx] = act

    return obs.reshape((5,5,5)), num_parts




def get_closest_goal_map_path(paths, start_map):
    goals_obs = []

    for path in paths:
        print(f"path {path}")
        goal_obs = np.zeros(shape=(125,))
        with open(path, "r") as f:
            for idx, line in enumerate(f.readlines()):
                if line.endswith('\n'):
                    clean_line = line[:-1]
                else:
                    clean_line = line

                block_type = clean_line.split(' ')[-1]
                goal_obs[idx] = action_space[block_type]

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

    return min_dist, closest_goal.reshape((5,5,5)), paths[min_idx]


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
            block_type = rev_action_space[int(act)]
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


action_space = {'recte4.dat': 0, '3023.dat': 1, '44674.dat': 2, '3021.dat': 3, '51719.dat': 4, '4081a.dat': 5, '4589.dat': 6, '50947.dat': 7, '4600.dat': 8, '51011.dat': 9, '50951.dat': 10, '2432.dat': 11, '30603.dat': 12, '30027a.dat': 13, '3020.dat': 14, '2412b.dat': 15, '30602.dat': 16, '3795.dat': 17, '3710.dat': 18, '41854.dat': 19, '30028.dat': 20, '48183.dat': 21, '3839a.dat': 22, '2412a.dat': 23, '54200.dat': 24, '3022.dat': 25, '43719.dat': 26, '3031.dat': 27, '6141.dat': 28, '6015.dat': 29, '6014.dat': 30, '2540.dat': 31, '3938.dat': 32, '3034.dat': 33, '32028.dat': 34, '3937.dat': 35, '6157.dat': 36}
rev_action_space = {0: 'recte4.dat', 1: '3023.dat', 2: '44674.dat', 3: '3021.dat', 4: '51719.dat', 5: '4081a.dat', 6: '4589.dat', 7: '50947.dat', 8: '4600.dat', 9: '51011.dat', 10: '50951.dat', 11: '2432.dat', 12: '30603.dat', 13: '30027a.dat', 14: '3020.dat', 15: '2412b.dat', 16: '30602.dat', 17: '3795.dat', 18: '3710.dat', 19: '41854.dat', 20: '30028.dat', 21: '48183.dat', 22: '3839a.dat', 23: '2412a.dat', 24: '54200.dat', 25: '3022.dat', 26: '43719.dat', 27: '3031.dat', 28: '6141.dat', 29: '6015.dat', 30: '6014.dat', 31: '2540.dat', 32: '3938.dat', 33: '3034.dat', 34: '32028.dat', 35: '3937.dat', 36: '6157.dat'}
num_trials = 10
size = 10
crop_size = 10
pad = crop_size//2
pad_value = 0
agent = keras.models.load_model(f'/Users/matt/legopcg/saved_models/racers_obs_5.h5')

# action_space = {'recte4.dat': 0, '3023.dat': 1, '44674.dat': 2, '3021.dat': 3, '51719.dat': 4, '4081a.dat': 5, '4589.dat': 6, '50947.dat': 7, '4600.dat': 8, '51011.dat': 9, '50951.dat': 10, '2432.dat': 11, '30603.dat': 12, '30027a.dat': 13, '3020.dat': 14, '2412b.dat': 15, '30602.dat': 16, '3795.dat': 17, '3710.dat': 18, '41854.dat': 19, '30028.dat': 20, '48183.dat': 21, '3839a.dat': 22, '2412a.dat': 23, '54200.dat': 24, '3022.dat': 25, '43719.dat': 26, '3031.dat': 27, '6141.dat': 28, '6015.dat': 29, '6014.dat': 30, '2540.dat': 31, '3938.dat': 32, '3034.dat': 33, '32028.dat': 34, '3937.dat': 35, '6157.dat': 36}
# rev_action_space = {0: 'recte4.dat', 1: '3023.dat', 2: '44674.dat', 3: '3021.dat', 4: '51719.dat', 5: '4081a.dat', 6: '4589.dat', 7: '50947.dat', 8: '4600.dat', 9: '51011.dat', 10: '50951.dat', 11: '2432.dat', 12: '30603.dat', 13: '30027a.dat', 14: '3020.dat', 15: '2412b.dat', 16: '30602.dat', 17: '3795.dat', 18: '3710.dat', 19: '41854.dat', 20: '30028.dat', 21: '48183.dat', 22: '3839a.dat', 23: '2412a.dat', 24: '54200.dat', 25: '3022.dat', 26: '43719.dat', 27: '3031.dat', 28: '6141.dat', 29: '6015.dat', 30: '6014.dat', 31: '2540.dat', 32: '3938.dat', 33: '3034.dat', 34: '32028.dat', 35: '3937.dat', 36: '6157.dat'}

# action_space = {'recte4.dat': 0, '3023.dat': 1, '44674.dat': 2, '3021.dat': 3, '51719.dat': 4, '4081a.dat': 5, '4589.dat': 6, '50947.dat': 7, '4600.dat': 8, '51011.dat': 9, '50951.dat': 10, '2432.dat': 11, '30603.dat': 12, '30027a.dat': 13, '3020.dat': 14, '2412b.dat': 15, '30602.dat': 16, '3795.dat': 17, '3710.dat': 18, '41854.dat': 19, '30028.dat': 20, '48183.dat': 21, '3839a.dat': 22, '2412a.dat': 23, '54200.dat': 24, '3022.dat': 25, '43719.dat': 26, '3031.dat': 27, '6141.dat': 28, '6015.dat': 29, '6014.dat': 30, '2540.dat': 31, '3938.dat': 32, '3034.dat': 34, '32028.dat': 35, '3937.dat': 36, '6157.dat': 37}
# rev_action_space = {0: 'recte4.dat', 1: '3023.dat', 2: '44674.dat', 3: '3021.dat', 4: '51719.dat', 5: '4081a.dat', 6: '4589.dat', 7: '50947.dat', 8: '4600.dat', 9: '51011.dat', 10: '50951.dat', 11: '2432.dat', 12: '30603.dat', 13: '30027a.dat', 14: '3020.dat', 15: '2412b.dat', 16: '30602.dat', 17: '3795.dat', 18: '3710.dat', 19: '41854.dat', 20: '30028.dat', 21: '48183.dat', 22: '3839a.dat', 23: '2412a.dat', 24: '54200.dat', 25: '3022.dat', 26: '43719.dat', 27: '3031.dat', 28: '6141.dat', 29: '6015.dat', 30: '6014.dat', 31: '2540.dat', 32: '3938.dat', 34: '3034.dat', 35: '32028.dat', 36: '3937.dat', 37: '6157.dat'}



actions_map = {'recte4.dat': 0, '30602.dat': 1, '43719.dat': 2, '3021.dat': 3, '3710.dat': 4, '2412a.dat': 5, '44674.dat': 6, '4589.dat': 7, '4081a.dat': 8, '3795.dat': 9, '3022.dat': 10, '4600.dat': 11, '6014.dat': 12, '6015.dat': 13, '6157.dat': 14, '32028.dat': 15, '30027a.dat': 16, '30028.dat': 17, '51719.dat': 18, '51011.dat': 19, '50951.dat': 20, '2432.dat': 21, '48183.dat': 22, '2540.dat': 23, '3023.dat': 24, '50947.dat': 25, '3031.dat': 26, '3034.dat': 27, '54200.dat': 28, '6141.dat': 29, '3020.dat': 30, '30603.dat': 31, '41854.dat': 32, '3937.dat': 33, '3938.dat': 34, '2412b.dat': 35, '3839a.dat': 36}
rev_arev_action_spacections_map = {0: 'recte4.dat', 1: '30602.dat', 2: '43719.dat', 3: '3021.dat', 4: '3710.dat', 5: '2412a.dat', 6: '44674.dat', 7: '4589.dat', 8: '4081a.dat', 9: '3795.dat', 10: '3022.dat', 11: '4600.dat', 12: '6014.dat', 13: '6015.dat', 14: '6157.dat', 15: '32028.dat', 16: '30027a.dat', 17: '30028.dat', 18: '51719.dat', 19: '51011.dat', 20: '50951.dat', 21: '2432.dat', 22: '48183.dat', 23: '2540.dat', 24: '3023.dat', 25: '50947.dat', 26: '3031.dat', 27: '3034.dat', 28: '54200.dat', 29: '6141.dat', 30: '3020.dat', 31: '30603.dat', 32: '41854.dat', 33: '3937.dat', 34: '3938.dat', 35: '2412b.dat', 36: '3839a.dat'}

generated_output_path = "/Users/matt/legopcg/data/generated/racers"
racers_path = '/Users/matt/legopcg/data/goals/racers'
mpd_output_path_root = f"/Users/matt/legopcg/data/generated/racers_detailed_trajectories"
paths = []
for file in os.listdir(racers_path):
    paths.append(f"{racers_path}/{file}")



for num_parts_targs in [i for i in range(18, 28)]: 
    mpd_output_path_target_level = f"{mpd_output_path_root}/target_{num_parts_targs}_trials"
    if not os.path.exists(mpd_output_path_root):
        os.makedirs(mpd_output_path_root)

    for i in range(1, num_trials+1):
        start_map, curr_num_pars = gen_random_start_state()
        min_dist, _, mpd_path_to_use = get_closest_goal_map_path(paths, start_map)

        mpd_output_path_trial_level = f"{mpd_output_path_target_level}/trial_{i}"
        if not os.path.exists(mpd_output_path_trial_level):
            os.makedirs(mpd_output_path_trial_level)

        states = []
        x, y, z = 0, 0, 0
        for step in range(125):
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
            if x >= 5:
                x = 0
                z += 1
                if z >= 5:
                    z = 0
                    y += 1
                    if y >= 5:
                        y = 0

        states.append(start_map)
        potential_min_dist, _, potential_mpd_path_to_use = get_closest_goal_map_path(paths, start_map)
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
            block_type = rev_action_space[int(act)]
            file_lines[j] += f" {block_type}"


        clean_mpd = '\n'.join(file_lines)

        similarity = int(((27.0-min_dist) / 27.0)*100.0)
        mpd_abs_output_path = f"/Users/matt/legopcg/data/generated/racers/trial_{i}_target_{num_parts_targs}_actual_{curr_num_pars}_similarity_{similarity}.mpd"
        to_mpd(mpd_path_to_use, mpd_abs_output_path, start_map)
            


                
    #             file_lines = []
    #             clean_mpd = None


    #             new_line_for_mpds = []
    #             for idx, file_line in enumerate(file_lines):
    #                 new_line_for_mpd = ' '.join(file_line.split(' ')[:-1])
    #                 print(f"new_line_for_mpd: {new_line_for_mpd}")
    #                 new_line_for_mpd += f" {state_vals[idx]}"
    #                 new_line_for_mpds.append(new_line_for_mpd)



    #             print(new_line_for_mpds)





    # for file in os.listdir("/Users/matt/legopcg/data/generated/vintage_car"):
    #     path = f"/Users/matt/legopcg/data/generated/vintage_car/{file}"
    #     trial_num = file.split("_")[-1].split(".txt")[0]

    #     mpds_file = f"/Users/matt/legopcg/data/generated/vintage_car/trial_{trial_num}_mpds"
    #     os.mkdir(mpds_file)

    #     with open(path, "r") as fp:
    #         the_states = fp.readlines()
    #         for s_i, a_state in enumerate(the_states):
    #             new_state = a_state
    #             if '\n' in a_state:
    #                 new_state = a_state.replace("\n", "")

    #             state_vals = []
    #             for act in new_state:
    #                 block_type = rev_action_space[int(act)]
    #                 state_vals.append(block_type)


                
    #             file_lines = []
    #             clean_mpd = None
    #             with open("/Users/matt/legopcg/data/templates/cleaned_vintage_car_small.mpd", "r") as fx:
    #                 for line in  fx.readlines():
    #                     if not line.startswith('1') and not line.endswith('.dat') and not line.endswith('.ldr'):
    #                         continue

    #                     clean_line = ""
    #                     if line.endswith('\n'):
    #                         clean_line = line[:-1]
    #                     else:
    #                         clean_line = line

    #                     file_lines.append(clean_line)

    #             new_line_for_mpds = []
    #             for idx, file_line in enumerate(file_lines):
    #                 new_line_for_mpd = ' '.join(file_line.split(' ')[:-1])
    #                 print(f"new_line_for_mpd: {new_line_for_mpd}")
    #                 new_line_for_mpd += f" {state_vals[idx]}"
    #                 new_line_for_mpds.append(new_line_for_mpd)



    #             print(new_line_for_mpds)

    #             clean_mpd = '\n'.join(new_line_for_mpds)

    #             with open(f"{mpds_file}/state_{s_i}.mpd", "w") as fmpd:
    #                 fmpd.write(clean_mpd)


            


"""

TODO

/Users/matt/legopcg/data/generated/vintage_car/states_trail_1.txt contains state (structure) at each step.
    Need to take each row (i.e. state and plug into the .mpd file) output a final file
"""





