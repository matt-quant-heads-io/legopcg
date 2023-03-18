import numpy as np
import pandas as pd
import random
import os
from collections import defaultdict
import math


action_space = {'recte4.dat': 0, '3023.dat': 1, '44674.dat': 2, '3021.dat': 3, '51719.dat': 4, '4081a.dat': 5, '4589.dat': 6, '50947.dat': 7, '4600.dat': 8, '51011.dat': 9, '50951.dat': 10, '2432.dat': 11, '30603.dat': 12, '30027a.dat': 13, '3020.dat': 14, '2412b.dat': 15, '30602.dat': 16, '3795.dat': 17, '3710.dat': 18, '41854.dat': 19, '30028.dat': 20, '48183.dat': 21, '3839a.dat': 22, '2412a.dat': 23, '54200.dat': 24, '3022.dat': 25, '43719.dat': 26, '3031.dat': 27, '6141.dat': 28, '6015.dat': 29, '6014.dat': 30, '2540.dat': 31, '3938.dat': 32, '3034.dat': 33, '32028.dat': 34, '3937.dat': 35, '6157.dat': 36}
rev_action_space = {0: 'recte4.dat', 1: '3023.dat', 2: '44674.dat', 3: '3021.dat', 4: '51719.dat', 5: '4081a.dat', 6: '4589.dat', 7: '50947.dat', 8: '4600.dat', 9: '51011.dat', 10: '50951.dat', 11: '2432.dat', 12: '30603.dat', 13: '30027a.dat', 14: '3020.dat', 15: '2412b.dat', 16: '30602.dat', 17: '3795.dat', 18: '3710.dat', 19: '41854.dat', 20: '30028.dat', 21: '48183.dat', 22: '3839a.dat', 23: '2412a.dat', 24: '54200.dat', 25: '3022.dat', 26: '43719.dat', 27: '3031.dat', 28: '6141.dat', 29: '6015.dat', 30: '6014.dat', 31: '2540.dat', 32: '3938.dat', 33: '3034.dat', 34: '32028.dat', 35: '3937.dat', 36: '6157.dat'}


racers_path = '/Users/matt/legopcg/data/goals/racers'
paths = []
for file in os.listdir(racers_path):
    paths.append(f"{racers_path}/{file}")

def gen_random_start_state():
    num_parts = 0
    obs = np.zeros(shape=125,)
    for idx, tile in enumerate(obs):
        model_size = random.choice([0,1,2])
        if model_size == 0:
            act = random.choice([0]*74 + [i for i in range(1,37)])
        elif model_size == 1:
            act = random.choice([0]*18 + [i for i in range(1,37)])
        else:
            act = random.choice([0,0,0,0] + [i for i in range(1,37)])
        if act != 0:
            num_parts += 1

        obs[idx] = act

    return obs.reshape((5,5,5)), num_parts


def calculate_hamming_distance(goal_obs, start_map):
    dist = 0
    for idx, val in enumerate(start_map.flatten()):
        if val != goal_obs[idx]:
            dist += 1

    return dist

 
def get_closest_goal_map(paths, start_map):
    goals_obs = []

    for path in paths:
        print(f"path {path}")
        goal_obs = np.zeros(shape=125,)
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
    for goal_obs in goals_obs:
        dist = calculate_hamming_distance(goal_obs, start_map)
        if dist < min_dist:
            min_dist = dist
            closest_goal = goal_obs

    return closest_goal.reshape((5,5,5))

def get_model_obs(map, pad, pad_value, size, x, y, z):
    padded = np.pad(map, pad, constant_values=pad_value)
    cropped = padded[y:y+size, x:x+size, z:z+size]
    obs = cropped

    return obs


def extend_mpd_maps():
    racers_path = '/Users/matt/legopcg/data/goals/racers'
    paths = []
    for file in os.listdir(racers_path):
        paths.append(f"{racers_path}/{file}")


    
    distinct_blocks = []
    clean_mpd = None
    filler_block = "recte4.dat"
    for path in paths:
        file_lines = []
        with open(path, "r") as f:
            for idx, line in  enumerate(f.readlines()):
                clean_line = ""
                if line.endswith('\n'):
                    clean_line = line[:-1]
                else:
                    clean_line = line

                file_lines.append(clean_line)
        final_line = file_lines[-1]
        prefix = final_line.split(' ')[:len(final_line.split(' '))-1]
        
        prefix.append(filler_block)
        new_line = ' '.join(prefix)
        num_padding_lines = (125 - len(file_lines))
        for idx in range(num_padding_lines):
            file_lines.append(new_line)
        
        clean_mpd = '\n'.join(file_lines)
        with open(path, "w") as f:
            f.write(clean_mpd)

        

        print(prefix)

    print([[len(i) for i in file_lines]])



def get_action_maps():
    racers_path = '/Users/matt/legopcg/data/goals/racers'
    paths = []
    for file in os.listdir(racers_path):
        paths.append(f"{racers_path}/{file}")

    distinct_blocks = []
    idx = 1

    actions_map = defaultdict(int)
    rev_actions_map = defaultdict(str)
    actions_map["recte4.dat"] = 0
    rev_actions_map[0] = "recte4.dat"
    
    for path in paths:
        file_lines = []
        with open(path, "r") as f:
            for line in f.readlines():
                clean_line = ""
                if line.endswith('\n'):
                    clean_line = line[:-1]
                else:
                    clean_line = line

                act = clean_line.split(' ')[-1]
                if not act in actions_map:
                    print(f"found act: {act}")
                    actions_map[act] = idx
                    rev_actions_map[idx] = act

                    idx += 1

                    print(f"idx now {idx}")

    return actions_map, rev_actions_map




    







# def order_mpd(file_lines, path):
#     ordered_res = []
#     for line in file_lines:
#         line_vals = line.split(' ')
#         x, y, z = line_vals[2:5]
#         print(f"x: {x}, y: {y}, z: {z}")
#         ordered_res.append(((float(x),float(y),float(z)), line))

#     ordered_res.sort(key=lambda k: k[0][1])

#     clean_mpd = [k[1] for k in ordered_res]
#     clean_mpd_str = '\n'.join(clean_mpd)

#     with open(path, "w") as f:
#         f.write(clean_mpd_str)

#     obs = np.zeros((27))
#     # for idx, act in enumerate(clean_mpd):
#     #     obs[idx] = vc_brick_to_obs_val_map[act.split(' ')[-1]]

#     return np.array(obs).reshape((3,3,3))



    # with open(path, "w") as f:
    #     f.write(clean_mpd)

# def get_action_space_mosaic(path="/Users/matt/legopcg/data/goals/AndyWarholsMarilynMonroeA.mpd"):
#     file_lines = []
#     distinct_blocks = []
#     clean_mpd = None
#     with open(path, "r") as f:
#         for line in  f.readlines():
#             if not line.startswith('1') and not line.endswith('.dat') and not line.endswith('.ldr'):
#                 continue

#             clean_line = ""
#             if line.endswith('\n'):
#                 clean_line = line[:-1]
#             else:
#                 clean_line = line

#             file_lines.append(clean_line)

#             # get the color at end of line
#             block_type = clean_line.split(' ')[1]
#             print(f"block_type: {block_type}")
#             # if not block_type.endswith('.dat') and not block_type.endswith('.ldr'):
#             #     raise ValueError

#             distinct_blocks.append(block_type)

#         num_distinct_blocks = len(set(distinct_blocks))
#         print(f"num_distinct_actions (colors): {num_distinct_blocks}")
#         print(f"set(distinct_blocks): {set(distinct_blocks)}")
#         print(f"board size: {len(file_lines)}")
#         clean_mpd = '\n'.join(file_lines)
#         print(f"\n\n\n")
#         print(clean_mpd)

#     with open("/Users/matt/legopcg/data/templates/test_mosaic.mpd", "w") as f:
#         f.write(clean_mpd)

    
actions_map = {'recte4.dat': 0, '30602.dat': 1, '43719.dat': 2, '3021.dat': 3, '3710.dat': 4, '2412a.dat': 5, '44674.dat': 6, '4589.dat': 7, '4081a.dat': 8, '3795.dat': 9, '3022.dat': 10, '4600.dat': 11, '6014.dat': 12, '6015.dat': 13, '6157.dat': 14, '32028.dat': 15, '30027a.dat': 16, '30028.dat': 17, '51719.dat': 18, '51011.dat': 19, '50951.dat': 20, '2432.dat': 21, '48183.dat': 22, '2540.dat': 23, '3023.dat': 24, '50947.dat': 25, '3031.dat': 26, '3034.dat': 27, '54200.dat': 28, '6141.dat': 29, '3020.dat': 30, '30603.dat': 31, '41854.dat': 32, '3937.dat': 33, '3938.dat': 34, '2412b.dat': 35, '3839a.dat': 36}
rev_actions_map = {0: 'recte4.dat', 1: '30602.dat', 2: '43719.dat', 3: '3021.dat', 4: '3710.dat', 5: '2412a.dat', 6: '44674.dat', 7: '4589.dat', 8: '4081a.dat', 9: '3795.dat', 10: '3022.dat', 11: '4600.dat', 12: '6014.dat', 13: '6015.dat', 14: '6157.dat', 15: '32028.dat', 16: '30027a.dat', 17: '30028.dat', 18: '51719.dat', 19: '51011.dat', 20: '50951.dat', 21: '2432.dat', 22: '48183.dat', 23: '2540.dat', 24: '3023.dat', 25: '50947.dat', 26: '3031.dat', 27: '3034.dat', 28: '54200.dat', 29: '6141.dat', 30: '3020.dat', 31: '30603.dat', 32: '41854.dat', 33: '3937.dat', 34: '3938.dat', 35: '2412b.dat', 36: '3839a.dat'}



def main():
    # get_action_space_mosaic()
    extend_mpd_maps()
    actions_map, rev_actions_map = get_action_maps()



    total_steps = 0
    state_act_dict = {f"col_{i}":[] for i in range(1000)}
    state_act_dict["move_diff"] = []
    state_act_dict["target"] = []

    obs_size = 5
    size = 10
    crop_size = 10
    pad = crop_size//2
    pad_value = 0



    
    while total_steps <= 1000000:
        x, y, z = 0, 0, 0

        start_map, _ = gen_random_start_state()
        obs = get_closest_goal_map(paths, start_map)

        step = 0
        while step <= 124:
            step += 1
            total_steps += 1
            print(f"total_steps: {total_steps}")
            repair_act = obs[y][x][z]
            state_act_dict["target"].append(int(repair_act))

            part_to_be_modified = obs[y][x][z]
            update_part = start_map[y][x][z]
            if update_part == 0 and repair_act!= 0:
                state_act_dict["move_diff"].append(2)
            elif repair_act == 0 and update_part != 0:
                state_act_dict["move_diff"].append(0)
            else:
                state_act_dict["move_diff"].append(1)





            obs[y][x][z] = start_map[y][x][z]
            
            curr_obs = get_model_obs(obs, pad, pad_value, size, x, y, z)

            for j, val in enumerate(curr_obs.flatten()):
                state_act_dict[f"col_{j}"].append(int(val))



            x += 1
            if x >= obs_size:
                x = 0
                z += 1
                if z >= obs_size:
                    z = 0
                    y += 1
                    if y >= obs_size:
                        y = 0

        if total_steps % 100000 == 0:
            df = pd.DataFrame(state_act_dict)
            df.to_csv(f"/Users/matt/legopcg/data/trajectories/racers/step_{total_steps}.csv", index=False)
            state_act_dict = {f"col_{i}":[] for i in range(1000)}
            state_act_dict["move_diff"] = []
            state_act_dict["target"] = []


    df = pd.DataFrame(state_act_dict)
    df.to_csv(f"/Users/matt/legopcg/data/trajectories/racers/step_{total_steps}.csv", index=False)







"""
    1) Run generate pod trajectory
        a) take in an .mpd and map it to a matrix for the obs

        b) iterate thru the matrix, record the action, and the observation, write it to a .csv
        c) code up inference, given a starting state, can it rebuild the car?



"""



                

if __name__ == '__main__':
    main()