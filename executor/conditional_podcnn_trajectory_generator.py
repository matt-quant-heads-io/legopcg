"""
PoDTrajectoryGenerator
    - This class generates training trajectories for PoD

"""
import os
import math
import random

import numpy as np
import pandas as pd

from configs.config import Config


class ConditionalPodCNNTrajectoryGenerator:
    def __init__(self, config):
        self.config = Config.from_json(config)
        self.action_space = self.config.data.action_space
        self.rev_action_space = self.config.data.rev_action_space
        self.goal_paths = self._get_goal_paths(self.config.data.goal_path)
        self.num_training_steps = self.config.data.num_training_steps

    def _get_goal_paths(self, goals_path):
        paths = []
        for file in os.listdir(goals_path):
            paths.append(f"{goals_path}/{file}")

        return paths

    def generate_trajectories(self):

        def gen_random_start_state():
            obs = np.zeros((27))
            for idx, tile in enumerate(obs):
                act = random.choice([0,0,0,0] + [i for i in range(1,40)])
                obs[idx] = act

            return obs.reshape((3,3,3))


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
                goal_obs = np.zeros(shape=(27,))
                with open(path, "r") as f:
                    for idx, line in enumerate(f.readlines()):
                        if line.endswith('\n'):
                            clean_line = line[:-1]
                        else:
                            clean_line = line

                        block_type = clean_line.split(' ')[-1]
                        goal_obs[idx] = self.action_space[block_type]

                goals_obs.append(goal_obs)

            closest_goal = goals_obs[0]
            min_dist = math.inf
            for goal_obs in goals_obs:
                dist = calculate_hamming_distance(goal_obs, start_map)
                if dist < min_dist:
                    min_dist = dist
                    closest_goal = goal_obs

            return closest_goal.reshape((3,3,3))

        def get_model_obs(map, pad, pad_value, size, x, y, z):
            padded = np.pad(map, pad, constant_values=pad_value)
            cropped = padded[y:y+size, x:x+size, z:z+size]
            obs = cropped

            return obs

        total_steps = 0
        state_act_dict = {f"col_{i}":[] for i in range(216)}
        state_act_dict["move_diff"] = []
        state_act_dict["target"] = []

        size = 6
        crop_size = 6
        pad = crop_size//2
        pad_value = 0



        
        while total_steps <= self.num_training_steps:
            x, y, z = 0, 0, 0

            start_map = gen_random_start_state()
            obs = get_closest_goal_map(self.goal_paths, start_map)

            step = 0
            while step <= 26:
                step += 1
                total_steps += 1
                print(f"total_steps: {total_steps}")
                repair_act = obs[y][x][z]
                state_act_dict["target"].append(int(repair_act))

                part_to_be_modified = obs[y][x][z]
                update_part = start_map[y][x][z]
                if part_to_be_modified == 0 and update_part != 0:
                    state_act_dict["move_diff"].append(1)
                elif update_part == 0 and part_to_be_modified != 0:
                    state_act_dict["move_diff"].append(-1)
                else:
                    state_act_dict["move_diff"].append(0)





                obs[y][x][z] = start_map[y][x][z]
                
                curr_obs = get_model_obs(obs, pad, pad_value, size, x, y, z)

                for j, val in enumerate(curr_obs.flatten()):
                    state_act_dict[f"col_{j}"].append(int(val))



                x += 1
                if x >= 3:
                    x = 0
                    z += 1
                    if z >= 3:
                        z = 0
                        y += 1
                        if y >= 3:
                            y = 0
        
        df = pd.DataFrame(state_act_dict)
        df.to_csv(f"/Users/matt/legopcg/data/trajectories/racers/step_{total_steps}.csv", index=False)




