import gym
from gym import spaces

from BasicRepresentation import BasicRepresentation

numOfBlockTypes = 3 # Think of a Generalized config file kinda soln laters
low = np.array([0,0,1]).astype(np.int8)
high = np.array([1,0,0]).astype(np.int8)

import numpy as np

class LegoPCGEnv(gym.Env):
    
    def __init__(self):

        # Length, width, height
        self._length = 10
        self._width = 10
        self._height = 10

        # Max changes allowed
        self._max_changes = 2
        
        # Define Observation Space
        self.observationSpace = spaces.Box(low=low, high=high, shape=(10,10,10))
        self._rep = BasicRepresentation()

        # Define Action Space
        self.actionSpace = spaces.Box(low,high)

        # Max Iterations
        self._maxIterations = self._max_changes * self._length * self._height * self._width

        # Initialize Number Of Iterations to zero
        self._numOfIterations = 0

    def reset(self):
        self._x = 0 
        self._y = 0
        self._z = 0
        self.observationSpace = spaces.Box(low=low, high=high, shape=(10,10,10))
        self._iteration = 0

        observation = self._rep.get_observation()
        # observation["heatmap"] = self._heatmap.copy()
        return observation

    def step(self, action):

        # Increment Number Of Iterations counter
        self.numOfIterations += 1
        
        # Save a copy of the old observation to calculate the reward
        previousObservation = self.observationSpace
        old_stats = self._rep_stats

        # Perform predicted Action
        # Update the current state to the new state based on the taken action
        change, x, y, z = self.update(action)

        # 
        if change > 0:
            self._changes += change
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))

        # Get observation?
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        
        # Compute Reward
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        
        # Determine if Episode is over or if changes > maximum changes or numOfIterations > 1000
        done = self._prob.get_episode_over(self._rep_stats,old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations # include this as well
        
        # Prepare debug information
        info = self._prob.get_debug_info(self._rep_stats,old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        
        # Return the values
        return observation, reward, done, info
    
    def update(self, action):
        change, x, y, z = self._rep.update(action)
        return change, x, y, z

    def step(self, action):
        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1.0
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        # calculate the values
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        done = self._prob.get_episode_over(self._rep_stats,old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info(self._rep_stats,old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        #return the values
        return observation, reward, done, info

        
