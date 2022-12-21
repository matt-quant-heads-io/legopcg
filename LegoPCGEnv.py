import gym
from gym import spaces

from BasicRepresentation import BasicRepresentation
from BasicProblem import BasicProblem
import numpy as np

import map_encodings

numOfBlockTypes = 3 
low = np.zeros(shape=(10,10,10))
high = np.ones(shape=(10,10,10))

import numpy as np

class LegoPCGEnv(gym.Env):
    
    def __init__(self):

        super(LegoPCGEnv, self).__init__()

        # Length, width, height
        self._length = 10
        self._width = 10
        self._height = 10
        self._changes = 0

        # Max changes allowed
        self._max_changes = 1
        
        # Define Observation Space
        self.observation_space = spaces.Box(low=0, high=3, shape=(10,10,10), dtype=np.float32)

        
        self._rep = BasicRepresentation()

        # Define Action Space
        self.action_space = spaces.Discrete(numOfBlockTypes + 1)

        # Max Iterations
        self._max_iterations = (self._max_changes * self._length * self._height * self._width) 
        
        # Initialize Number Of Iterations to zero
        self._numOfIterations = 0

        self._prob = BasicProblem()
        self._rep_stats = self._prob.get_stats(self._rep)
        
    def reset(self):
        self._rep.reset()
        self._rep_stats = self._prob.get_stats(self._rep)
        
        self._iteration = 0

        observation = self._rep.get_observation()['map']
        return observation

    def update(self, action):
        change, x, y, z = self._rep.update(action)
        return change, x, y, z

    def step(self, action):
        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats

        # update the current state to the new state based on the taken action
        change, x, y, z = self._rep.update(action)
        
        if change > 0:
            self._changes += change
            self._rep_stats = self._prob.get_stats(self._rep)

        # calculate the values
        observation = self._rep.get_observation()['map']
        # observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        
        done =  self._iteration >= self._max_iterations or self._prob.get_episode_over(self._rep_stats)
        info = self._prob.get_stats(self._rep)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        info['solved'] = self._prob.get_episode_over(self._rep_stats)
        
        return observation, reward, done, info

    def render(self, mode="lego"):
        pass

        
